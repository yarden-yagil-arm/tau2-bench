import json
from typing import Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
Make sure to use the latest state of the reservation in case updates have been made when following the policy.
""".strip()


SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
Do not ask the user for extra disambiguation if you can use tools to get the information you need.
""".strip()


STATE_JSON_STRUCTURE = """
{
  "user_state": {
    "request": "",
    "tasks_to_complete": [],
    "data_provided_by_user": {},
    "data_provided_by_tools": {},
    "entities": [],
    "facts": []
  },
  "policy_state": {
    "relevant_rules": [],
    "constraints": [],
    "required_checks": []
  },
  "conversation_summary": ""
}
""".strip()


STATE_SUMMARY_INSTRUCTION = """
You are a dialogue state tracker for a tool-using task-oriented agent.
Update the previous JSON state using the new messages and make sure the state is up to date.

The state should behave like a compact belief state:
- Preserve stable facts until the user or a tool result change them.
- Keep user-provided values separate from tool-observed facts.
- Keep only tool output fields that are relevant for deciding the next agent actions.
- Track policy rules that constrain or enable the active request.

<domain_policy>
{domain_policy}
</domain_policy>

<previous_state_json>
{state_json_structure}
</previous_state_json>
""".strip()



def _empty_state_json() -> str:
    return STATE_JSON_STRUCTURE


class SummaryAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    state_json: str = Field(default_factory=_empty_state_json)
    last_state_message_index: int = 0

SummaryAgentStateType = TypeVar("SummaryAgentStateType", bound="SummaryAgentState")


class SummaryAgent(
    LLMConfigMixin, HalfDuplexAgent[SummaryAgentStateType], Generic[SummaryAgentStateType]
):
    """
    A half-duplex LLM agent for turn-based conversations.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the SummaryAgent.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> SummaryAgentStateType:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return SummaryAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SummaryAgentStateType
    ) -> tuple[AssistantMessage, SummaryAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: SummaryAgentStateType
    ) -> AssistantMessage:
        """
        Generate the next message from a user or tool message.
        """
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceSummaryAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        state_json = self._generate_state_summary(state, state.state_json)
        messages += [
            SystemMessage(
                role="system",
                content=f"Current conversation state:\n {state_json}\nUse the state as helper to make better next step "
                        f"decision.",
            )
        ]
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message

    def _generate_state_summary(
        self,
        state: SummaryAgentStateType,
        last_state_json: str,
    ) -> str:
        """
        Update the compact JSON state from the messages not yet reflected in it.
        """
        new_messages = state.messages[state.last_state_message_index :]
        if not new_messages:
            return last_state_json

        messages = [
            SystemMessage(
                role="system",
                content=STATE_SUMMARY_INSTRUCTION.format(
                    domain_policy=self.domain_policy,
                    state_json_structure=last_state_json,
                ),
            ),
        ]
        assistant_message = generate(
            model=self.llm,
            tools=[],
            messages=messages,
            call_name="state_summary",
            **self.llm_args,
        )
        updated_state_json = assistant_message.content

        state.state_json = updated_state_json
        state.last_state_message_index = len(state.messages)
        return updated_state_json

    def _format_messages_for_state_update(
        self,
        messages: list[APICompatibleMessage],
        start_index: int = 0,
    ) -> str:
        return "\n\n".join(
            f"<message index=\"{idx}\" role=\"{message.role}\">\n{message}\n</message>"
            for idx, message in enumerate(messages, start=start_index)
        )


# =============================================================================
# AGENT FACTORY FUNCTIONS
# =============================================================================


def create_llm_agent(tools, domain_policy, **kwargs):
    """Factory function for SummaryAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return SummaryAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
