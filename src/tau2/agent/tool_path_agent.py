from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel

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
from tau2.utils.llm_utils import generate

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
Do not ask the user for extra disambiguation if you can use tools to get the information you need.
</policy>
""".strip()

TOOL_PATH_PROMPT = """"""

class PolicyInjectionAgentState(BaseModel):

    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


MemoryAgentStateType = TypeVar("MemoryAgentStateType", bound="PolicyInjectionAgentState")


class PolicyInjectionAgent(
    LLMConfigMixin, HalfDuplexAgent[MemoryAgentStateType], Generic[MemoryAgentStateType]
):
    """
    A half-duplex agent for turn-based conversations. The agent injects before each agent turn the most
    relevant policy rules from the domain policy.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the PolicyInjectionAgent.
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
    ) -> MemoryAgentStateType:
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
        return PolicyInjectionAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: MemoryAgentStateType
    ) -> tuple[AssistantMessage, MemoryAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _format_context(self, messages: list[APICompatibleMessage]) -> str:
        return "\n\n".join(
            f"<message index=\"{idx}\">\n{message}\n</message>"
            for idx, message in enumerate(messages)
        )

    def _generate_policy_rules(self, state: list[APICompatibleMessage]) -> str:
        messages = (state.system_messages + state.messages
                     + [SystemMessage(role="system", content=POLICY_RULES_INSTRUCTION)])
        assistant_message = generate(
            model=self.llm,
            tools=[],
            messages=messages,
            call_name="policy_rules",
            **self.llm_args,
        )
        return "Critical policy rules that must be followed exactly:\n" + assistant_message.content.strip()

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: MemoryAgentStateType
    ) -> AssistantMessage:
        """
        Generate the next message from a user or tool message.
        """
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = (state.system_messages + state.messages)
        policy_rules = self._generate_policy_rules(state)
        messages += [SystemMessage(role="system", content=policy_rules)]
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message


# =============================================================================
# AGENT FACTORY FUNCTIONS
# =============================================================================


def create_policy_injection_agent(tools, domain_policy, **kwargs):
    """Factory function for MemoryAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return PolicyInjectionAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
