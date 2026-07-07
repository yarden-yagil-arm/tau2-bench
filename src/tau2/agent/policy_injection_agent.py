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

"""
Research Notes:
1. Adding relevant policy in each agent turn improves results by 10%, tested on airline domain over 10 trails.
2. seems like it helps in medium tasks that the agent sometimes succeed, but not in hard tasks that usually fails (
e.g task 15: ~6 passed trails ~8 ->, vs. task 17: ~0 passed trails -> ~0).
3. adding to task 17 the specific required policy with rephrasing helps a lot, 9 passed trails.
"""

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

POLICY_RULES_INSTRUCTION = """
Your goal is to select the most relevant policy rules from the policy in order to help the agent following the 
policy, since it is very long. Given the conversation between the agent and user, which contains the policy, 
the user request and the state of 
the reservation, 
you should select the most relevant 
policy rules that are important for the to agent remember before its next action. Select up to 6 policy rules. Try to be concise, 
select only most relevant policy rules, 
you can summerize them and use 
a more strict policy phrasing.
Do not add any additional information or explanation.
""".strip()

"""
Task 17:
Critical policy rules that must be followed exactly:
# "1. If the user is changing cabin only and keeping the same flights, the agent must treat the change as allowed for
# all reservations, including basic economy.
# 2. Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed
# 3. The agent should ask which eligible payment method to use before updating reservation
# 4. Only one tool call can be made at each turn
# 5. 3 free checked bag for each economy passenger
# 6. Do not ask the user for extra disambiguation if you can use tools to get the information you need


original policy phrasing:

Critical policy rules that must be followed exactly:
1. Cabin cannot be changed if any flight in the reservation has already been flown.
In other cases, all reservations, including basic economy, can change cabin without changing the flights.
2. Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the actio
n details and obtain explicit user confirmation (yes) to proceed
3. If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons
4. You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time
5. If the booking user is a gold member:
2 free checked bag for each basic economy passenger
3 free checked bag for each economy passenger
6. If the user doesn't know their reservation id, the agent should help locate it using available tools
"""

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
