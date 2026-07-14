import json
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

POST_RESERVATION_LOOKUP_PROMPT = (
    "Most likely tool to call next: update_reservation_flights"
)

TOOL_PATH_PROMPT = """
Before taking action on the user's response, suggest optional tool-call trajectories
that explain which available tools should be used, and in what order, to solve the
task. Plan each trajectory forward from the current conversation state, rather than
from the beginning of the task. Take into consideration the domain policy, tool-call
dependencies, tools already called, and the results already available.

- Use the exact names of available tools and do not invent tools.
- If multiple viable tool sequences exist, suggest all of them.
- Do not call a tool in this response; only present the proposed tool trajectories.
- Each tool call should include tool name and arguments. If arguments are not known, use placeholders of <depends on previous tool call> or <requires user info>.

The tool calls already made are provided separately in chronological order:
<tool_calls_already_made_in_order>
{tool_calls_already_made}
</tool_calls_already_made_in_order>

Create a list where each entry contains an optional tool calls trajectory to solve the user request, the probability score of success for that trajectory, and any risks associated with that trajectory.
The probability score should be a float between 0 and 1, where 1 indicates a high likelihood of success and 0
indicates a low likelihood of success.
The risks should be a brief description of any potential issues or challenges that may arise when following that trajectory.
The dependencies is a dictionary were the keys are tools from the suggests trajectory, and the value of each tool is
a list of tools that must be called before it in order for it to succeed. If there are no dependencies, use an empty
list. If multiple tools depend on the same tool, for example tools B and C depend on tool A, and the output of tool A may require calling C before B than add tool C to the dependencies of tool B.
use the following format:
[
    {{
        "tools": ["tool_a", "tool_b", "tool_c"],
        "score": 0.9,
        "risks": "Dependencies between tool calls that can cause failure.",
        "dependencies": {{"tool_a": [], "tool_b": ["tool_a"], "tool_c": ["tool_a", "tool_b"]}}
    }},
    {{
        "tools": ["tool_b", "tool_c", "tool_a"],
        "score": 0.9,
        "risks": "Dependencies between tool calls that can cause failure.",
        "dependencies": {{"tool_a": [], "tool_b": ["tool_a"], "tool_c": ["tool_a", "tool_b"]}}
    }},
    {{
        "tools": ["tool_a", "tool_b", "tool_d"],
        "score": 0.9,
        "risks": "Dependencies between tool calls that can cause failure.",
        "dependencies": {{"tool_a": [], "tool_b": ["tool_a"], "tool_c": ["tool_a", "tool_b"]}}
    }}
]
""".strip()


class ToolPathAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    first_tool_calls_suggestions: Optional[list[dict]] = None
    high_trajectory_confidence: bool = True


ToolPathAgentStateType = TypeVar("ToolPathAgentStateType", bound="ToolPathAgentState")


class ToolPathAgent(
    LLMConfigMixin,
    HalfDuplexAgent[ToolPathAgentStateType],
    Generic[ToolPathAgentStateType],
):
    """A half-duplex agent that proposes tool paths for the user's request."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the ToolPathAgent.
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
    ) -> ToolPathAgentStateType:
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
        return ToolPathAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: ToolPathAgentStateType
    ) -> tuple[AssistantMessage, ToolPathAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_tools_trajectory(
        self, messages: list[APICompatibleMessage]
    ) -> list[dict]:
        """
        Generate the next message from a user or tool message.
        """
        tool_calls_already_made = [
            {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for message in messages
            if isinstance(message, AssistantMessage) and message.tool_calls
            for tool_call in message.tool_calls
        ]
        trajectory_prompt = TOOL_PATH_PROMPT.format(
            tool_calls_already_made=json.dumps(tool_calls_already_made, indent=2)
        )
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages
            + [SystemMessage(role="system", content=trajectory_prompt)],
            call_name="tool_path_response",
            **self.llm_args,
        )
        if assistant_message.content is None:
            raise ValueError("Tool trajectory response must contain JSON content.")
        try:
            trajectories = json.loads(assistant_message.content)
        except json.JSONDecodeError as error:
            raise ValueError("Tool trajectory response must be valid JSON.") from error
        if not isinstance(trajectories, list) or not all(
            isinstance(trajectory, dict) for trajectory in trajectories
        ):
            raise ValueError("Tool trajectory response must be a JSON list of objects.")
        return trajectories

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: ToolPathAgentStateType
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
        messages = state.system_messages + state.messages
        optional_tool_trajectories = self._generate_tools_trajectory(messages)
        if state.first_tool_calls_suggestions is None:
            state.first_tool_calls_suggestions = optional_tool_trajectories
            if len(optional_tool_trajectories) > 1:
                state.high_trajectory_confidence = False
            print("Tool Path Trajectory Suggestion:")
            print(json.dumps(optional_tool_trajectories, indent=2))
        called_tool_names = {
            tool_call.name
            for history_message in state.messages
            if isinstance(history_message, AssistantMessage)
            and history_message.tool_calls
            for tool_call in history_message.tool_calls
        }
        get_reservation_details_called = "get_reservation_details" in called_tool_names
        reservation_update_already_called = bool(
            {
                "update_reservation_flights",
                "update_reservation_passengers",
            }
            & called_tool_names
        )
        agent_messages = list(messages)
        if get_reservation_details_called and not reservation_update_already_called:
            agent_messages.append(
                SystemMessage(role="system", content=POST_RESERVATION_LOOKUP_PROMPT)
            )
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=agent_messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message


# =============================================================================
# AGENT FACTORY FUNCTIONS
# =============================================================================


def create_tool_path_agent(tools, domain_policy, **kwargs):
    """Factory function for ToolPathAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return ToolPathAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
