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
The tools is a list of the tools in the trajectory, in the order they should be called. Each tool should be represented as a dictionary with the following keys:
- "tool": The name of the tool.
- "goal": A brief description of what the tool is trying to achieve.
The probability score should be a float between 0 and 1, where 1 indicates a high likelihood of success and 0
indicates a low likelihood of success.
The risks should be a brief description of any potential issues or challenges that may arise when following that trajectory.
The dependencies is a dictionary were the keys are tools from the suggests trajectory, and the value of each tool is
a list of tools that must be called before it in order for it to succeed. If 
there are no dependencies, use an empty
list. If multiple tools depend on the same tool, for example tools B and C depend on tool A, and the output of tool A may require calling C before B than add tool C to the dependencies of tool B.
use the following format:
[
    {{
        "tools": [
            {{"tool": "tool_a", "goal": "What tool_a is trying to achieve."}},
            {{"tool": "tool_b", "goal": "What tool_b is trying to achieve."}},
            {{"tool": "tool_c", "goal": "What tool_c is trying to achieve."}}
        ],
        "score": 0.9,
        "risks": "Dependencies between tool calls that can cause failure.",
        "dependencies": {{"tool_a": [], "tool_b": ["tool_a"], "tool_c": ["tool_a", "tool_b"]}}
    }},
    {{
        "tools": [
            {{"tool": "tool_b", "goal": "What tool_b is trying to achieve."}},
            {{"tool": "tool_c", "goal": "What tool_c is trying to achieve."}},
            {{"tool": "tool_a", "goal": "What tool_a is trying to achieve."}}
        ],
        "score": 0.9,
        "risks": "Dependencies between tool calls that can cause failure.",
        "dependencies": {{"tool_a": [], "tool_b": ["tool_a"], "tool_c": ["tool_a", "tool_b"]}}
    }},
    {{
        "tools": [
            {{"tool": "tool_a", "goal": "What tool_a is trying to achieve."}},
            {{"tool": "tool_b", "goal": "What tool_b is trying to achieve."}},
            {{"tool": "tool_d", "goal": "What tool_d is trying to achieve."}}
        ],
        "score": 0.8,
        "risks": "Dependencies between tool calls that can cause failure.",
        "dependencies": {{"tool_a": [], "tool_b": ["tool_a"], "tool_c": ["tool_a", "tool_b"]}}
    }}
]
""".strip()


REQUIRED_TOOL_NAMES_TASK_17 = [
    "get_user_details", "get_reservation_details", "get_reservation_details", "get_reservation_details",
        "get_reservation_details", "update_reservation_flights",
        "update_reservation_passengers", "update_reservation_baggages",
]


class ToolPathAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    required_tool_calls: list[str]
    first_tool_calls_suggestions: Optional[list[dict]] = None
    current_trajectory_suggestion: Optional[dict] = None
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
            required_tool_calls=[],
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
        for trajectory in trajectories:
            for tool in trajectory.get("tools", []):
                if isinstance(tool, dict) and isinstance(tool.get("tool"), str):
                    tool["tool"] = tool["tool"].removeprefix("functions.")
        return trajectories

    @staticmethod
    def _get_highest_probability_trajectory(
        optional_tool_trajectories: list[dict],
    ) -> Optional[dict]:
        """Return the complete trajectory object with the highest score."""
        if not optional_tool_trajectories:
            return None

        for trajectory in optional_tool_trajectories:
            if not isinstance(trajectory.get("score"), (int, float)):
                raise ValueError("Every tool trajectory must have a numeric score.")
            tools = trajectory.get("tools")
            if not isinstance(tools, list) or not all(
                isinstance(tool, dict)
                and isinstance(tool.get("tool"), str)
                and isinstance(tool.get("goal"), str)
                for tool in tools
            ):
                raise ValueError(
                    "Every tool trajectory must have a tools list containing "
                    "objects with string 'tool' and 'goal' fields."
                )

        return max(
            optional_tool_trajectories,
            key=lambda trajectory: trajectory["score"],
        )

    def _generate_next_tool_policy_rules(
        self,
        agent_messages: list[APICompatibleMessage],
        required_tool_calls: list[str],
        current_trajectory_suggestion: Optional[dict],
    ) -> str | None:

        if len(required_tool_calls) == 0:
            return None
        next_required_tool = required_tool_calls[0]
        next_tool_goal = self._get_tool_goal(
            current_trajectory_suggestion, next_required_tool
        )
        next_action_instruction = f"""
        Your goal is to select the most relevant policy rules from the policy in order to help the agent following the 
        policy, since it is very long. Given the conversation between the agent and user, which contains the policy, 
        the user request and the state of the reservation, you should select the most relevant 
        policy rules that are important for the to agent remember before its next tool call, 
        focusing on policy rules that are relevant to the next tool call and the goal this tool is trying to achieve.
        Next tool to be called: {next_required_tool}
        Tool goal: {next_tool_goal}
        User Data: <list of required user info and preferences for the tool to succeed>
        Relevant Policy Rules:
    """.strip()

        policy_messages = agent_messages + [SystemMessage(role="system", content=next_action_instruction)]
        assistant_message = generate(
            model=self.llm,
            tools=[],
            messages=policy_messages,
            call_name="policy_rules",
            **self.llm_args,
        )
        return f"Critical policy rules that must be followed exactly in order to call {next_required_tool}:\n{assistant_message.content.strip()}"

    @staticmethod
    def _get_tool_goal(
        current_trajectory_suggestion: Optional[dict],
        tool_name: str,
    ) -> str:
        """Return a tool's goal from the current selected trajectory."""
        if not current_trajectory_suggestion:
            return (
                f"Determine the single atomic goal for {tool_name} from its tool "
                "schema and the current conversation state."
            )

        for tool in current_trajectory_suggestion.get("tools", []):
            if not isinstance(tool, dict):
                continue
            suggested_tool_name = str(tool.get("tool", ""))
            goal = tool.get("goal")
            if suggested_tool_name == tool_name and isinstance(goal, str):
                stripped_goal = goal.strip()
                if stripped_goal:
                    return stripped_goal

        return (
            f"Determine the single atomic goal for {tool_name} from its tool "
            "schema and the current conversation state."
        )

    def _generate_policy_rules(self, state: ToolPathAgentState) -> str:
        messages = (state.system_messages + state.messages + [SystemMessage(role="system", content=POLICY_RULES_INSTRUCTION)])
        assistant_message = generate(
            model=self.llm,
            tools=[],
            messages=messages,
            call_name="policy_rules",
            **self.llm_args,
        )
        return "Critical policy rules that must be followed exactly:\n" + assistant_message.content.strip()

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
        state.current_trajectory_suggestion = self._get_highest_probability_trajectory(optional_tool_trajectories)
        state.required_tool_calls = [tool["tool"] for tool in (state.current_trajectory_suggestion.get("tools", [])
                if state.current_trajectory_suggestion else [])]
        print("Tool Path Trajectory Suggestion:", state.required_tool_calls)
        if state.first_tool_calls_suggestions is None:
            state.first_tool_calls_suggestions = optional_tool_trajectories
            if len(optional_tool_trajectories) > 1:
                state.high_trajectory_confidence = False
        agent_messages = list(messages)
        next_tool_policy_rules = self._generate_next_tool_policy_rules(agent_messages, state.required_tool_calls, state.current_trajectory_suggestion,)
        if next_tool_policy_rules:
            agent_messages += [SystemMessage(role="system", content=next_tool_policy_rules)]
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
