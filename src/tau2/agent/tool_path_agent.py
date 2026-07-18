import json
from pathlib import Path
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
from tau2.data_model.simulation import Results
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.metrics.agent_metrics import is_successful
from tau2.utils import DATA_DIR
from tau2.utils.llm_utils import generate

PASSED_SIMULATIONS_PATHS = {
    "airline": DATA_DIR / "simulations" / "airline_llm_agent_10_trails" / "results.json",
    "retail": DATA_DIR / "simulations" / "retail_llm_agent_10_trails" / "results.json",
}

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
""".strip()


TOOL_PATH_PROMPT = """
Before taking action on the user's response, suggest a high probability next tool that is required in order to follow the policy and the user 
request. Take into consideration the current conversation state.
Use the exact names of available tools and do not invent tools.
-Do not call a tool in this response.
The suggested next tool call should include tool name and arguments for that tool call. If arguments are not known, use placeholders of <depends on 
previous tool call> or <requires user info>.

the returned selected tool should be a Json with the following keys:
- "tool": The name of the tool to call.
- "arguments": A dictionary of arguments to pass to the tool
- "goal": A short description of the goal of this tool call.

{gold_trajectory_instruction}
""".strip()

GOLD_TOOL_PATH_INSTRUCTION = """
Here is a gold tool call trajectory for the same task (a gold trajectory describes a verified solution for the same 
task):
{gold_tool_calls}

Use the gold trajectory to make a better next tool prediction while taking into consideration both the gold trajectory and the current 
conversation state. 
Do not blindly copy the gold trajectory, take it into consideration while also considering the previous tool calls that were made and the 
conversation state. In case you choose a tool which is as in the gold trajectory, use the same arguments as in the gold tool call if they are already available, 
otherwise use placeholders of <depends on previous tool call> or <requires user info> for the arguments that are not available yet.
""".strip()


REQUIRED_TOOL_NAMES_TASK_17 = [
    "get_user_details",
    "get_reservation_details",
    "get_reservation_details",
    "get_reservation_details",
    "get_reservation_details",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
]


class ToolPathAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    use_gold_trajectory: bool = True
    # print("Gold trajectory is on:", use_gold_trajectory)


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
        domain: str,
        llm_args: Optional[dict] = None,
        task: Optional[Task] = None,
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
        self.domain = domain
        self.task = task
        self.gold_trajectory = self.load_gold_tool_calls(self.task.id, PASSED_SIMULATIONS_PATHS[self.domain])

    @staticmethod
    def load_gold_tool_calls(task_id: str, simulations_path: str | Path) -> list[dict]:
        """Return tool names and arguments from the shortest passed trial."""
        results = Results.load(Path(simulations_path))
        passed_trials_tool_calls = []
        for simulation in results.simulations:
            if simulation.task_id != str(task_id) or simulation.reward_info is None or not is_successful(simulation.reward_info.reward):
                continue
            tool_calls = [
                {"tool": tool_call.name, "arguments": tool_call.arguments}
                for message in simulation.get_messages()
                if isinstance(message, AssistantMessage) and message.tool_calls
                for tool_call in message.tool_calls]
            passed_trials_tool_calls.append(tool_calls)
        return min(passed_trials_tool_calls, key=len) if passed_trials_tool_calls else []

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)

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

    @staticmethod
    def _get_tool_calls_already_made(messages: list[APICompatibleMessage],) -> list[dict]:
        """Return previously made tool calls in chronological order."""
        return [
            {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for message in messages
            if isinstance(message, AssistantMessage) and message.tool_calls
            for tool_call in message.tool_calls
        ]

    def _generate_tools_trajectory(
        self,
        messages: list[APICompatibleMessage],
        use_gold_trajectory: bool = False,
    ) -> list[dict]:
        """
        Generate the next message from a user or tool message.
        """
        tool_calls_already_made = self._get_tool_calls_already_made(messages)
        if use_gold_trajectory and len(self.gold_trajectory):
            trajectory_instruction = GOLD_TOOL_PATH_INSTRUCTION.format(gold_tool_calls=json.dumps(self.gold_trajectory, indent=2))
        else:
            trajectory_instruction = ""

        trajectory_prompt = TOOL_PATH_PROMPT.format(tool_calls_already_made=json.dumps(tool_calls_already_made, indent=2), gold_trajectory_instruction=trajectory_instruction)
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages + [SystemMessage(role="system", content=trajectory_prompt)],
            call_name="tool_path_response",
            **self.llm_args,
        )
        try:
            trajectories = json.loads(assistant_message.content)
            return trajectories
        except:
            return None
    @staticmethod
    def _get_highest_probability_trajectory(optional_tool_trajectories: list[dict]) -> Optional[dict]:
        if not optional_tool_trajectories:
            return None
        return optional_tool_trajectories[0]#max(optional_tool_trajectories, key=lambda trajectory: trajectory["score"])

    def _generate_next_tool_policy_rules(self, agent_messages: list[APICompatibleMessage], current_trajectory_suggestion: Optional[dict]) -> str | None:
        if not current_trajectory_suggestion:
            return None

        # suggested_tools = current_trajectory_suggestion.get("tools", [])
        next_required_tool = current_trajectory_suggestion#[0]
        next_required_tool_name = next_required_tool["tool"]
        next_required_tool_arguments = next_required_tool.get("arguments", {})
        next_tool_goal = next_required_tool["goal"]
        formatted_tool_arguments = json.dumps(next_required_tool_arguments, indent=2)
        next_action_instruction = f"""
        Your goal is to select the most relevant policy rules from the policy in order to help the agent following the 
        policy, since it is very long. Given the conversation between the agent and user, which contains the policy, 
        the user request and the state of the reservation, you should select the most relevant 
        policy rules that are important for the agent to remember before its next tool call to {next_required_tool_name}
        with arguments {formatted_tool_arguments}. The goal of this tool call is: {next_tool_goal}.
        Focus on policy rules that are relevant to this tool call and the goal it is trying to achieve.
        Try to be concise, select only most relevant policy rules, you can summerize them and use a more strict policy phrasing.
        Do not add any additional information or explanation.
    """.strip()

        policy_messages = agent_messages + [SystemMessage(role="system", content=next_action_instruction)]
        assistant_message = generate(
            model=self.llm,
            tools=[],
            messages=policy_messages,
            call_name="policy_rules",
            **self.llm_args,
        )
        return (
            f"Here is a highly relevant information that you should use to make a better next step:ֿ\n"
            f"- The next tool you will probably need to call is {next_required_tool_name} \n"
            f"- The arguments to make this tool call properly are: {next_required_tool_arguments} \n"
            f"- This tool call will help to achieve the goal: {next_tool_goal} \n"
            f"- Here are critical policy rules that must be followed exactly in order to call"
            f" {next_required_tool_name} properly, make sure to follow them:\n{assistant_message.content.strip()}"
        )

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
        optional_tool_trajectories = self._generate_tools_trajectory(messages, state.use_gold_trajectory)
        current_trajectory_suggestion = optional_tool_trajectories#self._get_highest_probability_trajectory(optional_tool_trajectories)
        agent_messages = list(messages)
        next_tool_policy_rules = self._generate_next_tool_policy_rules(agent_messages, current_trajectory_suggestion)
        b = self._generate_tools_trajectory(messages, state.use_gold_trajectory)
        # print("policy rules for next tool call:\n", next_tool_policy_rules)
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
        domain=kwargs.get("domain"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
