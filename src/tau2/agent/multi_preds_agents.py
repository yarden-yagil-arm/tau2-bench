import json
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import numpy as np
from litellm import embedding
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
from tau2.utils.llm_utils import generate, llm_log_mode

PASSED_SIMULATIONS_PATHS = {
    "airline": DATA_DIR / "simulations" / "airline_llm_agent_10_trails" / "results.json",
    "retail": DATA_DIR / "simulations" / "retail_llm_agent_10_trails" / "results.json",
}

SIMILARITY_EMBEDDING_MODEL = "text-embedding-3-small"

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

POLICY_RULES_INSTRUCTION = """
Your goal is to select the most relevant policy rules from the policy in order to help the agent following the 
policy, since it is very long. Given the conversation between the agent and user, which contains the policy, 
the user request and the state of the reservation, you should select the most relevant 
policy rules that are important for the agent to remember before its next action.
Focus on policy rules that are relevant to the current conversation state.
Try to be concise, select only most relevant policy rules, you can summerize them and use a more strict policy phrasing.
Do not add any additional information or explanation.
""".strip()

MULTI_RESPONSE_PROMPT = """
In case you decide to make a tool call, only return a string representation of it, without executing it.
""".strip()

NEXT_ACTION_HEURISTIC_PROMPT = """
Here is a gold agent trajectory for the same task (a gold trajectory describes a verified solution for the same 
task):
{gold_trajectory}

And here are next agent action candidates:
{options}
You task is to choose the candidate which is most similar in its semantic to one of the messages in the gold trajectory.
If you need to select a tool call out of 3 optional tool calls, select the one with the same tool and the same arguments as in the gold trajectory.

Return only the zero-based index of the most similar action candidate: 0, 1, or 2.
""".strip()

SELECTED_ACTION_PROMPT = """
Use the following selected action as the plan for your next response:
{selected_action}

If it describes a user-facing response, respond to the user accordingly. If it
describes a tool call, make the corresponding tool call with the specified
arguments. Continue to follow the policy and current conversation state.
""".strip()


class MultiPredsAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    use_gold_trajectory: bool = True


MultiPredsAgentStateType = TypeVar("MultiPredsAgentStateType", bound="MultiPredsAgentState")


class MultiPredsAgent(
    LLMConfigMixin,
    HalfDuplexAgent[MultiPredsAgentStateType],
    Generic[MultiPredsAgentStateType],
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
        Initialize the MultiPredsAgent.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.domain = domain
        self.task = task
        self.gold_trajectory = self.load_gold_agent_responses(self.task.id, PASSED_SIMULATIONS_PATHS[self.domain])

    @staticmethod
    def load_gold_agent_responses(task_id: str, simulations_path: str | Path) -> list[dict]:
        """Return agent responses and tool calls from the shortest passed trial."""
        results = Results.load(Path(simulations_path))
        passed_trials_responses = []
        for simulation in results.simulations:
            if simulation.task_id != str(task_id) or simulation.reward_info is None or not is_successful(simulation.reward_info.reward):
                continue
            gold_messages = [MultiPredsAgent._format_gold_message(message) for message in simulation.get_messages() if isinstance(message, AssistantMessage)]
            passed_trials_responses.append(gold_messages)
        return min(passed_trials_responses, key=len) if passed_trials_responses else []

    @staticmethod
    def _format_gold_message(message: AssistantMessage) -> dict:
        """Convert a gold agent response or tool call into a dictionary."""
        if message.tool_calls:
            return {"action_type": "tool_call", "action_content": [{"tool": tool_call.name, "arguments": tool_call.arguments} for tool_call in message.tool_calls]}
        return {"action_type": "agent_response", "action_content": message.content or ""}

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> MultiPredsAgentStateType:
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
        return MultiPredsAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: MultiPredsAgentStateType
    ) -> tuple[AssistantMessage, MultiPredsAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_policy_rules(self, agent_messages) -> str:
        messages = (agent_messages + [SystemMessage(role="system", content=POLICY_RULES_INSTRUCTION)])
        assistant_message = generate(
            model=self.llm,
            tools=[],
            messages=messages,
            call_name="policy_rules",
            **self.llm_args,
        )
        return "Critical policy rules that must be followed exactly:\n" + assistant_message.content.strip()

    def _generate_response_options(self, messages: list[APICompatibleMessage]) -> list:
        """Generate three likely next actions without making a tool call."""
        llm_args = {**self.llm_args, "logprobs": True, "top_logprobs": 3}
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages + [SystemMessage(role="system", content=MULTI_RESPONSE_PROMPT)],
            call_name="multi_response_options",
            **llm_args,
        )
        try:
            return response.choices
        except json.JSONDecodeError:
            raise ValueError("Response options must be valid JSON.")


    def next_action_heuristic(
        self,
        options: list,
        agent_messages: list[APICompatibleMessage],
    ) -> int:
        """Select the option with the highest cosine similarity to a gold item."""

        if not isinstance(options, list) or len(options) != 3 or any(not isinstance(option, dict) for option in options):
            raise ValueError("Response options must be exactly three dictionaries.")

        gold_trajectory = [gold_message["action_content"] for gold_message in self.gold_trajectory if gold_message["action_type"] == options[0]["action_type"]]
        option_contents = [option["action_content"] for option in options]
        gold_trajectory = [content if isinstance(content, str) else json.dumps(content, sort_keys=True) for content in gold_trajectory]
        option_contents = [content if isinstance(content, str) else json.dumps(content, sort_keys=True) for content in option_contents]
        if not gold_trajectory:
            raise ValueError(f"No gold items found for action type {options[0]['action_type']}.")
        embedding_response = embedding(model=SIMILARITY_EMBEDDING_MODEL, input=option_contents + gold_trajectory)
        embeddings = np.array([item["embedding"] for item in embedding_response["data"]], dtype=float)
        option_embeddings = embeddings[:len(option_contents)]
        gold_embeddings = embeddings[len(option_contents):]
        option_norms = np.linalg.norm(option_embeddings, axis=1, keepdims=True)
        gold_norms = np.linalg.norm(gold_embeddings, axis=1, keepdims=True)
        normalized_options = np.divide(option_embeddings, option_norms, out=np.zeros_like(option_embeddings), where=option_norms != 0)
        normalized_gold = np.divide(gold_embeddings, gold_norms, out=np.zeros_like(gold_embeddings), where=gold_norms != 0)
        similarities = normalized_options @ normalized_gold.T
        return int(np.argmax(np.max(similarities, axis=1)))

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: MultiPredsAgentStateType
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
        agent_messages = list(messages)
        # policy_rules = self._generate_policy_rules(agent_messages)
        # next_action_options = self._generate_response_options(agent_messages + [SystemMessage(role="system", content=policy_rules)])
        next_action_options = self._generate_response_options(agent_messages)
        selected_action_index = self.next_action_heuristic(next_action_options, agent_messages)
        selected_action = next_action_options[selected_action_index]
        print(selected_action["action_content"])
        agent_messages.append(SystemMessage(role="system", content=SELECTED_ACTION_PROMPT.format(selected_action=selected_action)))
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


def create_multi_preds_agent(tools, domain_policy, **kwargs):
    """Factory function for MultiPredsAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return MultiPredsAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        domain=kwargs.get("domain"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
