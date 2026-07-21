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
    ToolCall,
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
NUM_CANDIDATES = 5

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
        self.gold_tool_calls, self.gold_agent_messages = self._get_gold_tool_calls_and_agent_messages()
        self.gold_agent_message_embeddings = self._get_normalized_embeddings(self.gold_agent_messages)
        self.messages_candidates: list[list[AssistantMessage]] = []

    @staticmethod
    def load_gold_agent_responses(task_id: str, simulations_path: str | Path) -> list[AssistantMessage]:
        """Return agent responses and tool calls from the shortest passed trial."""
        results = Results.load(Path(simulations_path))
        passed_trials_responses = []
        for simulation in results.simulations:
            if simulation.task_id != str(task_id) or simulation.reward_info is None or not is_successful(simulation.reward_info.reward):
                continue
            passed_trials_responses.append(simulation.get_messages())
        return max(passed_trials_responses, key=len) if passed_trials_responses else []

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

    def _generate_response_options(self, messages: list[APICompatibleMessage]) -> list[AssistantMessage]:
        """Generate likely next actions without making a tool call."""
        llm_args = {
            **self.llm_args,
            "logprobs": True,
            "top_logprobs": 20,
            "n": NUM_CANDIDATES,
            "temperature": 1,
        }
        responses = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="multi_response_options",
            **llm_args,
        )
        if not isinstance(responses, list) or len(responses) != NUM_CANDIDATES or any(not isinstance(response, AssistantMessage) for response in responses):
            raise ValueError(f"Response options must be exactly {NUM_CANDIDATES} AssistantMessage objects.")
        self.messages_candidates.append(responses)
        return responses

    def _get_gold_tool_calls_and_agent_messages(self) -> tuple[list[ToolCall], list[str]]:
        """Split the gold trajectory into tool calls and agent text messages."""
        gold_tool_calls = []
        gold_agent_messages = []
        for gold_message in self.gold_trajectory:
            if not isinstance(gold_message, AssistantMessage):
                continue
            if gold_message.tool_calls:
                gold_tool_calls.extend(gold_message.tool_calls)
            else:
                gold_agent_messages.append(gold_message.content or "")
        return gold_tool_calls, gold_agent_messages

    def _is_tool_call_in_gold_tool_calls(self, tool_call: ToolCall) -> bool:
        """Return whether a tool call's name and arguments match a gold tool call."""
        return any(
            gold_tool_call.name == tool_call.name
            and gold_tool_call.arguments == tool_call.arguments
            for gold_tool_call in self.gold_tool_calls
        )

    @staticmethod
    def _get_normalized_embeddings(messages: list[str]) -> np.ndarray:
        """Generate normalized embeddings for messages."""
        embedding_response = embedding(model=SIMILARITY_EMBEDDING_MODEL, input=messages)
        embeddings = np.array([item["embedding"] for item in embedding_response["data"]], dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)


    def next_action_heuristic(
        self,
        candidates: list[AssistantMessage],
        agent_messages: list[APICompatibleMessage],
    ) -> int:
        """Select the option with the highest cosine similarity to a gold item."""

        if len(candidates) != NUM_CANDIDATES or any(not isinstance(option, AssistantMessage) for option in candidates):
            raise ValueError(f"Response candidates must be exactly {NUM_CANDIDATES} AssistantMessage objects.")
        num_candidates_with_tools = sum([bool(candidate.tool_calls) for candidate in candidates])
        if 1 <=  num_candidates_with_tools < NUM_CANDIDATES:
            # If more than half of the candidates have tool calls, filter to only those with tool calls; otherwise, filter to only those without tool calls.
            tool_call_filter = True if num_candidates_with_tools > NUM_CANDIDATES / 2 else False
            candidates = [candidate for candidate in candidates if bool(candidate.tool_calls) == tool_call_filter]


        if candidates[0].tool_calls:
            matching_candidate_indices = []
            for idx, c in enumerate(candidates):
                if any([self._is_tool_call_in_gold_tool_calls(tool_call) for tool_call in c.tool_calls]):
                    matching_candidate_indices.append(idx)
            if not matching_candidate_indices:
                print(f"Tool call does not match any gold tool call. Tool:", [tool_call.name for tool_call in candidates[0].tool_calls])
                return 0
            return min(matching_candidate_indices)

        candidate_contents = [candidate.content for candidate in candidates]
        candidate_embeddings = self._get_normalized_embeddings(candidate_contents)
        similarities = np.dot(candidate_embeddings, self.gold_agent_message_embeddings.T)
        candidate_similarities = np.max(similarities, axis=1)
        selected_candidate_index = int(np.argmax(candidate_similarities))
        selected_gold_message_index = int(np.argmax(similarities[selected_candidate_index]))
        print("Selected gold message:", self.gold_agent_messages[selected_gold_message_index])
        print("Selected candidate message:", candidates[selected_candidate_index].content)
        return selected_candidate_index

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
        next_action_options = self._generate_response_options(agent_messages)
        selected_action_index = self.next_action_heuristic(next_action_options, agent_messages)
        selected_action = next_action_options[selected_action_index]
        return selected_action

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
