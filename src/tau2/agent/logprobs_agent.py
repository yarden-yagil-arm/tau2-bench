import math
from typing import Generic, List, Optional, TypeVar

import tiktoken
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
from tau2.utils.llm_utils import generate

TOP_LOGPROBS = 20

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


class UncertaintyDetectionAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    tool_logprobs: dict[int, dict] = Field(default_factory=dict)
    uncertainty: Optional[float] = None

UncertaintyDetectionAgentStateType = TypeVar(
    "UncertaintyDetectionAgentStateType", bound="UncertaintyDetectionAgentState"
)


class UncertaintyDetectionAgent(
    LLMConfigMixin,
    HalfDuplexAgent[UncertaintyDetectionAgentStateType],
    Generic[UncertaintyDetectionAgentStateType],
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
        Initialize the UncertaintyDetectionAgent.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.encoding = tiktoken.encoding_for_model(self.llm)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UncertaintyDetectionAgentStateType:
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
        return UncertaintyDetectionAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: UncertaintyDetectionAgentStateType,
    ) -> tuple[AssistantMessage, UncertaintyDetectionAgentStateType]:
        """
        Respond to a user or tool message.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def calculate_uncertainty(
        self, selected_tool, messages: list[APICompatibleMessage]
    ) -> (str, dict[str, float]):
        """Return the top tool candidates and their number-token log probabilities."""
        tool_options = {
            index+1: tool_name
            for index, tool_name in enumerate(
                [*(tool.name for tool in self.tools)]
            )
        }
        logit_bias = {}
        for tool_num in tool_options:
            token_ids = self.encoding.encode(str(tool_num))
            if len(token_ids) != 1:
                raise ValueError(
                    f"Tool number {tool_num} is encoded as multiple tokens: {token_ids}"
                )
            logit_bias[token_ids[0]] = 20

        uncertainty_prompt = SystemMessage(
            role="system",
            content=(
                "Which of the following tools is the best tool to execute next?\n"
                "The tools are represented by this dictionary, where each key is "
                "a tool number and each value is its tool name:\n"
                f"{tool_options}\n. Respond with only the number "
                "of the best tool. I you think no tool should be called, and asking user for more information or "
                "preferences is the best next action, return 0. Do not make any "
                "tool call. Do not "
                "return the tool name or any other text, only a number that represents the "
                f"selected tool out of the available decimal numbers from the following list: {tool_options.keys()}."
            ),
        )
        llm_args = {
            **self.llm_args,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            # "temperature": 1.0,
        }
        selected_tool_num = str(next(key for key, value in tool_options.items() if value == selected_tool))
        pred_tool, trails = "", 1
        for trail in range(trails):
            if pred_tool == selected_tool_num:
                break
            response = generate(model=self.llm,
                                tools=self.tools,
                                messages=[*messages, uncertainty_prompt],
                                call_name="logprobs_response",
                                **llm_args,
            )
            pred_tool = response.content.strip()
            trail += 1
        tool_options[0] = "no_tool"
        token_logprobs = response.raw_data["choices"][0]["logprobs"]["content"][0]['top_logprobs']
        tool_logprobs: dict[str, tuple[float, int]] = {}
        for idx, candidate in enumerate(token_logprobs):
            token, logprob = candidate["token"].strip(), candidate["logprob"]
            if token.isdecimal() and int(token) in tool_options.keys():
                tool_name = tool_options[int(token)]
                if idx == 0 and logprob == 0.0:
                    return {tool_name: (logprob, idx)}
                if tool_name not in tool_logprobs: # in case multiple tokens map to the same tool, keep the highest logprob
                    tool_logprobs[tool_name] = (logprob, idx)
        return tool_logprobs

    @staticmethod
    def calculate_tool_entropy(
        tool_logprobs: dict[str, tuple[float, int]],
    ) -> float:
        """Calculate entropy over the available tool log probabilities in nats."""
        if not tool_logprobs:
            raise ValueError("Cannot calculate entropy without tool log probabilities")
        logprobs = [logprob for logprob, _ in tool_logprobs.values()]
        max_logprob = max(logprobs)
        weights = [
            math.exp(logprob - max_logprob) for logprob in logprobs
        ]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        entropy =  -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        return entropy / math.log(TOP_LOGPROBS)

    def _generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: UncertaintyDetectionAgentStateType,
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
        llm_args = {**self.llm_args}
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **llm_args,
        )
        if assistant_message.tool_calls is not None:
            selected_tool = assistant_message.tool_calls[0].name
            pred_tool_logprobs = self.calculate_uncertainty(selected_tool, messages)
            pred_next_tool = max(pred_tool_logprobs, key=pred_tool_logprobs.get)
            tool_entropy = self.calculate_tool_entropy(pred_tool_logprobs)
            state.uncertainty = max(
                value for value in (state.uncertainty, tool_entropy) if value is not None
            )
            # pred_tool_probabilities = {tool_name: math.exp(logprob) for tool_name, logprob in pred_tool_logprobs.items()}
            uncertainty_calculation = {
                "message_id": assistant_message.raw_data.get("id"),
                "turn_number": len(messages),
                "actual_selected_tool": selected_tool,
                "pred_next_tool": pred_next_tool,
                "pred_tool_logprobs": pred_tool_logprobs,
                # "pred_tool_probabilities": pred_tool_probabilities,
                "tool_entropy": tool_entropy,
                "conversation_uncertainty": state.uncertainty,
            }
            assistant_message.raw_data["uncertainty_calculation"] = (
                uncertainty_calculation
            )
            state.tool_logprobs[len(messages)] = uncertainty_calculation
            pred_text = "almost" if selected_tool in pred_tool_logprobs else "different"
            pred_text = "same" if selected_tool == pred_next_tool else pred_text
            if pred_tool_logprobs[pred_next_tool] != 0.0:
                print("KKKKKK")
            print(
                "RRRR",
                pred_text,
                selected_tool, tool_entropy,
                pred_tool_logprobs
            )
        return assistant_message


# =============================================================================
# AGENT FACTORY FUNCTIONS
# =============================================================================


def create_uncertainty_detection_agent(tools, domain_policy, **kwargs):
    """Factory function for UncertaintyDetectionAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return UncertaintyDetectionAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )

"""
high value->high uncertainty
0: -0.0, 0, 2.470694090843156e-18, 5.118000258136443e-16 , 
6: 1.2094334352606534e-08, 0, 0, 0
4: 0.185888, 6.261467577094791e-06 ,3.7392379806472186e-05, 0.063714
17: 0.1747544114156691, 0.00015, 0.04964, 1.8368781471046621e-06, 
15: 1.6274987274848018e-08(fail), 1.0837066733255517e-06 (pass), 8.97847326146008e-13 (fail), 1.863212459134395e-10 (
fail)


only no first token=0.0 logprob:
4: 0.000487, 0.0452, 0 (fail), 0.0044 
17: 0, 0, 1.332121684230914e-05 (pass), 0.0207
15: pass: 0, 0.00774, 0, 0.0253 fail: 0.2077
32: pass: 0, 0, 0.08782, 0.21433, 5.846656425294078e-07, fail: 0, 0.1724, 0
"""