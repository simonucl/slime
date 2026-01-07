"""
Trainable Agents for τ²-bench using official AgentGymEnv

This module implements trainable agents that work with τ²-bench's official
Gym interface (AgentGymEnv). This is the recommended approach as it:
- Uses τ²-bench's official API (guaranteed compatibility)
- Handles all orchestration internally (agent, user, environment)
- Matches the verifiers implementation pattern
- Much simpler than manual orchestration

Key components:
- AgentGymEnv: Official gym interface from τ²-bench
- Async LLM calls via sglang for training
- Proper token tracking for RL training
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tau2.data_model.tasks import Task
from tau2.gym import AgentGymEnv
from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent
from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from openai_tool_adapter import create_openai_adapter

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)

class Status(Enum):
    """Status of the agent-environment interaction"""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    """Result of a complete agent-environment interaction"""

    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: list[int] | None = None
    response_length: int = 0
    status: Status = Status.COMPLETED


class TrainableAgentTau2:
    """
    A trainable agent using τ²-bench's official AgentGymEnv.

    This agent uses τ²-bench's Gym interface which handles:
    - Environment management
    - User simulation
    - Tool execution
    - Orchestration (agent/user/env turns)

    The agent only needs to:
    - Generate actions via sglang
    - Track tokens for training
    - Convert results to slime format
    """

    def __init__(
        self,
        domain: str,
        task: Task,
        user_model: str,
        user_base_url: str,
        user_api_key_var: str,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        max_turns: int = 30,
    ):
        """Initialize the trainable agent with τ²-bench's AgentGymEnv

        Args:
            max_turns: Maximum total turns (user + agent messages) in the conversation
        """
        self.domain = domain
        self.task = task
        self.rollout_args = rollout_args
        self.sampling_params = sampling_params

        # Create τ²-bench's official Gym environment
        # This handles ALL orchestration internally!
        # max_steps in AgentGymEnv represents total conversation turns
        self.env = AgentGymEnv(
            domain=domain,
            task_id=task.id,
            max_steps=max_turns,
            solo_mode=False,  # We want user simulation
            user_llm=user_model,
            user_llm_args={
                "base_url": user_base_url,
                "api_key": os.getenv(user_api_key_var),
            },
        )

        self.openai_adapter = create_openai_adapter(parser_type="qwen")

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an LLM call to sglang server"""
        return await post(url, payload)

    def _parse_tool(self, response: str, tools_schema: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Parse tool calls from LLM response string.

        Args:
            response: Raw response text from LLM

        Returns:
            Parsed tool call result in OpenAI format
        """
        # Remove end token if present
        if response.endswith("<|im_end|>"):
            response = response[:-10]

        response = response.strip()

        openai_result = self.openai_adapter.parse_response_to_openai_format(response, tools_schema)
        return openai_result

    def _get_token_delta(
        self,
        tokenizer: AutoTokenizer,
        messages: list[dict],
        tools_schema: list[dict] | None,
    ) -> tuple[list[int], list[int]]:
        """
        Calculate token delta for the last message added to the conversation.

        This method computes the incremental tokens added by the most recent message,
        enabling token-in-token-out tracking during rollout. This ensures synchronization
        between generation and learning, preventing token mismatches.

        Args:
            tokenizer: Tokenizer instance
            messages: Current conversation messages (including the new message)
            tools_schema: Tool schemas for chat template

        Returns:
            Tuple of (token_ids, loss_mask) for the new message only
            - token_ids: New tokens added by last message
            - loss_mask: 1 for assistant tokens, 0 for user/tool/environment
        """
        # Apply chat template to current state
        curr = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools_schema,
        )

        # Get previous state (before last message)
        prev_messages = messages[:-1]

        # Determine if we need generation prompt for previous state
        if messages[-1]["role"] == "assistant":
            # Assistant message: previous state had generation prompt
            prev = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools_schema,
            )
        else:
            # Non-assistant: no generation prompt
            prev = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools_schema,
            )

        # Extract delta (text added by new message)
        delta_text = curr[len(prev) :]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)

        # Set loss mask based on role
        if messages[-1]["role"] == "assistant":
            loss_mask = [1] * len(new_tokens)  # Train on assistant tokens
        else:
            loss_mask = [0] * len(new_tokens)  # Don't train on user/tool/env tokens

        return new_tokens, loss_mask

    def _observation_to_messages(
        self, observation: list, system_prompt: str
    ) -> list[dict[str, Any]]:
        """
        Convert gym observation (list of Message objects) to OpenAI-style messages.

        The observation from AgentGymEnv is a list of Message objects (AssistantMessage,
        UserMessage, ToolMessage). We convert these directly to OpenAI format.

        Args:
            observation: List of Message objects from gym env
            system_prompt: System prompt from env info

        Returns:
            List of message dictionaries in OpenAI format
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Convert Message objects to OpenAI format
        if not observation:
            return messages

        for msg in observation:
            if isinstance(msg, AssistantMessage):
                if msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                elif msg.tool_calls:
                    # Convert tool calls to OpenAI format
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else str(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    messages.append({"role": "assistant", "tool_calls": tool_calls})
            elif isinstance(msg, UserMessage):
                if msg.content:
                    messages.append({"role": "user", "content": msg.content})
                elif msg.tool_calls:
                    # User tool calls - convert to OpenAI format
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else str(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    messages.append({"role": "user", "tool_calls": tool_calls})
            elif isinstance(msg, ToolMessage):
                # Tool messages need special handling in OpenAI format
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.id,
                        "content": msg.content or "",
                    }
                )

        return messages

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Reformulate tool call instruction for tau-bench environment.

        The default tool template assumes one or more function calls, but for
        tau-bench, at most one tool call or skip tool calls are the valid options.

        Args:
            text: Original tool instruction text

        Returns:
            Reformulated tool instruction text
        """
        return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)

    def _prepare_prompt_tokens(self, state: GenerateState, messages: list[dict[str, Any]], tools_schema: list[dict] | None) -> tuple[str, list[int]]:
        """
        Prepare prompt text and tokenize it.

        Args:
            state: GenerateState instance with tokenizer
            messages: Conversation messages

        Returns:
            Tuple of (prompt_text, prompt_token_ids)
        """
        prompt_text = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools_schema
        )
        # Reformulate tool call instruction for tau-bench
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, prompt_token_ids

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int]
    ) -> InteractionResult:
        """
        Build final result with already-accumulated tokens
        """
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join([msg.get('content', '') for msg in messages if msg.get("role") == "assistant" and msg.get('content')])
        res.response_length = len(loss_masks)

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"response_loss_mask_len={len(loss_masks)}, "
            f"prompt_token_len={len(prompt_token_ids)}, "
            f"response_token_len={len(response_token_ids)}, "
            f"response='{res.response[:100]}...'"
        )
        return res

    async def asolve(
        self,
        task: Task,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        max_turns: int = 30,
    ) -> InteractionResult:
        """
        Execute async agent-environment interaction using AgentGymEnv.

        This method uses INCREMENTAL TOKEN TRACKING (token-in-token-out):
        1. Resets the gym environment
        2. Loops: observe -> generate action -> TRACK TOKENS -> step environment -> TRACK TOKENS
        3. Tokens and loss masks accumulated during rollout, not after

        Args:
            task: τ²-bench task to solve
            rollout_args: Rollout configuration arguments
            sampling_params: LLM sampling parameters
            max_turns: Maximum total turns (user + agent messages) in the conversation.
                      Should match the max_steps passed to AgentGymEnv during initialization.

        Returns:
            InteractionResult containing the complete interaction trajectory
        """
        # Initialize
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"

        # Reset environment
        _, info = self.env.reset()

        # Get tools and policy from info
        tools = info.get("tools", [])
        policy = info.get("policy", "")

        # Convert Tool objects to OpenAI schema format for tokenizer
        # τ²-bench returns Tool objects, but tokenizer expects JSON schemas
        tools_schema = [tool.openai_schema for tool in tools] if tools else None

        system_prompt = SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=policy,
        )
        # Initialize token tracking lists (token-in-token-out)
        response_token_ids = []
        loss_masks = []

        # Get initial observation and build initial messages
        observation = self.env._agent.observation
        messages = self._observation_to_messages(observation, system_prompt)

        # Tokenize initial prompt (system message + any initial messages)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages, tools_schema)

        # Initialize tracking
        total_reward = 0.0
        terminated = False
        truncated = False

        # Initialize result
        res = InteractionResult(
            prompt=prompt_text, reward=0, messages=[], info={"task_id": task.id}
        )

        # Main interaction loop
        for turn in range(max_turns):
            # Prepare text for LLM
            text_input, text_token_ids = self._prepare_prompt_tokens(state, messages, tools_schema)

            # Call LLM via sglang
            try:
                payload = {"text": text_input, "sampling_params": sampling_params}
                output = await self._call_llm(url, payload)

                # Check for abort
                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    res.status = Status.ABORTED
                    return self._build_final_result(res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids)

                response = output["text"]
                openai_result = self._parse_tool(response, tools_schema)
                if not openai_result["success"]:
                    logger.warning(f"OpenAI adapter failed: {openai_result['error']}")
                    logger.warning(
                        f"rollout response: {response} can not be parsed into " f"tool calls {openai_result['error']}"
                    )
                    res.status = Status.ABORTED
                    return self._build_final_result(res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids)

                parsed = openai_result["parsed_result"]
                logger.debug(
                    f"Successfully parsed - normal_text: '{parsed['normal_text']}', "
                    f"calls: {parsed['calls']}"
                )

                # IMMEDIATELY track assistant tokens (token-in-token-out)
                # Add assistant response to conversation for tokenization
                messages.append({"role": "assistant", "content": response})

                # Get token delta for assistant response
                token_delta, mask_delta = self._get_token_delta(
                    state.tokenizer, messages, tools_schema
                )
                response_token_ids.extend(token_delta)
                loss_masks.extend(mask_delta)
                agent_content, function_calls = parsed["normal_text"], parsed["calls"]
                logger.debug(f"Creating action from - content: '{agent_content}', " f"calls: {function_calls}")
                if function_calls:
                    # Convert tool call dict to JSON string for env.step()
                    tool_call_dict = function_calls[0]
                    action, tool_called = json.dumps(tool_call_dict), True
                else:
                    tool_call_dict = None
                    action, tool_called = agent_content, False

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids)

            # Step environment with the action
            try:
                obs, reward, terminated, truncated, env_info = self.env.step(action)
                total_reward = reward  # Keep latest reward

                if tool_called:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_dict.get("id", str(uuid.uuid4())),
                        "content": obs,
                    })
                else:
                    messages.append(
                        {"role": "user", "content": obs},
                    )
                env_token_ids, env_loss_mask = self._get_token_delta(state.tokenizer, messages, tools_schema)
                response_token_ids.extend(env_token_ids)
                loss_masks.extend(env_loss_mask)
                info = {**info, **env_info}  # env_info is already a dict

            except Exception as e:
                logger.error(f"Environment step failed: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids)

            # Check termination
            if terminated:
                res.status = Status.COMPLETED
                break

            if truncated:
                res.status = Status.TRUNCATED
                break

        # Store number of agent turns for metrics
        num_turns = turn + 1 if terminated or truncated else turn
        info["turns"] = num_turns
        info["terminated"] = terminated
        info["truncated"] = truncated

        res = self._build_final_result(
            res,
            total_reward,
            info,
            messages,
            loss_masks,
            prompt_token_ids,
            response_token_ids,
        )

        return res


def agent_factory_tau2(
    domain: str,
    agent_type: str = "llm",
    user_model: str = "gpt-4o-mini",
    user_base_url: str = "https://api.openai.com/v1",
    user_api_key_var: str = "OPENAI_API_KEY",
    max_turns: int = 30,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
):
    """
    Factory function to create trainable τ²-bench agents.

    Args:
        domain: Domain name (retail, airline, telecom, mock)
        agent_type: Type of agent (llm, solo, gt) - currently only llm supported
        user_model: Model to use for user simulation
        user_base_url: Base URL for user model API
        user_api_key_var: Environment variable for API key
        max_turns: Maximum total turns (user + agent messages) per episode
        rollout_args: Rollout configuration
        sampling_params: LLM sampling parameters

    Returns:
        Agent factory that creates agents for specific tasks
    """
    if agent_type != "llm":
        raise NotImplementedError(f"Agent type {agent_type} not yet supported")

    class AgentFactory:
        def __init__(self):
            self.domain = domain
            self.user_model = user_model
            self.user_base_url = user_base_url
            self.user_api_key_var = user_api_key_var
            self.max_turns = max_turns
            self.rollout_args = rollout_args or {}
            self.sampling_params = sampling_params or {}

        async def asolve(self, task, rollout_args, sampling_params, max_turns=30):
            """Create agent for specific task and solve it"""
            agent = TrainableAgentTau2(
                domain=self.domain,
                task=task,
                user_model=self.user_model,
                user_base_url=self.user_base_url,
                user_api_key_var=self.user_api_key_var,
                rollout_args=rollout_args,
                sampling_params=sampling_params,
                max_turns=self.max_turns,
            )
            return await agent.asolve(task, rollout_args, sampling_params, max_turns)

    return AgentFactory()
