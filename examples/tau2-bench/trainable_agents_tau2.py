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
from tau2.run import get_tasks
from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

# Set up logger for this module
logger = logging.getLogger(__name__)


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

        # τ²-bench uses LiteLLM internally with tool.openai_schema
        # No need for custom tool parser - the gym env handles tool parsing internally

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an LLM call to sglang server"""
        return await post(url, payload)

    def _parse_llm_response(self, response: str) -> str:
        """
        Parse LLM response and return action string.

        τ²-bench AgentGymEnv accepts two formats:
        1. Plain text message: "Hello, how can I help?"
        2. Tool call (JSON format): '{"name": "search_flights", "arguments": {"origin": "NYC"}}'

        The gym env internally parses these - we just need to clean up the response.

        Args:
            response: Raw response text from LLM

        Returns:
            Action string for env.step()
        """
        # Remove end token if present
        if response.endswith("<|im_end|>"):
            response = response[:-10]

        # AgentGymEnv handles parsing internally - just return cleaned response
        return response.strip()

    def _compute_tokens_from_trajectory(
        self,
        messages: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        tools_schema: list[dict] | None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute token IDs and loss masks from complete message trajectory.

        This processes the full conversation and creates:
        - prompt_tokens: Initial system message
        - response_tokens: All conversation tokens after system message
        - loss_masks: 1 for assistant messages, 0 for user/tool messages

        Args:
            messages: Complete conversation in OpenAI format
            tokenizer: Tokenizer instance
            tools_schema: Tool schemas for chat template

        Returns:
            Tuple of (prompt_token_ids, response_token_ids, loss_masks)
        """
        # Tokenize just the system message as prompt
        system_only = messages[:1]
        prompt_text = tokenizer.apply_chat_template(
            system_only,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools_schema,
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Now process the rest of the conversation message by message
        response_token_ids = []
        loss_masks = []

        for i in range(1, len(messages)):
            # Get conversation up to this point
            prev_messages = messages[:i]
            curr_messages = messages[: i + 1]

            # Check if this is an assistant message
            is_assistant = curr_messages[-1]["role"] == "assistant"

            prev_text = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=is_assistant,
                tools=tools_schema,
            )

            # Tokenize current state
            curr_text = tokenizer.apply_chat_template(
                curr_messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools_schema,
            )

            # Get the delta
            if len(curr_text) > len(prev_text):
                delta_text = curr_text[len(prev_text) :]
                delta_tokens = tokenizer.encode(delta_text, add_special_tokens=False)

                response_token_ids.extend(delta_tokens)

                # Set loss mask: 1 for assistant, 0 for others
                if is_assistant:
                    loss_masks.extend([1] * len(delta_tokens))
                else:
                    loss_masks.extend([0] * len(delta_tokens))

        return prompt_token_ids, response_token_ids, loss_masks

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
        from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage

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

    async def asolve(
        self,
        task: Task,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        max_turns: int = 30,
    ) -> InteractionResult:
        """
        Execute async agent-environment interaction using AgentGymEnv.

        This method:
        1. Resets the gym environment
        2. Loops: observe -> generate action -> step environment
        3. Collects full trajectory
        4. Computes tokens from complete trajectory

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

        # Build system prompt
        system_prompt = f"<instructions>\n{policy}\n</instructions>"

        # Initialize tracking
        total_reward = 0.0
        terminated = False
        truncated = False

        # Initialize result
        res = InteractionResult(
            prompt="", reward=0, messages=[], info={"task_id": task.id}
        )

        # Main interaction loop
        for turn in range(max_turns):
            # Get actual Message objects from gym agent
            observation = self.env._agent.observation

            # Build current conversation
            conversation_messages = self._observation_to_messages(observation, system_prompt)

            # Prepare text for LLM
            text_input = state.tokenizer.apply_chat_template(
                conversation_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools_schema,
            )

            # Call LLM via sglang
            try:
                payload = {"text": text_input, "sampling_params": sampling_params}
                output = await self._call_llm(url, payload)

                # Check for abort
                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    res.status = Status.ABORTED
                    break

                response = output["text"]
                action = self._parse_llm_response(response)

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                res.status = Status.ABORTED
                break

            # Step environment with the action
            try:
                _, reward, terminated, truncated, info = self.env.step(action)
                total_reward = reward  # Keep latest reward

            except Exception as e:
                logger.error(f"Environment step failed: {e}")
                res.status = Status.ABORTED
                break

            # Check termination
            if terminated:
                res.status = Status.COMPLETED
                break

            if truncated:
                res.status = Status.TRUNCATED
                break

        # After interaction completes, process the full trajectory for training
        observation = self.env._agent.observation
        final_messages = self._observation_to_messages(observation, system_prompt)

        # Compute tokens and loss masks from complete trajectory
        prompt_token_ids, response_token_ids, loss_masks = self._compute_tokens_from_trajectory(
            final_messages, state.tokenizer, tools_schema
        )

        # Build prompt text (first conversation state)
        prompt_text = state.tokenizer.apply_chat_template(
            final_messages[:1],  # Just system message for prompt
            tokenize=False,
            add_generation_prompt=False,
            tools=tools_schema,
        )

        # Store number of agent turns for metrics
        num_turns = turn + 1 if terminated or truncated else turn

        res.prompt = prompt_text
        res.reward = total_reward
        res.info = {
            "task_id": task.id,
            "turns": num_turns,
            "terminated": terminated,
            "truncated": truncated,
            **info,
        }
        res.messages = final_messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join(
            [msg['content'] for msg in final_messages if msg["role"] == "assistant"]
        )
        res.response_length = len(loss_masks)

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
