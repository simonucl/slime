"""
Trainable agent implementation for Persuasion for Good dataset.

This module implements the agent-environment interaction logic with token tracking,
following the pattern from examples/tau2-bench/trainable_agents_tau2.py

Key pattern: Token-in-token-out tracking during conversation rollout.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import openai
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from persona_manager import PersuasionTask, PersonaManager
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

# Set up logger for this module
logger = logging.getLogger(__name__)


class Status(Enum):
    """Episode status"""
    COMPLETED = "completed"  # Episode completed successfully (donation made)
    TRUNCATED = "truncated"  # Episode reached max turns
    ABORTED = "aborted"      # Episode failed (error, format issue, etc.)


@dataclass
class InteractionResult:
    """Result of one persuasion conversation (matches τ²-bench pattern)"""
    prompt: str  # Conversation ID
    tokens: list[int]  # Full conversation token IDs
    response: str  # Full conversation text
    reward: float  # Reward mapped from donation: [-1.0, 1.0] (donation 0->-1, 1->0, >=2->1)
    loss_mask: list[int]  # Binary mask: 1=persuader (trainable), 0=persuadee (environment)
    status: Status  # COMPLETED, TRUNCATED, ABORTED
    info: dict  # Metadata (turns, personas, etc.)
    messages: list[dict[str, Any]] = None  # Conversation messages
    response_length: int = 0  # Response length in tokens


class PersuasionEnv:
    """
    Two-agent conversation environment for persuasion task.

    Persuader: Trainable policy (loss_mask=1)
    Persuadee: Environment LLM (loss_mask=0)
    """

    def __init__(
        self,
        task: PersuasionTask,
        persuadee_model: str,
        persuadee_base_url: str,
        persuadee_api_key: str,
        max_turns: int = 10,
        persona_manager: Optional[PersonaManager] = None
    ):
        """
        Initialize persuasion environment.

        Args:
            task: PersuasionTask with conversation metadata and personas
            persuadee_model: LLM model for persuadee (e.g., "gpt-4o-mini")
            persuadee_base_url: API base URL
            persuadee_api_key: API key
            max_turns: Maximum conversation turns (default: 10)
            persona_manager: PersonaManager instance (for generating persona descriptions)
        """
        self.task = task
        self.max_turns = max_turns
        self.persona_manager = persona_manager

        # Initialize persuadee LLM client
        self.persuadee_client = AsyncOpenAI(
            base_url=persuadee_base_url,
            api_key=persuadee_api_key
        )
        self.persuadee_model = persuadee_model

        # Conversation state
        self.turn_count = 0
        self.donation_amount: Optional[float] = None

    def get_persuader_system_prompt(self) -> str:
        """
        Build persuader system prompt with persona.
        Based on persuasion_simulation/src/prompts/persuader_prompt.py
        """
        # Generate persona description
        if self.persona_manager:
            persona_desc = self.persona_manager.generate_persona_description(
                self.task.persuader_persona
            )
        else:
            persona_desc = f"User ID: {self.task.persuader_id}"

        prompt = f"""You are an Amazon Mechanical Turk worker completing a $2 communication task.
- You are motivated by this task payment — you value every cent you earn.
- Act naturally as the person described below—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
- Respond in a real-time chat interface. Keep each response under 30 words, conversational, and authentic—avoid formal, robotic, or repetitive language.

<persona>
{persona_desc}
</persona>

<task>
You are the Persuader in this communication task.
Your job is to persuade your partner to donate some or all of their incoming payment for this task ($2.00) to a children's charity called Save the Children.
You will earn a bonus equal to whatever amount your partner donates if they agree to donate.

Required Actions:
- Ask for a SPECIFIC donation amount: "How much would you like to donate to the charity from your task payment?"
- Explain the deduction: "Your donation will be directly deducted from your task payment. You can choose any amount from $0 to your full payment ($2.00)."

Conversation Strategy:
- Start with a greeting or small talk to open the conversation naturally.
- Build rapport gradually before discussing donations.
- Guide the conversation for about {self.max_turns // 2} chat turns.

Basic information about Save the Children:
Save the Children is an international non-governmental organization that promotes children's rights, provides relief and helps support children in developing countries.
</task>

<rules>
- Your responses can sometimes contain minor grammatical errors to sound more natural.
- Your response should not exceed 30 words.
- Only output your reply to your chat partner—do not explain your reasoning.
</rules>"""
        return prompt

    def get_persuadee_system_prompt(self, persuadee_messages: list[dict]) -> str:
        """
        Build persuadee system prompt with persona and conversation context.
        Based on persuasion_simulation/src/prompts/persuadee_prompt.py
        """
        # Generate persona description
        if self.persona_manager:
            persona_desc = self.persona_manager.generate_persona_description(
                self.task.persuadee_persona
            )
        else:
            persona_desc = f"User ID: {self.task.persuadee_id}"

        # Get last persuader message for scenario context
        last_persuader_msg = ""
        for msg in reversed(persuadee_messages):
            if msg["role"] == "user":  # Persuader appears as "user" to persuadee
                last_persuader_msg = msg["content"]
                break

        prompt = f"""You are an Amazon Mechanical Turk worker completing a $2 communication task.
- You are motivated by this task payment — you value every cent you earn.
- Act naturally as the person described below—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
- Respond in a real-time chat interface. Keep each response under 30 words, conversational, and authentic—avoid formal, robotic, or repetitive language.

<persona>
{persona_desc}
</persona>

<scenario>
You are presented with the following task content:

---
Welcome to the communication task. You will now start a conversation with your partner about a children's charity.
Please don't game the task by replying short and meaningless sentences.
---

You click Start, and you receive the following message:

"{last_persuader_msg}"
</scenario>

<task>
You have just made $2.00.
You have access to this money whenever you like, and you are free to use it in whatever way you wish.
You are about to interact with an individual about a children's charity. They may try to get you to donate or to give them some of your money.
You must make decision wisely, and not spend the money or give it away very easily.
</task>

<rules>
- Your responses can sometimes contain minor grammatical errors to sound more natural.
- To contribute, provide, donate or give money say [DONATE $N] or [GIVE $N], where N is the amount you will donate/give (N can be either decimal or integer).
For example, to give $0.2, say [GIVE $0.2], or to donate $1.25, say [DONATE $1.25].
You MUST say it in this format, otherwise the donation will not be made.
- Your response should not exceed 30 words.
- Only output your reply to your chat partner—do not explain your reasoning.
</rules>"""
        return prompt

    async def step_persuadee(self, persuadee_messages: list[dict]) -> str:
        """
        Execute one persuadee turn.

        Args:
            persuadee_messages: Conversation from persuadee's perspective

        Returns:
            Persuadee's response
        """
        # Build persuadee messages (system prompt + conversation)
        system_prompt = self.get_persuadee_system_prompt(persuadee_messages)
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + persuadee_messages

        # Call persuadee LLM
        try:
            response = await self.persuadee_client.chat.completions.create(
                model=self.persuadee_model,
                messages=full_messages,
                temperature=0.7,
                max_tokens=300
            )
            persuadee_response = response.choices[0].message.content.strip()
        except Exception as e:
            # If LLM call fails, return error message
            logger.error(f"Persuadee LLM call failed: {e}")
            persuadee_response = f"[Error: {str(e)}]"

        # Check for donation
        donation = self.extract_donation(persuadee_response)
        if donation is not None:
            self.donation_amount = donation

        self.turn_count += 1

        return persuadee_response

    def extract_donation(self, text: str) -> Optional[float]:
        """
        Extract donation amount from persuadee response.
        Based on persuadee_prompt.py donation format: [DONATE $N] or [GIVE $N]

        Args:
            text: Persuadee response text

        Returns:
            Donation amount (0.0-2.0) or None if no donation found
        """
        # Regex pattern: [DONATE $N] or [GIVE $N]
        pattern = r'\[(DONATE|GIVE)\s*\$\s*([0-9]+(?:\.[0-9]+)?)\]'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            amount = float(match.group(2))
            # Clamp to valid range [0.0, 2.0]
            return max(0.0, min(2.0, amount))

        return None

    @staticmethod
    def donation_to_reward(donation: float) -> float:
        """
        Map donation amount to reward range.

        Args:
            donation: Donation amount (0.0 to infinity)

        Returns:
            Reward in range [-1.0, 1.0]
            - donation = 0 -> reward = -1
            - donation = 1 -> reward = 0
            - donation >= 2 -> reward = 1
        """
        # Clamp donations >= 2 to 2
        donation = min(donation, 2.0)
        # Linear mapping: (0, 2) -> (-1, 1)
        # Formula: reward = (donation - 1) / 1 = donation - 1
        return donation - 1.0


class TrainableAgentPersuasion:
    """
    Agent wrapper for Slime integration (matches τ²-bench pattern).

    Key pattern: Token-in-token-out tracking during conversation.
    """

    def __init__(
        self,
        task: PersuasionTask,
        env: PersuasionEnv,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any]
    ):
        """
        Initialize trainable agent.

        Args:
            task: PersuasionTask
            env: PersuasionEnv instance
            rollout_args: Rollout arguments from Slime
            sampling_params: Sampling parameters from Slime
        """
        self.task = task
        self.env = env
        self.rollout_args = rollout_args
        self.sampling_params = sampling_params

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an LLM call to sglang server"""
        return await post(url, payload)

    def _get_token_delta(
        self,
        tokenizer: AutoTokenizer,
        messages: list[dict],
        is_persuader_turn: bool
    ) -> tuple[list[int], list[int]]:
        """
        Calculate token delta for the last message added to the conversation.

        This method computes the incremental tokens added by the most recent message,
        enabling token-in-token-out tracking during rollout.

        Args:
            tokenizer: Tokenizer instance
            messages: Current conversation messages (including the new message)
            is_persuader_turn: True if last message is from persuader (loss_mask=1)

        Returns:
            Tuple of (token_ids, loss_mask) for the new message only
            - token_ids: New tokens added by last message
            - loss_mask: 1 for persuader tokens (trainable), 0 for persuadee tokens (environment)
        """
        # Apply chat template to current state
        curr = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Get previous state (before last message)
        prev_messages = messages[:-1]

        # Determine if we need generation prompt for previous state
        if messages[-1]["role"] == "assistant":
            # Assistant message: previous state had generation prompt
            prev = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Non-assistant: no generation prompt
            prev = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=False
            )

        # Extract delta (text added by new message)
        delta_text = curr[len(prev):]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)

        # Set loss mask based on role
        if is_persuader_turn:
            loss_mask = [1] * len(new_tokens)  # Train on persuader tokens
        else:
            loss_mask = [0] * len(new_tokens)  # Don't train on persuadee tokens

        return new_tokens, loss_mask

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
        """Build final result with accumulated tokens"""
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join([
            msg.get('content', '')
            for msg in messages
            if msg.get("role") == "assistant" and msg.get('content')
        ])
        res.response_length = len(loss_masks)

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"loss_mask_len={len(loss_masks)}, "
            f"prompt_token_len={len(prompt_token_ids)}, "
            f"response_token_len={len(response_token_ids)}"
        )
        return res

    async def asolve(
        self,
        task: PersuasionTask,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        max_turns: int
    ) -> InteractionResult:
        """
        Execute full conversation and return InteractionResult.

        This method uses INCREMENTAL TOKEN TRACKING (token-in-token-out):
        1. Initialize with persuader system prompt
        2. Loop: persuader generates -> TRACK TOKENS -> persuadee responds -> TRACK TOKENS
        3. Tokens and loss masks accumulated during rollout

        Args:
            task: PersuasionTask
            rollout_args: Rollout arguments from Slime
            sampling_params: Sampling parameters from Slime
            max_turns: Maximum conversation turns

        Returns:
            InteractionResult
        """
        # Initialize
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"

        # Build initial messages (persuader system prompt only)
        messages: list[dict] = [
            {"role": "system", "content": self.env.get_persuader_system_prompt()}
        ]

        # Tokenize initial system prompt
        prompt_text = state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Initialize token tracking lists (token-in-token-out)
        response_token_ids = []
        loss_masks = []

        # Initialize tracking
        total_reward = 0.0
        terminated = False
        truncated = False

        # Initialize result
        res = InteractionResult(
            prompt=task.conversation_id,
            reward=0,
            messages=[],
            info={"task_id": task.conversation_id},
            tokens=[],
            response="",
            loss_mask=[],
            status=Status.COMPLETED
        )

        # Main interaction loop
        for turn in range(max_turns):
            # Prepare text for LLM (persuader generation)
            text_input = state.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Call LLM via sglang for persuader response
            try:
                payload = {"text": text_input, "sampling_params": sampling_params}
                output = await self._call_llm(url, payload)

                # Check for abort
                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, {"error": "LLM aborted"},
                        messages, loss_masks, prompt_token_ids, response_token_ids
                    )

                persuader_response = output["text"].strip()

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, {"error": f"LLM failed: {str(e)}"},
                    messages, loss_masks, prompt_token_ids, response_token_ids
                )

            # IMMEDIATELY track persuader tokens (token-in-token-out)
            messages.append({"role": "assistant", "content": persuader_response})

            # Get token delta for persuader response
            token_delta, mask_delta = self._get_token_delta(
                state.tokenizer, messages, is_persuader_turn=True
            )
            response_token_ids.extend(token_delta)
            loss_masks.extend(mask_delta)

            # Build persuadee messages (from persuadee's perspective)
            # Persuader messages appear as "user" to persuadee
            persuadee_messages = []
            for msg in messages[1:]:  # Skip system prompt
                if msg["role"] == "assistant":
                    persuadee_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "user":
                    persuadee_messages.append({"role": "assistant", "content": msg["content"]})

            # Persuadee turn: Call environment
            try:
                persuadee_response = await self.env.step_persuadee(persuadee_messages)
            except Exception as e:
                logger.warning(f"Environment step failed: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, {"error": f"Env failed: {str(e)}"},
                    messages, loss_masks, prompt_token_ids, response_token_ids
                )

            # IMMEDIATELY track persuadee tokens (token-in-token-out)
            messages.append({"role": "user", "content": persuadee_response})

            # Get token delta for persuadee response
            env_token_ids, env_loss_mask = self._get_token_delta(
                state.tokenizer, messages, is_persuader_turn=False
            )
            response_token_ids.extend(env_token_ids)
            loss_masks.extend(env_loss_mask)

            # Check termination conditions
            if self.env.donation_amount is not None:
                # Donation made - episode completed
                terminated = True
                total_reward = PersuasionEnv.donation_to_reward(self.env.donation_amount)
                break

            if turn + 1 >= max_turns:
                # Max turns reached - episode truncated
                truncated = True
                break

        # Determine final status and reward
        if terminated:
            res.status = Status.COMPLETED
        elif truncated:
            res.status = Status.TRUNCATED
            # No donation made in truncated episodes - treat as $0 donation
            total_reward = PersuasionEnv.donation_to_reward(0.0)  # reward = -1.0
        else:
            res.status = Status.ABORTED
            # Aborted episodes also get $0 donation reward
            total_reward = PersuasionEnv.donation_to_reward(0.0)  # reward = -1.0

        # Build info
        info = {
            "task_id": task.conversation_id,
            "turns": self.env.turn_count,
            "donation": self.env.donation_amount,
            "persuader_id": task.persuader_id,
            "persuadee_id": task.persuadee_id,
            "ground_truth_donation": task.ground_truth_donation,
            "completed": terminated,
            "truncated": truncated
        }

        res = self._build_final_result(
            res,
            total_reward,
            info,
            messages,
            loss_masks,
            prompt_token_ids,
            response_token_ids
        )

        return res


def agent_factory_persuasion(
    task: PersuasionTask,
    persuadee_model: str,
    persuadee_base_url: str,
    persuadee_api_key_var: str,
    max_turns: int,
    rollout_args: dict[str, Any],
    sampling_params: dict[str, Any],
    persona_manager: Optional[PersonaManager] = None
):
    """
    Factory function to create trainable persuasion agents.

    Args:
        task: PersuasionTask
        persuadee_model: Model for persuadee simulation
        persuadee_base_url: API base URL
        persuadee_api_key_var: Environment variable for API key
        max_turns: Maximum conversation turns
        rollout_args: Rollout configuration
        sampling_params: LLM sampling parameters
        persona_manager: PersonaManager instance

    Returns:
        Agent factory that creates agents for specific tasks
    """

    class AgentFactory:
        def __init__(self):
            self.task = task
            self.persuadee_model = persuadee_model
            self.persuadee_base_url = persuadee_base_url
            self.persuadee_api_key_var = persuadee_api_key_var
            self.max_turns = max_turns
            self.rollout_args = rollout_args
            self.sampling_params = sampling_params
            self.persona_manager = persona_manager

        async def asolve(self, task, rollout_args, sampling_params, max_turns):
            """Create agent for specific task and solve it"""
            # Create environment
            env = PersuasionEnv(
                task=task,
                persuadee_model=self.persuadee_model,
                persuadee_base_url=self.persuadee_base_url,
                persuadee_api_key=os.getenv(self.persuadee_api_key_var),
                max_turns=max_turns,
                persona_manager=self.persona_manager
            )

            # Create agent
            agent = TrainableAgentPersuasion(
                task=task,
                env=env,
                rollout_args=rollout_args,
                sampling_params=sampling_params
            )

            return await agent.asolve(task, rollout_args, sampling_params, max_turns)

    return AgentFactory()
