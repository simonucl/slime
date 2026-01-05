"""
τ²-bench Integration for slime Training

This module provides the main interface for training agents in τ²-bench environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.

Based on the verifiers implementation pattern and adapted for τ²-bench's official API.
"""

import logging
import os
from typing import Any

from trainable_agents_tau2 import InteractionResult, Status, agent_factory_tau2

from slime.utils.types import Sample
from tau2.run import get_tasks

# Set up logger for this module
logger = logging.getLogger(__name__)

# τ²-bench configuration
# Can be overridden by environment variables
TAU2_CONFIGS = {
    "domain": os.getenv("TAU2_DOMAIN", "retail"),  # Select between ["retail", "airline", "telecom", "mock"]
    "agent_type": os.getenv("TAU2_AGENT_TYPE", "llm"),  # Select between ["llm", "solo", "gt"]
    "user_model": os.getenv("TAU2_USER_MODEL", "gpt-4o-mini"),  # Model for user simulator
    "user_base_url": os.getenv("TAU2_USER_BASE_URL", "https://api.openai.com/v1"),
    "user_api_key_var": os.getenv("TAU2_USER_API_KEY_VAR", "OPENAI_API_KEY"),
    "task_split": os.getenv("TAU2_TASK_SPLIT", "train"),  # Select between ["train", "test"]
    "max_steps": int(os.getenv("TAU2_MAX_STEPS", "200")),  # Maximum number of steps per episode
    "max_errors": int(os.getenv("TAU2_MAX_ERRORS", "10")),  # Maximum number of errors before termination
    "max_turns": int(os.getenv("TAU2_MAX_TURNS", "30")),  # Maximum number of agent turns
}

# Ensure API key is set
OPENAI_API_KEY = os.getenv(TAU2_CONFIGS["user_api_key_var"])
if OPENAI_API_KEY is None:
    logger.warning(f"{TAU2_CONFIGS['user_api_key_var']} is not set. User simulation may fail.")


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    This function transforms the τ²-bench interaction result into the format
    expected by slime's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from τ²-bench agent
        task_index: Index of the task being processed

    Returns:
        Sample object for slime training
    """
    # Map τ²-bench status to slime status
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    status = status_mapping.get(res.status, "aborted")

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for τ²-bench.

    This is the main entry point for slime training. It creates a τ²-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for τ²-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for τ²-bench task {task_index}")

    # Get the task from τ²-bench
    tasks = get_tasks(
        task_set_name=TAU2_CONFIGS["domain"],
        task_split_name=TAU2_CONFIGS["task_split"],
    )

    if task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} out of range. Available tasks: {len(tasks)}")

    task = tasks[task_index]
    logger.info(f"Task ID: {task.id}, Description: {task.user_scenario.persona if hasattr(task.user_scenario, 'persona') else 'N/A'}")

    # Create trainable agent with τ²-bench's official components
    agent = agent_factory_tau2(
        domain=TAU2_CONFIGS["domain"],
        agent_type=TAU2_CONFIGS["agent_type"],
        user_model=TAU2_CONFIGS["user_model"],
        user_base_url=TAU2_CONFIGS["user_base_url"],
        user_api_key_var=TAU2_CONFIGS["user_api_key_var"],
        max_steps=TAU2_CONFIGS["max_steps"],
        max_errors=TAU2_CONFIGS["max_errors"],
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # The task object is passed directly to the agent
    interaction_result = await agent.asolve(
        task=task,
        rollout_args=args,
        sampling_params=sampling_params,
        max_num_steps=TAU2_CONFIGS["max_turns"],
    )

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    logger.info(f"Finished agent-environment interaction for τ²-bench task {task_index}")
    logger.info(f"Reward: {result_sample.reward}, Status: {result_sample.status}")

    return result_sample
