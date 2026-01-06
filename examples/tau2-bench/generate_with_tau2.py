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
    "max_turns": int(os.getenv("TAU2_MAX_TURNS", "30")),  # Maximum total turns (user + agent messages)
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

    # Extract turns from info for metrics tracking
    num_turns = res.info.get("turns", 0) if res.info else 0

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata={**res.info, "num_turns": num_turns} if res.info else {"num_turns": num_turns},
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

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for τ²-bench.

    This is the main entry point for slime training. It creates a τ²-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Supports per-sample configuration via sample.metadata:
    - During evaluation: metadata_overrides from YAML config provide user_model, domain, etc.
    - During multi-domain training: metadata from JSONL provides domain and task_split
    - Single-domain training: uses global TAU2_CONFIGS

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field and optional metadata
        sampling_params: LLM sampling parameters
        evaluation: Whether this is an evaluation run (kept for backward compatibility)

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for τ²-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)

    # Get config from metadata if available (for eval or multi-domain training), else use global config
    if sample.metadata:
        # Use per-sample config from metadata (set in YAML for eval, or in JSONL for multi-domain training)
        user_model = sample.metadata.get("user_model", TAU2_CONFIGS["user_model"])
        user_base_url = sample.metadata.get("user_base_url", TAU2_CONFIGS["user_base_url"])
        user_api_key_var = sample.metadata.get("user_api_key_var", TAU2_CONFIGS["user_api_key_var"])
        domain = sample.metadata.get("domain", TAU2_CONFIGS["domain"])
        task_split = sample.metadata.get("task_split", TAU2_CONFIGS["task_split"])
        max_turns = sample.metadata.get("max_turns", TAU2_CONFIGS["max_turns"])
    else:
        # Use global config when no metadata is available
        user_model = TAU2_CONFIGS["user_model"]
        user_base_url = TAU2_CONFIGS["user_base_url"]
        user_api_key_var = TAU2_CONFIGS["user_api_key_var"]
        domain = TAU2_CONFIGS["domain"]
        task_split = TAU2_CONFIGS["task_split"]
        max_turns = TAU2_CONFIGS["max_turns"]

    # Get the task from τ²-bench
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split,
    )

    if task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} out of range. Available tasks: {len(tasks)}")

    task = tasks[task_index]

    # Create trainable agent with τ²-bench's official components
    agent = agent_factory_tau2(
        domain=domain,
        agent_type=TAU2_CONFIGS["agent_type"],
        user_model=user_model,
        user_base_url=user_base_url,
        user_api_key_var=user_api_key_var,
        max_turns=max_turns,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # The task object is passed directly to the agent
    interaction_result = await agent.asolve(
        task=task,
        rollout_args=args,
        sampling_params=sampling_params,
        max_turns=max_turns,
    )

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    return result_sample
