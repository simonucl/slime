"""
Persuasion for Good integration with Slime RL training.

Entry point for generating persuasion conversations following the τ²-bench pattern.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

from slime.utils.types import Sample
from persona_manager import PersonaManager, PersuasionTask, PersonaAttributes
from trainable_agents_persuasion import (
    InteractionResult,
    Status,
    agent_factory_persuasion,
)

# Set up logger
logger = logging.getLogger(__name__)

# Global configuration (matches τ²-bench pattern)
PERSUASION_CONFIGS = {
    # Corpus and task config
    "corpus_path": os.getenv(
        "PERSUASION_CORPUS_PATH",
        os.path.join(os.path.dirname(__file__), "data")
    ),
    "task_split": os.getenv("PERSUASION_TASK_SPLIT", "train"),

    # Persuadee simulator config
    "persuadee_model": os.getenv("PERSUASION_PERSUADEE_MODEL", "gpt-4o-mini"),
    "persuadee_base_url": os.getenv(
        "PERSUASION_PERSUADEE_BASE_URL",
        "https://api.openai.com/v1"
    ),
    "persuadee_api_key_var": os.getenv(
        "PERSUASION_PERSUADEE_API_KEY_VAR",
        "OPENAI_API_KEY"
    ),

    # Conversation config
    "max_turns": int(os.getenv("PERSUASION_MAX_TURNS", "10")),

    # Persuadee model rotation (optional, matches τ²-bench pattern)
    "persuadee_model_rotation": (
        os.getenv("PERSUASION_PERSUADEE_MODEL_ROTATION", "").split(",")
        if os.getenv("PERSUASION_PERSUADEE_MODEL_ROTATION")
        else None
    ),
}

# Global PersonaManager instance (lazy initialization)
_persona_manager: Optional[PersonaManager] = None


def get_persona_manager() -> PersonaManager:
    """Get or create global PersonaManager instance"""
    global _persona_manager
    if _persona_manager is None:
        corpus_path = PERSUASION_CONFIGS["corpus_path"]
        logger.info(f"Initializing PersonaManager from {corpus_path}")
        _persona_manager = PersonaManager(corpus_path)
    return _persona_manager


def deserialize_task(task_data: dict) -> PersuasionTask:
    """
    Deserialize task data from JSON dict to PersuasionTask object.

    Args:
        task_data: Dictionary containing full task data from JSONL

    Returns:
        PersuasionTask object
    """
    return PersuasionTask(
        conversation_id=task_data["conversation_id"],
        persuader_id=task_data["persuader_id"],
        persuadee_id=task_data["persuadee_id"],
        split=task_data["split"],
        persuader_persona=PersonaAttributes(**task_data["persuader_persona"]),
        persuadee_persona=PersonaAttributes(**task_data["persuadee_persona"]),
        ground_truth_donation=task_data.get("ground_truth_donation"),
        dialogue_id=task_data.get("dialogue_id")
    )


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Slime Sample format.

    Matches τ²-bench pattern from generate_with_tau2.py:46-98

    Args:
        res: InteractionResult from agent.asolve()
        task_index: Task index (conversation ID as int)

    Returns:
        Sample for Slime training
    """
    # Map Status enum to string
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }

    # Calculate metrics
    num_turns = res.info.get("turns", 0)
    completed = res.info.get("completed", False)
    truncated = res.info.get("truncated", False)

    # Build metadata
    metadata = {
        **res.info,
        "num_turns": num_turns,
        "completed": completed,
        "truncated": truncated,
        "donation": res.reward,
    }

    return Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status_mapping.get(res.status, "aborted"),
        metadata=metadata,
        response_length=res.response_length
    )


async def generate(
    args: dict[str, Any],
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False
) -> Sample:
    """
    Main entry point for Slime integration.

    Matches τ²-bench pattern from generate_with_tau2.py:101-199

    Configuration hierarchy (priority order):
    1. sample.metadata (eval overrides from YAML)
    2. Rotation (if enabled, training only)
    3. Global PERSUASION_CONFIGS

    Args:
        args: Rollout arguments from Slime
        sample: Input sample with full task data (JSON) in prompt field
        sampling_params: LLM sampling parameters
        evaluation: Whether this is evaluation mode

    Returns:
        Sample with interaction results
    """
    # Load task using index from sample.prompt
    # When --input-key index is used, sample.prompt contains the index as a string (e.g. "0", "1", "2")
    try:
        task_index = int(sample.prompt)

        # Get task split from metadata if available (for eval), else use global config
        if sample.metadata and "task_split" in sample.metadata:
            task_split = sample.metadata["task_split"]
        else:
            task_split = PERSUASION_CONFIGS["task_split"]

        # Load task file
        task_file = os.path.join(
            PERSUASION_CONFIGS["corpus_path"],
            f"{task_split}_tasks.jsonl"
        )

        # Read task at index
        with open(task_file, 'r') as f:
            for i, line in enumerate(f):
                if i == task_index:
                    task_data = json.loads(line)
                    task = deserialize_task(task_data)
                    break
            else:
                raise ValueError(f"Task index {task_index} not found in {task_file}")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load task: {e}")
        return Sample(
            index=sample.index,
            prompt=sample.prompt if sample.prompt else f"index:{sample.index}",
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            status="aborted",
            metadata={"error": f"Task loading failed: {str(e)}"},
            response_length=0
        )

    conversation_id = task.conversation_id

    # Configuration hierarchy: metadata > rotation > global
    if sample.metadata:
        # Priority 1: Metadata overrides (for eval)
        persuadee_model = sample.metadata.get(
            "persuadee_model",
            PERSUASION_CONFIGS["persuadee_model"]
        )
        persuadee_base_url = sample.metadata.get(
            "persuadee_base_url",
            PERSUASION_CONFIGS["persuadee_base_url"]
        )
        persuadee_api_key_var = sample.metadata.get(
            "persuadee_api_key_var",
            PERSUASION_CONFIGS["persuadee_api_key_var"]
        )
        max_turns = sample.metadata.get(
            "max_turns",
            PERSUASION_CONFIGS["max_turns"]
        )
    else:
        # Priority 2: Check rotation (training only)
        rotation_models = PERSUASION_CONFIGS.get("persuadee_model_rotation")
        if rotation_models and len(rotation_models) > 1 and not evaluation:
            # Rotate based on sample index
            model_idx = sample.index % len(rotation_models)
            persuadee_model = rotation_models[model_idx].strip()
            logger.info(
                f"Using rotated persuadee model {model_idx + 1}/{len(rotation_models)}: "
                f"{persuadee_model}"
            )
            # Use global config for other params
            persuadee_base_url = PERSUASION_CONFIGS["persuadee_base_url"]
            persuadee_api_key_var = PERSUASION_CONFIGS["persuadee_api_key_var"]
        else:
            # Priority 3: Global config
            persuadee_model = PERSUASION_CONFIGS["persuadee_model"]
            persuadee_base_url = PERSUASION_CONFIGS["persuadee_base_url"]
            persuadee_api_key_var = PERSUASION_CONFIGS["persuadee_api_key_var"]

        max_turns = PERSUASION_CONFIGS["max_turns"]

    logger.info(
        f"Processing conversation {conversation_id}: "
        f"persuader={task.persuader_id}, persuadee={task.persuadee_id}, "
        f"model={persuadee_model}"
    )

    # Create agent via factory
    # Note: PersonaManager is lazy-initialized inside PersuasionEnv if needed for persona descriptions
    agent = agent_factory_persuasion(
        task=task,
        persuadee_model=persuadee_model,
        persuadee_base_url=persuadee_base_url,
        persuadee_api_key_var=persuadee_api_key_var,
        max_turns=max_turns,
        rollout_args=args,
        sampling_params=sampling_params,
        persona_manager=get_persona_manager()  # Still needed for persona description generation
    )

    # Execute conversation
    try:
        interaction_result = await agent.asolve(
            task=task,
            rollout_args=args,
            sampling_params=sampling_params,
            max_turns=max_turns
        )
    except Exception as e:
        logger.error(f"Error executing conversation {conversation_id}: {e}")
        # Return aborted sample
        return Sample(
            index=sample.index,
            prompt=conversation_id,
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            status="aborted",
            metadata={"error": str(e)},
            response_length=0
        )

    # Convert to Sample
    result_sample = res_to_sample(interaction_result, sample.index)

    logger.info(
        f"Completed conversation {conversation_id}: "
        f"status={result_sample.status}, reward={result_sample.reward:.2f}, "
        f"turns={result_sample.metadata.get('num_turns', 0)}"
    )

    return result_sample


# For backward compatibility with Slime's dynamic import
__all__ = ["generate", "PERSUASION_CONFIGS"]
