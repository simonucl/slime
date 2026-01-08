#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openai
from tqdm.asyncio import tqdm

# Add examples/persuasion to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from persona_manager import PersonaAttributes, PersonaManager, PersuasionTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one persuasion conversation."""
    example_id: str  # conversation_id
    prompt: dict  # Full task data
    completion: dict  # Full conversation history
    reward: float  # Donation amount
    error: str | None  # Error message if failed
    evaluation_criteria: dict  # Detailed metrics
    metadata: dict  # Additional info (turns, models, etc.)


def deserialize_task(task_data: dict) -> PersuasionTask:
    """Deserialize task data from JSON dict to PersuasionTask object."""
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


def build_persuader_system_prompt(
    task: PersuasionTask,
    persona_manager: PersonaManager,
    max_turns: int
) -> str:
    """
    Build persuader system prompt matching training exactly.

    Based on trainable_agents_persuasion.py get_persuader_system_prompt()
    """
    persona_desc = persona_manager.generate_persona_description(task.persuader_persona)

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
- Guide the conversation for about {max_turns // 2} chat turns.

Basic information about Save the Children:
Save the Children is an international non-governmental organization that promotes children's rights, provides relief and helps support children in developing countries.
</task>

<rules>
- Your responses can sometimes contain minor grammatical errors to sound more natural.
- Your response should not exceed 30 words.
- Only output your reply to your chat partner—do not explain your reasoning.
</rules>"""
    return prompt


def build_persuadee_system_prompt(
    task: PersuasionTask,
    persona_manager: PersonaManager,
    last_persuader_msg: str
) -> str:
    """
    Build persuadee system prompt matching training exactly.

    Based on trainable_agents_persuasion.py get_persuadee_system_prompt()
    """
    persona_desc = persona_manager.generate_persona_description(task.persuadee_persona)

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


def extract_donation_amount(text: str) -> float | None:
    """
    Extract donation amount from persuadee response.

    Based on persuasion_simulation/src/evaluation/ground_truth/extractor.py
    """
    import re

    # Pattern: [DONATE $X] or [GIVE $X]
    patterns = [
        r'\[DONATE\s+\$?(\d+(?:\.\d{1,2})?)\]',
        r'\[GIVE\s+\$?(\d+(?:\.\d{1,2})?)\]',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


async def run_conversation(
    task: PersuasionTask,
    persona_manager: PersonaManager,
    persuader_client: openai.AsyncOpenAI,
    persuader_model: str,
    persuadee_client: openai.AsyncOpenAI,
    persuadee_model: str,
    max_turns: int = 10,
    temperature: float = 1.0,
    timeout_per_turn: float = 60.0,
) -> EvalResult:
    """
    Run one persuasion conversation and return detailed results.

    Args:
        task: PersuasionTask with personas and metadata
        persona_manager: PersonaManager for generating persona descriptions
        persuader_client: OpenAI client for persuader
        persuader_model: Model name for persuader
        persuadee_client: OpenAI client for persuadee
        persuadee_model: Model name for persuadee
        max_turns: Maximum conversation turns
        temperature: Sampling temperature

    Returns:
        EvalResult with conversation details and metrics
    """
    # Build persuader system prompt (static)
    persuader_system = build_persuader_system_prompt(task, persona_manager, max_turns)

    # Conversation history
    persuader_messages = [{"role": "system", "content": persuader_system}]
    # Persuadee messages will be rebuilt each turn with dynamic prompt

    conversation_history = []
    donation_amount = 0.0
    error_message = None
    num_turns = 0

    try:
        # Initial persuader message
        for turn in range(max_turns):
            num_turns = turn + 1

            # Persuader turn
            try:
                persuader_response = await asyncio.wait_for(
                    persuader_client.chat.completions.create(
                        model=persuader_model,
                        messages=persuader_messages,
                        temperature=temperature,
                        max_tokens=512,
                    ),
                    timeout=timeout_per_turn
                )
                persuader_content = persuader_response.choices[0].message.content
            except asyncio.TimeoutError:
                error_message = f"Persuader API timeout after {timeout_per_turn}s"
                logger.error(f"Task {task.conversation_id}: {error_message}")
                break
            except Exception as e:
                error_message = f"Persuader API error: {str(e)}"
                logger.error(f"Task {task.conversation_id}: {error_message}")
                break

            persuader_messages.append({"role": "assistant", "content": persuader_content})
            conversation_history.append({
                "role": "persuader",
                "content": persuader_content,
                "turn": turn + 1
            })

            # Build dynamic persuadee system prompt with last persuader message
            persuadee_system = build_persuadee_system_prompt(
                task, persona_manager, persuader_content
            )

            # Rebuild persuadee messages with dynamic system prompt
            persuadee_messages = [{"role": "system", "content": persuadee_system}]

            # Add conversation history from persuadee's perspective
            for msg in conversation_history:
                if msg["role"] == "persuader":
                    persuadee_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "persuadee":
                    persuadee_messages.append({"role": "assistant", "content": msg["content"]})

            # Persuadee turn
            try:
                persuadee_response = await asyncio.wait_for(
                    persuadee_client.chat.completions.create(
                        model=persuadee_model,
                        messages=persuadee_messages,
                        temperature=temperature,
                        max_tokens=512,
                    ),
                    timeout=timeout_per_turn
                )
                persuadee_content = persuadee_response.choices[0].message.content
            except asyncio.TimeoutError:
                error_message = f"Persuadee API timeout after {timeout_per_turn}s"
                logger.error(f"Task {task.conversation_id}: {error_message}")
                break
            except Exception as e:
                error_message = f"Persuadee API error: {str(e)}"
                logger.error(f"Task {task.conversation_id}: {error_message}")
                break

            persuader_messages.append({"role": "user", "content": persuadee_content})
            conversation_history.append({
                "role": "persuadee",
                "content": persuadee_content,
                "turn": turn + 1
            })

            # Check for donation
            donation = extract_donation_amount(persuadee_content)
            if donation is not None:
                donation_amount = donation
                break

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(f"Task {task.conversation_id}: {error_message}")

    # Build evaluation criteria
    evaluation_criteria = {
        "donation_amount": donation_amount,
        "success": donation_amount > 0.0,
        "turns_used": num_turns,
        "completed": error_message is None,
    }

    # Compare with ground truth if available
    if task.ground_truth_donation is not None:
        evaluation_criteria["ground_truth_donation"] = task.ground_truth_donation
        evaluation_criteria["donation_difference"] = donation_amount - task.ground_truth_donation

    return EvalResult(
        example_id=task.conversation_id,
        prompt={
            "conversation_id": task.conversation_id,
            "persuader_persona": asdict(task.persuader_persona),
            "persuadee_persona": asdict(task.persuadee_persona),
        },
        completion={
            "messages": conversation_history,
            "final_donation": donation_amount,
        },
        reward=donation_amount,
        error=error_message,
        evaluation_criteria=evaluation_criteria,
        metadata={
            "persuader_model": persuader_model,
            "persuadee_model": persuadee_model,
            "max_turns": max_turns,
            "temperature": temperature,
            "ground_truth_donation": task.ground_truth_donation,
        }
    )


async def evaluate_tasks(
    tasks: list[PersuasionTask],
    persona_manager: PersonaManager,
    persuader_model: str,
    persuader_base_url: str,
    persuader_api_key: str,
    persuadee_model: str,
    persuadee_base_url: str,
    persuadee_api_key: str,
    max_concurrent: int = 128,
    max_turns: int = 10,
    temperature: float = 1.0,
    timeout_per_turn: float = 60.0,
) -> list[EvalResult]:
    """
    Evaluate multiple tasks concurrently.

    Returns list of EvalResult objects.
    """
    # Create API clients
    persuader_client = openai.AsyncOpenAI(
        base_url=persuader_base_url,
        api_key=persuader_api_key,
    )
    persuadee_client = openai.AsyncOpenAI(
        base_url=persuadee_base_url,
        api_key=persuadee_api_key,
    )

    # Run evaluations with concurrency limit and progress tracking
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task):
        async with semaphore:
            logger.info(f"Evaluating task {task.conversation_id}")
            return await run_conversation(
                task=task,
                persona_manager=persona_manager,
                persuader_client=persuader_client,
                persuader_model=persuader_model,
                persuadee_client=persuadee_client,
                persuadee_model=persuadee_model,
                max_turns=max_turns,
                temperature=temperature,
                timeout_per_turn=timeout_per_turn,
            )

    # Create all tasks
    tasks_coroutines = [run_with_semaphore(task) for task in tasks]

    # Run with progress bar that updates as each episode completes
    results = []
    for coro in tqdm.as_completed(tasks_coroutines, total=len(tasks), desc="Evaluating conversations", unit="episode"):
        result = await coro
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone evaluation for Persuasion for Good"
    )

    # Persuader (policy) configuration
    parser.add_argument(
        "--persuader-model",
        type=str,
        required=True,
        help="Persuader model name (e.g., 'openrouter/deepseek/deepseek-chat')"
    )
    parser.add_argument(
        "--persuader-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Persuader API base URL"
    )
    parser.add_argument(
        "--persuader-api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for persuader API key"
    )

    # Persuadee (environment) configuration
    parser.add_argument(
        "--persuadee-model",
        type=str,
        required=True,
        help="Persuadee model name (e.g., 'gpt-4o-mini')"
    )
    parser.add_argument(
        "--persuadee-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Persuadee API base URL"
    )
    parser.add_argument(
        "--persuadee-api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for persuadee API key"
    )

    # Data configuration
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to persuasionforgood_corpus directory"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to test_tasks.jsonl file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for results JSONL file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (for testing)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--timeout-per-turn",
        type=float,
        default=60.0,
        help="Timeout in seconds for each API call (default: 60.0)"
    )

    args = parser.parse_args()

    # Get API keys from environment
    persuader_api_key = os.getenv(args.persuader_api_key_var)
    if not persuader_api_key:
        raise ValueError(f"Environment variable {args.persuader_api_key_var} not set")

    persuadee_api_key = os.getenv(args.persuadee_api_key_var)
    if not persuadee_api_key:
        raise ValueError(f"Environment variable {args.persuadee_api_key_var} not set")

    # Load PersonaManager
    logger.info(f"Loading PersonaManager from {args.corpus_path}")
    persona_manager = PersonaManager(args.corpus_path)

    # Load tasks
    logger.info(f"Loading tasks from {args.data_path}")
    tasks = []
    with open(args.data_path, "r") as f:
        for line in f:
            task_data = json.loads(line)
            task = deserialize_task(task_data)
            tasks.append(task)

    if args.limit:
        tasks = tasks[:args.limit]

    logger.info(f"Loaded {len(tasks)} tasks")
    logger.info(f"Persuader model: {args.persuader_model}")
    logger.info(f"Persuadee model: {args.persuadee_model}")

    # Run evaluation
    results = asyncio.run(
        evaluate_tasks(
            tasks=tasks,
            persona_manager=persona_manager,
            persuader_model=args.persuader_model,
            persuader_base_url=args.persuader_base_url,
            persuader_api_key=persuader_api_key,
            persuadee_model=args.persuadee_model,
            persuadee_base_url=args.persuadee_base_url,
            persuadee_api_key=persuadee_api_key,
            max_concurrent=args.max_concurrent,
            max_turns=args.max_turns,
            temperature=args.temperature,
            timeout_per_turn=args.timeout_per_turn,
        )
    )

    # Write results to JSONL
    logger.info(f"Writing results to {args.output_path}")
    with open(args.output_path, "w") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")

    # Compute and log statistics
    successful = sum(1 for r in results if r.reward > 0.0)
    total_reward = sum(r.reward for r in results)
    avg_reward = total_reward / len(results) if results else 0.0
    errors = sum(1 for r in results if r.error is not None)

    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Total tasks: {len(results)}")
    logger.info(f"  Successful: {successful} ({successful/len(results)*100:.1f}%)")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Average donation: ${avg_reward:.2f}")
    logger.info(f"  Total donations: ${total_reward:.2f}")

    # Ground truth comparison if available
    gt_results = [r for r in results if r.metadata.get("ground_truth_donation") is not None]
    if gt_results:
        gt_avg = sum(r.metadata["ground_truth_donation"] for r in gt_results) / len(gt_results)
        logger.info(f"  Ground truth avg: ${gt_avg:.2f}")
        logger.info(f"  Difference: ${avg_reward - gt_avg:.2f}")


if __name__ == "__main__":
    main()
