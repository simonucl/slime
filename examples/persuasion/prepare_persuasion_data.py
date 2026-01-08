"""
Prepare Persuasion for Good dataset for Slime training.

This script creates JSONL files with full task data (including personas) for training and evaluation.
Unlike tau2-bench (which uses indices), we serialize complete task information to avoid extra loading.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from persona_manager import PersonaManager


def serialize_task(task) -> dict:
    """
    Serialize PersuasionTask to JSON-serializable dict.

    Includes all task information: conversation_id, personas, split, ground_truth_donation.
    This avoids needing to reload PersonaManager during training.
    """
    return {
        "conversation_id": task.conversation_id,
        "persuader_id": task.persuader_id,
        "persuadee_id": task.persuadee_id,
        "split": task.split,
        "persuader_persona": asdict(task.persuader_persona),
        "persuadee_persona": asdict(task.persuadee_persona),
        "ground_truth_donation": task.ground_truth_donation,
        "dialogue_id": task.dialogue_id
    }


def create_persuasion_tasks(
    corpus_path: str,
    split: str,
    output_path: str,
    limit: int = None
):
    """
    Create JSONL file with full task data for training or evaluation.

    Args:
        corpus_path: Path to persuasionforgood_corpus directory
        split: "train" or "test"
        output_path: Output JSONL file path
        limit: Optional limit on number of tasks (for testing)
    """
    print(f"Loading Persuasion for Good corpus from: {corpus_path}")
    persona_manager = PersonaManager(corpus_path)

    print(f"Getting tasks for split: {split}")
    tasks = persona_manager.get_tasks_by_split(split)

    if limit:
        tasks = tasks[:limit]

    print(f"Found {len(tasks)} tasks for split '{split}'")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Write JSONL with full task data (not just indices)
    with open(output_path, 'w') as f:
        for idx, task in enumerate(tasks):
            # Each line contains complete task information including personas
            # This avoids extra loading during training (unlike tau2-bench's index-only approach)
            task_data = serialize_task(task)
            # Add index field for compatibility with --input-key index
            task_data["index"] = str(idx)
            f.write(json.dumps(task_data) + "\n")

    print(f"Created {output_path} with {len(tasks)} tasks")

    # Print statistics
    donations = [t.ground_truth_donation for t in tasks if t.ground_truth_donation is not None]
    if donations:
        print(f"\nGround truth donation statistics:")
        print(f"  Tasks with donations: {len(donations)}/{len(tasks)}")
        print(f"  Min donation: ${min(donations):.2f}")
        print(f"  Max donation: ${max(donations):.2f}")
        print(f"  Avg donation: ${sum(donations) / len(donations):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Persuasion for Good dataset for Slime training"
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to persuasionforgood_corpus directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSONL files (default: same as corpus-path)"
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Limit number of training tasks (for testing)"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit number of test tasks (for testing)"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.corpus_path

    # Create training tasks
    print("=" * 60)
    print("Creating training tasks")
    print("=" * 60)
    create_persuasion_tasks(
        corpus_path=args.corpus_path,
        split="train",
        output_path=os.path.join(output_dir, "train_tasks.jsonl"),
        limit=args.train_limit
    )

    # Create test tasks
    print("\n" + "=" * 60)
    print("Creating test tasks")
    print("=" * 60)
    create_persuasion_tasks(
        corpus_path=args.corpus_path,
        split="test",
        output_path=os.path.join(output_dir, "test_tasks.jsonl"),
        limit=args.test_limit
    )

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nTask files created in: {output_dir}")
    print(f"  - train_tasks.jsonl (with full task data)")
    print(f"  - test_tasks.jsonl (with full task data)")


if __name__ == "__main__":
    main()
