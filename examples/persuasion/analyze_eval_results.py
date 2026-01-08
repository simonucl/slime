#!/usr/bin/env python3
"""
Analyze evaluation results from eval_persuasion.py

Usage:
    python analyze_eval_results.py eval_results/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """
    Load all JSONL result files from directory.

    Returns: {filename: [result_dicts]}
    """
    results = {}
    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        file_results = []
        with open(jsonl_file, "r") as f:
            for line in f:
                file_results.append(json.loads(line))
        results[jsonl_file.name] = file_results

    return results


def compute_statistics(results: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for a set of results."""
    if not results:
        return {}

    total = len(results)
    successful = sum(1 for r in results if r["reward"] > 0.0)
    errors = sum(1 for r in results if r["error"] is not None)

    rewards = [r["reward"] for r in results]
    total_reward = sum(rewards)
    avg_reward = total_reward / total if total > 0 else 0.0
    max_reward = max(rewards) if rewards else 0.0

    turns = [r["evaluation_criteria"]["turns_used"] for r in results]
    avg_turns = sum(turns) / len(turns) if turns else 0.0

    # Ground truth comparison if available
    gt_donations = [
        r["metadata"]["ground_truth_donation"]
        for r in results
        if r["metadata"].get("ground_truth_donation") is not None
    ]
    gt_avg = sum(gt_donations) / len(gt_donations) if gt_donations else None

    return {
        "total": total,
        "successful": successful,
        "success_rate": successful / total if total > 0 else 0.0,
        "errors": errors,
        "error_rate": errors / total if total > 0 else 0.0,
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "total_reward": total_reward,
        "avg_turns": avg_turns,
        "ground_truth_avg": gt_avg,
        "donation_difference": avg_reward - gt_avg if gt_avg is not None else None,
    }


def print_statistics(filename: str, stats: dict[str, Any]):
    """Pretty print statistics for one result file."""
    print(f"\n{'=' * 80}")
    print(f"File: {filename}")
    print(f"{'=' * 80}")

    print(f"Total tasks:        {stats['total']}")
    print(f"Successful:         {stats['successful']} ({stats['success_rate']*100:.1f}%)")
    print(f"Errors:             {stats['errors']} ({stats['error_rate']*100:.1f}%)")
    print(f"Average donation:   ${stats['avg_reward']:.2f}")
    print(f"Max donation:       ${stats['max_reward']:.2f}")
    print(f"Total donations:    ${stats['total_reward']:.2f}")
    print(f"Average turns:      {stats['avg_turns']:.1f}")

    if stats.get("ground_truth_avg") is not None:
        print(f"\nGround truth comparison:")
        print(f"  GT average:       ${stats['ground_truth_avg']:.2f}")
        print(f"  Difference:       ${stats['donation_difference']:.2f}")


def compare_models(all_stats: dict[str, dict[str, Any]]):
    """Print comparison table across all model combinations."""
    print(f"\n{'=' * 80}")
    print("Model Comparison")
    print(f"{'=' * 80}")

    # Extract model names from filenames
    # Format: persuader_MODEL__persuadee_MODEL.jsonl
    model_pairs = []
    for filename in sorted(all_stats.keys()):
        parts = filename.replace(".jsonl", "").split("__")
        if len(parts) == 2:
            persuader = parts[0].replace("persuader_", "")
            persuadee = parts[1].replace("persuadee_", "")
            model_pairs.append((persuader, persuadee, all_stats[filename]))

    # Print table
    print(f"\n{'Persuader':<40} {'Persuadee':<40} {'Avg $':<8} {'Success%':<10} {'Avg Turns':<10}")
    print("-" * 110)

    for persuader, persuadee, stats in model_pairs:
        avg_reward = stats.get("avg_reward", 0.0)
        success_rate = stats.get("success_rate", 0.0)
        avg_turns = stats.get("avg_turns", 0.0)

        print(f"{persuader:<40} {persuadee:<40} ${avg_reward:<7.2f} {success_rate*100:<9.1f}% {avg_turns:<10.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Persuasion for Good evaluation results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing result JSONL files"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-file statistics"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Directory {args.results_dir} does not exist")
        sys.exit(1)

    # Load all results
    all_results = load_results(args.results_dir)

    if not all_results:
        print(f"No JSONL files found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(all_results)} result files")

    # Compute statistics for each file
    all_stats = {}
    for filename, results in all_results.items():
        stats = compute_statistics(results)
        all_stats[filename] = stats

        if args.detailed:
            print_statistics(filename, stats)

    # Print comparison table
    compare_models(all_stats)

    print()


if __name__ == "__main__":
    main()
