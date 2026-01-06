"""
Prepare τ²-bench data for slime training.

This script creates JSONL files containing task indices for each domain and split.
Unlike tau1-bench which requires mock data generation, τ²-bench tasks are loaded
directly from the official repository, so we only need to create index files.

Usage:
    python prepare_tau2_data.py --output_dir /root/tau2_bench_data

Output:
    Creates JSONL files with format: {"index": 0}, {"index": 1}, ...
    Files created:
    - retail_train_tasks.jsonl
    - retail_dev_tasks.jsonl
    - retail_test_tasks.jsonl
    - airline_test_tasks.jsonl
    - telecom_test_tasks.jsonl
    - mock_test_tasks.jsonl
"""

import argparse
import json
import os
from pathlib import Path

from tau2.run import get_tasks


def prepare_domain_split_data(
    domain: str,
    split: str,
    output_dir: Path,
    include_domain_metadata: bool = False,
) -> int:
    """
    Prepare data for a specific domain and split.

    Args:
        domain: Domain name (retail, airline, telecom, mock)
        split: Split name (train, dev, test)
        output_dir: Output directory for JSONL files
        include_domain_metadata: If True, include domain and task_split in metadata

    Returns:
        Number of tasks created
    """
    # Load tasks from τ²-bench
    try:
        tasks = get_tasks(task_set_name=domain, task_split_name=split)
    except Exception as e:
        print(f"Warning: Could not load {domain}/{split}: {e}")
        return 0

    if not tasks:
        print(f"Warning: No tasks found for {domain}/{split}")
        return 0

    # Create output file
    output_file = output_dir / f"{domain}_{split}_tasks.jsonl"

    # Write task indices
    with open(output_file, "w") as f:
        for i in range(len(tasks)):
            if include_domain_metadata:
                entry = {
                    "index": i,
                    "metadata": {
                        "domain": domain,
                        "task_split": split,
                    }
                }
            else:
                entry = {"index": i}
            f.write(json.dumps(entry) + "\n")

    print(f"Created {output_file} with {len(tasks)} tasks")
    return len(tasks)


def create_multi_domain_dataset(
    domains: list[str],
    split: str,
    output_dir: Path,
    output_name: str = None,
) -> int:
    """
    Create a multi-domain dataset by concatenating tasks from multiple domains.

    Each task entry includes metadata indicating which domain it belongs to,
    so the generate function can load the correct task.

    Args:
        domains: List of domain names to concatenate (e.g., ["retail", "telecom"])
        split: Split name (train, test)
        output_dir: Output directory for JSONL files
        output_name: Name for output file (default: "multi_domain_{split}_tasks.jsonl")

    Returns:
        Total number of tasks created
    """
    if output_name is None:
        output_name = f"multi_domain_{split}_tasks.jsonl"

    output_file = output_dir / output_name
    total_tasks = 0

    print(f"\nCreating multi-domain dataset: {output_file}")
    print(f"  Domains: {', '.join(domains)}")
    print(f"  Split: {split}")

    with open(output_file, "w") as f:
        for domain in domains:
            try:
                tasks = get_tasks(task_set_name=domain, task_split_name=split)
            except Exception as e:
                print(f"  Warning: Could not load {domain}/{split}: {e}")
                continue

            if not tasks:
                print(f"  Warning: No tasks found for {domain}/{split}")
                continue

            # Write tasks with domain metadata
            for i in range(len(tasks)):
                entry = {
                    "index": i,
                    "metadata": {
                        "domain": domain,
                        "task_split": split,
                    }
                }
                f.write(json.dumps(entry) + "\n")

            print(f"  Added {len(tasks)} tasks from {domain}")
            total_tasks += len(tasks)

    print(f"Created {output_file} with {total_tasks} total tasks")
    return total_tasks


def main():
    parser = argparse.ArgumentParser(description="Prepare τ²-bench data for slime training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/tau2_bench_data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["retail", "airline", "telecom", "mock"],
        help="Domains to prepare (default: all)",
    )
    parser.add_argument(
        "--multi-domain-train",
        type=str,
        nargs="+",
        default=None,
        help="Create multi-domain training dataset from specified domains (e.g., --multi-domain-train retail telecom)",
    )
    parser.add_argument(
        "--multi-domain-name",
        type=str,
        default=None,
        help="Output name for multi-domain dataset (default: multi_domain_train_tasks.jsonl)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing τ²-bench data for slime training")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Domains: {', '.join(args.domains)}")
    print()

    # Domain-split mapping (which splits are available for each domain)
    domain_splits = {
        "retail": ["train", "test"],     # retail has train and test
        "airline": ["train", "test"],    # airline has train and test
        "telecom": ["train", "test"],    # telecom has train and test
        "mock": ["test"],                # mock only has test split
    }

    total_tasks = 0
    summary = []

    # Process each domain
    for domain in args.domains:
        if domain not in domain_splits:
            print(f"Warning: Unknown domain '{domain}', skipping")
            continue

        splits = domain_splits[domain]
        domain_tasks = 0

        print(f"\nProcessing domain: {domain}")
        print("-" * 80)

        for split in splits:
            num_tasks = prepare_domain_split_data(domain, split, output_dir)
            domain_tasks += num_tasks
            total_tasks += num_tasks

            if num_tasks > 0:
                summary.append(f"  {domain}/{split}: {num_tasks} tasks")

        print(f"Total for {domain}: {domain_tasks} tasks")

    # Create multi-domain dataset if requested
    if args.multi_domain_train:
        print()
        print("=" * 80)
        multi_tasks = create_multi_domain_dataset(
            domains=args.multi_domain_train,
            split="train",
            output_dir=output_dir,
            output_name=args.multi_domain_name,
        )
        summary.append(f"  Multi-domain train: {multi_tasks} tasks ({', '.join(args.multi_domain_train)})")
        total_tasks += multi_tasks

    # Print summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for line in summary:
        print(line)
    print(f"\nTotal tasks created: {total_tasks}")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("1. Set your API key for user simulation:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print()
    if args.multi_domain_train:
        multi_file = args.multi_domain_name or "multi_domain_train_tasks.jsonl"
        print("2. Update run_qwen3_4B.sh to use multi-domain dataset:")
        print(f"   --prompt-data {output_dir}/{multi_file}")
    else:
        print("2. Update run_qwen3_4B.sh with the correct paths:")
        print(f"   --prompt-data {output_dir}/retail_train_tasks.jsonl")
    print()
    print("3. Run training:")
    print("   bash examples/tau2-bench/run_qwen3_4B.sh")
    print("=" * 80)


if __name__ == "__main__":
    main()
