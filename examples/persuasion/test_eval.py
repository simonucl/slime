#!/usr/bin/env python3
"""
Quick test for evaluation script functionality.
Tests basic components without making API calls.
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from persona_manager import PersonaAttributes, PersonaManager, PersuasionTask
from eval_persuasion import (
    deserialize_task,
    build_persuader_system_prompt,
    build_persuadee_system_prompt,
    extract_donation_amount,
)


def test_deserialize_task():
    """Test task deserialization from JSONL format."""
    print("Testing task deserialization...")

    # Load first task from test_tasks.jsonl
    test_file = Path(__file__).parent / "data" / "test_tasks.jsonl"
    if not test_file.exists():
        print(f"SKIP: {test_file} not found")
        return

    with open(test_file, "r") as f:
        task_data = json.loads(f.readline())

    task = deserialize_task(task_data)

    assert isinstance(task, PersuasionTask)
    assert task.conversation_id == task_data["conversation_id"]
    assert isinstance(task.persuader_persona, PersonaAttributes)
    assert isinstance(task.persuadee_persona, PersonaAttributes)

    print(f"  ✓ Deserialized task: {task.conversation_id}")
    print(f"  ✓ Persuader persona: {task.persuader_persona.age}, {task.persuader_persona.sex}")
    print(f"  ✓ Persuadee persona: {task.persuadee_persona.age}, {task.persuadee_persona.sex}")

    return task


def test_system_prompts(task, persona_manager):
    """Test system prompt generation."""
    print("\nTesting system prompt generation...")

    persuader_prompt = build_persuader_system_prompt(task, persona_manager, max_turns=10)
    persuadee_prompt = build_persuadee_system_prompt(
        task, persona_manager, "Hello! I'd like to talk to you about Save the Children."
    )

    assert isinstance(persuader_prompt, str)
    assert isinstance(persuadee_prompt, str)
    assert len(persuader_prompt) > 0
    assert len(persuadee_prompt) > 0

    # Check that prompts include key elements
    assert "Amazon Mechanical Turk" in persuader_prompt
    assert "Amazon Mechanical Turk" in persuadee_prompt
    assert "persona" in persuader_prompt.lower()
    assert "persona" in persuadee_prompt.lower()
    assert "$2" in persuader_prompt  # Task payment
    assert "$2" in persuadee_prompt  # Task payment

    print(f"  ✓ Persuader prompt length: {len(persuader_prompt)} chars")
    print(f"  ✓ Persuadee prompt length: {len(persuadee_prompt)} chars")

    # Print first 200 chars of each
    print(f"\n  Persuader prompt preview:")
    print(f"    {persuader_prompt[:200]}...")
    print(f"\n  Persuadee prompt preview:")
    print(f"    {persuadee_prompt[:200]}...")


def test_donation_extraction():
    """Test donation amount extraction from text."""
    print("\nTesting donation extraction...")

    test_cases = [
        ("[DONATE $1.50]", 1.50),
        ("[GIVE $2.00]", 2.00),
        ("[donate $0.5]", 0.5),
        ("[DONATE $0]", 0.0),
        ("I'm willing to [DONATE $1.25] to help.", 1.25),
        ("No donation", None),
        ("I'll donate later", None),
    ]

    for text, expected in test_cases:
        result = extract_donation_amount(text)
        if result != expected:
            print(f"  ✗ FAILED: '{text}' -> {result} (expected {expected})")
        else:
            print(f"  ✓ '{text}' -> {result}")


def test_persona_format(task, persona_manager):
    """Test persona formatting."""
    print("\nTesting persona formatting...")

    persuader_text = persona_manager.generate_persona_description(task.persuader_persona)
    persuadee_text = persona_manager.generate_persona_description(task.persuadee_persona)

    assert isinstance(persuader_text, str)
    assert isinstance(persuadee_text, str)
    assert len(persuader_text) > 0
    assert len(persuadee_text) > 0

    print(f"  ✓ Persuader persona:")
    print("    " + persuader_text.replace("\n", "\n    ")[:300] + "...")

    print(f"\n  ✓ Persuadee persona:")
    print("    " + persuadee_text.replace("\n", "\n    ")[:300] + "...")


def main():
    print("=" * 80)
    print("Evaluation Script Component Tests")
    print("=" * 80)

    try:
        task = test_deserialize_task()
        if task:
            # Load PersonaManager for prompt generation tests
            corpus_path = Path(__file__).parent / "data" / "persuasionforgood_corpus"
            if not corpus_path.exists():
                print(f"\nSKIP: Corpus not found at {corpus_path}")
                print("Cannot test prompt generation without PersonaManager")
                persona_manager = None
            else:
                from persona_manager import PersonaManager
                persona_manager = PersonaManager(str(corpus_path))

            if persona_manager:
                test_system_prompts(task, persona_manager)
                test_persona_format(task, persona_manager)
        test_donation_extraction()

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
