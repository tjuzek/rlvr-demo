"""
Download MBPP and format it as an RLVR dataset for code generation.

MBPP (Mostly Basic Python Problems) has ~1000 problems, each with:
  - A natural language description
  - A reference solution
  - Test assertions

We format this for RLVR: the prompt asks the model to write code,
and the verifier executes the code against the test assertions.

Usage:
    python download_dataset.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "data"
SEED = 42


def format_for_rlvr(example: dict, source: str = "sanitized") -> dict:
    """Convert an MBPP example into RLVR format.

    The sanitized split uses 'prompt' for the description,
    the full split uses 'text'.
    """
    description = example.get("prompt") or example.get("text", "")
    user_prompt = (
        f"Write a Python function to solve the following problem.\n\n"
        f"{description}\n\n"
        f"Your response should contain only the function definition. "
        f"Do not include test cases or example usage."
    )

    # Sanitized split has test_imports; full split has test_setup_code
    test_imports = example.get("test_imports", [])
    test_setup = example.get("test_setup_code", "")

    return {
        "messages": [{"role": "user", "content": user_prompt}],
        "ground_truth": example["test_list"],
        "test_imports": test_imports if test_imports else [],
        "test_setup_code": test_setup if test_setup else "",
        "reference_code": example["code"],
        "task_id": example["task_id"],
        "dataset": "mbpp",
        "source": source,
    }


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    # The sanitized split has verified, high-quality problems
    # It has train/validation/test splits
    print(f"  Splits: {list(ds.keys())}")
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data)} examples")

    # Use the provided splits
    train_examples = [format_for_rlvr(ex, "sanitized") for ex in ds["train"]]
    val_examples = [format_for_rlvr(ex, "sanitized") for ex in ds["validation"]]
    test_examples = [format_for_rlvr(ex, "sanitized") for ex in ds["test"]]

    # Also load the full (non-sanitized) dataset for extra training data
    ds_full = load_dataset("google-research-datasets/mbpp", "full")
    print(f"\n  Full dataset: {len(ds_full['train'])} examples")

    # Get task IDs already in sanitized set
    sanitized_ids = set()
    for split in ds.values():
        for ex in split:
            sanitized_ids.add(ex["task_id"])

    # Add non-sanitized examples to training pool
    extra_train = []
    for ex in ds_full["train"]:
        if int(ex["task_id"]) not in sanitized_ids:
            extra_train.append(format_for_rlvr(ex, "full"))

    print(f"  Extra training examples (non-sanitized): {len(extra_train)}")

    # Combine training data: sanitized train + validation + extra
    # Keep sanitized test set clean for evaluation
    all_train = train_examples + val_examples + extra_train
    random.shuffle(all_train)

    print(f"\nFinal splits:")
    print(f"  Train: {len(all_train)} examples")
    print(f"  Test:  {len(test_examples)} examples")

    # Save
    train_path = OUTPUT_DIR / "code_rlvr_train.json"
    test_path = OUTPUT_DIR / "code_rlvr_test.json"

    with open(train_path, "w") as f:
        json.dump(all_train, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved training set -> {train_path}")

    with open(test_path, "w") as f:
        json.dump(test_examples, f, indent=2, ensure_ascii=False)
    print(f"  Saved test set -> {test_path}")

    # Print a few examples
    print("\n--- Example (train) ---")
    ex = all_train[0]
    print(f"  Prompt: {ex['messages'][0]['content'][:150]}...")
    print(f"  Tests:  {ex['ground_truth']}")
    print(f"  Reference: {ex['reference_code'][:150]}...")

    print("\n--- Example (test) ---")
    ex = test_examples[0]
    print(f"  Prompt: {ex['messages'][0]['content'][:150]}...")
    print(f"  Tests:  {ex['ground_truth']}")


if __name__ == "__main__":
    main()
