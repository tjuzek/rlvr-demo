"""
Prepare GSM8K data for RLVR:
  - Training: allenai/RLVR-GSM (7,473 examples, 8-shot CoT prompts, numeric ground truth).
  - Held-out eval: openai/gsm8k test split (1,319 examples), formatted with the same 8-shot prefix.

Both splits are saved to math-rlvr/data/ so the pipeline is self-contained
for Lambda (no dependence on the parent rlvr/data/ directory).

Usage:
    python prepare_data.py
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "data"


def extract_prefix(prompt: str) -> str:
    """Grab everything before the final 'Question:' in an 8-shot CoT prompt."""
    idx = prompt.rfind("Question:")
    if idx == -1:
        raise ValueError("No 'Question:' marker in prompt — unexpected format.")
    return prompt[:idx]


def gsm8k_answer_to_int(answer: str) -> str:
    """Extract the final numeric answer after the GSM8K '####' marker."""
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", answer)
    if not m:
        raise ValueError(f"No '####' marker in GSM8K answer: {answer[:100]}...")
    return m.group(1).replace(",", "")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading training set (allenai/RLVR-GSM)...")
    train_ds = load_dataset("allenai/RLVR-GSM", split="train")
    print(f"  {len(train_ds)} training examples")

    train_out = []
    for i, ex in enumerate(train_ds):
        train_out.append({
            "task_id": i,
            "messages": ex["messages"],
            "ground_truth": str(ex["ground_truth"]).strip(),
            "dataset": ex.get("dataset", "gsm8k"),
        })

    train_path = OUTPUT_DIR / "rlvr_gsm_train.json"
    with open(train_path, "w") as f:
        json.dump(train_out, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {train_path}")

    prefix = extract_prefix(train_out[0]["messages"][0]["content"])
    print(f"  8-shot prefix length: {len(prefix)} chars")

    print("\nDownloading held-out test set (openai/gsm8k main test)...")
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  {len(test_ds)} test examples")

    test_out = []
    for i, ex in enumerate(test_ds):
        question = ex["question"].strip()
        ground_truth = gsm8k_answer_to_int(ex["answer"])
        prompt_text = prefix + f"Question: {question}"
        test_out.append({
            "task_id": i,
            "messages": [{"role": "user", "content": prompt_text}],
            "ground_truth": ground_truth,
            "reference_answer": ex["answer"],
            "question": question,
            "dataset": "gsm8k",
        })

    test_path = OUTPUT_DIR / "rlvr_gsm_test.json"
    with open(test_path, "w") as f:
        json.dump(test_out, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {test_path}")

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_out)} examples -> {train_path.name}")
    print(f"  Test:  {len(test_out)} examples -> {test_path.name}")

    print("\n--- Example (train) ---")
    ex = train_out[0]
    tail = ex["messages"][0]["content"][-180:]
    print(f"  Prompt tail: ...{tail}")
    print(f"  Ground truth: {ex['ground_truth']}")

    print("\n--- Example (test) ---")
    ex = test_out[0]
    tail = ex["messages"][0]["content"][-180:]
    print(f"  Prompt tail: ...{tail}")
    print(f"  Ground truth: {ex['ground_truth']}")
    print(f"  Reference answer tail: ...{ex['reference_answer'][-100:]}")


if __name__ == "__main__":
    main()
