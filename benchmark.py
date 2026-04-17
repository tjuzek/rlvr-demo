"""
Pre/post RLVR benchmark for code generation.

Evaluates a model on the MBPP test set using pass@1:
  - Generate code for each problem
  - Execute against test assertions (using the verifier)
  - Report accuracy

Run before training for baseline, after training for comparison.

Usage:
    # Baseline evaluation
    python benchmark.py --model allenai/OLMo-7B-Instruct

    # After RLVR training (with LoRA adapter)
    python benchmark.py --model allenai/OLMo-7B-Instruct --adapter ./output/checkpoint

    # Quick smoke test (5 problems)
    python benchmark.py --model allenai/OLMo-7B-Instruct --quick

    # Compare two runs
    python benchmark.py --compare results/baseline.json results/post_rlvr.json

Authored by Anthropic's Claude Opus 4.6 via the Claude Code CLI.
See README.md for full attribution. Maintainer: Tommie Juzek <tjuzek@fsu.edu>.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import extract_code_from_response, verify_code

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_model(model_name: str, adapter_path: str | None = None):
    """Load model and tokenizer, optionally with a LoRA adapter."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a code response from the model."""
    messages = [{"role": "user", "content": prompt}]

    # Use chat template if available, otherwise raw prompt
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def evaluate(
    model,
    tokenizer,
    examples: list[dict],
    max_examples: int | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate model on MBPP test examples.

    Returns a results dict with accuracy, per-example details, and timing.
    """
    if max_examples:
        examples = examples[:max_examples]

    results = []
    correct = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        prompt = ex["messages"][0]["content"]

        # Generate
        gen_start = time.time()
        response = generate_code(model, tokenizer, prompt)
        gen_time = time.time() - gen_start

        # Extract code from response
        code = extract_code_from_response(response)

        # Verify
        vresult = verify_code(
            code,
            ex["ground_truth"],
            ex.get("test_imports"),
            ex.get("test_setup_code", ""),
        )

        passed = vresult["passed"]
        if passed:
            correct += 1

        result_entry = {
            "task_id": ex["task_id"],
            "passed": passed,
            "num_tests_passed": vresult["num_passed"],
            "num_tests_total": vresult["num_tests"],
            "generated_code": code[:1000],
            "raw_response": response[:1500],
            "reference_code": ex["reference_code"],
            "error": vresult["error"][:300] if vresult["error"] else None,
            "gen_time_s": round(gen_time, 1),
        }
        results.append(result_entry)

        status = "PASS" if passed else "FAIL"
        acc_so_far = correct / (i + 1) * 100
        print(f"  [{i+1:3d}/{len(examples)}] Task {ex['task_id']:4d}: {status}  "
              f"(running: {acc_so_far:.1f}%, {gen_time:.1f}s)")

        if verbose and not passed:
            print(f"           Error: {vresult['error'][:100] if vresult['error'] else 'unknown'}")

    elapsed = time.time() - t0
    accuracy = correct / len(examples) * 100

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "elapsed_s": round(elapsed, 1),
        "avg_gen_time_s": round(elapsed / len(examples), 1),
        "results": results,
    }


def spot_check(results: dict, n: int = 5):
    """Print detailed spot checks: some passes, some failures."""
    entries = results["results"]

    passes = [r for r in entries if r["passed"]]
    fails = [r for r in entries if not r["passed"]]

    print(f"\n{'='*70}")
    print(f"SPOT CHECK — {n} passes, {n} failures")
    print(f"{'='*70}")

    for label, subset in [("PASS", passes[:n]), ("FAIL", fails[:n])]:
        for r in subset:
            print(f"\n--- [{label}] Task {r['task_id']} ---")
            print(f"Tests: {r['num_tests_passed']}/{r['num_tests_total']} passed")
            print(f"Generated ({r['gen_time_s']}s):")
            for line in r["generated_code"].split("\n")[:10]:
                print(f"  {line}")
            if r["error"]:
                print(f"Error: {r['error'][:150]}")


def compare_runs(before_path: str, after_path: str):
    """Compare two saved benchmark runs."""
    before = json.load(open(before_path))
    after = json.load(open(after_path))

    print(f"\n{'='*70}")
    print(f"COMPARISON: {before['model']} (before vs. after RLVR)")
    print(f"{'='*70}")
    print(f"  Before: {before['accuracy']:.1f}% ({before['correct']}/{before['total']})")
    print(f"  After:  {after['accuracy']:.1f}% ({after['correct']}/{after['total']})")
    print(f"  Delta:  {after['accuracy'] - before['accuracy']:+.1f}%")
    print()

    # Find problems that flipped
    before_map = {r["task_id"]: r for r in before["results"]}
    after_map = {r["task_id"]: r for r in after["results"]}

    fixed = []
    regressed = []
    for task_id in before_map:
        if task_id in after_map:
            b = before_map[task_id]["passed"]
            a = after_map[task_id]["passed"]
            if not b and a:
                fixed.append(task_id)
            elif b and not a:
                regressed.append(task_id)

    print(f"  Fixed (fail -> pass):      {len(fixed)}")
    print(f"  Regressed (pass -> fail):  {len(regressed)}")

    if fixed:
        print(f"\n  Fixed tasks: {fixed[:20]}")
    if regressed:
        print(f"  Regressed tasks: {regressed[:20]}")


def run_pipeline_checks(test_data: list[dict]):
    """Run spot checks on the pipeline itself (no model needed)."""
    print("\n--- Pipeline Health Checks ---")

    # Check 1: Dataset integrity
    print(f"\n[CHECK 1] Dataset integrity")
    print(f"  Test examples loaded: {len(test_data)}")
    assert len(test_data) > 0, "No test data!"
    for key in ["messages", "ground_truth", "reference_code", "task_id"]:
        assert key in test_data[0], f"Missing key: {key}"
    print(f"  Required fields present: OK")

    # Check 2: Reference solutions pass verifier
    print(f"\n[CHECK 2] Reference solutions (sample of 20)")
    sample = test_data[:20]
    passes = 0
    for ex in sample:
        r = verify_code(
            ex["reference_code"],
            ex["ground_truth"],
            ex.get("test_imports"),
            ex.get("test_setup_code", ""),
        )
        if r["passed"]:
            passes += 1
    print(f"  {passes}/{len(sample)} reference solutions pass ({passes/len(sample)*100:.0f}%)")
    assert passes == len(sample), "Some reference solutions fail!"

    # Check 3: Corrupted examples fail verifier
    corrupted_path = DATA_DIR / "code_rlvr_corrupted.json"
    if corrupted_path.exists():
        corrupted = json.load(open(corrupted_path))
        print(f"\n[CHECK 3] Corrupted examples (sample of 20)")
        sample_c = corrupted[:20]
        fails = 0
        for ex in sample_c:
            r = verify_code(
                ex["corrupted_code"],
                ex["ground_truth"],
                ex.get("test_imports"),
                ex.get("test_setup_code", ""),
            )
            if not r["passed"]:
                fails += 1
        print(f"  {fails}/{len(sample_c)} corrupted solutions fail ({fails/len(sample_c)*100:.0f}%)")

    print(f"\n--- All pipeline checks passed ---\n")


def main():
    parser = argparse.ArgumentParser(description="RLVR Code Benchmark")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--quick", action="store_true", help="Run on 5 examples only")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved result files")
    parser.add_argument("--checks-only", action="store_true",
                        help="Run pipeline health checks only (no model needed)")
    args = parser.parse_args()

    # Load test data
    test_path = DATA_DIR / "code_rlvr_test.json"
    test_data = json.load(open(test_path))

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return

    if args.checks_only:
        run_pipeline_checks(test_data)
        return

    if not args.model:
        parser.error("--model is required (unless using --compare or --checks-only)")

    # Run pipeline checks first
    run_pipeline_checks(test_data)

    # Load model
    model, tokenizer = load_model(args.model, args.adapter)

    # Determine example count
    max_ex = 5 if args.quick else args.max_examples

    # Evaluate
    label = "post-RLVR" if args.adapter else "baseline"
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {args.model} ({label})")
    print(f"{'='*70}")

    results = evaluate(model, tokenizer, test_data, max_ex, args.verbose)

    print(f"\n{'='*70}")
    print(f"RESULTS: pass@1 = {results['accuracy']:.1f}% "
          f"({results['correct']}/{results['total']}) "
          f"in {results['elapsed_s']}s")
    print(f"{'='*70}")

    # Spot checks
    spot_check(results, n=3)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(
        RESULTS_DIR / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_data = {
        "model": args.model,
        "adapter": args.adapter,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        **results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
