"""
Pre/post RLVR benchmark for GSM8K math.

Generates answers for each held-out GSM8K problem, extracts the final
numeric answer, compares to ground truth, reports pass@1.

Output schema matches code-rlvr/benchmark.py so make_report.py works unchanged.

Usage:
    # Baseline
    python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct \
        --output results/baseline.json

    # Post-RLVR (with LoRA adapter)
    python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct \
        --adapter output/final --output results/post_rlvr.json

    # Quick smoke test
    python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct --quick

    # Compare two runs
    python benchmark.py --compare results/baseline.json results/post_rlvr.json

    # Pipeline health checks (no GPU)
    python benchmark.py --checks-only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import extract_answer, verify_answer

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_model(model_name: str, adapter_path: str | None = None):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]

    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=3072).to(model.device)

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
    if max_examples:
        examples = examples[:max_examples]

    results = []
    correct = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        prompt = ex["messages"][0]["content"]

        gen_start = time.time()
        response = generate_answer(model, tokenizer, prompt)
        gen_time = time.time() - gen_start

        vresult = verify_answer(response, ex["ground_truth"])
        passed = vresult["passed"]
        if passed:
            correct += 1

        result_entry = {
            "task_id": ex["task_id"],
            "passed": passed,
            "predicted": vresult["predicted"],
            "expected": vresult["expected"],
            "generated_code": response[:1500],  # reuse field name for make_report.py compat
            "raw_response": response[:2000],
            "reference_answer": ex.get("reference_answer", "")[:800],
            "question": ex.get("question", ""),
            "error": vresult["error"],
            # Kept for schema parity with code-rlvr results:
            "num_tests_passed": 1 if passed else 0,
            "num_tests_total": 1,
            "gen_time_s": round(gen_time, 1),
        }
        results.append(result_entry)

        status = "PASS" if passed else "FAIL"
        acc_so_far = correct / (i + 1) * 100
        print(f"  [{i+1:4d}/{len(examples)}] Task {ex['task_id']:4d}: {status}  "
              f"pred={vresult['predicted']!s:>10}  gt={ex['ground_truth']:>8}  "
              f"(running: {acc_so_far:.1f}%, {gen_time:.1f}s)")

        if verbose and not passed:
            print(f"           response tail: ...{response[-120:]}")

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


def spot_check(results: dict, n: int = 3):
    entries = results["results"]
    passes = [r for r in entries if r["passed"]]
    fails = [r for r in entries if not r["passed"]]

    print(f"\n{'='*70}")
    print(f"SPOT CHECK — {n} passes, {n} failures")
    print(f"{'='*70}")

    for label, subset in [("PASS", passes[:n]), ("FAIL", fails[:n])]:
        for r in subset:
            print(f"\n--- [{label}] Task {r['task_id']} ---")
            print(f"  predicted: {r['predicted']!r}   expected: {r['expected']!r}")
            print(f"  response tail ({r['gen_time_s']}s):")
            tail = r["generated_code"][-300:]
            for line in tail.split("\n"):
                print(f"    {line}")


def compare_runs(before_path: str, after_path: str):
    before = json.load(open(before_path))
    after = json.load(open(after_path))

    print(f"\n{'='*70}")
    print(f"COMPARISON: {before.get('model', '?')} (before vs. after RLVR)")
    print(f"{'='*70}")
    print(f"  Before: {before['accuracy']:.2f}% ({before['correct']}/{before['total']})")
    print(f"  After:  {after['accuracy']:.2f}% ({after['correct']}/{after['total']})")
    print(f"  Delta:  {after['accuracy'] - before['accuracy']:+.2f}pp")
    print()

    before_map = {r["task_id"]: r for r in before["results"]}
    after_map = {r["task_id"]: r for r in after["results"]}

    fixed, regressed = [], []
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
        print(f"\n  Fixed tasks (first 20): {fixed[:20]}")
    if regressed:
        print(f"  Regressed tasks (first 20): {regressed[:20]}")


def run_pipeline_checks(test_data: list[dict]):
    print("\n--- Pipeline Health Checks ---")

    print(f"\n[CHECK 1] Dataset integrity")
    print(f"  Test examples loaded: {len(test_data)}")
    assert len(test_data) > 0, "No test data!"
    for key in ["messages", "ground_truth", "task_id"]:
        assert key in test_data[0], f"Missing key: {key}"
    print(f"  Required fields present: OK")

    print(f"\n[CHECK 2] Verifier extracts ground truth from reference answers (20 samples)")
    sample = test_data[:20]
    hits = 0
    for ex in sample:
        ref = ex.get("reference_answer", "")
        if not ref:
            continue
        extracted = extract_answer(ref)
        if extracted is not None and extracted.lstrip("-0").lstrip(".") == \
                ex["ground_truth"].lstrip("-0").lstrip("."):
            hits += 1
    print(f"  {hits}/{len(sample)} reference answers parse correctly "
          f"({hits/len(sample)*100:.0f}%)")
    if hits < len(sample) * 0.9:
        print(f"  WARNING: verifier regex may be too strict — inspect failures.")

    print(f"\n[CHECK 3] Verifier round-trip on ground truth strings")
    for ex in sample[:5]:
        gt = ex["ground_truth"]
        synthetic = f"So the answer is {gt}."
        r = verify_answer(synthetic, gt)
        assert r["passed"], f"Round-trip failed for {gt}: {r}"
    print(f"  5/5 round-trips pass")

    print(f"\n--- All pipeline checks passed ---\n")


def main():
    parser = argparse.ArgumentParser(description="RLVR Math Benchmark (GSM8K)")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Run on 5 examples only")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()

    test_path = DATA_DIR / "rlvr_gsm_test.json"

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return

    if not test_path.exists():
        raise SystemExit(
            f"Test set not found at {test_path}. Run `python prepare_data.py` first."
        )

    test_data = json.load(open(test_path))

    if args.checks_only:
        run_pipeline_checks(test_data)
        return

    if not args.model:
        parser.error("--model is required (unless using --compare or --checks-only)")

    run_pipeline_checks(test_data)

    model, tokenizer = load_model(args.model, args.adapter)

    max_ex = 5 if args.quick else args.max_examples

    label = "post-RLVR" if args.adapter else "baseline"
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {args.model} ({label}) on GSM8K test")
    print(f"{'='*70}")

    results = evaluate(model, tokenizer, test_data, max_ex, args.verbose)

    print(f"\n{'='*70}")
    print(f"RESULTS: pass@1 = {results['accuracy']:.2f}% "
          f"({results['correct']}/{results['total']}) "
          f"in {results['elapsed_s']}s")
    print(f"{'='*70}")

    spot_check(results, n=3)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(
        RESULTS_DIR / f"{label.replace('-','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_data = {
        "model": args.model,
        "adapter": args.adapter,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "dataset": "gsm8k",
        **results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
