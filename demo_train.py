"""
RLVR Live Demo — fail-safe, fast, visual.

Designed to run during a live presentation (2-3 min on an A10).
Uses expert iteration (rejection sampling + SFT) to demonstrate
the RLVR concept: generate -> verify -> learn from correct -> repeat.

Produces:
  - Live terminal output with progress bars and emoji-free status
  - An HTML report with interactive Plotly charts (training curves)
  - A JSON log for the presentation to consume

Usage:
    # Standard demo (OLMo-1B, ~2 min on A10)
    python demo_train.py

    # Dry run — no GPU, uses mock data to test the visual pipeline
    python demo_train.py --dry-run

    # Custom model
    python demo_train.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "demo_output"
DATA_DIR = Path(__file__).parent / "data"

# ---- Demo configuration ----
# These are deliberately small for speed + reliability

DEFAULT_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
NUM_PROBLEMS = 20          # training problems (small for speed)
NUM_EVAL = 15              # evaluation problems
NUM_ROUNDS = 3             # expert iteration rounds
SAMPLES_PER_PROBLEM = 4    # generations per problem per round
MAX_NEW_TOKENS = 384
SFT_EPOCHS_PER_ROUND = 2
SFT_LR = 2e-5
LORA_R = 8
LORA_ALPHA = 16


def generate_html_report(log: dict, output_path: Path):
    """Generate a self-contained HTML report with Plotly charts."""
    rounds = log["rounds"]
    round_nums = [r["round"] for r in rounds]
    accuracies = [r["eval_accuracy"] for r in rounds]
    rewards = [r["avg_reward"] for r in rounds]
    correct_samples = [r["correct_samples_collected"] for r in rounds]

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>RLVR Demo Results</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{
    font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
    background: #09090b;
    color: #e4e4e7;
    margin: 0;
    padding: 2rem;
  }}
  h1 {{ color: #4d8eff; font-size: 1.8rem; }}
  h2 {{ color: #a1a1aa; font-size: 1.2rem; font-weight: normal; margin-top: 2rem; }}
  .chart {{ width: 100%; max-width: 800px; height: 350px; margin: 1rem 0; }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
  }}
  .stat {{
    background: #141418;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 1rem;
  }}
  .stat-value {{ font-size: 2rem; color: #4d8eff; }}
  .stat-label {{ font-size: 0.85rem; color: #71717a; }}
  .example {{
    background: #141418;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
  }}
  .pass {{ border-left: 3px solid #22c55e; }}
  .fail {{ border-left: 3px solid #ef4444; }}
  pre {{ margin: 0.5rem 0; white-space: pre-wrap; color: #a1a1aa; }}
  .label {{ color: #71717a; font-size: 0.75rem; text-transform: uppercase; }}
</style>
</head>
<body>

<h1>RLVR Code Demo Results</h1>
<p style="color:#71717a">Model: {log['model']} | {log['timestamp']}</p>

<div class="stats">
  <div class="stat">
    <div class="stat-value">{accuracies[0]:.0f}%</div>
    <div class="stat-label">Baseline pass@1</div>
  </div>
  <div class="stat">
    <div class="stat-value">{accuracies[-1]:.0f}%</div>
    <div class="stat-label">Final pass@1</div>
  </div>
  <div class="stat">
    <div class="stat-value">+{accuracies[-1] - accuracies[0]:.0f}%</div>
    <div class="stat-label">Improvement</div>
  </div>
  <div class="stat">
    <div class="stat-value">{log['total_time_s']:.0f}s</div>
    <div class="stat-label">Total time</div>
  </div>
</div>

<h2>Training Curve: pass@1 over expert iteration rounds</h2>
<div id="accuracy-chart" class="chart"></div>

<h2>Reward signal: fraction of correct generations per round</h2>
<div id="reward-chart" class="chart"></div>

<h2>Correct samples collected per round (SFT training data)</h2>
<div id="samples-chart" class="chart"></div>

<script>
const plotLayout = {{
  paper_bgcolor: '#09090b',
  plot_bgcolor: '#141418',
  font: {{ color: '#a1a1aa', family: 'JetBrains Mono, monospace', size: 12 }},
  xaxis: {{ gridcolor: '#27272a', title: 'Round' }},
  yaxis: {{ gridcolor: '#27272a' }},
  margin: {{ l: 60, r: 20, t: 20, b: 50 }},
}};

Plotly.newPlot('accuracy-chart', [{{
  x: {json.dumps(round_nums)},
  y: {json.dumps(accuracies)},
  type: 'scatter',
  mode: 'lines+markers',
  line: {{ color: '#4d8eff', width: 3 }},
  marker: {{ size: 10 }},
}}], {{...plotLayout, yaxis: {{...plotLayout.yaxis, title: 'pass@1 (%)', range: [0, 100]}}}});

Plotly.newPlot('reward-chart', [{{
  x: {json.dumps(round_nums)},
  y: {json.dumps(rewards)},
  type: 'scatter',
  mode: 'lines+markers',
  line: {{ color: '#22c55e', width: 3 }},
  marker: {{ size: 10 }},
}}], {{...plotLayout, yaxis: {{...plotLayout.yaxis, title: 'Avg reward', range: [0, 1]}}}});

Plotly.newPlot('samples-chart', [{{
  x: {json.dumps(round_nums)},
  y: {json.dumps(correct_samples)},
  type: 'bar',
  marker: {{ color: '#4d8eff' }},
}}], {{...plotLayout, yaxis: {{...plotLayout.yaxis, title: 'Correct samples'}}}});
</script>

<h2>Before / After Examples</h2>
"""

    for ex in log.get("examples", []):
        status_before = "pass" if ex["before_passed"] else "fail"
        status_after = "pass" if ex["after_passed"] else "fail"
        html += f"""
<div class="example {status_before}">
  <div class="label">Round 0 (baseline) — {"PASS" if ex["before_passed"] else "FAIL"}</div>
  <strong>Task {ex['task_id']}: {ex['prompt'][:100]}...</strong>
  <pre>{ex['before_code'][:400]}</pre>
</div>
<div class="example {status_after}">
  <div class="label">Round {log['rounds'][-1]['round']} (after RLVR) — {"PASS" if ex["after_passed"] else "FAIL"}</div>
  <pre>{ex['after_code'][:400]}</pre>
</div>
<br>
"""

    html += """
</body>
</html>
"""
    output_path.write_text(html)


def dry_run():
    """Generate report with mock data — for testing visuals without GPU."""
    print("\n=== DRY RUN (no GPU, mock data) ===\n")

    log = {
        "model": "mock-model",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_time_s": 42.0,
        "rounds": [
            {"round": 0, "eval_accuracy": 15.0, "avg_reward": 0.15,
             "correct_samples_collected": 0, "sft_loss": None},
            {"round": 1, "eval_accuracy": 25.0, "avg_reward": 0.28,
             "correct_samples_collected": 12, "sft_loss": 1.8},
            {"round": 2, "eval_accuracy": 35.0, "avg_reward": 0.38,
             "correct_samples_collected": 18, "sft_loss": 1.4},
            {"round": 3, "eval_accuracy": 45.0, "avg_reward": 0.48,
             "correct_samples_collected": 22, "sft_loss": 1.1},
        ],
        "examples": [
            {
                "task_id": 602,
                "prompt": "Write a python function to find the first repeated character",
                "before_passed": False,
                "before_code": "def first_repeated_char(s):\n    return s[0]  # wrong",
                "after_passed": True,
                "after_code": "def first_repeated_char(str1):\n  for index,c in enumerate(str1):\n    if str1[:index+1].count(c) > 1:\n      return c",
            },
            {
                "task_id": 604,
                "prompt": "Write a function to reverse words in a string",
                "before_passed": False,
                "before_code": "def reverse_words(s):\n    return s[::-1]  # reverses chars not words",
                "after_passed": True,
                "after_code": 'def reverse_words(s):\n    return " ".join(s.split()[::-1])',
            },
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / "demo_report.html"
    generate_html_report(log, html_path)
    print(f"Report: {html_path}")

    json_path = OUTPUT_DIR / "demo_log.json"
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log:    {json_path}")
    print("\nOpen the HTML file in a browser to preview the visuals.")


def run_demo(model_name: str):
    """Run the full live demo with real model training."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset as HFDataset
    from verifier import extract_code_from_response, verify_code

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- Load data ----
    print("\n[1/6] Loading data...")
    train_data = json.load(open(DATA_DIR / "code_rlvr_train.json"))
    test_data = json.load(open(DATA_DIR / "code_rlvr_test.json"))

    random.seed(42)
    train_subset = random.sample(train_data, min(NUM_PROBLEMS, len(train_data)))
    eval_subset = random.sample(test_data, min(NUM_EVAL, len(test_data)))
    print(f"       Train: {len(train_subset)} problems, Eval: {len(eval_subset)} problems")

    # ---- Load model ----
    print(f"\n[2/6] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Helper functions ----
    def generate_one(prompt_text: str) -> str:
        messages = [{"role": "user", "content": prompt_text}]
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"### Instruction:\n{prompt_text}\n\n### Response:\n"

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    def evaluate_model(examples: list[dict]) -> tuple[float, list[dict]]:
        correct = 0
        details = []
        for ex in examples:
            prompt = ex["messages"][0]["content"]
            response = generate_one(prompt)
            code = extract_code_from_response(response)
            result = verify_code(
                code, ex["ground_truth"],
                ex.get("test_imports"), ex.get("test_setup_code", ""),
            )
            if result["passed"]:
                correct += 1
            details.append({
                "task_id": ex["task_id"],
                "prompt": prompt[:120],
                "code": code[:500],
                "passed": result["passed"],
            })
        accuracy = correct / len(examples) * 100
        return accuracy, details

    # ---- Expert iteration loop ----
    log = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rounds": [],
        "examples": [],
    }

    # Round 0: Baseline
    print(f"\n[3/6] Baseline evaluation ({len(eval_subset)} problems)...")
    baseline_acc, baseline_details = evaluate_model(eval_subset)
    print(f"       Baseline pass@1: {baseline_acc:.1f}%")

    log["rounds"].append({
        "round": 0,
        "eval_accuracy": baseline_acc,
        "avg_reward": baseline_acc / 100,
        "correct_samples_collected": 0,
        "sft_loss": None,
    })

    # Rounds 1..N: Generate -> Verify -> SFT -> Eval
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n[{3 + round_num}/6] Round {round_num}/{NUM_ROUNDS}")

        # Generate + verify
        print(f"       Generating {SAMPLES_PER_PROBLEM} samples x "
              f"{len(train_subset)} problems...")
        correct_pairs = []
        total_generated = 0
        total_correct = 0

        for ex in train_subset:
            prompt = ex["messages"][0]["content"]
            for _ in range(SAMPLES_PER_PROBLEM):
                response = generate_one(prompt)
                code = extract_code_from_response(response)
                result = verify_code(
                    code, ex["ground_truth"],
                    ex.get("test_imports"), ex.get("test_setup_code", ""),
                )
                total_generated += 1
                if result["passed"]:
                    total_correct += 1
                    # Build SFT training pair
                    correct_pairs.append({
                        "prompt": prompt,
                        "completion": code,
                    })

        avg_reward = total_correct / max(total_generated, 1)
        print(f"       Correct: {total_correct}/{total_generated} "
              f"(reward: {avg_reward:.2f})")

        if not correct_pairs:
            print("       No correct samples — skipping SFT this round")
            acc, _ = evaluate_model(eval_subset)
            log["rounds"].append({
                "round": round_num,
                "eval_accuracy": acc,
                "avg_reward": avg_reward,
                "correct_samples_collected": 0,
                "sft_loss": None,
            })
            continue

        # SFT on correct samples
        print(f"       SFT on {len(correct_pairs)} correct samples "
              f"({SFT_EPOCHS_PER_ROUND} epochs)...")

        def tokenize_pair(pair):
            messages = [
                {"role": "user", "content": pair["prompt"]},
                {"role": "assistant", "content": pair["completion"]},
            ]
            if tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False
                )
            else:
                text = (f"### Instruction:\n{pair['prompt']}\n\n"
                        f"### Response:\n{pair['completion']}")

            tokens = tokenizer(text, truncation=True, max_length=1024,
                               padding="max_length", return_tensors="pt")
            tokens["labels"] = tokens["input_ids"].clone()
            return {k: v.squeeze(0) for k, v in tokens.items()}

        tokenized = [tokenize_pair(p) for p in correct_pairs]
        sft_dataset = HFDataset.from_dict({
            k: [t[k] for t in tokenized]
            for k in tokenized[0].keys()
        })

        sft_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / f"sft_round_{round_num}"),
            num_train_epochs=SFT_EPOCHS_PER_ROUND,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=SFT_LR,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=model,
            args=sft_args,
            train_dataset=sft_dataset,
        )

        train_result = trainer.train()
        sft_loss = train_result.training_loss

        print(f"       SFT loss: {sft_loss:.3f}")

        # Evaluate
        print(f"       Evaluating...")
        acc, details = evaluate_model(eval_subset)
        print(f"       pass@1: {acc:.1f}% (delta: {acc - baseline_acc:+.1f}%)")

        log["rounds"].append({
            "round": round_num,
            "eval_accuracy": acc,
            "avg_reward": avg_reward,
            "correct_samples_collected": len(correct_pairs),
            "sft_loss": sft_loss,
        })

    # ---- Collect before/after examples ----
    print(f"\n[6/6] Collecting before/after examples...")
    _, final_details = evaluate_model(eval_subset)

    for i in range(min(5, len(baseline_details))):
        b = baseline_details[i]
        a = final_details[i]
        log["examples"].append({
            "task_id": b["task_id"],
            "prompt": b["prompt"],
            "before_passed": b["passed"],
            "before_code": b["code"],
            "after_passed": a["passed"],
            "after_code": a["code"],
        })

    # ---- Save results ----
    total_time = time.time() - t_start
    log["total_time_s"] = round(total_time, 1)

    html_path = OUTPUT_DIR / "demo_report.html"
    generate_html_report(log, html_path)

    json_path = OUTPUT_DIR / "demo_log.json"
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # Save the adapter
    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # ---- Final summary ----
    rounds = log["rounds"]
    print(f"\n{'='*60}")
    print(f"DEMO COMPLETE in {total_time:.0f}s")
    print(f"{'='*60}")
    print(f"  Baseline:     {rounds[0]['eval_accuracy']:.1f}%")
    print(f"  Final:        {rounds[-1]['eval_accuracy']:.1f}%")
    print(f"  Improvement:  {rounds[-1]['eval_accuracy'] - rounds[0]['eval_accuracy']:+.1f}%")
    print(f"\n  Report: {html_path}")
    print(f"  Log:    {json_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"\n  Open the HTML report in a browser to see training curves!")


def main():
    parser = argparse.ArgumentParser(description="RLVR Live Demo")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate report with mock data (no GPU needed)")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        run_demo(args.model)


if __name__ == "__main__":
    main()
