"""
RLVR training: fine-tune a language model on code generation with
verifiable rewards using GRPO (Group Relative Policy Optimization).

The pipeline:
  1. Load OLMo-7B (or any HF model) with LoRA + 4-bit quantization
  2. For each prompt, generate N completions
  3. Score each with the code verifier (pass=1, fail=0)
  4. Update the policy using GRPO — reward is relative to the group

This uses TRL's GRPOTrainer for clean integration.

Usage:
    # Full training on A10 (24GB)
    python train.py

    # Quick test (3 steps, 2 samples per prompt)
    python train.py --quick

    # Custom model
    python train.py --model allenai/Olmo-3-7B-Instruct

    # Resume from checkpoint
    python train.py --resume output/checkpoint-100

Authored by Anthropic's Claude (Opus 4.6 for the GRPO training loop,
Opus 4.7 for the MetricsCallback) via the Claude Code CLI.
See README.md for full attribution. Maintainer: Tommie Juzek <tjuzek@fsu.edu>.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from verifier import extract_code_from_response, verify_code


class MetricsCallback(TrainerCallback):
    """Stream trainer log_history to a JSONL file as logs are emitted.

    Writing per-log (not per-step) keeps the file small and means a mid-run
    crash still leaves a plottable partial curve — important for a one-shot
    demo run on an ephemeral Lambda instance.
    """

    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, **logs}
        with open(self.out_path, "a") as f:
            f.write(json.dumps(record) + "\n")

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---- Default Configuration ----

# Model: OLMo-2 7B Instruct (the Tulu 3 base model family)
# Swap this one line to try other models:
#   allenai/Olmo-3-7B-Instruct      (newest OLMo)
#   allenai/OLMo-7B-0724-Instruct-hf  (older, stable)
#   meta-llama/Llama-3.2-3B-Instruct  (smaller, for testing)
DEFAULT_MODEL = "allenai/OLMo-2-1124-7B-Instruct"

# GRPO hyperparameters
NUM_GENERATIONS = 4       # completions per prompt (the "group" in GRPO)
MAX_NEW_TOKENS = 512      # max tokens per completion
LEARNING_RATE = 5e-6
NUM_TRAIN_STEPS = 200     # total optimization steps
BATCH_SIZE = 4            # prompts per batch
GRADIENT_ACCUMULATION = 2
KL_COEFF = 0.05           # KL penalty vs reference model
WARMUP_RATIO = 0.1
LOGGING_STEPS = 5
SAVE_STEPS = 50
EVAL_STEPS = 50

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def build_dataset(data_path: Path) -> Dataset:
    """Load MBPP training data and format for GRPOTrainer.

    GRPOTrainer expects a dataset with a 'prompt' column where each
    entry is a list of message dicts (chat format).
    """
    raw = json.load(open(data_path))

    prompts = []
    metadata = []
    for ex in raw:
        prompts.append(ex["messages"])
        metadata.append({
            "task_id": ex["task_id"],
            "ground_truth": ex["ground_truth"],
            "test_imports": ex.get("test_imports", []),
            "test_setup_code": ex.get("test_setup_code", ""),
        })

    return Dataset.from_dict({
        "prompt": prompts,
        "task_id": [m["task_id"] for m in metadata],
        "ground_truth": [json.dumps(m["ground_truth"]) for m in metadata],
        "test_imports": [json.dumps(m["test_imports"]) for m in metadata],
        "test_setup_code": [m["test_setup_code"] for m in metadata],
    })


def make_reward_function():
    """Create the reward function for GRPO.

    This closure returns a function that:
      1. Extracts code from each completion
      2. Runs the code verifier against test assertions
      3. Returns binary rewards (1.0 = pass, 0.0 = fail)
    """
    _call_count = [0]
    _total_reward = [0.0]
    _total_count = [0]

    def reward_fn(prompts, completions, **kwargs):
        """Score completions by executing them against test assertions."""
        rewards = []
        task_ids = kwargs.get("task_id", [None] * len(prompts))
        ground_truths = kwargs.get("ground_truth", [])
        test_imports_list = kwargs.get("test_imports", [])
        test_setup_list = kwargs.get("test_setup_code", [])

        for i, completion in enumerate(completions):
            # Extract the text content from the completion
            if isinstance(completion, list):
                # Chat format: list of message dicts
                text = completion[-1]["content"] if completion else ""
            else:
                text = str(completion)

            code = extract_code_from_response(text)

            # Parse the ground truth back from JSON string
            try:
                tests = json.loads(ground_truths[i]) if i < len(ground_truths) else []
            except (json.JSONDecodeError, IndexError):
                tests = []

            try:
                imports = json.loads(test_imports_list[i]) if i < len(test_imports_list) else []
            except (json.JSONDecodeError, IndexError):
                imports = []

            setup = test_setup_list[i] if i < len(test_setup_list) else ""

            if not tests:
                rewards.append(0.0)
                continue

            result = verify_code(code, tests, imports, setup, timeout=10)
            rewards.append(result["reward"])

        # Track running stats
        _call_count[0] += 1
        _total_reward[0] += sum(rewards)
        _total_count[0] += len(rewards)
        if _call_count[0] % 10 == 0:
            avg = _total_reward[0] / max(_total_count[0], 1)
            print(f"  [Reward tracker] Calls: {_call_count[0]}, "
                  f"Running avg reward: {avg:.3f} "
                  f"(total: {_total_count[0]} completions)")

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="RLVR Training with GRPO")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 steps, 2 generations")
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip 4-bit quantization (needs more VRAM)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode overrides
    num_generations = 2 if args.quick else NUM_GENERATIONS
    max_steps = 3 if args.quick else args.steps
    save_steps = 1 if args.quick else SAVE_STEPS
    eval_steps = 1 if args.quick else EVAL_STEPS

    print(f"{'='*70}")
    print(f"RLVR Training with GRPO")
    print(f"{'='*70}")
    print(f"  Model:           {args.model}")
    print(f"  Steps:           {max_steps}")
    print(f"  Generations/prompt: {num_generations}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Output:          {output_dir}")
    print(f"  Quantized:       {not args.no_quantize}")
    print()

    # ---- Load dataset ----
    print("Loading training data...")
    train_path = DATA_DIR / "code_rlvr_train.json"
    dataset = build_dataset(train_path)
    print(f"  {len(dataset)} training prompts")

    # ---- Model config ----
    quantization_config = None
    if not args.no_quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # ---- LoRA config ----
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # ---- GRPO config ----
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=num_generations,
        max_completion_length=MAX_NEW_TOKENS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=max_steps,
        beta=KL_COEFF,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=save_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        # Generation config
        temperature=0.7,
        top_p=0.95,
    )

    # ---- Build reward function ----
    reward_fn = make_reward_function()

    # ---- Create trainer ----
    print("Loading model and creating trainer...")
    metrics_callback = MetricsCallback(output_dir / "metrics.jsonl")
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        callbacks=[metrics_callback],
        model_init_kwargs={
            "quantization_config": quantization_config,
            "torch_dtype": torch.bfloat16,
        },
    )

    print(f"  Model loaded on: {trainer.model.device}")
    print(f"  Trainable params: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    print(f"  Total params: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # ---- Train ----
    print(f"\n{'='*70}")
    print("Starting GRPO training...")
    print(f"{'='*70}\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ---- Save ----
    print(f"\nSaving final model to {output_dir}/final...")
    trainer.save_model(str(output_dir / "final"))

    # Save training config for reproducibility
    config = {
        "model": args.model,
        "num_generations": num_generations,
        "max_steps": max_steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "kl_coeff": KL_COEFF,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Checkpoint: {output_dir / 'final'}")
    print(f"\nNext step: run the benchmark to compare before/after:")
    print(f"  python benchmark.py --model {args.model} --adapter {output_dir / 'final'}")


if __name__ == "__main__":
    main()
