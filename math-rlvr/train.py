"""
RLVR training: fine-tune OLMo-2-7B-Instruct on GSM8K math with GRPO.

The pipeline:
  1. Load OLMo-2-7B-Instruct with LoRA + 4-bit quantization.
  2. For each prompt, generate N completions.
  3. Score each with the math verifier (final-answer exact-match = 1, else 0).
  4. Update the policy using GRPO — reward is relative to the group.

This uses TRL's GRPOTrainer for clean integration.

Usage:
    # Full training on Lambda A10 (24GB)
    python train.py

    # Quick test (3 steps, 2 samples per prompt)
    python train.py --quick

    # Custom model
    python train.py --model meta-llama/Llama-3.2-3B-Instruct

    # Resume from checkpoint
    python train.py --resume output/checkpoint-100
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from verifier import verify_answer


class MetricsCallback(TrainerCallback):
    """Stream trainer log_history to a JSONL file as logs are emitted."""

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

DEFAULT_MODEL = "allenai/OLMo-2-1124-7B-Instruct"

NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 512
LEARNING_RATE = 5e-6
NUM_TRAIN_STEPS = 200
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
KL_COEFF = 0.05
WARMUP_RATIO = 0.1
LOGGING_STEPS = 5
SAVE_STEPS = 50

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def build_dataset(data_path: Path) -> Dataset:
    """Load GSM8K training data and format for GRPOTrainer."""
    raw = json.load(open(data_path))

    return Dataset.from_dict({
        "prompt": [ex["messages"] for ex in raw],
        "task_id": [ex["task_id"] for ex in raw],
        "ground_truth": [str(ex["ground_truth"]) for ex in raw],
    })


def make_reward_function():
    """Score completions by extracting the final number and comparing to ground truth."""
    _call_count = [0]
    _total_reward = [0.0]
    _total_count = [0]

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        ground_truths = kwargs.get("ground_truth", [])

        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            else:
                text = str(completion)

            gt = ground_truths[i] if i < len(ground_truths) else ""
            result = verify_answer(text, gt)
            rewards.append(result["reward"])

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
    parser = argparse.ArgumentParser(description="RLVR Math Training (GSM8K) with GRPO")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 steps, 2 generations")
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_generations = 2 if args.quick else NUM_GENERATIONS
    max_steps = 3 if args.quick else args.steps
    save_steps = 1 if args.quick else SAVE_STEPS

    print(f"{'='*70}")
    print(f"RLVR Math Training with GRPO")
    print(f"{'='*70}")
    print(f"  Model:              {args.model}")
    print(f"  Steps:              {max_steps}")
    print(f"  Generations/prompt: {num_generations}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Output:             {output_dir}")
    print(f"  Quantized:          {not args.no_quantize}")
    print()

    print("Loading training data...")
    train_path = DATA_DIR / "rlvr_gsm_train.json"
    if not train_path.exists():
        raise SystemExit(
            f"Training data not found at {train_path}. "
            f"Run `python prepare_data.py` first."
        )
    dataset = build_dataset(train_path)
    print(f"  {len(dataset)} training prompts")

    quantization_config = None
    if not args.no_quantize:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

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
        temperature=0.7,
        top_p=0.95,
        model_init_kwargs={
            "quantization_config": quantization_config,
            "dtype": torch.bfloat16,
        },
    )

    reward_fn = make_reward_function()

    print("Loading model and creating trainer...")
    metrics_callback = MetricsCallback(output_dir / "metrics.jsonl")
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        callbacks=[metrics_callback],
    )

    print(f"  Model loaded on: {trainer.model.device}")
    print(f"  Trainable params: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    print(f"  Total params: {sum(p.numel() for p in trainer.model.parameters()):,}")

    print(f"\n{'='*70}")
    print("Starting GRPO training...")
    print(f"{'='*70}\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    print(f"\nSaving final model to {output_dir}/final...")
    trainer.save_model(str(output_dir / "final"))

    config = {
        "model": args.model,
        "dataset": "gsm8k",
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
