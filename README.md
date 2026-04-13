# Code RLVR: Reinforcement Learning from Verifiable Rewards on Code

Proof-of-concept: fine-tune OLMo-7B to write better Python code using
RLVR — the same technique behind Tulu 3, applied to code generation.

The verifier is simple: execute the code against test assertions. Pass = reward 1, fail = reward 0.

## Quick start (Lambda A10, 24GB)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download dataset (MBPP formatted for RLVR)
python download_dataset.py

# 3. Create corrupted examples (for presentation)
python create_corruptions.py

# 4. Run pipeline checks
python benchmark.py --checks-only

# 5. Baseline evaluation
python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct --max-examples 50

# 6. Train with GRPO
python train.py

# 7. Post-training evaluation
python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct \
    --adapter output/final --max-examples 50

# 8. Compare
python benchmark.py --compare results/baseline_*.json results/post_rlvr_*.json
```

## Files

| File | Purpose |
|------|---------|
| `download_dataset.py` | Download MBPP, format as RLVR dataset |
| `create_corruptions.py` | Generate synthetically corrupted code |
| `verifier.py` | Execute code against tests (the reward function) |
| `benchmark.py` | Pre/post evaluation + pipeline health checks |
| `train.py` | GRPO training with LoRA on OLMo-7B |

## Hardware

| Setup | Works? |
|-------|--------|
| Lambda A10 (24GB) | Yes — comfortable |
| Local 16GB GPU | Tight — use `--no-quantize` off, or try a 3B model |
| CPU only | Dataset + verifier work; training needs GPU |

## Swapping models

Edit the `DEFAULT_MODEL` line in `train.py`, or pass `--model`:

```bash
python train.py --model allenai/Olmo-3-7B-Instruct
python train.py --model meta-llama/Llama-3.2-3B-Instruct
```

## How it works

1. **Dataset**: 417 coding problems from MBPP with test assertions
2. **Verifier**: execute generated code in subprocess, check assertions
3. **Training (GRPO)**: for each prompt, generate N completions, score with verifier, update policy to favor passing completions
4. **Evaluation**: pass@1 on 257 held-out MBPP test problems
