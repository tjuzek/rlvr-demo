# Math RLVR: Reinforcement Learning from Verifiable Rewards on GSM8K

Rerun of the Tulu-3-style RLVR recipe on **grade-school math** instead of
code. Same base model (OLMo-2-7B-Instruct), same GRPO + LoRA training
loop, same orchestration — but a numeric-answer verifier in place of the
code-execution verifier.

Why a rerun: the original `code-rlvr/` pipeline produced near-zero
gains (2.7% → 3.1% on MBPP) because OLMo-2-Instruct is not code-tuned —
the baseline was too low for GRPO to find reward-variance groups. On
GSM8K the same base model has a ~75% baseline, which is where Tulu 3
reported meaningful RLVR uplift.

See [`../code-rlvr/`](../code-rlvr/) for the original code experiment
and its report — both runs are archived together in this repo.

## Quick start (Lambda A10, 24GB)

```bash
# One-shot full pipeline: data prep -> baseline -> train -> post-eval -> report
bash run_all.sh

# Include --push to commit the run artefacts back to git
bash run_all.sh --push
```

Individual steps:

```bash
pip install -r requirements.txt
python prepare_data.py                         # fetch train + official GSM8K test
python benchmark.py --checks-only              # sanity-check verifier + data
python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct \
    --output results/baseline.json
python train.py                                # 200 steps GRPO, ~2-3h on A10
python benchmark.py --model allenai/OLMo-2-1124-7B-Instruct \
    --adapter output/final --output results/post_rlvr.json
python benchmark.py --compare results/baseline.json results/post_rlvr.json
python make_report.py                          # writes results/gsm8k_update.html
```

## Files

| File | Purpose |
|------|---------|
| `verifier.py` | Extract final number from a completion, compare to ground truth |
| `prepare_data.py` | Download RLVR-GSM (train) and openai/gsm8k (test) |
| `benchmark.py` | Pre/post evaluation on GSM8K test (1,319 problems) |
| `train.py` | GRPO training with LoRA on OLMo-2-7B-Instruct |
| `make_report.py` | Dark-mode HTML report (`results/gsm8k_update.html`) |
| `run_all.sh` | Lambda orchestration |

## How the math verifier works

Given a model completion like

> *"In April, Natalia sold 48 clips. In May she sold 48/2 = 24. In total: 48 + 24 = 72 clips. The answer is 72."*

the verifier tries, in order:

1. GSM8K canonical marker: `#### 72`
2. CoT-style tail: `"answer is 72"` (case-insensitive)
3. LaTeX boxed: `\boxed{72}`
4. Fallback: the last number in the completion

The extracted number is compared to ground truth as a float (with
1e-4 tolerance) so `"72"`, `"72.0"`, and `"72."` all match.

No subprocess execution, no sandbox — much faster per training step
than the code verifier.

## AI Attribution

This entire pipeline was authored by **Anthropic's Claude** (Opus 4.7)
via the Claude Code CLI. The maintainer (Tommie Juzek,
`tjuzek@fsu.edu`) directed the work, reviewed the output, ran the
pipeline on Lambda, and is accountable for any errors. But the code
itself and the structural decisions were Claude's.
