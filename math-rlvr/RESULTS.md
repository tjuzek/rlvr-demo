# RLVR on GSM8K — Post-Talk Update

Follow-up to the SC-AI Seminar talk. The demo during the talk used the
code-rlvr pipeline (MBPP); that base model (OLMo-2-7B-Instruct) wasn't
code-tuned, so RLVR had no reward-variance groups to learn from and the
pre/post delta sat in the noise (2.7% → 3.1%). This update reruns the
same recipe on **GSM8K math**, where the same base model has a healthy
baseline — the regime the Tulu 3 paper targets.

## Headline numbers

*Pending Lambda run — updated automatically after `bash run_all.sh --push`.*

| | Baseline | Post-RLVR | Δ |
|---|---:|---:|---:|
| GSM8K test (1,319) | *TBD* | *TBD* | *TBD* |

## Setup

- **Base model:** `allenai/OLMo-2-1124-7B-Instruct`
- **Training data:** `allenai/RLVR-GSM` (7,473 examples, 8-shot CoT prompts)
- **Held-out eval:** `openai/gsm8k` test split (1,319 problems)
- **Trainer:** TRL `GRPOTrainer` with 4-bit QLoRA (r=16)
- **Hyperparameters:** 4 generations/prompt, lr 5e-6, KL coeff 0.05, 200 steps
- **Hardware:** Lambda Cloud A10 (24GB)
- **Verifier:** regex-based final-answer extraction + numeric equality

## Full report

[**→ Interactive HTML report with charts**](results/gsm8k_update.html)

Includes the pre/post bar chart, reward-curve during training, KL
divergence from reference, policy-loss and gradient-norm trajectories,
per-problem flip analysis, and example before/after completions for
problems the model learned to solve.

## Reproduce

```bash
git clone https://github.com/tjuzek/rlvr-demo.git
cd rlvr-demo/math-rlvr
bash run_all.sh --push   # ~4-5 hours on A10
```

## Attribution

Pipeline authored by Anthropic's Claude (Opus 4.7) via Claude Code,
directed by Tommie Juzek (`tjuzek@fsu.edu`). The scientific
interpretation and any errors are Tom's.
