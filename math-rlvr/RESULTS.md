# RLVR on GSM8K — Post-Talk Update

Follow-up to the SC-AI Seminar talk. The demo during the talk used the
code-rlvr pipeline (MBPP); that base model (OLMo-2-7B-Instruct) wasn't
code-tuned, so RLVR had no reward-variance groups to learn from and the
pre/post delta sat in the noise (2.7% → 3.1%). This update reruns the
same recipe on **GSM8K math**, where the same base model has a healthy
baseline — the regime the Tulu 3 paper targets.

## Headline numbers

| | Baseline | Post-RLVR | Δ |
|---|---:|---:|---:|
| GSM8K test (1,319) | **82.64%** (1090) | **82.11%** (1083) | **−0.53pp** (−7) |

Under the hood: 200 GRPO steps moved 107 problems (8% of the test set) —
**50 fail→pass** and **57 pass→fail**. The net is within noise. Training
reward stayed flat at ~0.89 and KL from the reference policy stayed
~1e-4, so the adapter barely drifted from the base model.

### Why the delta is ~zero

With an 82.6% baseline and 4 generations per prompt, the chance a group
of 4 is all-correct is `0.826⁴ ≈ 47%`. GRPO's advantage term is zero
for groups with zero reward variance — so roughly half of training groups
contribute no gradient. The training logs confirm this:
`frac_reward_zero_std` sat at **0.6–1.0** throughout the 200 steps.

This is the mirror of the MBPP failure mode in [`../code-rlvr/`](../code-rlvr/):
there the baseline was too low (near-zero), so groups were all-fail; here
it's high enough that groups are often all-pass. RLVR works best on the
middle band — somewhere around 30–70% pass rate — where GRPO sees enough
reward variance to learn from. OLMo-2-Instruct on stock GSM8K with 8-shot
CoT sits above that sweet spot.

### What did change

- 107 individual problems flipped pass/fail
- Training reward stayed stable (0.87–0.95 across 800 completions)
- KL divergence from reference: max ~0.001 (extremely low)
- Loss oscillated around zero, grad norm typically <0.01

So the LoRA adapter did learn *something* — but the net effect on the
aggregate accuracy metric was indistinguishable from measurement noise.

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
