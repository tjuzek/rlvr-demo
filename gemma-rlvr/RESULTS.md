# Gemma-2-2B RLVR on GSM8K — Variance-band run

Third experiment in the `rlvr-demo` repo. Same GSM8K + GRPO + LoRA
recipe as the OLMo run in [`../math-rlvr/`](../math-rlvr/), but with
deliberate choices meant to land inside GRPO's reward-variance sweet
spot: smaller base model with a mid-band baseline (Gemma-2-2B-IT,
~51% on GSM8K), **G=8** generations per group, temperature **1.0**.

## Headline numbers

| | Baseline | Post-RLVR | Δ |
|---|---:|---:|---:|
| GSM8K test (1,319) | **58.53%** (772/1319) | **58.23%** (768/1319) | **−0.30pp** |

Fixed (fail → pass): **74**. Regressed (pass → fail): **78**. Within noise.

## Recipe

| Knob | Value |
|---|---|
| Base model | `google/gemma-2-2b-it` |
| Generations per prompt (G) | 8 |
| Sampling temperature | 1.0 |
| Training steps | 200 |
| Learning rate | 5e-6 |
| KL coefficient (β) | 0.05 |
| LoRA r / α | 16 / 32 |
| Quantization | 4-bit NF4 |
| Hardware | Lambda Cloud A10 (24GB) |

## Prediction vs. reality

Going in, the back-of-envelope estimate from
`p^G + (1-p)^G` for Gemma-2-2B-IT on GSM8K was:

```
p ≈ 0.51      # Gemma-2-2B-IT baseline on GSM8K
G = 8
p^G + (1-p)^G ≈ 0.51^8 + 0.49^8 ≈ 1.8%   ← predicted dead-group share
```

The empirical baseline landed at **p = 0.585**, even better-positioned
in theory (dead-group share ≈ 1.4%). But the actual run logged:

- `frac_reward_zero_std`: mean **52.5%**, median **60%** across 40 logged steps

So ~47% of groups carried gradient — **~2.4× more signal per step than
the OLMo run** (~20% mixed there), but nowhere near the 98% the
independence assumption predicted. Per-prompt difficulty clusters
outcomes: some prompts are "easy" for Gemma (all 8 completions pass),
others are "hard" (all 8 fail). The independent-Bernoulli model
underestimates zero-variance groups substantially.

## Why the delta is still ~zero

With 2.4× more learning signal than the OLMo run, the expectation was
a visible improvement. Training-trace diagnostics show why it didn't
land:

| Metric | Value | Reading |
|---|---|---|
| `reward` mean (40 log rows) | **0.565** | No clear upward trajectory (0.4 → 0.725, noisy) |
| `kl` max | **0.0011** | Policy barely moved from reference |
| `grad_norm` max | **1.43** | Updates applied, but small |
| `frac_reward_zero_std` mean | **0.525** | ~2.4× the signal density of OLMo |

The KL stays pinned near zero — the combination of `lr=5e-6`,
`kl_coeff=0.05`, and 200 total steps was too cautious to meaningfully
shift the policy, even with abundant reward variance. **Reward
variance was necessary but not sufficient.** Learning rate / step
count / KL coefficient turned out to be the binding constraint in this
configuration.

## Full report

[**→ Interactive HTML report with charts**](results/gemma_gsm8k_report.html)

Includes: recipe panel, pre/post bar chart, reward-variance
(`frac_reward_zero_std`) trajectory, training reward curve,
KL divergence, policy-loss and grad-norm, fail↔pass flip counts,
and example before/after completions.

## How this fits with the other two experiments

| Experiment | Baseline | G | `frac_reward_zero_std` | Net Δ |
|---|---:|---:|---:|---:|
| [`code-rlvr/`](../code-rlvr/) (MBPP) | 2.7% | 4 | ~89% predicted (all-fail) | +0.4pp |
| [`math-rlvr/`](../math-rlvr/) (OLMo-2 / GSM8K) | 82.6% | 4 | **~80% observed** | −0.5pp |
| **`gemma-rlvr/`** (Gemma / GSM8K) | **58.5%** | 8 | **~52.5% observed** | **−0.3pp** |

All three landed within noise. Cutting zero-variance groups by
~1.5× (OLMo 80% → Gemma 52%) produced no measurable pass@1 gain —
the binding constraint for a 200-step, `lr=5e-6`, `β=0.05` GRPO run
turns out to be policy movement, not reward sparsity. KL stayed under
0.0011 for the entire Gemma run.

## Reproduce

```bash
git clone https://github.com/tjuzek/rlvr-demo.git
cd rlvr-demo/gemma-rlvr
bash run_all.sh --push   # ~3-4 hours on A10
```

## Attribution

Pipeline authored by Anthropic's Claude (Opus 4.7) via Claude Code,
directed by Tommie Juzek (`tjuzek@fsu.edu`). Scientific interpretation
and errors are Tom's.
