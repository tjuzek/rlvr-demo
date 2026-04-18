# Gemma-2-2B RLVR on GSM8K — Variance-band run

Third experiment in the `rlvr-demo` repo. Same GSM8K + GRPO + LoRA
recipe as the OLMo run in [`../math-rlvr/`](../math-rlvr/), but with
deliberate choices meant to land inside GRPO's reward-variance sweet
spot: smaller base model with a mid-band baseline (Gemma-2-2B-IT,
~51% on GSM8K), **G=8** generations per group, temperature **1.0**.

## Headline numbers

*Pending Lambda run — updated automatically after `bash run_all.sh --push`.*

| | Baseline | Post-RLVR | Δ |
|---|---:|---:|---:|
| GSM8K test (1,319) | *TBD* | *TBD* | *TBD* |

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

## The prediction

The OLMo-2-7B-Instruct run on GSM8K logged
`frac_reward_zero_std ≈ 0.80` throughout training (80% of groups with
zero variance). A back-of-envelope estimate for this run:

```
p ≈ 0.51      # Gemma-2-2B-IT baseline on GSM8K
G = 8
p^G + (1-p)^G ≈ 0.51^8 + 0.49^8 ≈ 1.1% + 0.7% = 1.8%
```

So we expect roughly **98% of groups to be mixed-reward**. If that
prediction holds, GRPO has ~50× more usable signal per step than in
the OLMo run.

## Full report

[**→ Interactive HTML report with charts**](results/gemma_gsm8k_report.html)

Includes: recipe panel, pre/post bar chart, reward-variance
(`frac_reward_zero_std`) trajectory, training reward curve,
KL divergence, policy-loss and grad-norm, fail↔pass flip counts,
and example before/after completions.

## How this fits with the other two experiments

| Experiment | Baseline | G | Dead-group share | Net Δ |
|---|---:|---:|---:|---:|
| [`code-rlvr/`](../code-rlvr/) (MBPP) | 2.7% | 4 | ~89% all-fail | +0.4pp |
| [`math-rlvr/`](../math-rlvr/) (OLMo-2 / GSM8K) | 82.6% | 4 | ~80% observed | −0.5pp |
| **`gemma-rlvr/`** (Gemma / GSM8K) | *TBD* | 8 | *TBD* | *TBD* |

The first two runs landed on opposite edges of the variance band and
both came out within noise. This one tries to land inside.

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
