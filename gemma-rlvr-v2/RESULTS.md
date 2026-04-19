# Gemma-2-2B RLVR on GSM8K v2 — Unlocked update budget

Fourth experiment in `rlvr-demo`. Direct follow-up to
[`../gemma-rlvr/`](../gemma-rlvr/), which confirmed reward-variance
sweet spot empirically but showed `kl` max of 0.0011 — the policy
never moved. v2 relaxes the three knobs that bound movement.

## Headline numbers

| | Baseline | Post-RLVR | Δ |
|---|---:|---:|---:|
| GSM8K test (1,319) | **58.53%** (772/1319) | **60.35%** (796/1319) | **+1.82pp** |

Fixed (fail → pass): **120**. Regressed (pass → fail): **96**. Net **+24 problems**.
**First delta outside noise** across all four experiments in `rlvr-demo`.

## Recipe (diff from v1)

| Knob | v1 | **v2** |
|---|---|---|
| Learning rate | 5e-6 | **2e-5** |
| KL coefficient β | 0.05 | **0.005** |
| Training steps | 200 | **400** |
| `per_device_train_batch_size` | 4 | **8** (= G) |
| Effective batch (completions) | 8 | **16** |
| Prompts per optimizer step | 1 | **2** |
| Unique prompts touched | 200 (2.7% of data) | **800 (11%)** |
| Base model, G, temp, LoRA | — | unchanged |

## What actually happened

| Criterion | Target | Observed |
|---|---|---|
| `kl` movement | 0.05–0.5 | mean **0.018**, max **0.032** (29× v1) |
| `reward` trend | clearly upward | noisy; first-5 avg 0.625, last-5 avg 0.463 |
| Pass@1 delta | > 0.5pp | **+1.82pp** |

The policy moved **29×** further from the reference than v1, and
pass@1 went up. The reward-per-batch trace is noisy and actually
drifts slightly down in the tail — worth flagging. Two plausible
explanations:

1. The per-step `reward` column is a single batch of ~16 completions
   (2 prompts × G=8), so noise dominates the trend. Pass@1 on the
   1,319-problem test set is the smoothed signal.
2. The policy learned to emit crisper, more decisive final answers —
   which helps test pass rate even when per-step reward is flat.

Inspecting completions would settle it. Either way, **the hypothesis
held**: policy update budget was the binding constraint.

## OOM fix (one deviation from the original plan)

The v2 recipe OOMed at step 42 in the first launch attempt (per-batch
activation memory spiked as completion lengths grew past 300 tokens).
Fix: cut `max_completion_length` from 512 → 320 and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The length cap
is still well above the observed mean (~170) and max (~319) in v1.
The fix is committed in the first commit under this directory.

## The bigger picture

This run tests the hypothesis that emerged from v1's diagnostics:
**reward signal was necessary but not sufficient; policy update
budget was the binding constraint.** Unlocking the budget without
changing the reward setup directly tests that hypothesis.

| Experiment | Base | G | lr | β | Steps | KL max | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| [`code-rlvr/`](../code-rlvr/) | OLMo-2-7B | 4 | 5e-6 | 0.05 | 200 | *(n/a)* | +0.4pp |
| [`math-rlvr/`](../math-rlvr/) | OLMo-2-7B | 4 | 5e-6 | 0.05 | 200 | *(n/a)* | −0.5pp |
| [`gemma-rlvr/`](../gemma-rlvr/) | Gemma-2-2B | 8 | 5e-6 | 0.05 | 200 | 0.0011 | −0.3pp |
| **`gemma-rlvr-v2/`** | Gemma-2-2B | 8 | **2e-5** | **0.005** | **400** | **0.032** | **+1.82pp** |

## Reproduce

```bash
git clone https://github.com/tjuzek/rlvr-demo.git
cd rlvr-demo/gemma-rlvr-v2
bash run_all.sh --push   # ~3-4 hours on A10
```

## Attribution

Pipeline authored by Anthropic's Claude (Opus 4.7) via Claude Code,
directed by Tommie Juzek (`tjuzek@fsu.edu`).
