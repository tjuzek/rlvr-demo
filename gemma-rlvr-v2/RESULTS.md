# Gemma-2-2B RLVR on GSM8K v2 — Unlocked update budget

Fourth experiment in `rlvr-demo`. Direct follow-up to
[`../gemma-rlvr/`](../gemma-rlvr/), which confirmed reward-variance
sweet spot empirically but showed `kl` max of 0.0011 — the policy
never moved. v2 relaxes the three knobs that bound movement.

## Headline numbers

*Pending Lambda run — updated automatically after
`bash run_all.sh --push`.*

| | Baseline | Post-RLVR | Δ |
|---|---:|---:|---:|
| GSM8K test (1,319) | 58.53% (from v1, same model) | *TBD* | *TBD* |

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

## Success criteria

Not just pass@1. Primary metrics:

1. **`kl` rises into 0.05–0.5 range** during training. v1 stayed under
   0.002 end-to-end.
2. **`reward` mean trends upward** across the full run, not just noise.
3. **Pass@1 moves** by more than the ~0.5pp noise floor seen in the
   first three runs.

## The bigger picture

This run tests the hypothesis that emerged from v1's diagnostics:
**reward signal was necessary but not sufficient; policy update
budget was the binding constraint.** Unlocking the budget without
changing the reward setup directly tests that hypothesis.

| Experiment | Base | G | lr | β | Steps | Δ |
|---|---|---:|---:|---:|---:|---:|
| [`code-rlvr/`](../code-rlvr/) | OLMo-2-7B | 4 | 5e-6 | 0.05 | 200 | +0.4pp |
| [`math-rlvr/`](../math-rlvr/) | OLMo-2-7B | 4 | 5e-6 | 0.05 | 200 | −0.5pp |
| [`gemma-rlvr/`](../gemma-rlvr/) | Gemma-2-2B | 8 | 5e-6 | 0.05 | 200 | −0.3pp |
| **`gemma-rlvr-v2/`** | Gemma-2-2B | 8 | **2e-5** | **0.005** | **400** | *TBD* |

## Reproduce

```bash
git clone https://github.com/tjuzek/rlvr-demo.git
cd rlvr-demo/gemma-rlvr-v2
bash run_all.sh --push   # ~3-4 hours on A10
```

## Attribution

Pipeline authored by Anthropic's Claude (Opus 4.7) via Claude Code,
directed by Tommie Juzek (`tjuzek@fsu.edu`).
