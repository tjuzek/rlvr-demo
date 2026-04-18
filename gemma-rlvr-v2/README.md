# Gemma RLVR v2: Unlocked update budget

Fourth experiment. Same GSM8K + GRPO + LoRA recipe as
[`../gemma-rlvr/`](../gemma-rlvr/) — same base model (Gemma-2-2B-IT),
same G=8, same temperature 1.0 — but with the three knobs that bound
policy movement in v1 relaxed.

## Why v2

The v1 run landed well inside the reward-variance sweet spot
(`frac_reward_zero_std` ~52% vs OLMo's ~80%) and still showed
no learning. The diagnostic that cracked it was `kl` max **0.0011**
across 200 steps — **the policy never moved**.

With `lr=5e-6`, `β=0.05`, and only 200 steps, the TRL GRPO default
recipe is too cautious for LoRA on a small model. v2 keeps the
"variance band" framing from v1 but answers a different question:
*given enough update budget, does RLVR on GSM8K produce learning?*

## What changed

| Knob | v1 (`gemma-rlvr/`) | v2 (this run) | Rationale |
|---|---|---|---|
| Learning rate | 5e-6 | **2e-5** | 4× — within safe LoRA range |
| KL coefficient β | 0.05 | **0.005** | 10× lower — stop anchoring to reference |
| Training steps | 200 | **400** | 2× — more gradient updates |
| `per_device_train_batch_size` | 4 | **8** | = `num_generations`, makes batch math unambiguous (2 prompts per optimizer step instead of 1) |
| Model, G, temp, LoRA config | — | unchanged | Clean ablation |

## What we expect to see

The success criterion is **qualitative, not just pass@1**:
`kl` should rise into the 0.05–0.5 range during training (v1 stayed
under 0.002 end-to-end). If it does and pass@1 still doesn't move,
we've learned something deeper — reward is misaligned with learning,
or the LoRA subspace doesn't cover the capability.

Failure modes to watch for:
- **Reward collapse**: `reward` mean drops to random-guess level;
  policy has hacked the verifier. Diagnosis: inspect a few completions.
- **KL explosion**: `kl` past 5. Gradient clipping should catch it,
  but means the run is useless from that point.
- **No movement**: `kl` still under 0.01 after step 100 — even the
  unlocked config is too cautious. Next lever would be `lr=5e-5`.

## Quick start

```bash
bash run_all.sh --push   # ~3-4 hours on A10
```

Baseline is not re-run; `../gemma-rlvr/results/baseline.json` is
copied (same base model, same benchmark prompts, so result is
identical — 58.53%).

## Files

Same structure as `../gemma-rlvr/`. The only files that meaningfully
changed are `train.py` (hyperparameter constants) and `run_all.sh`
(baseline reuse).

## AI Attribution

Pipeline authored by **Anthropic's Claude** (Opus 4.7) via the Claude
Code CLI. Directed by Tommie Juzek (`tjuzek@fsu.edu`).
