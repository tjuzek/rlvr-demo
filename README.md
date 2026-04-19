# rlvr-demo

Four RLVR experiments from follow-up work to the SC-AI Seminar talk on
*Reinforcement Learning from Verifiable Rewards* (Tom & Lan Li,
April 2026). The same GRPO + LoRA training recipe, varied along the
axes that determine whether GRPO has signal to learn from:
the base model (so: the baseline pass rate), group size `G`,
sampling temperature, and — finally, in run 4 — the policy update
budget (learning rate, KL coefficient, step count).

| Experiment | Task · Model | G | Base pass@1 | Post-RLVR pass@1 | Δ |
|---|---|---:|---:|---:|---:|
| [`code-rlvr/`](code-rlvr/) | MBPP · OLMo-2-7B-Instruct | 4 | 2.7% | 3.1% | +0.4pp |
| [`math-rlvr/`](math-rlvr/) | GSM8K · OLMo-2-7B-Instruct | 4 | 82.6% | 82.1% | −0.5pp |
| [`gemma-rlvr/`](gemma-rlvr/) | GSM8K · Gemma-2-2B-IT | 8 | 58.5% | 58.2% | −0.3pp |
| [`gemma-rlvr-v2/`](gemma-rlvr-v2/) | GSM8K · Gemma-2-2B-IT | 8 | 58.5% | **60.4%** | **+1.8pp** |

## The through-line

GRPO's advantage term is reward minus group mean. When every completion
in a group gets the same reward, that term is zero — the step contributes
no gradient. So **the fraction of mixed-reward groups** during training
determines how much signal the run actually carries.

For a baseline pass rate `p` and group size `G`, the dead-group share is
`p^G + (1-p)^G`:

| Experiment | `p` | `G` | Predicted dead share | Observed `frac_reward_zero_std` |
|---|---:|---:|---:|---:|
| `code-rlvr/` | 0.027 | 4 | 89% | *(not logged)* |
| `math-rlvr/` | 0.826 | 4 | 46% | **~80%** |
| `gemma-rlvr/` | 0.585 | 8 | 1.4% | **~52%** |

The independence model (`p^G + (1-p)^G`) underestimates zero-variance
groups — per-prompt difficulty clusters outcomes. Even so, Gemma cut
zero-variance from OLMo's 80% to 52% (~2.4× more signal per step) and
still landed within noise. With `lr=5e-6`, `β=0.05`, and 200 steps,
KL max was **0.0011** — the policy never meaningfully moved. Reward
variance was necessary but not sufficient.

## Run 4: unlock the update budget

[`gemma-rlvr-v2/`](gemma-rlvr-v2/) keeps run-3's model, G, temperature,
and reward setup unchanged, but unlocks the three knobs that bounded
policy movement: `lr=2e-5` (4×), `β=0.005` (10× lower), 400 steps (2×).
KL max jumps to **0.032** — 29× v1's ceiling — and pass@1 moves
**+1.82pp (58.53% → 60.35%)**, the first delta outside the ~0.5pp
noise floor seen in the first three runs.

Fixed (fail → pass): 120. Regressed (pass → fail): 96. Net +24 problems.

## What to read first

- **[`gemma-rlvr-v2/`](gemma-rlvr-v2/)** — the only run that clearly learns.
  Same recipe as `gemma-rlvr/` but with policy update budget unlocked.
  **+1.82pp** pass@1, KL max 0.032 (29× v1's ceiling).
- **[`gemma-rlvr/`](gemma-rlvr/)** — the deliberate variance-band attempt.
  Smaller base model (58.5% baseline), G=8, temperature 1.0. Cut
  zero-variance groups by ~1.5×; still within noise. KL stayed ~0.
  Surfaces the binding constraint that v2 then relaxes.
- **[`math-rlvr/`](math-rlvr/)** — same GSM8K verifier with OLMo-2-7B-Instruct.
  Baseline too high; most groups all-pass. Net Δ within noise.
- **[`code-rlvr/`](code-rlvr/)** — the original talk demo.
  Baseline too low; most groups all-fail. Net Δ within noise.

Each subdirectory is self-contained — its own `run_all.sh`,
`requirements.txt`, `README.md`, `RESULTS.md`, and report HTML.

## Reading order for the reports

1. [`gemma-rlvr-v2/results/gemma_gsm8k_report.html`](gemma-rlvr-v2/results/gemma_gsm8k_report.html) — update budget unlocked (+1.82pp)
2. [`gemma-rlvr/results/gemma_gsm8k_report.html`](gemma-rlvr/results/gemma_gsm8k_report.html) — variance-band attempt
3. [`math-rlvr/results/gsm8k_update.html`](math-rlvr/results/gsm8k_update.html) — OLMo-2 on GSM8K
4. [`code-rlvr/results/grpo_report.html`](code-rlvr/results/grpo_report.html) — original MBPP run

## Infrastructure

Both pipelines target **Lambda A10 (24GB)** with 4-bit QLoRA. Workflow:

```bash
# One-time
git clone https://github.com/tjuzek/rlvr-demo.git
cd rlvr-demo/<experiment>

# Run
bash run_all.sh --push   # push writes results back to the repo
```

## Attribution

Both pipelines were written by **Anthropic's Claude** (Opus 4.6 and
4.7) via the Claude Code CLI, directed by Tommie Juzek. See each
subdirectory's README for per-file attribution.

## Talk

The accompanying presentation repository (reveal.js slides, the
`rlvr/` companion that drove the seminar) is not included here —
this repo is just the compute artefacts.
