# rlvr-demo

Two RLVR experiments from the SC-AI Seminar talk on
*Reinforcement Learning from Verifiable Rewards* (Tom & Lan Li,
April 2026) — the same training recipe applied to two different
verifiable-reward domains. Same base model, same GRPO + LoRA loop,
different verifiers and different signals.

| Experiment | Task | Verifier | Base pass@1 | Post-RLVR pass@1 |
|---|---|---|---:|---:|
| [`code-rlvr/`](code-rlvr/) | MBPP code generation | Python subprocess, assertions | 2.7% | 3.1% |
| [`math-rlvr/`](math-rlvr/) | GSM8K math reasoning | Regex + numeric equality | 82.6% | 82.1% |

## What to read first

- **[math-rlvr/](math-rlvr/)** — the main experiment, post-talk follow-up.
  Rerun on a dataset where OLMo-2-Instruct has real signal to improve.
- **[code-rlvr/](code-rlvr/)** — the original experiment shown during the
  talk. Interesting as a negative result: RLVR can't rescue a baseline
  that's near-zero because GRPO has no reward-variance groups to learn from.

Each subdirectory is self-contained — its own `run_all.sh`,
`requirements.txt`, `README.md`, and report HTML.

## Why both experiments are here

Both runs show — from opposite sides — that RLVR depends on **reward
variance inside a group**. GRPO's advantage is zero when all generations
in a group get the same reward; those groups contribute no gradient.

- On **MBPP** with a 2.7% baseline, `0.973⁴ ≈ 90%` of groups are all-fail.
- On **GSM8K** with an 82.6% baseline, `0.826⁴ ≈ 47%` of groups are all-pass.
  (Empirically we saw `frac_reward_zero_std` ≈ 0.6–1.0 in training logs —
  even higher than the naïve estimate.)

In both regimes the adapter barely moves. RLVR's sweet spot is the middle
band where enough groups are mixed-reward for GRPO to learn from — which
is why the Tulu 3 paper targets specific task/model pairings where the
baseline sits in that window.

Read the two reports together to see both failure modes.

## Reading order for the reports

1. [`math-rlvr/results/gsm8k_update.html`](math-rlvr/results/gsm8k_update.html) — post-talk update
2. [`code-rlvr/results/grpo_report.html`](code-rlvr/results/grpo_report.html) — original run

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
