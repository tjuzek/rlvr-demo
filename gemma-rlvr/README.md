# Gemma RLVR: Hunting the variance band on GSM8K

Third experiment in this repo. Same GSM8K + GRPO + LoRA recipe as
[`../math-rlvr/`](../math-rlvr/), but with deliberate choices aimed at
keeping GRPO's reward-variance signal high:

| Knob | `math-rlvr/` (OLMo-2) | `gemma-rlvr/` (this run) |
|---|---|---|
| Base model | `allenai/OLMo-2-1124-7B-Instruct` | `google/gemma-2-2b-it` |
| GSM8K baseline | ~82% | ~51% (predicted) |
| Generations per prompt (G) | 4 | **8** |
| Sampling temperature | 0.7 | **1.0** |
| Training steps | 200 | 200 |

## Why these choices

GRPO's advantage term is zero for any group where all G completions get
the same reward — so zero-variance groups contribute no gradient. With
a baseline pass rate `p` and a group size `G`, the share of "dead"
groups is `p^G + (1-p)^G`:

| Model baseline `p` | G=4 dead share | G=8 dead share |
|---:|---:|---:|
| 0.03 (MBPP) | 89% | 79% |
| 0.51 (Gemma/GSM8K) | 13% | **2%** |
| 0.82 (OLMo/GSM8K) | 46% | 23% |

Gemma-2-2B-IT at ~51% on GSM8K with G=8 sits in the sweet spot:
~98% of groups should carry gradient, versus ~54% in the OLMo run.
Higher temperature (1.0 vs 0.7) further increases within-group
disagreement even when `p` is close to 0 or 1.

## Quick start (Lambda A10, 24GB)

```bash
# One-shot full pipeline
bash run_all.sh --push   # ~3-4 hours on A10
```

Individual steps:

```bash
pip install -r requirements.txt
python prepare_data.py                              # fetch train + test
python benchmark.py --checks-only                   # sanity-check verifier
python benchmark.py --model google/gemma-2-2b-it \
    --output results/baseline.json
python train.py                                     # G=8, temp=1.0, 200 steps
python benchmark.py --model google/gemma-2-2b-it \
    --adapter output/final --output results/post_rlvr.json
python benchmark.py --compare results/baseline.json results/post_rlvr.json
python make_report.py                               # results/gemma_gsm8k_report.html
```

## Files

| File | Purpose |
|------|---------|
| `verifier.py` | Numeric-answer extraction + float equality (shared with math-rlvr) |
| `prepare_data.py` | Download RLVR-GSM train + GSM8K test |
| `benchmark.py` | Pre/post evaluation on 1,319 GSM8K problems |
| `train.py` | GRPO + LoRA + QLoRA; `--num-generations` and `--temperature` CLI knobs |
| `make_report.py` | Dark-mode HTML report with recipe panel + variance chart |
| `run_all.sh` | Lambda orchestration |

## What's different from `math-rlvr/`

- **Defaults**: G=8, temperature=1.0, `google/gemma-2-2b-it`
- **CLI knobs**: `train.py --num-generations N --temperature T`
- **`attn_implementation="eager"`**: Gemma-2 sliding-window attention
  needs eager (flash-attn support is partial for this architecture)
- **Report**: adds a "Recipe" panel (model + knobs) and a
  `frac_reward_zero_std` chart, which is the single most diagnostic
  signal for whether the run landed inside the variance band

## AI Attribution

Pipeline authored by **Anthropic's Claude** (Opus 4.7) via the Claude
Code CLI. Directed by Tommie Juzek (`tjuzek@fsu.edu`), who reviewed,
ran on Lambda, and is accountable for any errors.
