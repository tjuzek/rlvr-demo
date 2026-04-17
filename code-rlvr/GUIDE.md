# Step-by-Step Guide: Running the Code RLVR Pipeline

This guide walks you through everything from setup to presenting results.
There are three phases: **local prep**, **Lambda training**, and **presenting**.

---

## Phase 1: Local Prep (your laptop, no GPU needed)

### 1.1 Create a GitHub repo

Go to GitHub and create a new repo (e.g. `code-rlvr`). Then push:

```bash
cd /home/tom/claudecode/rlvr/code-rlvr
git remote add origin git@github.com:<your-username>/code-rlvr.git
git push -u origin main
```

### 1.2 Test the visual pipeline locally

You can preview the demo report without a GPU:

```bash
python3 demo_train.py --dry-run
```

This creates `demo_output/demo_report.html` — open it in a browser to check
that the charts and dark theme render correctly. This is exactly what the real
demo produces, just with mock data.

### 1.3 Verify the data pipeline locally

```bash
pip install datasets
python3 download_dataset.py       # downloads MBPP from HuggingFace
python3 create_corruptions.py     # creates corrupted code examples
python3 benchmark.py --checks-only  # runs all pipeline health checks
```

Expected output:
- `data/code_rlvr_train.json` — 417 training problems
- `data/code_rlvr_test.json` — 257 test problems
- `data/code_rlvr_corrupted.json` — 388 corrupted examples
- All pipeline checks should say "passed"

---

## Phase 2: Lambda Training (A10 GPU, 24GB)

### 2.1 Spin up a Lambda instance

- GPU: A10 (24GB) or better
- OS: Ubuntu with CUDA
- Cost: ~$0.60/hr for A10

### 2.2 Clone and run

SSH into the Lambda instance, then:

```bash
git clone git@github.com:<your-username>/code-rlvr.git
cd code-rlvr
```

### 2.3 Option A: Full training (recommended, ~1-2 hours)

This runs the complete pipeline — baseline eval, GRPO training, post-eval:

```bash
bash run_all.sh
```

What happens:
1. Installs dependencies (`pip install -r requirements.txt`)
2. Downloads MBPP data + creates corrupted examples
3. Runs pipeline health checks
4. Evaluates baseline pass@1 on the test set
5. Trains with GRPO + LoRA (200 steps)
6. Evaluates post-training pass@1
7. Prints the comparison

Results are saved to:
- `results/baseline.json` — baseline benchmark
- `results/post_rlvr.json` — post-training benchmark
- `output/final/` — trained LoRA adapter

### 2.4 Option B: Demo only (fast, ~2-3 min)

If you just want the demo results quickly:

```bash
bash run_all.sh --demo
```

This runs `demo_train.py` which uses expert iteration (faster than GRPO).
Produces `demo_output/demo_report.html` with interactive charts.

### 2.5 Push results back to GitHub

```bash
bash run_all.sh --demo --push
# or after a full run:
git add -A results/ demo_output/
git commit -m "Training results"
git push
```

### 2.6 Pull results on your laptop

```bash
# On your laptop
cd /home/tom/claudecode/rlvr/code-rlvr
git pull
```

Now you have the results locally for the presentation.

---

## Phase 3: Presenting

### 3.1 Show pre-computed results

Open `demo_output/demo_report.html` in a browser. It has:
- Stat cards: baseline vs. final pass@1, improvement, time
- Training curve (pass@1 over rounds)
- Reward signal chart
- Before/after code examples

The report uses the same dark theme as the presentation slides.

### 3.2 Live demo during the talk

If you want to run the demo live (via SSH to Lambda):

```bash
# On Lambda (via SSH from your laptop)
cd code-rlvr
python3 demo_train.py
```

The terminal output is clean and readable:
```
[1/6] Loading data...
[2/6] Loading model: allenai/OLMo-2-1124-7B-Instruct
[3/6] Baseline evaluation (15 problems)...
       Baseline pass@1: 20.0%
[4/6] Round 1/3
       Generating 4 samples x 20 problems...
       Correct: 18/80 (reward: 0.22)
       SFT on 18 correct samples (2 epochs)...
       SFT loss: 1.82
       Evaluating...
       pass@1: 33.3% (delta: +13.3%)
...
DEMO COMPLETE in 147s
  Baseline:     20.0%
  Final:        46.7%
  Improvement:  +26.7%
```

Then open the HTML report to show the charts.

### 3.3 Key talking points

**The dataset**: "417 Python coding problems from MBPP. Each has a description
and test assertions. That's all we need — prompts with a way to check the
answer."

**The verifier**: "We run the generated code in a subprocess and check the
test assertions. Pass = reward 1, fail = reward 0. No reward model, no human
labels."

**The corrupted examples**: "We took correct solutions and applied mechanical
bugs — off-by-one errors, wrong operators, returning None. The verifier
catches all of them. This is what the model's output looks like before
training."

**The training**: "We generate multiple solutions per problem, keep the ones
that pass, and fine-tune on those. Repeat. The model learns to write code
that passes tests."

**The result**: "pass@1 went from X% to Y% — the model got better at writing
correct code, using only automated verification as the training signal."

---

## Troubleshooting

**`run_all.sh` fails on pip install**: Make sure you have CUDA-compatible
PyTorch. On Lambda, this should be pre-installed. If not:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Out of memory on A10**: The default config (OLMo-7B, 4-bit, LoRA r=16)
should fit. If not, try reducing batch size in `train.py` or switch to a
smaller model:
```bash
bash run_all.sh --model=meta-llama/Llama-3.2-3B-Instruct
```

**`demo_train.py` crashes mid-demo**: Run `--dry-run` first to verify the
visual pipeline works. The dry-run produces the same HTML report with mock
data, so you always have a fallback.

**No internet on Lambda**: Download the model and data locally first, then
transfer. Or use `huggingface-cli download allenai/OLMo-2-1124-7B-Instruct`
to pre-cache the model.

---

## File Reference

| File | What it does |
|------|-------------|
| `download_dataset.py` | Downloads MBPP, formats as RLVR JSON |
| `create_corruptions.py` | Deterministic code corruptions for presentation |
| `verifier.py` | Executes code + checks test assertions |
| `benchmark.py` | Pre/post evaluation, pipeline checks, comparison |
| `train.py` | Full GRPO + LoRA training (real, slow) |
| `demo_train.py` | Fast expert iteration demo with HTML report |
| `run_all.sh` | One-command pipeline (supports `--demo`, `--push`) |
| `requirements.txt` | Python dependencies |
| `GUIDE.md` | This file |
