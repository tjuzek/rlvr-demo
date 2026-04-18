#!/bin/bash
#
# RLVR Math Pipeline (Gemma-v2, update-budget unlocked) — Lambda A10.
#
# Run 4 in the rlvr-demo series. Same base model and variance setup as
# gemma-rlvr/, but with lr=2e-5 (4x), KL coeff=0.005 (10x smaller),
# 400 steps (2x), and batch_size=8 (= G) for unambiguous batch math.
# Prior run had KL max 0.0011 — policy barely moved. This recipe
# aims to let it actually drift.
#
# Baseline is NOT re-run: we reuse ../gemma-rlvr/results/baseline.json
# since the base model is identical (Gemma-2-2B-IT).
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh --push
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PUSH=false
MODEL="google/gemma-2-2b-it"

for arg in "$@"; do
    case $arg in
        --push) PUSH=true ;;
        --model=*) MODEL="${arg#*=}" ;;
    esac
done

echo "========================================"
echo "  RLVR Math Pipeline (gemma-rlvr-v2)"
echo "  Model:       $MODEL"
echo "  Recipe:      lr=2e-5, beta=0.005, 400 steps, G=8"
echo "========================================"
echo ""

# ---- Step 1: Install dependencies ----
echo "[1/6] Installing dependencies..."
pip install -q -r requirements.txt
echo "       Done."

# ---- Step 2: Prepare data ----
echo ""
echo "[2/6] Preparing data..."
if [ ! -f data/rlvr_gsm_train.json ] || [ ! -f data/rlvr_gsm_test.json ]; then
    python3 prepare_data.py
else
    echo "       Data already prepared, skipping download."
fi

# ---- Step 3: Pipeline health checks ----
echo ""
echo "[3/6] Running pipeline checks..."
python3 benchmark.py --checks-only

# ---- Step 4: Baseline (reuse from gemma-rlvr/) ----
echo ""
echo "[4/6] Baseline: reusing ../gemma-rlvr/results/baseline.json (same base model)..."
mkdir -p results
if [ ! -f results/baseline.json ]; then
    if [ -f ../gemma-rlvr/results/baseline.json ]; then
        cp ../gemma-rlvr/results/baseline.json results/baseline.json
        echo "       Copied baseline from gemma-rlvr/ (58.53%)."
    else
        echo "       gemma-rlvr baseline missing — running fresh baseline..."
        python3 benchmark.py --model "$MODEL" --output results/baseline.json
    fi
else
    echo "       baseline.json already present, skipping."
fi

# ---- Step 5: GRPO training ----
echo ""
echo "[5/6] GRPO training (update-budget unlocked)..."
python3 train.py --model "$MODEL"

# ---- Step 6: Post-training evaluation + report ----
echo ""
echo "[6/6] Post-training evaluation..."
python3 benchmark.py --model "$MODEL" \
    --adapter output/final \
    --output results/post_rlvr.json

# Compare
echo ""
echo "========================================"
echo "  COMPARISON"
echo "========================================"
python3 benchmark.py --compare results/baseline.json results/post_rlvr.json

# Build themed HTML report
echo ""
echo "[*] Building themed report..."
python3 make_report.py

# ---- Optional: push results to git ----
if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing results to git..."
    git config user.email "tjuzek@fsu.edu"
    git config user.name "Tommie Juzek"
    git add -f results/ output/training_config.json output/metrics.jsonl
    git commit -m "Add gemma-rlvr-v2 training results $(date +%Y-%m-%d)" || true
    git push || echo "  (push failed — fetch artefacts manually via scp)"
    echo "Pushed!"
fi

echo ""
echo "========================================"
echo "  PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "  Baseline:    results/baseline.json"
echo "  Post-RLVR:   results/post_rlvr.json"
echo "  Adapter:     output/final/"
echo "  Report:      results/gemma_gsm8k_report.html"
echo "  Metrics:     output/metrics.jsonl"
