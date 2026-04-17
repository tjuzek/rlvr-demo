#!/bin/bash
#
# RLVR Math Pipeline — run everything on Lambda A10.
#
# Usage:
#   # Full pipeline: prepare data -> baseline -> train -> post-eval -> report
#   bash run_all.sh
#
#   # Push results to git when done
#   bash run_all.sh --push
#
#   # Override base model
#   bash run_all.sh --model=meta-llama/Llama-3.2-3B-Instruct
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PUSH=false
MODEL="allenai/OLMo-2-1124-7B-Instruct"

for arg in "$@"; do
    case $arg in
        --push) PUSH=true ;;
        --model=*) MODEL="${arg#*=}" ;;
    esac
done

echo "========================================"
echo "  RLVR Math Pipeline (GSM8K)"
echo "  Model: $MODEL"
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

# ---- Step 4: Baseline evaluation ----
echo ""
echo "[4/6] Baseline evaluation (1,319 GSM8K test problems)..."
python3 benchmark.py --model "$MODEL" --output results/baseline.json

# ---- Step 5: GRPO training ----
echo ""
echo "[5/6] GRPO training..."
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
echo "[*] Building post-talk update report..."
python3 make_report.py

# ---- Optional: push results to git ----
if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing results to git..."
    # results/, output/, and demo_output/ are in .gitignore to avoid bulk artifacts;
    # selectively force-include the small ones worth archiving.
    git add -f results/ output/training_config.json output/metrics.jsonl
    git commit -m "Add RLVR math training results $(date +%Y-%m-%d)"
    git push
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
echo "  Report:      results/gsm8k_update.html"
echo "  Metrics:     output/metrics.jsonl"
