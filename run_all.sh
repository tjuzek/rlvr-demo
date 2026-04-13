#!/bin/bash
#
# RLVR Full Pipeline — run everything on Lambda A10.
#
# Usage:
#   # Full training pipeline
#   bash run_all.sh
#
#   # Demo only (fast, for presentation)
#   bash run_all.sh --demo
#
#   # Push results to git when done
#   bash run_all.sh --push
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="full"
PUSH=false
MODEL="allenai/OLMo-2-1124-7B-Instruct"

for arg in "$@"; do
    case $arg in
        --demo) MODE="demo" ;;
        --push) PUSH=true ;;
        --model=*) MODEL="${arg#*=}" ;;
    esac
done

echo "========================================"
echo "  RLVR Code Pipeline"
echo "  Mode: $MODE | Model: $MODEL"
echo "========================================"
echo ""

# ---- Step 1: Install dependencies ----
echo "[1/6] Installing dependencies..."
pip install -q -r requirements.txt
echo "       Done."

# ---- Step 2: Download + prepare data ----
echo ""
echo "[2/6] Preparing data..."
if [ ! -f data/code_rlvr_train.json ]; then
    python3 download_dataset.py
else
    echo "       Data already exists, skipping download."
fi

if [ ! -f data/code_rlvr_corrupted.json ]; then
    python3 create_corruptions.py
else
    echo "       Corrupted examples already exist, skipping."
fi

# ---- Step 3: Pipeline health checks ----
echo ""
echo "[3/6] Running pipeline checks..."
python3 benchmark.py --checks-only

if [ "$MODE" = "demo" ]; then
    # ---- Demo mode: fast demo_train.py ----
    echo ""
    echo "[4/6] Running live demo training..."
    python3 demo_train.py --model "$MODEL"

    echo ""
    echo "[5/6] Skipped (demo mode)"
    echo "[6/6] Skipped (demo mode)"
else
    # ---- Full mode: baseline -> train -> post-eval ----

    # Step 4: Baseline
    echo ""
    echo "[4/6] Baseline evaluation..."
    python3 benchmark.py --model "$MODEL" --output results/baseline.json

    # Step 5: Train
    echo ""
    echo "[5/6] GRPO training..."
    python3 train.py --model "$MODEL"

    # Step 6: Post-training evaluation
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
fi

# ---- Optional: push results to git ----
if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing results to git..."
    git add -A results/ demo_output/ output/training_config.json
    git commit -m "Add RLVR training results $(date +%Y-%m-%d)"
    git push
    echo "Pushed!"
fi

echo ""
echo "========================================"
echo "  PIPELINE COMPLETE"
echo "========================================"

if [ "$MODE" = "demo" ]; then
    echo ""
    echo "  Demo report: demo_output/demo_report.html"
    echo "  Demo log:    demo_output/demo_log.json"
else
    echo ""
    echo "  Baseline:    results/baseline.json"
    echo "  Post-RLVR:   results/post_rlvr.json"
    echo "  Adapter:     output/final/"
fi
