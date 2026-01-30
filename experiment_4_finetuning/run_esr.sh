#!/bin/bash
# Run ESR experiments on fine-tuned masked-ratio models
# Uses same features/prompts from base model but recalibrates thresholds
#
# Usage:
#   ./run_esr.sh --gpu 0 --pct 10,20,30
#   ./run_esr.sh --gpu 2 --pct 50,60,70,80,90
#   ./run_esr.sh -g 0 -p 10,20,30,40,50
#
# Arguments:
#   --gpu, -g      GPU device ID (0-5)
#   --pct, -p      Comma-separated list of percentages (10-90)
#   --merged, -m   Directory containing merged models (default: ./merged-models)
#   --judge, -j    Judge model to use (e.g., 'haiku', 'sonnet')

set -e

# Parse arguments
GPU=""
PCT_LIST=""
MERGED_DIR="./merged-models"
JUDGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        --pct|-p)
            PCT_LIST="$2"
            shift 2
            ;;
        --merged|-m)
            MERGED_DIR="$2"
            shift 2
            ;;
        --judge|-j)
            JUDGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --gpu <GPU_ID> --pct <pct1,pct2,...> [--merged <dir>] [--judge <model>]"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$GPU" ]; then
    echo "Error: --gpu is required"
    echo "Usage: $0 --gpu <GPU_ID> --pct <pct1,pct2,...> [--merged <dir>] [--judge <model>]"
    exit 1
fi

if [ -z "$PCT_LIST" ]; then
    echo "Error: --pct is required"
    echo "Usage: $0 --gpu <GPU_ID> --pct <pct1,pct2,...> [--merged <dir>] [--judge <model>]"
    exit 1
fi

# Build judge argument if specified
JUDGE_ARG=""
if [ -n "$JUDGE" ]; then
    JUDGE_ARG="--judge $JUDGE"
fi

# Convert comma-separated list to array
IFS=',' read -ra PCT_ARRAY <<< "$PCT_LIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate venv
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=$GPU

# Base results file to get features/prompts from (Llama 8B with ~285 features, 1425 trials)
BASE_RESULTS="experiment_results/experiment_results_Meta-Llama-3.1-8B-Instruct_20251104_165624.json"

# Convert relative merged dir to absolute path from parent directory
if [[ "$MERGED_DIR" != /* ]]; then
    MERGED_DIR="experiment_4_finetuning/$MERGED_DIR"
fi

if [ ! -f "$BASE_RESULTS" ]; then
    echo "Error: Base results file not found: $BASE_RESULTS"
    exit 1
fi

echo "=========================================="
echo "[GPU $GPU] ESR Experiments: ${PCT_ARRAY[*]}"
echo "=========================================="
echo "Base results: $BASE_RESULTS"
echo "Merged models dir: $MERGED_DIR"
[ -n "$JUDGE" ] && echo "Judge model: $JUDGE"
echo ""

for pct in "${PCT_ARRAY[@]}"; do
    model_path="${MERGED_DIR}/masked-ratio-${pct}pct-merged"
    
    if [ ! -d "$model_path" ]; then
        echo ""
        echo "[GPU $GPU] [${pct}%] Model not found: $model_path (skipping)"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "[GPU $GPU] Running ESR on masked-ratio-${pct}pct-merged"
    echo "=========================================="
    
    python experiment_1_esr.py 8b \
        --model-path "$model_path" \
        --from-results "$BASE_RESULTS" \
        --recalibrate-thresholds \
        $JUDGE_ARG
    
    echo "[GPU $GPU] [${pct}%] Complete!"
done

echo ""
echo "=========================================="
echo "[GPU $GPU] All ESR experiments complete!"
echo "=========================================="

