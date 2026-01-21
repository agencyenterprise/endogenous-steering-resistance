#!/bin/bash
# Train masked-ratio self-correction models
# Mixes masked self-correction examples with normal response examples
#
# Usage:
#   ./train.sh --gpu 0 --pct 10,20,30
#   ./train.sh --gpu 2 --pct 50,60,70,80,90
#   ./train.sh -g 0 -p 10,20,30,40,50
#
# Arguments:
#   --gpu, -g    GPU device ID (0-5)
#   --pct, -p    Comma-separated list of percentages (10-90)
#                These refer to the % of masked self-correction data in the mix

set -e

# Parse arguments
GPU=""
PCT_LIST=""

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --gpu <GPU_ID> --pct <pct1,pct2,...>"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$GPU" ]; then
    echo "Error: --gpu is required"
    echo "Usage: $0 --gpu <GPU_ID> --pct <pct1,pct2,...>"
    exit 1
fi

if [ -z "$PCT_LIST" ]; then
    echo "Error: --pct is required"
    echo "Usage: $0 --gpu <GPU_ID> --pct <pct1,pct2,...>"
    exit 1
fi

# Convert comma-separated list to array
IFS=',' read -ra PCT_ARRAY <<< "$PCT_LIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=$GPU

echo "=========================================="
echo "Training on GPU $GPU: ${PCT_ARRAY[*]}"
echo "=========================================="
echo ""
echo "Config type: masked-ratio (mixed masked self-correction + normal responses)"
echo ""

for pct in "${PCT_ARRAY[@]}"; do
    echo ""
    echo "=========================================="
    echo "[GPU $GPU] Training masked-ratio-${pct}pct..."
    echo "  (${pct}% masked self-correction, $((100-pct))% normal responses)"
    echo "=========================================="
    
    config_file="config_masked_ratio_${pct}pct.yml"
    
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file not found: $config_file"
        echo "Run 'python setup_masked_ratio_sweep.py' first!"
        exit 1
    fi
    
    uv run axolotl train "$config_file"
    
    echo "[GPU $GPU] Completed: masked-ratio-${pct}pct"
done

echo ""
echo "=========================================="
echo "[GPU $GPU] All training complete!"
echo "=========================================="

