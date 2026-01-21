#!/bin/bash
# Merge LoRA adapters from masked-ratio sweep
#
# Usage:
#   ./merge.sh --pct 10,20,30
#   ./merge.sh --pct 50,60,70,80,90
#   ./merge.sh -p 10,20,30,40,50,60,70,80,90
#
# Arguments:
#   --pct, -p    Comma-separated list of percentages to merge (10-90)
#   --output, -o Output directory for merged models (default: ./merged-models)

set -e

# Parse arguments
PCT_LIST=""
MERGED_DIR="./merged-models"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pct|-p)
            PCT_LIST="$2"
            shift 2
            ;;
        --output|-o)
            MERGED_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --pct <pct1,pct2,...> [--output <dir>]"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$PCT_LIST" ]; then
    echo "Error: --pct is required"
    echo "Usage: $0 --pct <pct1,pct2,...> [--output <dir>]"
    exit 1
fi

# Convert comma-separated list to array
IFS=',' read -ra PCT_ARRAY <<< "$PCT_LIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Merging LoRA adapters: ${PCT_ARRAY[*]}"
echo "Output directory: $MERGED_DIR"
echo "=========================================="

mkdir -p "$MERGED_DIR"

for pct in "${PCT_ARRAY[@]}"; do
    adapter_dir="./outputs-lora-8b-self-correction/masked-ratio-${pct}pct"
    output_dir="${MERGED_DIR}/masked-ratio-${pct}pct-merged"
    
    if [ ! -d "$adapter_dir" ]; then
        echo ""
        echo "Warning: Adapter not found: $adapter_dir (skipping)"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Merging masked-ratio-${pct}pct..."
    echo "=========================================="
    
    uv run python merge_lora_adapter.py "$adapter_dir" "$output_dir"
    
    echo "Complete: $output_dir"
done

echo ""
echo "=========================================="
echo "All merges complete!"
echo "=========================================="
ls -la "$MERGED_DIR"

