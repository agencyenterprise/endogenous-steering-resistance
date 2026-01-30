#!/bin/bash
# Master script to run all experiments for the ESR paper
# This will take approximately 185-245 GPU hours total

set -e

echo "===== ESR Paper: Running All Experiments ====="
echo "This will take several days of GPU compute time."
echo "Estimated total: 185-245 GPU hours (~$2-3k on cloud)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Create output directories
mkdir -p experiment_results
mkdir -p plots

# Experiment 1: ESR across models (60-80 GPU hours)
echo "===== Experiment 1: ESR Across Models ====="
echo "Running Llama-3.3-70B (30-40 GPU hours)..."
python -m experiment_01_esr 70b

echo "Running Llama-3.1-8B (8-12 GPU hours)..."
python -m experiment_01_esr 8b

echo "Running Gemma-2-27B (8-10 GPU hours)..."
VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_FLASH_ATTN_VERSION=3 python -m experiment_01_esr gemma-27b

echo "Running Gemma-2-9B (6-8 GPU hours)..."
VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_FLASH_ATTN_VERSION=3 python -m experiment_01_esr gemma-9b

echo "Running Gemma-2-2B (4-6 GPU hours)..."
VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_FLASH_ATTN_VERSION=3 python -m experiment_01_esr gemma-2b

# Experiment 2: Boost level ablation (15-20 GPU hours)
echo "===== Experiment 2: Boost Level Ablation ====="
python -m experiment_02_multi_boost 70b

# Experiment 3: Off-topic detector ablation (30-40 GPU hours)
echo "===== Experiment 3: Off-Topic Detector Ablation ====="
cd experiment_03_off_topic_detectors
echo "Finding off-topic detectors..."
python find_off_topic_detectors.py 70b
echo "Running ablation experiment..."
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors.json
cd ..

# Experiment 4: Fine-tuning (40-60 GPU hours)
echo "===== Experiment 4: Fine-Tuning ====="
cd experiment_04_finetuning
echo "Generating synthetic training data..."
python setup_masked_ratio_sweep.py
echo "Training models (this will take the longest)..."
bash train.sh
echo "Evaluating fine-tuned models..."
bash run_esr.sh
cd ..

# Experiment 5: Meta-prompting (15-20 GPU hours)
echo "===== Experiment 5: Meta-Prompting ====="
BASELINE_FILE=$(ls -t experiment_results/experiment_results_Meta-Llama-3.3-70B-Instruct_*.json | head -1)
python -m experiment_05_prompt_variants 70b --from-results "$BASELINE_FILE"

# Experiment 6: Sequential activations (< 1 GPU hour)
echo "===== Experiment 6: Sequential Activations ====="
python -m experiment_06_sequential_activations

# Experiment 7: Cross-judge validation (requires OpenRouter API key)
echo "===== Experiment 7: Cross-Judge Validation ====="
if [ -n "$OPENROUTER_API_KEY" ]; then
    python experiment_07_cross_judge/run_cross_judge.py --n-samples 1000
else
    echo "Skipping (OPENROUTER_API_KEY not set)"
fi

# Experiment 8: No-steering baseline (10-15 GPU hours)
echo "===== Experiment 8: No-Steering Baseline ====="
BASELINE_FILE=$(ls -t experiment_results/experiment_results_Meta-Llama-3.3-70B-Instruct_*.json | head -1)
python -m experiment_08_no_steering_baseline 70b --from-results "$BASELINE_FILE"

# Experiment 9: Activation statistics (3-5 GPU hours)
echo "===== Experiment 9: Activation Statistics ====="
python experiment_09_activation_stats/run_activation_stats.py all

# Experiment 10: Random latent ablation control (15-20 GPU hours)
echo "===== Experiment 10: Random Latent Ablation Control ====="
BASELINE_FILE=$(ls -t experiment_results/experiment_results_Meta-Llama-3.3-70B-Instruct_*.json | head -1)
python experiment_10_random_latent_control/run_random_latent_control.py run-random --from-results "$BASELINE_FILE" --n-sets 3
python experiment_10_random_latent_control/run_random_latent_control.py analyze

# Generate all plots
echo "===== Generating Plots ====="
python plotting/plot_all.py

echo ""
echo "===== All Experiments Complete ====="
echo "Results saved to: experiment_results/"
echo "Plots saved to: plots/"
