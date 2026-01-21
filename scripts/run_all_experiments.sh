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
python experiment_1_esr.py 70b

echo "Running Llama-3.1-8B (8-12 GPU hours)..."
python experiment_1_esr.py 8b

echo "Running Gemma-2-27B (8-10 GPU hours)..."
./python_for_gemma.sh experiment_1_esr.py gemma-27b

echo "Running Gemma-2-9B (6-8 GPU hours)..."
./python_for_gemma.sh experiment_1_esr.py gemma-9b

echo "Running Gemma-2-2B (4-6 GPU hours)..."
./python_for_gemma.sh experiment_1_esr.py gemma-2b

# Experiment 2: Boost level ablation (15-20 GPU hours)
echo "===== Experiment 2: Boost Level Ablation ====="
python experiment_2_multi_boost.py 70b

# Experiment 3: Off-topic detector ablation (30-40 GPU hours)
echo "===== Experiment 3: Off-Topic Detector Ablation ====="
cd experiment_3_off_topic_detectors
echo "Finding off-topic detectors..."
python find_off_topic_detectors.py 70b
echo "Running ablation experiment..."
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors.json
cd ..

# Experiment 4: Fine-tuning (40-60 GPU hours)
echo "===== Experiment 4: Fine-Tuning ====="
cd experiment_4_finetuning
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
python experiment_5_prompt_variants.py 70b --from-results "$BASELINE_FILE"

# Experiment 6: Sequential activations (3-4 GPU hours)
echo "===== Experiment 6: Sequential Activations ====="
cd experiment_6_sequential_activations
python extract_episodes.py
python collect_activations.py
python collect_baseline_activations.py
python analyze_activations.py
python plot_activations.py
cd ..

# Experiment 7: OTD activation statistics (2-3 GPU hours)
echo "===== Experiment 7: OTD Activation Statistics ====="
cd experiment_7_otd_statistics
python collect_activations.py
python analyze_activations.py
python generate_otd_table.py
cd ..

# Experiment 8: Random ablation control (15-20 GPU hours)
echo "===== Experiment 8: Random Ablation Control ====="
cd experiment_8_random_ablation_control
python run_ablation_experiment.py
python analyze_results.py
python create_plot.py
cd ..

# Cross-judge validation (requires OpenRouter API key)
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "===== Cross-Judge Validation ====="
    python regrade_cross_judge.py
else
    echo "Skipping cross-judge validation (OPENROUTER_API_KEY not set)"
fi

# Generate all plots
echo "===== Generating Plots ====="
python plotting/plot_exp1.py
python plotting/plot_exp2.py
python plotting/plot_exp3.py
python plotting/plot_exp4.py
python plotting/plot_exp5.py
python plotting/plot_exp6.py
python plotting/plot_exp7.py
python plotting/plot_exp8.py

echo ""
echo "===== All Experiments Complete ====="
echo "Results saved to: experiment_results/"
echo "Plots saved to: plots/"
