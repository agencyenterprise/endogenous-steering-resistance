# Experiment 9: Self-Correction Activation Statistics

This experiment analyzes SAE latent activations during self-correction episodes
in LLM responses to test hypotheses about off-topic detector (OTD) and backtracking latents.

## Overview

When steered off-topic, language models sometimes spontaneously self-correct within
a single response. This analysis tests:
1. OTD latents fire during off-topic content and decrease after self-correction
2. Backtracking latents fire specifically at the correction point

## Data

- **Source**: Llama-3.3-70B experiment 1 results (from haiku judge folder)
- **Episodes**: ~150 successful 2-attempt self-correction episodes
- **Latents**: 26 OTD latents (from data/off_topic_detectors_old.json) + 16 backtracking latents

## Pipeline

The analysis runs in multiple steps:

```bash
# Step 1: Find backtracking latents (keyword search)
python run_activation_stats.py find-backtracking

# Step 2: Extract self-correction episodes from exp1 results
python run_activation_stats.py extract-episodes

# Step 3: Annotate episode boundaries with Claude
python run_activation_stats.py annotate-boundaries [--limit N]

# Step 4: Collect per-token activations (requires GPU)
python run_activation_stats.py collect-activations [--limit N]

# Step 5: Analyze activation statistics
python run_activation_stats.py analyze

# Step 6: Collect baseline activations
python run_activation_stats.py collect-baseline [--limit N]

# Step 7: Generate tables
python run_activation_stats.py generate-output

# Or run the full pipeline:
python run_activation_stats.py all

# Step 8: Generate plots (use plotting/plot_exp9.py)
python plotting/plot_exp9.py --output-dir plots/
```

## Output

- Tables: `data/experiment_results/claude_haiku_4_5_20251001_judge/activation_stats/tables/`
- Data: `data/experiment_results/claude_haiku_4_5_20251001_judge/activation_stats/`
- Plots: `plots/experiment_9_*.png` (via `plotting/plot_exp9.py`)
