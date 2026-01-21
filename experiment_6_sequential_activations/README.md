# Self-Correction Activation Statistics

This directory contains the analysis pipeline for collecting and analyzing SAE latent activations during self-correction episodes in LLM responses.

## Overview

When steered off-topic, language models sometimes spontaneously self-correct within a single response, producing patterns like:

```
[Off-topic content]... Wait, I made a mistake! [Correction phrase]... [On-topic content]
```

This analysis tests the hypothesis that:
1. **Off-topic detector (OTD) latents** fire during off-topic content and decrease after self-correction
2. **Backtracking latents** fire specifically at the correction point

## Data

- **Source**: Llama-3.3-70B experiment results from AGI-1516-esr-with-vllm
- **Episodes**: ~150 successful 2-attempt self-correction episodes
  - Filtered for `len(attempts) == 2` and `final_score > first_score`
- **Latents tracked**:
  - 25 Off-Topic Detectors (separability-based, AUC=1.0)
  - 16 Backtracking latents (keyword-based search)

## Pipeline

### Step 1: Find Backtracking Latents
```bash
python find_backtracking_latents.py
```

Searches `llama-70b-goodfire-l50.csv` for latents related to self-correction using weighted keyword matching:
- High weight: "self-correct", "backtrack", "correct a", "apologize"
- Medium weight: "mistake", "incorrect", "revise"
- Excludes: HTTP errors, stack traces, encoding errors

**Output**: `data/self-correction/backtracking_latent_candidates.json`

### Step 2: Extract Episodes
```bash
python extract_episodes.py
```

Extracts self-correction episodes from experiment results.

**Filtering criteria**:
- Exactly 2 attempts in `score.attempts`
- Final attempt score > first attempt score (successful self-correction)

**Output**: `data/self-correction/episodes.json`

### Step 3: Annotate Boundaries
```bash
python annotate_boundaries.py [--limit N]
```

Uses Claude to identify character positions for:
- `off_topic_start`: Where off-topic content begins
- `correction_start`: Where self-correction language starts
- `on_topic_start`: Where on-topic content resumes

**Output**: `data/self-correction/episodes_annotated.json`

### Step 4: Collect Activations
```bash
python collect_activations.py [--limit N]
```

Runs vLLM forward passes with `is_feature_decode=True` to extract per-token SAE activations.

**Requirements**: GPU with sufficient memory for Llama-70B

**Output**:
- `data/self-correction/episode_metadata.json`
- `data/self-correction/activations/*.npz`

### Step 5: Analyze Statistics
```bash
python analyze_activations.py
```

Computes:
- Mean activation per latent in each region (off-topic, correction, on-topic)
- Wilcoxon signed-rank test (non-parametric): off-topic > on-topic
- Paired t-test (parametric comparison)
- Effect sizes: Cohen's d, Cliff's delta

**Output**: `data/self-correction/analysis_results.json`

### Step 6: Collect Baseline Activations
```bash
python collect_baseline_activations.py [--limit N]
```

Collects activations from non-self-correction episodes (single attempt, score >= 50) for comparison.

**Output**: `data/self-correction/baseline_stats.json`

### Step 7: Generate Output
```bash
python generate_tables.py
python plot_activations.py
```

Generates:
- LaTeX tables for paper
- Visualization plots (each plot generates a JSON data file first, then renders PNG from it)

**Output**:
- `data/self-correction/tables/*.tex`
- `data/self-correction/plots/*.png` (with corresponding `.json` and `.md` sidecar files)

## Statistical Tests

### Hypothesis 1: OTDs fire less after self-correction
- **Test**: One-tailed Wilcoxon signed-rank test
- **Alternative**: off-topic activation > on-topic activation
- **Expected**: Significant (p < 0.05) with positive effect size

### Hypothesis 2: Backtracking latents peak during correction
- **Test**: Compare correction region mean to off-topic and on-topic
- **Expected**: correction > off-topic and correction > on-topic

## File Structure

```
self-correction-activation-statistics/
├── README.md                        # This file
├── find_backtracking_latents.py     # Step 1
├── extract_episodes.py              # Step 2
├── annotate_boundaries.py           # Step 3
├── collect_activations.py           # Step 4
├── analyze_activations.py           # Step 5
├── collect_baseline_activations.py  # Step 6
├── generate_tables.py               # Step 7
└── plot_activations.py              # Step 7

data/self-correction/
├── backtracking_latent_candidates.json
├── episodes.json
├── episodes_annotated.json
├── episode_metadata.json
├── analysis_results.json
├── baseline_stats.json
├── activations/
│   └── ep_XXXX.npz
├── tables/
│   └── *.tex
└── plots/
    ├── aligned_overlay.png
    ├── aligned_overlay.json
    ├── aligned_overlay.md
    ├── baseline_comparison.png
    ├── baseline_comparison.json
    └── baseline_comparison.md
```

## Latent Groups

### Off-Topic Detectors (25 latents)
Found via separability analysis (AUC-ROC = 1.0):
```python
[6527, 7517, 9005, 10304, 11390, 17250, 17481, 17516, 23093, 24684,
 26312, 28403, 28540, 37234, 37536, 38956, 39926, 40119, 40792, 44845,
 45078, 56830, 58565, 59483, 61420]
```

### Backtracking Latents (16 latents)
Found via keyword search (score >= 5):
```python
[5852, 18311, 45478, 52694, 57675, 63162,  # score 8-11
 3994, 3473, 5215, 7318, 53491,            # score 6-7
 890, 1719, 28564, 34597, 33044]           # score 5
```

Key labels:
- 5852: "The assistant needs to apologize and correct a mistake" (score=11)
- 52694: "The assistant is apologizing and correcting a mistake in its previous response" (score=8)
- 33044: "Sarcastic backtracking after provocative statements" (known from experiment_6)

## Dependencies

```
anthropic  # For Claude API (boundary annotation)
numpy
scipy
matplotlib
transformers  # For tokenizer
vllm  # For activation collection
```

## Usage Example

Full pipeline:
```bash
# Step 1: Find backtracking latents
python find_backtracking_latents.py

# Step 2: Extract episodes
python extract_episodes.py

# Step 3: Annotate boundaries (uses Claude API)
python annotate_boundaries.py

# Step 4: Collect activations (requires GPU)
python collect_activations.py

# Step 5: Analyze
python analyze_activations.py

# Step 6: Collect baseline activations (requires GPU)
python collect_baseline_activations.py

# Step 7: Generate output
python generate_tables.py
python plot_activations.py
```

For testing with a subset:
```bash
python annotate_boundaries.py --limit 10
python collect_activations.py --limit 10
python collect_baseline_activations.py --limit 10
```
