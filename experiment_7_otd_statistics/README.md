# OTD Activation Statistics

This folder contains analysis of the off-topic detectors used in AGI-1516 Experiment 3,
and tools to create and test new off-topic detector sets.

Data files are stored in the parent directory's `data/` folder.

## Background

### The Original 26 Off-Topic Detectors

The original experiment ablates 27 off-topic detector latents (26 unique; latent 40119 appears twice).
These latents are from `old_off_topic_detectors_for_ablation.json`, sourced from an earlier
branch (`old_experiment_circuitsvis_branch`).

**The 26 unique latents:**
```
3675, 7517, 9168, 10304, 11977, 15375, 17481, 17516, 24684, 27331,
28540, 33044, 34002, 34765, 37536, 40119, 40792, 41038, 45078, 46037,
49897, 54311, 58565, 59483, 61116, 61420
```

### Origin of the Original Detectors

Investigation traced these latents to the AGI-1138 "Endogenous Steering Resistance" experiment,
which used Goodfire's now-deprecated `contrast()` API. The 26 came from **two different contrast experiments**:

1. **Shuffling Contrast** (baseline vs shuffled responses): Found latents that activate more on
   off-topic (shuffled) responses than on-topic responses.

2. **ESR Contrast** (successful self-correction vs baseline): Found latents involved in the
   self-correction mechanism itself, not just off-topic detection.

### Reverse-Engineering the Selection Criteria

We tested whether the original 25 detectors (excluding 33044) could be recovered using
**Separability (AUC-ROC)** - the probability that a random off-topic sample has higher
activation than a random on-topic sample.

**Results:**
- Best single derangement recovers **13/25** (52%) in top-25 by Separability
- Union across 10 different derangements recovers **15/25** (60%) in top-25
- The remaining **10 latents** cannot be recovered via separability alone

**The 10 "missing" latents** (not recoverable via separability):
```
3675, 11977, 15375, 27331, 34002, 41038, 46037, 49897, 54311, 61116
```

These have labels suggesting they're **self-correction triggers**, not off-topic detectors:
- 15375: "Expressions of dismay or realizing mistakes"
- 27331: "The assistant is positioning itself as helpful"
- 34002: "The assistant needs clarification or is offering a"
- 41038: "Assistant response needs termination due to repetition"
- 61116: "The assistant is being stubborn or failing to follow"

## New Off-Topic Detectors (Separability-Based)

We created a new set using a clear, reproducible methodology:

**Selection criteria:**
1. Compute Separability (AUC-ROC) for each latent: P(shuffled activation > normal activation)
2. Select all latents with **Separability ≥ 1.0** (perfect separation)

**The 25 separability-based detectors** (Separability = 1.0):
```
6527, 7517, 9005, 10304, 11390, 17250, 17481, 17516, 23093, 24684,
26312, 28403, 28540, 37234, 37536, 38956, 39926, 40119, 40792, 44845,
45078, 56830, 58565, 59483, 61420
```

This is saved to `data/off_topic_detectors_separability.json`.

**Overlap with original 26:** 13 latents overlap

## Ablation Results

We ran the full ablation experiment (1000 features, ~4300 trials) with the separability-based OTDs:

| Condition | MSI | Multi-attempt % | ESR Reduction |
|-----------|-----|-----------------|---------------|
| Baseline (no ablation) | 0.53 | 1.60% | - |
| **Separability ablation (25 OTDs)** | **0.24** | **0.78%** | **54%** |

**Key finding:** The separability-based OTDs reduce ESR by ~54%. This is a significant
reduction achieved using only latents with perfect separability (AUC-ROC = 1.0) between
shuffled and normal prompt-response pairs.

The 25 separability-based detectors are a "purer" set than the original off-topic detectors
 - they are definitionally off-topic detectors (they perfectly distinguish off-topic from 
 on-topic responses). The original 26 included additional latents from ESR contrast that 
 may have been self-correction triggers rather than off-topic detectors.

## Usage

### Step 1: Collect activation data (requires GPU)

```bash
cd /workspace/ast-goodfire/experiments/AGI-1635-esr-appendices/otd-activation-statistics
python collect_activations.py
```

Options:
- `--prompts <file>` - Use different prompts file (default: prompts.txt)
- `--output-dir <dir>` - Save to different directory (default: ../data/)

This will:
1. Generate unsteered responses to prompts
2. Create shuffled (off-topic) pairs using 10 derangements
3. Collect SAE activations for ALL ~65K latents (max-pooled across tokens)
4. Save to `<output-dir>/activations_all_latents.npz`

### Step 2: Analyze the data

```bash
python analyze_activations.py
```

Options:
- `--data-dir <dir>` - Load from different directory (default: ../data/)

This will:
1. Report stats for the 25 off-topic detectors from the original experiment
2. Show top 50 latents by Separability (AUC-ROC)
3. Show which original detectors are recovered

### Step 3: Run ablation experiment (requires GPU)

```bash
cd /workspace/ast-goodfire/experiments/AGI-1516-esr-with-vllm/experiment_3_off_topic_detectors

python experiment_3_with_ablation.py 70b \
  --ablate /workspace/ast-goodfire/experiments/AGI-1635-esr-appendices/data/off_topic_detectors_separability.json \
  --from-results ../experiment_results/experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_180833_with_ablation.json
```

This replays the exact same trials as the original ablation but with the new OTDs.

### Step 4: Generate LaTeX table for paper appendix

```bash
python generate_otd_table.py --output ../data/otd_activation_table.tex
```

This generates a LaTeX table showing activation statistics for each of the 25 OTDs.

## Files

- `collect_activations.py` - Collects SAE activations (GPU required)
- `analyze_activations.py` - Analyzes collected activations, computes Separability
- `generate_otd_table.py` - Generates LaTeX table of activation statistics
- `prompts.txt` - 38 prompts for activation collection
- `requirements.txt` - Python dependencies

### Data files (in ../data/)

- `old_off_topic_detectors_for_ablation.json` - The original 27 detectors (26 unique)
- `off_topic_detectors_separability.json` - New 25 detectors (Separability ≥ 1.0)
- `otd_activation_table.tex` - LaTeX table for paper appendix (generated)
- `llama-70b-goodfire-l50.csv` - Latent labels
- `activations_all_latents.npz` - Collected activation data (generated)
- `normal_responses.json` - Cached responses (generated)
