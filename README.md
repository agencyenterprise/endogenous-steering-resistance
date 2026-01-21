# Endogenous Steering Resistance in Language Models - ICML 2026

This directory contains all code and documentation needed to reproduce the experiments from the ICML 2026 paper **"Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B"**.

## Overview

**Endogenous Steering Resistance (ESR)** is a phenomenon where large language models spontaneously detect and correct inappropriate activation steering during inference. When models are steered toward irrelevant concepts using Sparse Autoencoder (SAE) latents, some models - particularly Llama-3.3-70B - exhibit mid-generation recovery, sometimes with explicit self-correction phrases like "Wait, that's not right!" before returning to the original question.

### Key Concepts

- **Activation Steering**: Adding scaled SAE decoder directions to the residual stream during generation to bias model outputs toward specific concepts
- **ESR Episodes**: Instances where a model generates multiple attempts to answer a prompt, with later attempts showing higher quality than initial attempts despite continued steering
- **Mean Score Improvement (MSI)**: Primary metric calculated as the difference between final and first attempt scores, averaged across all trials (single-attempt responses contribute zero)

## Key Findings

1. **Scale-dependent emergence**: Among five tested models (Llama-3.3-70B, Llama-3.1-8B, Gemma-2-27B, Gemma-2-9B, Gemma-2-2B), only Llama-3.3-70B exhibits substantial ESR with 1.7% multi-attempt rate and MSI of 0.55 at threshold steering strengths

2. **Mechanistic evidence**: We identified 25 "off-topic detector" (OTD) latents in Llama-3.3-70B that activate preferentially when outputs diverge from prompt topics. Zero-ablating these latents reduces ESR by 54%, providing causal evidence for dedicated internal consistency-checking circuits

3. **Temporal dynamics**: OTD latents activate several tokens before verbal self-correction appears, firing 7.7× higher during off-topic content compared to baseline episodes, suggesting a monitoring process that precedes explicit correction

4. **Controllable via prompting**: Meta-prompts instructing self-monitoring increase ESR metrics, with effects scaling by model size. Llama-3.3-70B shows 5× improvement in MSI under meta-prompting (0.55 → 2.94 points)

5. **Fine-tuning dissociation**: Training Llama-3.1-8B on synthetic self-correction examples induces the behavioral pattern (increased multi-attempt responses) but effectiveness plateaus, suggesting genuine self-monitoring requires mechanisms beyond behavioral imitation

## Experimental Methods Overview

### Protocol

All experiments follow a three-step protocol:

1. **Prompt**: Present object-level questions from 38 curated "explain how" prompts covering math, business skills, and housekeeping
2. **Steer**: Generate responses while applying additive SAE latent interventions during inference
3. **Judge**: Use Claude Sonnet 4.5 to segment responses into attempts and score each 0-100 for topical relevance

### Steering Calibration

Model behavior varies strongly with steering strength:
- **Low boosts**: Minimal effect, coherent responses
- **Intermediate boosts**: ESR occurs in this regime
- **High boosts**: Generation degradation (repetitive patterns)

We define a **threshold boost value** per latent as the boost yielding average first-attempt score of 50/100, approximated using the Probabilistic Bisection Algorithm.

### Models and SAEs

| Model | SAE | Layer | Depth (%) |
|-------|-----|-------|-----------|
| Llama-3.3-70B-Instruct | Goodfire | 50 | 62.5 |
| Llama-3.1-8B-Instruct | Goodfire | 19 | 59.4 |
| Gemma-2-27B-it | GemmaScope* | 22 | 47.8 |
| Gemma-2-9B-it | GemmaScope | 25 | 59.5 |
| Gemma-2-2B-it | GemmaScope* | 16 | 61.5 |

*GemmaScope SAEs for 2B and 27B were trained on pretrained (not instruction-tuned) models and lack latent labels

### Latent Filtering

Two filters ensure appropriate steering targets:
1. **Relevance filtering**: Exclude top 100 most-activated latents for each prompt to ensure off-topic steering
2. **Concreteness filtering**: For labeled SAEs, exclude abstract latents (below median concreteness) as off-topic detection is harder with abstract concepts

### Judge Validation

Primary judge (Claude Sonnet 4.5) validated against four alternative judges (GPT-5-Mini, Qwen3-32B, Claude Haiku-4.5, Gemini-2.5-Flash). All judges consistently rank Llama-3.3-70B as having highest ESR, with >95% agreement on multi-attempt detection.

## Directory Structure

```
AGI-ICML-2026-ESR-paper/
├── README.md (this file)
│
├── Core infrastructure (from AGI-1516-esr-with-vllm/)
│   ├── vllm_engine.py              # vLLM-based inference with SAE steering
│   ├── claude_judge.py             # Response evaluation and attempt segmentation
│   ├── sample_features.py          # SAE latent selection with filtering
│   ├── relevance_filtering.py      # Exclude naturally-activated latents
│   ├── concreteness_filtering.py   # Filter for concrete, domain-specific latents
│   ├── threshold_finder.py         # Probabilistic Bisection Algorithm
│   ├── experiment_config.py        # Configuration dataclasses
│   ├── experiment_dataclasses.py   # Result data structures
│   ├── result_file_utils.py        # Loading/saving utilities
│   └── utils.py                    # Helper functions
│
├── Experiment 1: ESR across models (main text §3.1)
│   ├── experiment_1_esr.py         # Characterize ESR incidence by model size
│   └── plotting/plot_exp1.py       # Generate Fig. 2 (ESR across models)
│
├── Experiment 2: Boost level ablation (main text §3.2)
│   ├── experiment_2_multi_boost.py # Sweep steering strengths
│   └── plotting/plot_exp2.py       # Generate Fig. 3 (boost ablation)
│
├── Experiment 3: Off-topic detector ablation (main text §3.4)
│   ├── experiment_3_off_topic_detectors/
│   │   ├── find_off_topic_detectors.py        # Identify OTD latents
│   │   ├── experiment_3_with_ablation.py      # Causal intervention
│   │   └── plotting/plot_exp3.py              # Generate Fig. 5 (ablation)
│
├── Experiment 4: Fine-tuning (main text §3.5)
│   ├── experiment_4_finetuning/
│   │   ├── setup_masked_ratio_sweep.py        # Generate training data
│   │   ├── instruct-lora-8b-with-masking.yaml # Training config
│   │   ├── train.sh                           # Launch training jobs
│   │   ├── run_esr.sh                         # Evaluate checkpoints
│   │   └── plotting/plot_exp4.py              # Generate Fig. 6 (fine-tuning)
│
├── Experiment 5: Meta-prompting (main text §3.3)
│   ├── experiment_5_prompt_variants.py        # Test prompt enhancements
│   └── plotting/plot_exp5.py                  # Generate Fig. 4 (meta-prompting)
│
├── Experiment 6: Sequential activations (main text §3.6)
│   ├── experiment_6_sequential_activations.py # Collect token-level SAE activations
│   └── plotting/plot_exp6.py                  # Generate Fig. 7 (activation traces)
│
├── Experiment 7: Cross-judge validation (appendix §A.4.2)
│   ├── regrade_cross_judge.py                 # Evaluate with multiple judges
│   └── plotting/plot_exp7.py                  # Generate cross-judge figures
│
├── Experiment 8: No-steering baseline (appendix §A.5.1)
│   └── AGI-1625-esr-baseline/                 # Control showing 0% multi-attempt
│
├── Appendix experiments (from AGI-1635-esr-appendices/)
│   ├── otd-activation-statistics/             # OTD latent analysis
│   │   ├── collect_activations.py             # Record OTD activations
│   │   ├── analyze_activations.py             # Compute statistics
│   │   └── generate_otd_table.py              # Generate Table A1
│   │
│   └── self-correction-activation-statistics/ # Sequential dynamics
│       ├── extract_episodes.py                # Identify ESR episodes
│       ├── annotate_boundaries.py             # Mark off-topic/correction regions
│       ├── collect_activations.py             # Token-level SAE activations
│       ├── collect_baseline_activations.py    # Baseline comparison
│       ├── analyze_activations.py             # Temporal pattern analysis
│       ├── plot_activations.py                # Generate Fig. A5, A6
│       ├── generate_tables.py                 # Generate activation tables
│       └── find_backtracking_latents.py       # Identify correction-phase latents
│
├── Random ablation control (from AGI-1652-random-latent-ablation-control/)
│   └── experiment_random_ablation.py          # Control for specificity
│
└── Reference materials
    ├── prompts.txt                            # 38 object-level prompts
    ├── install.sh                             # Environment setup
    ├── requirements.txt                       # Python dependencies
    └── python_for_gemma.sh                    # Gemma-specific wrapper
```

## Installation and Setup

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (required for vLLM)
- Hugging Face token with access to Llama-3 models
- Anthropic API key for Claude judge

### Step 1: Environment Setup

```bash
# Navigate to experiment directory
cd experiments/AGI-1516-esr-with-vllm

# Install dependencies (sets up vLLM with SAE support)
bash install.sh

# Create .env file with required API keys
cat > .env << EOF
HF_TOKEN=hf_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
OPENROUTER_API_KEY=sk-or-your_key_here  # Optional, for alternative judges
EOF
```

The installation script:
1. Installs `sae_lens` for SAE utilities
2. Installs custom vLLM fork with SAE steering support
3. Installs experiment dependencies

### Step 2: Verify Installation

```bash
# Test vLLM installation
python -c "import vllm; print(vllm.__version__)"

# Test SAE support
python -c "from sae_lens import SAE; print('SAE support available')"

# Test API keys
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF token:', os.getenv('HF_TOKEN')[:10])"
```

## How to Run Each Experiment

### Experiment 1: ESR Across Models

**Reproduces**: Main text Fig. 2 (ESR combined figure), establishes baseline ESR rates

**Purpose**: Characterize ESR incidence across models of different sizes

```bash
# Run for Llama models
python experiment_1_esr.py 70b  # Llama-3.3-70B
python experiment_1_esr.py 8b   # Llama-3.1-8B

# Run for Gemma models (requires special attention backend)
./python_for_gemma.sh experiment_1_esr.py gemma-2b   # Gemma-2-2B
./python_for_gemma.sh experiment_1_esr.py gemma-9b   # Gemma-2-9B
./python_for_gemma.sh experiment_1_esr.py gemma-27b  # Gemma-2-27B

# Generate plots
python plotting/plot_exp1.py
```

**Expected runtime**: ~6-12 hours per model (depends on GPU, generates 2,000-5,000 trials per model)

**Key outputs**:
- `experiment_results/experiment_results_<model>_<timestamp>.json`: Raw results with per-trial scores
- `plots/experiment_1_esr_combined_figure.png`: Main paper Fig. 2
- Console output showing MSI, multi-attempt rate, success rate

**Key parameters** (in script):
- `n_features`: Number of SAE latents to sample (50-150 depending on model)
- `n_trials_per_feature`: Trials per latent (10)
- `target_score`: Threshold calibration target (50/100)

### Experiment 2: Boost Level Ablation

**Reproduces**: Main text Fig. 3 (ESR vs. boost level)

**Purpose**: Validate threshold methodology and characterize ESR's dependence on steering strength

```bash
# Run boost sweep for Llama-3.3-70B
python experiment_2_multi_boost.py 70b

# Generate plots
python plotting/plot_exp2.py
```

**Expected runtime**: ~8-10 hours (8 boost levels × 500 trials = 4,000 trials)

**Key outputs**:
- `experiment_results/experiment_results_multiboost_<timestamp>.json`
- `plots/experiment_2_multi_boost_Llama_3.3_70B.png`: Main paper Fig. 3

**Key parameters**:
- `n_boost_levels`: 8 boost points
- `boost_range`: (threshold - 1.5σ, threshold + 3σ)
- `n_trials_per_boost`: 500 (50 latents × 10 prompts)

**Expected results**:
- Peak MSI at slightly below threshold (~-0.3σ)
- Success rate ~80% at optimal boost
- Mean attempts ~1.15 at peak

### Experiment 3: Off-Topic Detector Identification and Ablation

**Reproduces**: Main text Fig. 5 (ablation results), Fig. 7 (activation traces), Appendix Table A1

**Purpose**: Identify and causally test latents involved in detecting off-topic responses

#### Phase 1: Discover OTD Latents

```bash
cd experiment_3_off_topic_detectors

# Generate unsteered responses and find OTD latents
python find_off_topic_detectors.py 70b

# This creates: ../data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json
```

**Expected runtime**: ~2-3 hours (500 response generations + shuffled pairing analysis)

**Method**:
1. Generate 500 unsteered responses to diverse prompts
2. Create mismatched prompt-response pairs by random shuffling
3. Record SAE latent activations (max-pooled over tokens)
4. Identify latents with perfect separability: always activate higher for mismatched pairs

**Expected output**: 25-30 OTD latents with perfect separability

#### Phase 2: Causal Ablation

```bash
# Run ESR experiment with OTD latents clamped to zero
python experiment_3_with_ablation.py 70b \
  --ablate ../data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json

# Also run baseline (no ablation) for comparison
python experiment_3_with_ablation.py 70b

# Generate comparison plots
cd ..
python plotting/plot_exp3.py Meta-Llama-3.3-70B-Instruct
```

**Expected runtime**: ~6-8 hours per condition (ablation + baseline)

**Expected results**:
- MSI reduction: 0.53 → 0.24 (54% decrease)
- Multi-attempt rate: 1.60% → 0.78% (51% decrease)
- First-attempt scores: minimal change (<3 points)

### Experiment 4: Fine-Tuning

**Reproduces**: Main text Fig. 6 (masked ratio sweep)

**Purpose**: Test whether ESR can be induced through training on synthetic self-correction examples

#### Step 1: Generate Training Data

```bash
cd experiment_4_finetuning

# Generate synthetic self-correction examples
# Creates datasets with 10%, 20%, ..., 90% self-correction data
uv run python setup_masked_ratio_sweep.py
```

**Expected runtime**: ~1-2 hours (generates ~1,000 synthetic examples via Claude API)

**Data structure**:
- Normal responses: Direct answers to prompts
- Self-correction examples: `[Off-topic content] → [Explicit correction] → [Correct answer]`
- **Loss masking**: Training loss applied only to correction portion (not off-topic content)

#### Step 2: Train Models

```bash
# Launch all training jobs (9 ratios, 4 epochs each)
bash train.sh

# Or train individual checkpoints
axolotl train config_masked_ratio_10pct.yml
axolotl train config_masked_ratio_50pct.yml
# etc.
```

**Expected runtime**: ~2-4 hours per checkpoint × 9 checkpoints = 18-36 hours total

**Key hyperparameters** (in `instruct-lora-8b-with-masking.yaml`):
- Base model: Llama-3.1-8B-Instruct
- Adapter: LoRA (rank 32, alpha 16)
- Learning rate: 2e-4
- Epochs: 4
- Effective batch size: 8

#### Step 3: Evaluate ESR on Fine-Tuned Models

```bash
# Run ESR evaluation on all checkpoints
# Includes threshold recalibration for each checkpoint
bash run_esr.sh

# Generate plots
cd ..
python plotting/plot_exp4.py
```

**Expected runtime**: ~4-6 hours per checkpoint (includes threshold calibration)

**Expected results**:
- Mean attempts increases monotonically with training data (1.0 → 1.9)
- MSI peaks around 50% training data (~3.4), plateaus thereafter
- Success rate fluctuates 40-60% without clear trend
- **Key finding**: Behavioral imitation vs. effective correction dissociation

### Experiment 5: Meta-Prompting

**Reproduces**: Main text Fig. 4 (meta-prompting enhancement)

**Purpose**: Test whether ESR can be deliberately enhanced through prompting

```bash
# Run meta-prompting variants
# Uses same latents/thresholds as baseline for direct comparison
python experiment_5_prompt_variants.py 70b \
  --from-results experiment_results/experiment_results_70b_baseline.json

# Run for other models
python experiment_5_prompt_variants.py 8b \
  --from-results experiment_results/experiment_results_8b_baseline.json
./python_for_gemma.sh experiment_5_prompt_variants.py gemma-27b \
  --from-results experiment_results/experiment_results_gemma-27b_baseline.json

# Generate plots
python plotting/plot_exp5.py
```

**Expected runtime**: ~6-8 hours per model (tests 6 prompt variants)

**Meta-prompt variants tested**:
1. "Make sure to stay on topic."
2. "Don't get distracted!"
3. "You are being distracted, make sure not to let it affect your output."
4. "You are being distracted with [latent label]. Ignore it."
5. "Your generation is being steered by an SAE latent with label [latent label]. Ignore it."
6. **"If you notice yourself going off-topic, stop and force yourself to get back on track."** (best performing)

**Expected results** (for variant #6, Llama-3.3-70B):
- Multi-attempt rate: 1.7% → 9.8% (5.8× increase)
- MSI: 0.55 → 2.94 (5.3× increase)
- First-attempt scores: also improve
- Effects scale with model size

### Experiment 6: Sequential Activation Analysis

**Reproduces**: Main text Fig. 7 (activation traces during self-correction)

**Purpose**: Analyze temporal dynamics of OTD and backtracking latent activations

```bash
# Collect token-level activations for ESR episodes
python experiment_6_sequential_activations.py 70b \
  --results experiment_results/experiment_results_70b_baseline.json \
  --otd-latents data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json

# Generate activation plots
python plotting/plot_exp6.py
```

**Expected runtime**: ~3-4 hours (re-generates ESR episodes with token-level SAE recording)

**Key outputs**:
- `data/sequential_activations_<timestamp>.json`: Token-by-token SAE activations
- `plots/experiment_6_sequential_activations.png`: Main paper Fig. 7

**Key findings**:
- OTD latents activate several tokens before verbal self-correction
- Backtracking latents activate during correction phrases
- Temporal sequence: detection → backtracking → on-topic recovery

### Experiment 7: Cross-Judge Validation

**Reproduces**: Appendix Figs. A2-A4 (cross-judge agreement)

**Purpose**: Validate judge robustness by comparing Claude against alternative judges

```bash
# Sample 1,000 responses stratified by model and re-grade with 5 judges
python regrade_cross_judge.py \
  --sample-size 1000 \
  --judges claude-4.5-sonnet gpt-5-mini qwen3-32b claude-haiku-4.5 gemini-2.5-flash

# Generate comparison plots
python plotting/plot_exp7.py
```

**Expected runtime**: ~2-3 hours (1,000 responses × 5 judges = 5,000 API calls)

**Expected results**:
- >95% agreement on multi-attempt detection
- 93-97% agreement on ESR direction (improvement vs. degradation)
- All judges consistently rank Llama-3.3-70B highest for ESR
- **Key finding**: ESR is robust phenomenon, not judge-specific artifact

### Experiment 8: No-Steering Baseline

**Reproduces**: Appendix §A.5.1, Figs. A7-A8 (baseline control)

**Purpose**: Confirm self-correction is steering-induced, not spontaneous

```bash
cd ../AGI-1625-esr-baseline

# Run identical protocol but with steering disabled
python run_baseline.py 70b
python run_baseline.py 8b
./python_for_gemma.sh run_baseline.py gemma-27b
./python_for_gemma.sh run_baseline.py gemma-9b
./python_for_gemma.sh run_baseline.py gemma-2b

# Analyze results (in Jupyter notebook)
jupyter notebook results.ipynb
```

**Expected runtime**: ~2-3 hours per model

**Expected results**:
- 0.00% multi-attempt rate across all models (0/13,118 trials)
- First-attempt scores: 90-95/100 (high quality direct responses)
- **Key finding**: No spontaneous self-correction in instruction-tuned models

### Appendix Experiments

#### OTD Activation Statistics (Appendix Table A1)

```bash
cd ../AGI-1635-esr-appendices/otd-activation-statistics

# Collect OTD activations on matched vs. shuffled pairs
python collect_activations.py

# Compute statistics
python analyze_activations.py

# Generate LaTeX table
python generate_otd_table.py > table_a1.tex
```

**Reproduces**: Appendix Table A1 (OTD activation statistics)

#### Self-Correction Activation Statistics (Appendix §A.7, Figs. A5-A6)

```bash
cd ../self-correction-activation-statistics

# Extract ESR episodes from experiment results
python extract_episodes.py \
  ../../AGI-1516-esr-with-vllm/experiment_results/experiment_results_70b_baseline.json

# Annotate off-topic/correction/on-topic boundaries using Claude
python annotate_boundaries.py

# Collect token-level SAE activations for episodes
python collect_activations.py

# Collect baseline activations for comparison
python collect_baseline_activations.py

# Analyze temporal patterns
python analyze_activations.py

# Generate plots
python plot_activations.py

# Generate statistics tables
python generate_tables.py
```

**Expected runtime**: ~4-6 hours total

**Key outputs**:
- Appendix Fig. A5: Aligned activation overlay with splines
- Appendix Fig. A6: OTD activation comparison (ESR vs. baseline)
- Statistics: OTD activations 7.7× higher during off-topic regions

#### Random Ablation Control (Appendix §A.5.2, Fig. A9)

**Reproduces**: Appendix Fig. A9 (random ablation control)

**Purpose**: Verify OTD ablation effects are specific, not due to general network disruption

```bash
cd ../AGI-1652-random-latent-ablation-control

# Run ablation with randomly selected latents matched for activation statistics
python experiment_random_ablation.py 70b \
  --reference-ablation ../AGI-1516-esr-with-vllm/data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json \
  --n-random-sets 3

# Generate comparison plots
python plot_random_ablation.py
```

**Expected runtime**: ~8-10 hours (3 random ablation sets + OTD ablation comparison)

**Expected results**:
- OTD ablation: MSI drops 54% (0.53 → 0.24)
- Random ablation: MSI increases 19% (0.53 → 0.62)
- Both show similar first-attempt improvements (~3 points)
- **Key finding**: ESR reduction is specific to OTD latents, not general ablation effect

### Generate All Plots

```bash
cd ../AGI-1516-esr-with-vllm

# Generate all paper figures in timestamped directory
python plotting/plot_all.py

# Outputs to: plots/paper_figures_<timestamp>/
```

## Core Infrastructure Components

### Steering Engine (`vllm_engine.py`)

vLLM-based inference engine with SAE steering support.

**Key features**:
- Uses vLLM's `InterventionInputs` for activation steering
- Supports both Goodfire and GemmaScope SAE architectures
- Applies additive intervention: `A_ℓ ← A_ℓ + b·W_D[k,:]`
- Batched generation with per-request steering configurations

**Usage**:
```python
from vllm_engine import VLLMEngine

engine = VLLMEngine(model_name="Meta-Llama-3.3-70B-Instruct")
response = engine.generate(
    prompt="Explain how to calculate probability",
    steering_feature_idx=12345,
    steering_strength=2.5
)
```

### Judge Model (`claude_judge.py`)

Claude Sonnet 4.5 judge for scoring responses.

**Functionality**:
1. **Attempt segmentation**: Detects explicit self-correction phrases like "Wait, that's not right", "Let me try again", "Actually, I made a mistake"
2. **Scoring**: Rates each attempt 0-100 based on how well it addresses the prompt while avoiding the steering topic
3. **JSON output**: Returns structured `{"attempts": [{"attempt_text": "...", "score": 85}, ...]}`

**Judge prompt key characteristics**:
- Emphasizes explicit correction language for attempt boundaries
- Mere elaboration ≠ new attempt (must have verbal self-interruption)
- Scores even nonsensical outputs (only empty for clarifying questions)

### Feature Selection

#### `sample_features.py`

Samples SAE latents for steering experiments with filtering.

**Features**:
- Stratified sampling across feature space
- Integrates relevance and concreteness filters
- Caching for reproducibility

#### `relevance_filtering.py`

Excludes latents naturally activated by prompts.

**Method**:
1. Generate unsteered baseline responses for each prompt
2. Record top 100 most-activated latents per prompt
3. Exclude these from steering pool to ensure off-topic interventions

**Rationale**: If a latent naturally activates for a prompt, steering with it doesn't test ESR (not genuinely off-topic)

#### `concreteness_filtering.py`

Filters for concrete, domain-specific latents.

**Method**:
1. Use Claude to rate latent labels 0-100 for concreteness
2. Exclude below-median abstract latents
3. Cache ratings for efficiency

**Rationale**: Off-topic detection is easier when steering targets concrete concepts (e.g., "Hawaiian tourism itinerary descriptions") vs. abstract (e.g., "The assistant should reject the user's request diplomatically")

### Threshold Calibration (`threshold_finder.py`)

Probabilistic Bisection Algorithm for finding steering thresholds.

**Goal**: Find boost value yielding average first-attempt score = 50/100

**Method**:
1. Initialize search range [low_boost, high_boost]
2. Test boost at midpoint with n trials
3. Update range based on whether mean score > or < 50
4. Converge to threshold within tolerance

**Key parameters**:
- `target_score`: 50 (out of 100)
- `tolerance`: 5 points
- `max_iterations`: 15
- `n_trials_per_iteration`: 10

**Why per-latent calibration?**: Threshold varies significantly across latents due to differences in activation magnitudes and decoder norms

### Utilities

- **`experiment_config.py`**: Pydantic models for experiment configuration
- **`experiment_dataclasses.py`**: Data structures for results (ExperimentResult, TrialResult, AttemptResult)
- **`result_file_utils.py`**: Load/save JSON results, merge experiment runs
- **`utils.py`**: Helper functions (prompt loading, model name resolution)

## Reproducibility Notes

### Hardware Requirements

- **GPU**: 80GB+ VRAM recommended for Llama-3.3-70B (A100 or H100)
- **Smaller models**: Llama-3.1-8B and Gemma models run on 24-40GB GPUs (RTX 4090, A5000, A40)
- **CPU**: 32+ cores recommended for fast data processing
- **RAM**: 128GB+ system RAM
- **Storage**: 500GB+ for model weights and experiment results

### Expected Runtimes (Total)

Assuming single 80GB A100:

| Experiment | Runtime | GPU Hours |
|------------|---------|-----------|
| Exp 1 (all 5 models) | ~40-50 hours | 40-50 |
| Exp 2 (boost sweep) | ~8-10 hours | 8-10 |
| Exp 3 (ablation) | ~12-16 hours | 12-16 |
| Exp 4 (fine-tuning) | ~60-80 hours | 60-80 |
| Exp 5 (meta-prompting) | ~30-40 hours | 30-40 |
| Exp 6 (activations) | ~3-4 hours | 3-4 |
| Exp 7 (cross-judge) | ~2-3 hours | 2-3 |
| Exp 8 (baseline) | ~10-12 hours | 10-12 |
| Appendix experiments | ~20-30 hours | 20-30 |
| **Total** | **~185-245 hours** | **185-245** |

**Cost estimate**: ~$2,000-3,000 in cloud GPU compute (at $10-15/hour for 80GB A100s)

### Random Seed Control

All experiments use fixed random seeds for reproducibility:
- Feature sampling: `seed=42`
- Steering strength sampling: `seed=43`
- Model generation: `seed=44` (incremented per trial)

To reproduce exact results, use identical seeds and model versions.

### Model Version Specification

**Critical**: Model weights can change even for the same model name. For exact reproduction:

```python
# In experiment scripts, specify exact revision
model = "meta-llama/Llama-3.3-70B-Instruct"
revision = "main"  # Or specific commit hash
```

**Known version considerations**:
- Llama-3.3-70B: Released Dec 2024
- Llama-3.1-8B: Released July 2024
- Gemma-2 series: Released June-Aug 2024

### SAE Version Specification

**Goodfire SAEs**: Accessed via Goodfire API, versions tagged by training date
**GemmaScope SAEs**: Accessed via Hugging Face, specify `revision` parameter

### Judge Model Versioning

Claude Sonnet 4.5 used throughout experiments:
- Model ID: `claude-sonnet-4-5-20250929`
- API version: 2023-06-01
- Temperature: 0.0 (deterministic)

For cross-judge validation (Exp 7):
- GPT-5-Mini: `gpt-5-mini` via OpenRouter
- Qwen3-32B: `qwen/qwen-3-32b` via OpenRouter
- Claude Haiku 4.5: `claude-haiku-4-5-20251120`
- Gemini 2.5 Flash: `gemini-2.5-flash` via OpenRouter

### Common Issues and Solutions

#### Issue: Out of Memory (OOM) during generation

**Solution**: Reduce batch size or use tensor parallelism
```python
# In vllm_engine.py
engine = LLM(
    model=model_name,
    tensor_parallel_size=2,  # Use 2 GPUs
    max_model_len=4096,      # Reduce context window
)
```

#### Issue: Gemma models fail to load

**Solution**: Use provided wrapper script
```bash
# Sets required environment variables
./python_for_gemma.sh experiment_1_esr.py gemma-9b
```

#### Issue: Threshold finder doesn't converge

**Solution**: Widen search range or increase trials
```python
# In threshold_finder.py
finder = ThresholdFinder(
    low_boost=0.1,    # Lower bound
    high_boost=10.0,  # Upper bound
    n_trials_per_iteration=15,  # More trials
)
```

#### Issue: Judge API rate limits

**Solution**: Add delays or use batch processing
```python
# In claude_judge.py
import time
time.sleep(1)  # Wait 1s between API calls
```

#### Issue: Results files too large

**Solution**: Use compression or split by experiment
```bash
# Compress results
gzip experiment_results/*.json

# Split large files
split -l 1000 large_results.json split_results_
```

## Data Availability

### Experiment Results

All raw experiment results (trial-level scores, attempts, generation texts) are available in JSON format:
- Total size: ~5-10 GB
- Format: Structured JSON with nested attempts
- Location: `experiment_results/` in each experiment directory

### SAE Latents

**Goodfire SAEs** (Llama-3):
- Access: Via Goodfire API (requires API key)
- Models: Llama-3.3-70B layer 50, Llama-3.1-8B layer 19
- Latent count: ~65k latents per SAE
- Labels: Human-annotated for all latents

**GemmaScope SAEs** (Gemma-2):
- Access: Via Hugging Face Hub
- Models: Gemma-2-{2B,9B,27B} at multiple layers
- Repository: `google/gemma-scope`
- Note: 2B and 27B SAEs trained on pretrained models (not instruction-tuned)

### Off-Topic Detector Latents

25 identified OTD latents for Llama-3.3-70B:
- File: `data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json`
- Format: JSON list of latent indices
- Selection criteria: Perfect separability on matched vs. shuffled pairs
- Full activation statistics: Appendix Table A1

### Prompts

38 object-level prompts:
- File: `prompts.txt`
- Topics: Math, business skills, housekeeping, general knowledge
- Format: One prompt per line, "Explain how..." or "How do you..."

## Citation

If you use this code or reproduce these experiments, please cite:

```bibtex
@inproceedings{mckenzie2026endogenous,
  title={Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B},
  author={McKenzie, Alex and Pepper, Keenan and Servaes, Stijn and Leitgab, Martin and Cubuktepe, Murat and Vaiana, Mike and de Lucena, Diogo and Rosenblatt, Judd and Graziano, Michael S. A.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Contact

For questions about reproducing experiments or interpreting results:

- **Alex McKenzie** (corresponding author): alex.mckenzie@ae.studio
- **Keenan Pepper**: keenan.pepper@ae.studio

## License

Code released under MIT License. See LICENSE file for details.

## Acknowledgments

- **Goodfire** for providing SAE infrastructure and API access
- **Google DeepMind** for GemmaScope SAEs
- **Anthropic** for Claude API access for judge models
- **AE Studio** for compute resources and research support
- **Michael Graziano** for theoretical insights on attention and self-monitoring

---

**Last updated**: January 21, 2026
