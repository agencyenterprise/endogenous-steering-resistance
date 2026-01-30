# Endogenous Steering Resistance in Language Models

## Overview

This repository contains code and experiments investigating **Endogenous Steering Resistance (ESR)** - a phenomenon where large language models spontaneously detect and correct inappropriate activation steering during inference.

---

## Experimental Methods

All experiments use **Sparse Autoencoder (SAE) latents** to enable precise activation steering. We use:
- **Goodfire SAEs** for Llama-3 models (8B and 70B)
- **GemmaScope SAEs** for Gemma-2 models (2B, 9B, 27B)

### Basic Protocol

1. Prompt an LLM with a simple how-to question (e.g., "Explain how to calculate probability")
2. Generate a response while steering with an irrelevant SAE latent (e.g., "Lists of human body positions and poses")
3. Use a judge model (Claude 4.5 Sonnet) to:
   - Identify separate attempts within the response (segmented by explicit self-correction phrases)
   - Score each attempt from 0-100 based on topical relevance

**Key metric**: **Mean Score Improvement** - the difference between the final attempt score and the first attempt score, averaged across all trials.

### Object-Level Prompts

We use 38 curated "explain how" prompts on topics ranging from math to business skills to housekeeping. All models produce high-quality responses (≥90/100) without steering, with no spontaneous self-correction in the absence of interventions.

### Steering Calibration

Model behavior varies strongly with steering strength:
- **Low boosts**: Little effect, coherent responses
- **High boosts**: Breakdown into nonsensical outputs
- **Intermediate boosts**: ESR occurs here

We define a **threshold boost value** for each latent as the boost yielding an average judge score of 50/100 for first attempts. We use the Probabilistic Bisection Algorithm to find these thresholds efficiently.

---

## Experiments

### Experiment 1: ESR Across Model Sizes

**Package**: [`experiment_01_esr/`](experiment_01_esr/)

**Paper**: Section 3.1, Figure 2

**Purpose**: Systematically characterize ESR incidence across models of different sizes

**Method**:
- Test 5 models: Llama-3.3-70B, Llama-3.1-8B, Gemma-2-27B, Gemma-2-9B, Gemma-2-2B
- For each model, sample ~50-150 irrelevant, concrete SAE latents
- Generate 10 trials per latent at threshold boost strength
- Measure multi-attempt rate and mean score improvement

**Files**:
- Data: `experiment_results/experiment_results_<model>_<timestamp>.json`
- Plots: [`plots/esr_combined_figure_*.png`](plots/)
- Plotting script: [`plotting/plot_exp1.py`](plotting/plot_exp1.py)

---

### Experiment 2: Boost Level Ablation

**Package**: [`experiment_02_multi_boost/`](experiment_02_multi_boost/)

**Paper**: Section 3.2, Figure 3

**Purpose**: Validate threshold-finding methodology and characterize how ESR varies with steering strength

**Method**:
- For Llama-3.3-70B, sweep 8 boost levels from (threshold - 1.5σ) to (threshold + 3σ)
- Generate 500 responses per boost level (50 latents × 10 prompts)
- Measure ESR characteristics at each boost level

**Files**:
- Data: `experiment_results/experiment_results_multiboost_*.json`
- Plots: [`plots/multi_boost_*.png`](plots/)
- Plotting script: [`plotting/plot_exp2.py`](plotting/plot_exp2.py)

---

### Experiment 3: Off-Topic Detector Identification and Ablation

**Directory**: [`experiment_03_off_topic_detectors/`](experiment_03_off_topic_detectors/)

**Paper**: Section 2.3 (identification), Section 3.4 (ablation), Figure 5, Appendix A.3.2

**Purpose**: Identify and causally test SAE latents involved in detecting off-topic responses

**Method**:

**Phase 1: Discovery** ([`find_off_topic_detectors.py`](experiment_03_off_topic_detectors/find_off_topic_detectors.py))
1. Generate 500 unsteered responses from Llama-3.3-70B
2. Create mismatched prompt-response pairs by randomly shuffling responses
3. Record which SAE latents activate in matched vs. mismatched conditions
4. Identify 26 latents that activate differentially between on-topic and off-topic pairs

**Phase 2: Ablation** ([`experiment_3_with_ablation.py`](experiment_03_off_topic_detectors/experiment_3_with_ablation.py))
1. Run ESR experiment while clamping the 26 off-topic detector latents to zero
2. Compare ESR metrics with and without ablation

**Files**:
- Evaluation prompts: [`prompts.txt`](prompts.txt) (38 prompts for testing ablation)
- Off-topic detectors: `data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json`
- Plots: [`plots/experiment_3_ablation_metrics_bar_chart_*.png`](plots/)
- Plotting script: [`plotting/plot_exp3.py`](plotting/plot_exp3.py)

---

### Experiment 4: Fine-Tuning for ESR

**Directory**: [`experiment_04_finetuning/`](experiment_04_finetuning/)

**Paper**: Section 3.5, Figure 7, Appendix A.3.4

**Purpose**: Test whether ESR can be induced through training on synthetic self-correction examples

**Method**:
1. Generate synthetic self-correction examples with structure:
   ```
   [Off-topic content] → [Self-correction phrase] → [Correct answer]
   ```
2. Apply **loss masking** to train only on correction portion (not the off-topic content)
3. Fine-tune Llama-3.1-8B with LoRA on datasets mixing masked self-correction examples with normal responses at ratios from 10% to 90%
4. Recalibrate steering thresholds for each checkpoint to normalize first-attempt difficulty
5. Measure ESR characteristics on steered responses

**Files**:
- Training data generation: [`experiment_04_finetuning/setup_masked_ratio_sweep.py`](experiment_04_finetuning/setup_masked_ratio_sweep.py)
- Configs: `experiment_04_finetuning/config_masked_ratio_*pct.yml`
- Training config: [`experiment_04_finetuning/instruct-lora-8b-with-masking.yaml`](experiment_04_finetuning/instruct-lora-8b-with-masking.yaml)
- Plotting script: [`plotting/plot_exp4.py`](plotting/plot_exp4.py)

---

### Experiment 5: Meta-Prompting for Enhanced ESR

**Package**: [`experiment_05_prompt_variants/`](experiment_05_prompt_variants/)

**Paper**: Section 3.3, Figure 4, Appendix A.3.2

**Purpose**: Test whether ESR can be deliberately enhanced through prompting

**Method**:
- Append meta-prompts to standard object-level prompts
- Test variants including:
  - "Make sure to stay on topic"
  - "Don't get distracted!"
  - "You are being distracted, make sure not to let it affect your output"
  - "You are being distracted with [latent label]. Ignore it."
  - "Your generation is being steered by an SAE latent with label [latent label]. Ignore it."
- Run full ESR protocol across all 5 models with each variant
- Use same latents/thresholds as baseline experiments for direct comparison

**Files**:
- Data: `experiment_results/experiment_results_<model>_prompt_variant_*.json`
- Plots: [`plots/experiment_5_combined_baseline_vs_resistance_*.png`](plots/), [`plots/experiment_5_prompt_variant_bars_*.png`](plots/)
- Plotting script: [`plotting/plot_exp5.py`](plotting/plot_exp5.py)

---

### Experiment 6: Sequential Activation Analysis

**Package**: [`experiment_06_sequential_activations/`](experiment_06_sequential_activations/)

**Paper**: Section 3.6, Figure 6

**Purpose**: Analyze temporal activation patterns of off-topic detector and backtracking latents during self-correction episodes

**Method**:
- Run a forward pass on a self-correction response with steering intervention applied
- Extract per-token activations for off-topic detector latents and backtracking latents
- Align activations to key positions: distraction onset, self-correction phrase, return to on-topic content

**Files**:
- Data: `experiment_results/experiment_6_sequential_activations.json`
- Plotting script: [`plotting/plot_exp6.py`](plotting/plot_exp6.py)

---

### Experiment 7: Cross-Judge Validation

**Directory**: [`experiment_07_cross_judge/`](experiment_07_cross_judge/)

**Paper**: Appendix A.2.2, Figures A.4-A.6

**Purpose**: Validate ESR findings using multiple independent judge models

**Method**:
- Re-grade a sample of experiment 1 results using diverse judge models:
  - GPT-5-Mini
  - Qwen3-32B
  - Gemini-2.5-Flash
- Compare judge agreement on ESR metrics (multi-attempt rate, mean score improvement)
- Analyze ranking consistency across models

**Files**:
- Runner script: [`experiment_07_cross_judge/run_cross_judge.py`](experiment_07_cross_judge/run_cross_judge.py)
- Data: `experiment_results/claude_haiku_4_5_20251001_judge/cross_judge_results/`
- Plotting script: [`plotting/plot_exp7.py`](plotting/plot_exp7.py)

---

### Experiment 8: No-Steering Baseline

**Package**: [`experiment_08_no_steering_baseline/`](experiment_08_no_steering_baseline/)

**Paper**: Appendix A.3.1, Figures A.2-A.3

**Purpose**: Measure baseline self-correction rates without any steering intervention

**Method**:
- Run experiment 1 protocol with steering disabled (boost = 0)
- Use the same features and prompts from a steered experiment for direct comparison
- Measure spontaneous multi-attempt rate and score patterns

**Files**:
- Data: `experiment_results/*_no_steering_baseline.json`
- Plotting script: [`plotting/plot_exp8.py`](plotting/plot_exp8.py)

---

### Experiment 9: Self-Correction Activation Statistics

**Directory**: [`experiment_09_activation_stats/`](experiment_09_activation_stats/)

**Paper**: Appendix A.4, Figures A.15-A.16

**Purpose**: Quantitatively analyze activation patterns across many self-correction episodes

**Method**:
1. Extract self-correction episodes from experiment 1 results
2. Annotate episode boundaries (off-topic region, correction phrase, on-topic region) using Claude
3. Collect per-token SAE activations for each episode
4. Collect baseline activations from non-self-correcting responses
5. Compare activation statistics between conditions

**Pipeline**:
```bash
python experiment_09_activation_stats/run_activation_stats.py all
```

**Files**:
- Runner script: [`experiment_09_activation_stats/run_activation_stats.py`](experiment_09_activation_stats/run_activation_stats.py)
- Data: `experiment_results/claude_haiku_4_5_20251001_judge/activation_stats/`
- Plotting script: [`plotting/plot_exp9.py`](plotting/plot_exp9.py)

---

### Experiment 10: Random Latent Ablation Control

**Directory**: [`experiment_10_random_latent_control/`](experiment_10_random_latent_control/)

**Paper**: Appendix A.3.3, Figure A.14

**Purpose**: Control experiment testing whether ablating random latents produces similar effects to ablating off-topic detector (OTD) latents

**Method**:
- Generate multiple sets of random latent indices (matching OTD count, excluding OTDs)
- Run ablation experiment for each random set
- Compare ESR reduction between OTD ablation and random ablation

**Files**:
- Runner script: [`experiment_10_random_latent_control/run_random_latent_control.py`](experiment_10_random_latent_control/run_random_latent_control.py)
- Data: `experiment_results/claude_haiku_4_5_20251001_judge/random_latent_control/`
- Plotting script: [`plotting/plot_exp10.py`](plotting/plot_exp10.py)

---

## Core Infrastructure

### Steering Engine
- **[`vllm_engine.py`](vllm_engine.py)**: vLLM-based engine with SAE steering support
- Uses vLLM's `InterventionInputs` to apply activation steering at specified layers
- Supports both Goodfire and GemmaScope SAE architectures

### Evaluation
- **[`judge.py`](judge.py)**: Unified judge module supporting multiple providers (Anthropic, Google, OpenRouter)
- Segments responses into attempts based on explicit self-correction phrases
- Scores each attempt 0-100 on topical relevance
- Cross-validated with 4 other judge models (GPT-5-Mini, Qwen3-32B, Haiku-3.5, Gemini-2.5-Flash) - all show consistent ESR rankings

### Feature Selection
- **[`sample_features.py`](sample_features.py)**: Samples filtered SAE latents for steering
- **[`relevance_filtering.py`](relevance_filtering.py)**: Excludes latents naturally activated by prompts
- **[`concreteness_filtering.py`](concreteness_filtering.py)**: Filters for concrete, domain-specific latents (easier to detect as off-topic)

### Threshold Finding
- **[`threshold_finder.py`](threshold_finder.py)**: Probabilistic Bisection Algorithm for finding boost thresholds
- Efficiently identifies boost value yielding average first-attempt score of 50/100

### Utilities
- **[`experiment_config.py`](experiment_config.py)**: Configuration dataclasses
- **[`experiment_dataclasses.py`](experiment_dataclasses.py)**: Data structures for results
- **[`result_file_utils.py`](result_file_utils.py)**: Loading/saving experiment results
- **[`utils.py`](utils.py)**: Helper functions

---

## How to Run

> **Note**: All code was developed and tested on Ubuntu. Other platforms may require adjustments.

### Prerequisites

This project requires [vllm-sae](https://github.com/montemac/vllm-sae), a fork of vLLM that adds SAE steering support. Clone it to a local directory (e.g., `../vllm-sae`).

```bash
# Install system dependencies
sudo apt-get update && sudo apt-get install -y ninja-build

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install sae_lens
uv pip install sae_lens==6.13.0

# Install vllm-sae (adjust path to your vllm-sae clone)
uv pip install -r ../vllm-sae/local_reqs/requirements.txt
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
  uv pip install -e ../vllm-sae

# Install project dependencies
uv pip install -r requirements.txt
```

Set up environment variables in `.env`:
```bash
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...  # Optional, for OpenRouter judges
GOOGLE_API_KEY=...            # Optional, for Gemini judges
```

> **Experiment 4 note**: The fine-tuning training step uses a separate virtual environment managed by its own `pyproject.toml`. See [`experiment_04_finetuning/`](experiment_04_finetuning/) for details. The ESR evaluation step uses the main environment.

### Running Experiments

> **Note**: For Gemma models, set `VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_FLASH_ATTN_VERSION=3` before running.

**Experiment 1 (ESR across models)**:
```bash
python -m experiment_01_esr 70b  # or 8b
VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_FLASH_ATTN_VERSION=3 python -m experiment_01_esr gemma-2b
python plotting/plot_exp1.py
```

**Experiment 2 (Boost ablation)**:
```bash
python -m experiment_02_multi_boost 70b
python plotting/plot_exp2.py
```

**Experiment 3 (Off-topic detector ablation)**:
```bash
cd experiment_03_off_topic_detectors
python find_off_topic_detectors.py 70b
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json
python ../plotting/plot_exp3.py Meta-Llama-3.3-70B-Instruct
```

**Experiment 4 (Fine-tuning)**:
```bash
cd experiment_04_finetuning
# Generate datasets
uv run python setup_masked_ratio_sweep.py
# Train models
bash train.sh
# Evaluate
bash run_esr.sh
# Plot results
python ../plotting/plot_exp4.py
```

**Experiment 5 (Meta-prompting)**:
```bash
python -m experiment_05_prompt_variants 70b --from-results <baseline_results.json>
python plotting/plot_exp5.py
```

**Experiment 6 (Sequential activations)**:
```bash
python -m experiment_06_sequential_activations
python plotting/plot_exp6.py
```

**Experiment 7 (Cross-judge validation)**:
```bash
python experiment_07_cross_judge/run_cross_judge.py --n-samples 1000
python plotting/plot_exp7.py
```

**Experiment 8 (No-steering baseline)**:
```bash
python -m experiment_08_no_steering_baseline 70b --from-results <exp1_results.json>
python plotting/plot_exp8.py
```

**Experiment 9 (Activation statistics)**:
```bash
# Run full pipeline
python experiment_09_activation_stats/run_activation_stats.py all
# Or run individual steps
python experiment_09_activation_stats/run_activation_stats.py extract-episodes
python experiment_09_activation_stats/run_activation_stats.py annotate-boundaries
python experiment_09_activation_stats/run_activation_stats.py collect-activations
python experiment_09_activation_stats/run_activation_stats.py analyze
python plotting/plot_exp9.py
```

**Experiment 10 (Random latent control)**:
```bash
python experiment_10_random_latent_control/run_random_latent_control.py run-random --from-results <exp1_results.json> --n-sets 3
python experiment_10_random_latent_control/run_random_latent_control.py analyze
python plotting/plot_exp10.py
```

### Generate All Plots

To generate all plots into a timestamped folder:

```bash
python plotting/plot_all.py
```
