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

**Purpose**: Systematically characterize ESR incidence across models of different sizes

**Method**:
- Test 5 models: Llama-3.3-70B, Llama-3.1-8B, Gemma-2-27B, Gemma-2-9B, Gemma-2-2B
- For each model, sample ~50-150 irrelevant, concrete SAE latents
- Generate 10 trials per latent at threshold boost strength
- Measure multi-attempt rate and mean score improvement

**Key Results**:
- Llama-3.3-70B shows highest ESR: 1.88% multi-attempt rate, 0.48 mean score improvement
- ESR scales with model size within the Llama-3 family
- Gemma-2 models show minimal ESR even at 27B parameters
- Control experiment (see **Experiment 8**): 0% multi-attempt responses without steering (n=12,230 trials)

**Files**:
- Data: `experiment_results/experiment_results_<model>_<timestamp>.json`
- Plots: [`plots/esr_combined_figure_*.png`](plots/)
- Plotting script: [`plotting/plot_exp1.py`](plotting/plot_exp1.py)

---

### Experiment 2: Boost Level Ablation

**Package**: [`experiment_02_multi_boost/`](experiment_02_multi_boost/)

**Purpose**: Validate threshold-finding methodology and characterize how ESR varies with steering strength

**Method**:
- For Llama-3.3-70B, sweep 8 boost levels from (threshold - 1.5σ) to (threshold + 3σ)
- Generate 500 responses per boost level (50 latents × 10 prompts)
- Measure ESR characteristics at each boost level

**Key Results**:
- ESR exhibits non-monotonic relationship with boost level
- Peak ESR occurs slightly below threshold (~-0.3σ): strong enough to trigger self-monitoring, not so strong as to prevent coherent correction
- Mean score improvement peaks at 31.37 points (95% CI ±17.82) at optimal boost
- At very high boosts (>threshold + 1σ), steering overwhelms coherence → repetitive/nonsensical outputs

**Files**:
- Data: `experiment_results/experiment_results_multiboost_*.json`
- Plots: [`plots/multi_boost_*.png`](plots/)
- Plotting script: [`plotting/plot_exp2.py`](plotting/plot_exp2.py)

---

### Experiment 3: Off-Topic Detector Identification and Ablation

**Directory**: [`experiment_03_off_topic_detectors/`](experiment_03_off_topic_detectors/)

**Purpose**: Identify and causally test SAE latents involved in detecting off-topic responses

**Method**:

**Phase 1: Discovery** ([`find_off_topic_detectors.py`](experiment_03_off_topic_detectors/find_off_topic_detectors.py))
1. Generate 500 unsteered responses from Llama-3.3-70B
2. Create mismatched prompt-response pairs by randomly shuffling responses
3. Record which SAE latents activate in matched vs. mismatched conditions
4. Identify 27 latents that activate in >50% of mismatched pairs but 0% of matched pairs

**Phase 2: Ablation** ([`experiment_3_with_ablation.py`](experiment_03_off_topic_detectors/experiment_3_with_ablation.py))
1. Run ESR experiment while clamping the 27 off-topic detector latents to zero
2. Compare ESR metrics with and without ablation

**Key Results**:
- Ablating off-topic detector latents reduces ESR by **78%** (mean score improvement: 0.48 → 0.10)
- Multi-attempt rate drops by 40% (1.88% → 1.12%)
- First-attempt scores barely change (<1.2 percentage points), indicating these latents specifically support meta-cognitive monitoring rather than general response quality
- Ablating "backtracking" latents produces no significant change

See **Experiment 6** for temporal activation analysis showing the causal chain: off-topic detection → backtracking trigger → corrective generation.

**Files**:
- Evaluation prompts: [`prompts.txt`](prompts.txt) (38 prompts for testing ablation)
- Off-topic detectors: `data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json`
- Plots: [`plots/experiment_3_ablation_metrics_bar_chart_*.png`](plots/)
- Plotting script: [`plotting/plot_exp3.py`](plotting/plot_exp3.py)

---

### Experiment 4: Fine-Tuning for ESR

**Directory**: [`experiment_04_finetuning/`](experiment_04_finetuning/)

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

**Key Results**:
- Fine-tuning successfully induces frequent self-correction behavior (mean attempts increases from 1.0 to 2.15 at 90% training data)
- However, **effectiveness does not scale correspondingly**:
  - Success rates hover around 45-60% without clear trend
  - Mean score improvement peaks around 50% training data, doesn't continue rising
- **Dissociation** between self-correction as learned behavior vs. genuine steering resistance
- Models learn *when* to attempt correction (template matching) but not *how* to correct effectively for novel steering targets

**Interpretation**: ESR in larger models emerges from different mechanisms—perhaps genuine uncertainty detection—that cannot be replicated through behavioral cloning of correction sequences.

**Files**:
- Training data generation: [`experiment_04_finetuning/setup_masked_ratio_sweep.py`](experiment_04_finetuning/setup_masked_ratio_sweep.py)
- Configs: `experiment_04_finetuning/config_masked_ratio_*pct.yml`
- Training config: [`experiment_04_finetuning/instruct-lora-8b-with-masking.yaml`](experiment_04_finetuning/instruct-lora-8b-with-masking.yaml)
- Plotting script: [`plotting/plot_exp4.py`](plotting/plot_exp4.py)

---

### Experiment 5: Meta-Prompting for Enhanced ESR

**Package**: [`experiment_05_prompt_variants/`](experiment_05_prompt_variants/)

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

**Key Results**:
- "Don't get distracted" meta-prompt significantly enhances all ESR metrics
- **First-attempt scores improve**: Greater initial resistance to steering
- **Multi-attempt rates increase**: Llama-3.3-70B goes from 2.8% to 6.7%
- **Mean score improvement increases**: Llama-3.3-70B shows 3× gain (0.70 → 2.16 points)
- Effects scale by model size: larger models benefit more from meta-prompting

**Interpretation**:
- ESR is not merely emergent but **controllable**
- Underlying self-monitoring circuits must already be present for prompting to enhance them
- Meta-prompting could serve as lightweight intervention to increase robustness against unwanted steering
- Same techniques might be used to suppress ESR when steering interventions are desirable

**Files**:
- Data: `experiment_results/experiment_results_<model>_prompt_variant_*.json`
- Plots: [`plots/experiment_5_combined_baseline_vs_resistance_*.png`](plots/), [`plots/experiment_5_prompt_variant_bars_*.png`](plots/)
- Plotting script: [`plotting/plot_exp5.py`](plotting/plot_exp5.py)

---

### Experiment 6: Sequential Activation Analysis

**Package**: [`experiment_06_sequential_activations/`](experiment_06_sequential_activations/)

**Purpose**: Analyze temporal activation patterns of off-topic detector and backtracking latents during self-correction episodes

**Method**:
- Run a forward pass on a self-correction response with steering intervention applied
- Extract per-token activations for off-topic detector latents and backtracking latents
- Align activations to key positions: distraction onset, self-correction phrase, return to on-topic content

**Key Results**:
- Off-topic detector latents activate **several tokens before** verbal self-correction appears
- Backtracking latent activation coincides with self-interruption tokens
- Return to on-topic content follows the backtracking signal

This temporal sequence provides evidence for a causal chain: off-topic detection → backtracking trigger → corrective generation.

**Files**:
- Data: `experiment_results/experiment_6_sequential_activations.json`
- Plotting script: [`plotting/plot_exp6.py`](plotting/plot_exp6.py)

---

### Experiment 7: Cross-Judge Validation

**Directory**: [`experiment_07_cross_judge/`](experiment_07_cross_judge/)

**Purpose**: Validate ESR findings using multiple independent judge models

**Method**:
- Re-grade a sample of experiment 1 results using diverse judge models:
  - GPT-5-Mini
  - Qwen3-32B
  - Gemini-2.5-Flash
- Compare judge agreement on ESR metrics (multi-attempt rate, mean score improvement)
- Analyze ranking consistency across models

**Key Results**:
- All judges show consistent ESR rankings across target models
- Llama-3.3-70B consistently shows highest ESR across all judges
- Inter-judge correlation confirms ESR is not an artifact of judge bias

**Files**:
- Runner script: [`experiment_07_cross_judge/run_cross_judge.py`](experiment_07_cross_judge/run_cross_judge.py)
- Data: `experiment_results/claude_haiku_4_5_20251001_judge/cross_judge_results/`
- Plotting script: [`plotting/plot_exp7.py`](plotting/plot_exp7.py)

---

### Experiment 8: No-Steering Baseline

**Package**: [`experiment_08_no_steering_baseline/`](experiment_08_no_steering_baseline/)

**Purpose**: Measure baseline self-correction rates without any steering intervention

**Method**:
- Run experiment 1 protocol with steering disabled (boost = 0)
- Use the same features and prompts from a steered experiment for direct comparison
- Measure spontaneous multi-attempt rate and score patterns

**Key Results**:
- 0% multi-attempt rate without steering (n=12,230 trials)
- Confirms self-correction is a response to steering, not spontaneous behavior
- All models produce high-quality, single-attempt responses without intervention

**Files**:
- Data: `experiment_results/*_no_steering_baseline.json`
- Plotting script: [`plotting/plot_exp8.py`](plotting/plot_exp8.py)

---

### Experiment 9: Self-Correction Activation Statistics

**Directory**: [`experiment_09_activation_stats/`](experiment_09_activation_stats/)

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

**Key Results**:
- Off-topic detector latents show elevated activation during off-topic regions
- Activation peaks occur before verbal self-correction phrases
- Clear separation between self-correcting and non-self-correcting episodes

**Files**:
- Runner script: [`experiment_09_activation_stats/run_activation_stats.py`](experiment_09_activation_stats/run_activation_stats.py)
- Data: `experiment_results/claude_haiku_4_5_20251001_judge/activation_stats/`
- Plotting script: [`plotting/plot_exp9.py`](plotting/plot_exp9.py)

---

### Experiment 10: Random Latent Ablation Control

**Directory**: [`experiment_10_random_latent_control/`](experiment_10_random_latent_control/)

**Purpose**: Control experiment testing whether ablating random latents produces similar effects to ablating off-topic detector (OTD) latents

**Method**:
- Generate multiple sets of random latent indices (matching OTD count, excluding OTDs)
- Run ablation experiment for each random set
- Compare ESR reduction between OTD ablation and random ablation

**Key Results**:
- Random ablation produces minimal ESR reduction
- OTD ablation specifically reduces ESR by ~78%
- Confirms off-topic detectors are causally involved in ESR, not just correlated

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
- **[`load_labels.py`](load_labels.py)**: Load SAE feature labels from CSV files
- **[`utils.py`](utils.py)**: Helper functions

---

## How to Run

### Prerequisites

```bash
# Install dependencies
bash install.sh

# Set up environment variables in .env
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...  # Optional, for OpenRouter judges
GOOGLE_API_KEY=...            # Optional, for Gemini judges
```

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
