# Experiment Organization

All experiments are numbered for clarity and map directly to paper sections.

## Main Experiments

### Experiment 1: ESR Across Models (§3.1)
**Script**: `experiment_1_esr.py`
**Paper Figure**: Figure 2
- Tests ESR incidence across 5 models (Llama-3.3-70B, Llama-3.1-8B, Gemma-2-27B/9B/2B)
- Finds Llama-3.3-70B shows highest ESR: 1.7% multi-attempt rate, MSI = 0.55

### Experiment 2: Boost Level Ablation (§3.2)  
**Script**: `experiment_2_multi_boost.py`
**Paper Figure**: Figure 3
- Sweeps 8 steering strengths from threshold-1.5σ to threshold+3σ
- Shows ESR peaks at intermediate boost levels

### Experiment 3: Off-Topic Detector Ablation (§3.4)
**Directory**: `experiment_3_off_topic_detectors/`
**Paper Figure**: Figure 5
- Identifies 25 OTD latents that detect off-topic responses
- Ablating them reduces ESR by 54%

### Experiment 4: Fine-Tuning (§3.5)
**Directory**: `experiment_4_finetuning/`
**Paper Figure**: Figure 7
- Trains Llama-3.1-8B on synthetic self-correction examples
- Tests 9 mixing ratios (10%-90% self-correction data)

### Experiment 5: Meta-Prompting (§3.3)
**Script**: `experiment_5_prompt_variants.py`
**Paper Figure**: Figure 4
- Tests 6 meta-prompt variants for enhancing ESR
- Shows 5× MSI improvement for Llama-3.3-70B

## Appendix Experiments

### Experiment 6: Sequential Activation Analysis (§3.6 + Appendix A.4)
**Directory**: `experiment_6_sequential_activations/`
**Paper Figure**: Figure 6
- Token-level analysis of OTD and backtracking latent activations
- Shows OTDs fire 7.7× higher during off-topic content

### Experiment 7: OTD Activation Statistics (Appendix A.3.5)
**Directory**: `experiment_7_otd_statistics/`
**Paper Table**: Table 4
- Computes activation statistics for 25 OTD latents
- Compares matched vs. shuffled (off-topic) prompt-response pairs

### Experiment 8: Random Ablation Control (Appendix A.3.6)
**Directory**: `experiment_8_random_ablation_control/`
**Paper Figure**: Appendix random ablation figure
- Tests whether OTD ablation effects are specific
- Compares ablating OTD latents vs. randomly selected latents

## Cross-Cutting Scripts

### Cross-Judge Validation (Appendix A.2.2)
**Script**: `regrade_cross_judge.py`
- Validates primary judge (Claude Sonnet 4.5) against 4 alternative judges
- Shows consistent ESR rankings across all judges

### No-Steering Baseline (Appendix A.3.1)
**Run**: `experiment_1_esr.py --no-steering`
- Control showing 0% multi-attempt responses without steering
- Confirms ESR is induced by steering, not spontaneous

## Running Experiments

### Individual experiments
```bash
# Experiment 1-2, 5
python experiment_1_esr.py 70b
python experiment_2_multi_boost.py 70b
python experiment_5_prompt_variants.py 70b

# Experiment 3
cd experiment_3_off_topic_detectors
python find_off_topic_detectors.py 70b
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors_v2.json

# Experiment 4
cd experiment_4_finetuning
bash train.sh && bash run_esr.sh

# Experiment 6
cd experiment_6_sequential_activations
python extract_episodes.py && python collect_activations.py
python analyze_activations.py && python plot_activations.py

# Experiment 7
cd experiment_7_otd_statistics
python collect_activations.py && python generate_otd_table.py

# Experiment 8
cd experiment_8_random_ablation_control
python run_ablation_experiment.py && python create_plot.py
```

### All experiments
```bash
bash scripts/run_all_experiments.sh
```

## Plotting

Generate all figures:
```bash
python plotting/plot_all.py
```

Or individual figures:
```bash
python plotting/plot_exp1.py  # Figure 2
python plotting/plot_exp2.py  # Figure 3
python plotting/plot_exp3.py  # Figure 5
python plotting/plot_exp4.py  # Figure 7
python plotting/plot_exp5.py  # Figure 4
python plotting/plot_exp6.py  # Figure 6
```
