# Folder Structure Overview

## Naming Convention

This folder is named `AGI-ICML-2026-ESR-paper-code` to clearly indicate:
- **AGI-ICML-2026**: Conference and year
- **ESR**: Endogenous Steering Resistance (the phenomenon studied)
- **paper-code**: Contains all code to reproduce the paper

## Experiment Numbering

Experiments are numbered 1-8 for clarity:

| # | Name | Paper Section | Description |
|---|------|---------------|-------------|
| 1 | `experiment_1_esr.py` | §3.1 | ESR across model sizes |
| 2 | `experiment_2_multi_boost.py` | §3.2 | Boost level ablation |
| 3 | `experiment_3_off_topic_detectors/` | §3.4 | OTD identification & ablation |
| 4 | `experiment_4_finetuning/` | §3.5 | Fine-tuning for ESR |
| 5 | `experiment_5_prompt_variants.py` | §3.3 | Meta-prompting enhancement |
| 6 | `experiment_6_sequential_activations/` | §3.6 + App. A.4 | Token-level analysis |
| 7 | `experiment_7_otd_statistics/` | Appendix A.3.5 | OTD activation stats |
| 8 | `experiment_8_random_ablation_control/` | Appendix A.3.6 | Random ablation control |

## Why This Structure?

### Before (confusing)
```
appendices/
├── self-correction-activation-statistics/  # What number?
└── otd-activation-statistics/              # What number?
random_ablation_control/                    # What number?
```

### After (clear)
```
experiment_6_sequential_activations/        # Clearly experiment 6
experiment_7_otd_statistics/                # Clearly experiment 7
experiment_8_random_ablation_control/       # Clearly experiment 8
```

## Complete Structure

```
AGI-ICML-2026-ESR-paper-code/
│
├── Documentation
│   ├── README.md                       # Main documentation
│   ├── README_EXPERIMENTS.md           # Experiment guide
│   ├── STRUCTURE.md                    # This file
│   ├── PAPER_MAPPING.md                # Map figures to code
│   ├── QUICKSTART.md                   # Getting started
│   ├── MANIFEST.md                     # Complete inventory
│   └── CITATION.bib                    # BibTeX citation
│
├── Core Infrastructure
│   ├── vllm_engine.py
│   ├── claude_judge.py
│   ├── threshold_finder.py
│   ├── sample_features.py
│   ├── relevance_filtering.py
│   ├── concreteness_filtering.py
│   ├── experiment_config.py
│   ├── experiment_dataclasses.py
│   ├── gemma_models_and_saes.py
│   ├── utils.py
│   ├── prompts.txt
│   └── requirements.txt
│
├── Main Experiments (1-5)
│   ├── experiment_1_esr.py
│   ├── experiment_2_multi_boost.py
│   ├── experiment_3_off_topic_detectors/
│   ├── experiment_4_finetuning/
│   └── experiment_5_prompt_variants.py
│
├── Appendix Experiments (6-8)
│   ├── experiment_6_sequential_activations/
│   ├── experiment_7_otd_statistics/
│   └── experiment_8_random_ablation_control/
│
├── Analysis & Plotting
│   ├── plotting/
│   │   ├── plot_all.py
│   │   ├── plot_exp1.py through plot_exp8.py
│   │   └── plot_utils.py
│   └── regrade_cross_judge.py
│
├── Data & Configuration
│   ├── data/
│   │   ├── off_topic_detectors_v2.json
│   │   ├── off_topic_detectors_separability.json
│   │   └── normal_responses.json
│   └── .env.example
│
└── Scripts
    └── scripts/
        └── run_all_experiments.sh
```

## Finding Experiments

### By Paper Section
- Introduction (§1): See `PAPER_MAPPING.md`
- Methods (§2): See core infrastructure files
- Results §3.1: `experiment_1_esr.py`
- Results §3.2: `experiment_2_multi_boost.py`
- Results §3.3: `experiment_5_prompt_variants.py`
- Results §3.4: `experiment_3_off_topic_detectors/`
- Results §3.5: `experiment_4_finetuning/`
- Results §3.6: `experiment_6_sequential_activations/`

### By Figure Number
- Figure 1: Manual selection from Experiment 1
- Figure 2: `experiment_1_esr.py` → `plotting/plot_exp1.py`
- Figure 3: `experiment_2_multi_boost.py` → `plotting/plot_exp2.py`
- Figure 4: `experiment_5_prompt_variants.py` → `plotting/plot_exp5.py`
- Figure 5: `experiment_3_off_topic_detectors/` → `plotting/plot_exp3.py`
- Figure 6: `experiment_6_sequential_activations/` → `plotting/plot_exp6.py`
- Figure 7: `experiment_4_finetuning/` → `plotting/plot_exp4.py`

### By Research Question
- "Does ESR exist?": Experiment 1
- "How does ESR vary with steering strength?": Experiment 2
- "Can we enhance ESR?": Experiment 5
- "What circuits implement ESR?": Experiments 3, 6, 7
- "Can we induce ESR via training?": Experiment 4
- "Is OTD ablation specific?": Experiment 8

## Navigation Tips

1. **Start here**: `README.md` or `QUICKSTART.md`
2. **Find specific code**: `PAPER_MAPPING.md`
3. **Understand structure**: This file
4. **Run experiments**: `README_EXPERIMENTS.md`
5. **See all files**: `MANIFEST.md`

## Historical Note

This folder consolidates code from three previous experiments:
- `AGI-1516-esr-with-vllm`: Core infrastructure + Experiments 1-5
- `AGI-1635-esr-appendices`: Experiments 6-7
- `AGI-1652-random-latent-ablation-control`: Experiment 8

The reorganization makes the code structure match the paper structure for easier navigation and reproduction.
