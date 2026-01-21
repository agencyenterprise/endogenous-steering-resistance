# Experiment Folder Contents

## Overview

This folder contains all code and data needed to reproduce the results in:

**"Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B"**
McKenzie et al., ICML 2026

## Directory Structure

```
AGI-ICML-2026-ESR-paper/
│
├── README.md                          # Main documentation
├── PAPER_MAPPING.md                   # Maps paper sections to code
├── CITATION.bib                       # BibTeX citation
├── LICENSE                            # MIT License
├── MANIFEST.md                        # This file
├── .env.example                       # Example environment variables
│
├── Core Infrastructure (15 files)
│   ├── vllm_engine.py                 # vLLM-based steering engine
│   ├── claude_judge.py                # Judge model for scoring responses
│   ├── experiment_config.py           # Configuration dataclasses
│   ├── experiment_dataclasses.py      # Result data structures
│   ├── threshold_finder.py            # Probabilistic Bisection Algorithm
│   ├── sample_features.py             # SAE latent sampling
│   ├── relevance_filtering.py         # Filter relevant latents
│   ├── concreteness_filtering.py      # Filter abstract latents
│   ├── gemma_models_and_saes.py       # Gemma model configs
│   ├── utils.py                       # Helper functions
│   ├── prompts.txt                    # 38 object-level prompts
│   ├── requirements.txt               # Python dependencies
│   ├── install.sh                     # Installation script
│   └── python_for_gemma.sh            # Gemma environment wrapper
│
├── Main Experiments (4 files + 2 dirs)
│   ├── experiment_1_esr.py            # §3.1: ESR across models
│   ├── experiment_2_multi_boost.py    # §3.2: Boost level ablation
│   ├── experiment_5_prompt_variants.py # §3.3: Meta-prompting
│   ├── regrade_cross_judge.py         # Appendix A.2.2: Cross-judge validation
│   ├── experiment_3_off_topic_detectors/  # §3.4: OTD ablation
│   │   ├── find_off_topic_detectors.py
│   │   ├── experiment_3_with_ablation.py
│   │   └── prompts_otd_discovery.txt
│   └── experiment_4_finetuning/       # §3.5: Fine-tuning for ESR
│       ├── setup_masked_ratio_sweep.py
│       ├── generate_prompted_self_correction_data.py
│       ├── train.sh
│       ├── run_esr.sh
│       ├── off_topic_subjects.txt
│       ├── dataset_normal_responses.json
│       └── config_masked_ratio_*.yml (9 configs)
│
├── Appendix Experiments
│   └── appendices/
│       ├── self-correction-activation-statistics/  # §3.6 + Appendix A.4
│       │   ├── extract_episodes.py           # Extract self-correction episodes
│       │   ├── collect_activations.py        # Collect token-level activations
│       │   ├── collect_baseline_activations.py
│       │   ├── annotate_boundaries.py        # Manual boundary annotation
│       │   ├── analyze_activations.py        # Statistical analysis
│       │   ├── plot_activations.py           # Generate activation plots
│       │   ├── find_backtracking_latents.py  # Identify backtracking latents
│       │   └── generate_tables.py            # Create LaTeX tables
│       │
│       └── otd-activation-statistics/        # Appendix A.3.5
│           ├── collect_activations.py        # Collect OTD activations
│           ├── analyze_activations.py        # Compute statistics
│           └── generate_otd_table.py         # Create OTD table for paper
│
├── Random Ablation Control (Appendix A.3.6)
│   └── random_ablation_control/
│       ├── random_latent_ablation.py    # Random latent selection
│       ├── run_ablation_experiment.py   # Run control experiment
│       ├── analyze_results.py           # Statistical comparison
│       ├── create_plot.py               # Generate comparison plot
│       └── README.md                    # Control experiment details
│
├── Plotting Scripts
│   └── plotting/
│       ├── plot_all.py                  # Generate all plots at once
│       ├── plot_exp1.py                 # Figure 2: ESR across models
│       ├── plot_exp1_include_degraded.py # Appendix: Robustness check
│       ├── plot_exp2.py                 # Figure 3: Boost ablation
│       ├── plot_exp3.py                 # Figure 5: OTD ablation
│       ├── plot_exp4.py                 # Figure 7: Fine-tuning results
│       ├── plot_exp5.py                 # Figure 4: Meta-prompting
│       ├── plot_exp6.py                 # Figure 6: Sequential activations
│       ├── plot_exp7.py                 # Appendix: Cross-judge validation
│       ├── plot_exp8.py                 # Appendix: No-steering baseline
│       ├── plot_utils.py                # Shared plotting utilities
│       └── plot_combined_prompting_finetuning.py
│
├── Data Files
│   └── data/
│       ├── off_topic_detectors_v2.json          # 25 OTD latents (original)
│       ├── off_topic_detectors_separability.json # 25 OTD latents (separability)
│       └── normal_responses.json                 # Baseline unsteered responses
│
├── Scripts
│   └── scripts/
│       └── run_all_experiments.sh       # Master script to run everything
│
├── Output Directories (created during execution)
│   ├── experiment_results/              # JSON files with trial-level data
│   ├── plots/                           # Generated figures
│   ├── checkpoints/                     # Fine-tuned model checkpoints
│   └── merged_models/                   # LoRA-merged models
│
└── Logs and Caches (gitignored)
    ├── .venv/                           # Virtual environment
    ├── __pycache__/                     # Python bytecode
    └── *.log                            # Execution logs
```

## File Counts by Category

| Category | Count | Description |
|----------|-------|-------------|
| Core infrastructure | 14 files | Steering, judging, sampling, utilities |
| Main experiment scripts | 6 files | Experiments 1, 2, 3, 4, 5, cross-judge |
| Appendix experiment scripts | 13 files | Sequential activations, OTD stats, random ablation |
| Plotting scripts | 11 files | Generate all paper figures |
| Configuration files | 11 files | Fine-tuning configs, prompts, data |
| Data files | 3 files | OTD latents, baseline responses |
| Documentation | 5 files | README, mapping, citation, manifest, license |
| **Total source files** | **63** | |

## Key Files for Reproduction

### Quick Start (5 files to understand first)
1. `README.md` - Main documentation
2. `PAPER_MAPPING.md` - Connect paper to code
3. `experiment_config.py` - Model and experiment configuration
4. `prompts.txt` - The 38 evaluation prompts
5. `scripts/run_all_experiments.sh` - Run everything

### Core Components (4 files)
1. `vllm_engine.py` - Implements SAE-based activation steering
2. `claude_judge.py` - Scores and segments model responses
3. `threshold_finder.py` - Calibrates steering strengths
4. `experiment_dataclasses.py` - Data structures for results

### Critical Experiments (3 files)
1. `experiment_1_esr.py` - Main ESR characterization (Figure 2)
2. `experiment_3_off_topic_detectors/experiment_3_with_ablation.py` - Causal evidence (Figure 5)
3. `experiment_5_prompt_variants.py` - Controllability (Figure 4)

## Data Dependencies

### Required for Experiments
- **Models**: Downloaded automatically from HuggingFace (requires `HF_TOKEN`)
  - Llama-3.3-70B-Instruct (~140GB)
  - Llama-3.1-8B-Instruct (~16GB)
  - Gemma-2-2B/9B/27B-it (~4/18/54GB)

- **SAEs**: Accessed via API or downloaded
  - Goodfire SAEs (Llama models)
  - GemmaScope SAEs (Gemma models)

### Provided in Repository
- `prompts.txt` - 38 object-level prompts
- `data/off_topic_detectors_v2.json` - 25 identified OTD latents
- `data/off_topic_detectors_separability.json` - Alternative OTD set
- `data/normal_responses.json` - Baseline responses for OTD discovery
- `experiment_4_finetuning/off_topic_subjects.txt` - 50 off-topic subjects
- `experiment_4_finetuning/dataset_normal_responses.json` - Normal training data

### Generated During Experiments
- Experiment results: `experiment_results/*.json`
- Plots: `plots/*.png`
- Fine-tuned checkpoints: `checkpoints/masked_ratio_*pct/`
- Activation data: `appendices/*/data/`

## Compute Requirements Summary

| Experiment | GPU Hours | GPU Type | Cost Estimate |
|------------|-----------|----------|---------------|
| Experiment 1 (all models) | 60-80 | A100-80GB | $600-800 |
| Experiment 2 (boost ablation) | 15-20 | A100-80GB | $150-200 |
| Experiment 3 (OTD ablation) | 30-40 | A100-80GB | $300-400 |
| Experiment 4 (fine-tuning) | 40-60 | A100-80GB | $400-600 |
| Experiment 5 (meta-prompting) | 15-20 | A100-80GB | $150-200 |
| Appendix experiments | 10-15 | A100-80GB | $100-150 |
| Random ablation control | 15-20 | A100-80GB | $150-200 |
| No-steering baseline | 8-12 | A100-80GB | $80-120 |
| **Total** | **185-245** | **A100-80GB** | **$2,000-3,000** |

## Version Control

This folder is self-contained and version-controlled. Key dependencies:
- Python 3.11+
- PyTorch 2.1+
- vLLM 0.6.0+
- transformers 4.40+
- See `requirements.txt` for full list

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mckenzie2026esr,
  title={{Endogenous Resistance to Activation Steering in Language Models}},
  author={McKenzie, Alex and Pepper, Keenan and Servaes, Stijn and ...},
  booktitle={ICML},
  year={2026}
}
```

See `CITATION.bib` for full citation.

## Support

For questions or issues:
- Check `README.md` for detailed instructions
- See `PAPER_MAPPING.md` for connecting paper claims to code
- Open an issue on GitHub (if public) or contact authors

## License

MIT License - See `LICENSE` file for details.

---

Last updated: January 2026
Corresponds to: ICML 2026 submission
