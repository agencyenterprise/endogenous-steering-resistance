# AGI-ICML-2026-ESR-paper-code - Complete Index

**Purpose**: All code to reproduce "Endogenous Resistance to Activation Steering in Language Models" (ICML 2026)

## Start Here

| Document | Purpose | Size | Read Time |
|----------|---------|------|-----------|
| **QUICKSTART.md** | Get running in 15 minutes | 9KB | 5 min |
| **STRUCTURE.md** | Understand organization | 6KB | 3 min |
| **README_EXPERIMENTS.md** | Experiment reference | 4KB | 3 min |

## Complete Documentation

| Document | Purpose | Size |
|----------|---------|------|
| README.md | Comprehensive guide | 32KB |
| PAPER_MAPPING.md | Map figures to code | 11KB |
| MANIFEST.md | File inventory | 10KB |
| QUICKSTART.md | Getting started | 9KB |
| STRUCTURE.md | Folder organization | 6KB |
| README_EXPERIMENTS.md | Experiment guide | 4KB |
| CHANGES.md | Recent reorganization | 3KB |
| CITATION.bib | BibTeX citation | 1KB |
| LICENSE | MIT License | 1KB |
| .env.example | Environment template | 1KB |

**Total documentation**: ~78KB across 10 files

## Experiments (8 total)

### Main Text Experiments (1-5)

| # | Name | Paper | Type | What It Does |
|---|------|-------|------|--------------|
| 1 | experiment_1_esr.py | §3.1 | Script | Tests ESR across 5 models |
| 2 | experiment_2_multi_boost.py | §3.2 | Script | Sweeps steering strengths |
| 3 | experiment_3_off_topic_detectors/ | §3.4 | Dir | Finds & ablates OTD latents |
| 4 | experiment_4_finetuning/ | §3.5 | Dir | Trains on self-correction data |
| 5 | experiment_5_prompt_variants.py | §3.3 | Script | Tests meta-prompts |

### Appendix Experiments (6-8)

| # | Name | Paper | Type | What It Does |
|---|------|-------|------|--------------|
| 6 | experiment_6_sequential_activations/ | §3.6 + A.4 | Dir | Token-level activation analysis |
| 7 | experiment_7_otd_statistics/ | Appendix A.3.5 | Dir | OTD activation statistics |
| 8 | experiment_8_random_ablation_control/ | Appendix A.3.6 | Dir | Random ablation control |

## Core Infrastructure (14 files)

| File | Purpose |
|------|---------|
| vllm_engine.py | vLLM-based inference with SAE steering |
| claude_judge.py | Response evaluation and segmentation |
| experiment_config.py | Configuration dataclasses |
| experiment_dataclasses.py | Result data structures |
| threshold_finder.py | Probabilistic Bisection Algorithm |
| sample_features.py | SAE latent sampling |
| relevance_filtering.py | Filter naturally-activated latents |
| concreteness_filtering.py | Filter abstract latents |
| gemma_models_and_saes.py | Gemma model configurations |
| utils.py | Helper functions |
| prompts.txt | 38 evaluation prompts |
| requirements.txt | Python dependencies |
| install.sh | Installation script |
| python_for_gemma.sh | Gemma environment wrapper |

## Plotting & Analysis (12 files)

| Directory | Files |
|-----------|-------|
| plotting/ | plot_all.py, plot_exp1-8.py, plot_utils.py, plot_combined_*.py |
| . | regrade_cross_judge.py (cross-judge validation) |

## Data Files (3 files)

| File | Content |
|------|---------|
| data/off_topic_detectors_v2.json | 25 OTD latents (original discovery) |
| data/off_topic_detectors_separability.json | 25 OTD latents (separability-based) |
| data/normal_responses.json | Baseline unsteered responses |

## Scripts (1 file)

| File | Purpose |
|------|---------|
| scripts/run_all_experiments.sh | Master script to run all experiments |

## Quick Navigation

### I want to...

**...get started quickly**
→ Read `QUICKSTART.md` (5 min read, 15 min to first results)

**...understand the folder structure**
→ Read `STRUCTURE.md` (explains naming and organization)

**...find code for a specific paper figure**
→ Read `PAPER_MAPPING.md` (maps all figures to code)

**...run a specific experiment**
→ Read `README_EXPERIMENTS.md` (commands for each experiment)

**...see what files exist**
→ Read `MANIFEST.md` (complete inventory with descriptions)

**...understand what changed recently**
→ Read `CHANGES.md` (reorganization details)

**...reproduce everything**
→ Run `bash scripts/run_all_experiments.sh` (~185-245 GPU hours)

### Common Tasks

| Task | Command |
|------|---------|
| Install dependencies | `bash install.sh` |
| Run quick test | `python experiment_1_esr.py 8b --num-features 5 --num-prompts 5` |
| Reproduce Figure 2 | `python experiment_1_esr.py 70b && python plotting/plot_exp1.py` |
| Reproduce Figure 5 | `cd experiment_3_off_topic_detectors && python experiment_3_with_ablation.py 70b` |
| Generate all plots | `python plotting/plot_all.py` |
| Run all experiments | `bash scripts/run_all_experiments.sh` |

## Paper Sections → Code

| Paper Section | Folder/File |
|---------------|-------------|
| §1 Introduction | Examples from Experiment 1 |
| §2 Methods | Core infrastructure + `experiment_config.py` |
| §3.1 ESR Across Models | `experiment_1_esr.py` |
| §3.2 Boost Ablation | `experiment_2_multi_boost.py` |
| §3.3 Meta-Prompting | `experiment_5_prompt_variants.py` |
| §3.4 OTD Ablation | `experiment_3_off_topic_detectors/` |
| §3.5 Fine-Tuning | `experiment_4_finetuning/` |
| §3.6 Sequential Activations | `experiment_6_sequential_activations/` |
| Appendix A.2.2 (Cross-Judge) | `regrade_cross_judge.py` |
| Appendix A.3.1 (No-Steering) | `experiment_1_esr.py --no-steering` |
| Appendix A.3.5 (OTD Stats) | `experiment_7_otd_statistics/` |
| Appendix A.3.6 (Random Abl.) | `experiment_8_random_ablation_control/` |

## Figures → Code

| Figure | Script | Plotting |
|--------|--------|----------|
| Figure 1 | Manual selection from Exp 1 | LaTeX |
| Figure 2 | experiment_1_esr.py | plotting/plot_exp1.py |
| Figure 3 | experiment_2_multi_boost.py | plotting/plot_exp2.py |
| Figure 4 | experiment_5_prompt_variants.py | plotting/plot_exp5.py |
| Figure 5 | experiment_3_off_topic_detectors/ | plotting/plot_exp3.py |
| Figure 6 | experiment_6_sequential_activations/ | plotting/plot_exp6.py |
| Figure 7 | experiment_4_finetuning/ | plotting/plot_exp4.py |

## Compute Requirements

| Experiment | GPU Hours | Cost* |
|------------|-----------|-------|
| Experiment 1 (all models) | 60-80 | $600-800 |
| Experiment 2 | 15-20 | $150-200 |
| Experiment 3 | 30-40 | $300-400 |
| Experiment 4 | 40-60 | $400-600 |
| Experiment 5 | 15-20 | $150-200 |
| Experiment 6-8 | 25-35 | $250-350 |
| **Total** | **185-245** | **$2,000-3,000** |

*At $10/GPU-hour for A100-80GB

## File Counts

| Category | Count |
|----------|-------|
| Documentation | 10 files |
| Core infrastructure | 14 files |
| Main experiments | 5 items (2 scripts, 3 dirs) |
| Appendix experiments | 3 directories |
| Plotting scripts | 11 files |
| Data files | 3 files |
| Scripts | 1 file |

**Total**: ~100+ source files across all experiments

## Status

✅ **Complete and ready for reproduction**
- All 8 experiments implemented
- All documentation complete
- All plotting code ready
- Data files included
- Master run script provided

## Citation

See `CITATION.bib` for BibTeX citation.

## License

MIT License - See `LICENSE` file.

## Contact

See paper for author contact information.

---

**Last updated**: January 21, 2026
**Corresponds to**: ICML 2026 submission
**Git repo**: experiments/AGI-ICML-2026-ESR-paper-code/
