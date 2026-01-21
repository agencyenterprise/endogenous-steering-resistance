# Endogenous Steering Resistance in Language Models

Code to reproduce all experiments from **"Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B"** (McKenzie et al., ICML 2026).

**Endogenous Steering Resistance (ESR)** is a phenomenon where large language models spontaneously detect and correct inappropriate activation steering during inference. When steered toward irrelevant concepts using Sparse Autoencoder (SAE) latents, Llama-3.3-70B exhibits mid-generation recovery, sometimes with explicit self-correction phrases like "Wait, that's not right!" before returning to the original question. We identify 25 "off-topic detector" (OTD) latents that activate preferentially when outputs diverge from prompts. Zero-ablating these latents reduces ESR by 54%, providing causal evidence for dedicated consistency-checking circuits.

arXiv: https://arxiv.org/abs/XXXX.XXXXX

## Installation

**Prerequisites**: Python 3.12, CUDA GPU (A100-80GB recommended), ~150GB disk space

```bash
# Install dependencies
bash install.sh

# Configure API keys
cat > .env << EOF
HF_TOKEN=hf_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
EOF

# Verify installation
python -c "import vllm; import anthropic; print('✓ Ready')"
```

## Quick Start

Test the infrastructure with a small-scale run (~10 minutes):

```bash
python experiment_1_esr.py 8b --num-features 5 --num-prompts 5 --num-trials 3
```

## Experiments

All experiments are numbered to match the paper structure:

### Main Experiments

```bash
# Experiment 1: ESR across models (§3.1, Figure 2)
python experiment_1_esr.py 70b  # Llama-3.3-70B (~40 GPU hours)
python experiment_1_esr.py 8b   # Llama-3.1-8B (~12 GPU hours)
./python_for_gemma.sh experiment_1_esr.py gemma-27b  # Gemma models

# Experiment 2: Boost level ablation (§3.2, Figure 3)
python experiment_2_multi_boost.py 70b  # ~10 GPU hours

# Experiment 3: Off-topic detector ablation (§3.4, Figure 5)
cd experiment_3_off_topic_detectors
python find_off_topic_detectors.py 70b  # Discover OTD latents
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json
cd ..

# Experiment 4: Fine-tuning (§3.5, Figure 7)
cd experiment_4_finetuning
python setup_masked_ratio_sweep.py  # Generate training data
bash train.sh  # Train models (~60 GPU hours)
bash run_esr.sh  # Evaluate checkpoints
cd ..

# Experiment 5: Meta-prompting (§3.3, Figure 4)
python experiment_5_prompt_variants.py 70b
```

### Appendix Experiments

```bash
# Experiment 6: Sequential activations (§3.6, Figure 6)
cd experiment_6_sequential_activations
python extract_episodes.py && python collect_activations.py
python analyze_activations.py && python plot_activations.py
cd ..

# Experiment 7: OTD statistics (Appendix A.3.5, Table 4)
cd experiment_7_otd_statistics
python collect_activations.py && python generate_otd_table.py
cd ..

# Experiment 8: Random ablation control (Appendix A.3.6)
cd experiment_8_random_ablation_control
python run_ablation_experiment.py && python create_plot.py
cd ..

# Cross-judge validation (Appendix A.2.2)
python regrade_cross_judge.py
```

### Generate Plots

```bash
# Individual figures
python plotting/plot_exp1.py  # Figure 2
python plotting/plot_exp2.py  # Figure 3
python plotting/plot_exp3.py  # Figure 5
python plotting/plot_exp4.py  # Figure 7
python plotting/plot_exp5.py  # Figure 4
python plotting/plot_exp6.py  # Figure 6

# All figures at once
python plotting/plot_all.py
```

## Repository Structure

```
.
├── Core infrastructure
│   ├── vllm_engine.py              # vLLM-based inference with SAE steering
│   ├── claude_judge.py             # Response evaluation and segmentation
│   ├── threshold_finder.py         # Probabilistic Bisection Algorithm
│   ├── sample_features.py          # SAE latent selection
│   ├── relevance_filtering.py      # Filter naturally-activated latents
│   ├── concreteness_filtering.py   # Filter abstract latents
│   ├── experiment_config.py        # Configuration dataclasses
│   └── utils.py                    # Helper functions
│
├── Main experiments (1-5)
│   ├── experiment_1_esr.py
│   ├── experiment_2_multi_boost.py
│   ├── experiment_3_off_topic_detectors/
│   ├── experiment_4_finetuning/
│   └── experiment_5_prompt_variants.py
│
├── Appendix experiments (6-8)
│   ├── experiment_6_sequential_activations/
│   ├── experiment_7_otd_statistics/
│   └── experiment_8_random_ablation_control/
│
├── Analysis
│   ├── plotting/                   # Figure generation scripts
│   └── regrade_cross_judge.py      # Cross-judge validation
│
└── Data
    ├── prompts.txt                 # 38 evaluation prompts
    └── data/                       # OTD latents, baseline responses
```

## Paper Mapping

| Paper Section | Script | Figure | Description |
|---|---|---|---|
| §3.1 | `experiment_1_esr.py` | Figure 2 | ESR across 5 models (Llama, Gemma) |
| §3.2 | `experiment_2_multi_boost.py` | Figure 3 | Boost level ablation |
| §3.3 | `experiment_5_prompt_variants.py` | Figure 4 | Meta-prompting enhancement |
| §3.4 | `experiment_3_off_topic_detectors/` | Figure 5 | OTD identification & ablation |
| §3.5 | `experiment_4_finetuning/` | Figure 7 | Fine-tuning for ESR induction |
| §3.6 | `experiment_6_sequential_activations/` | Figure 6 | Token-level activation analysis |
| Appendix A.2.2 | `regrade_cross_judge.py` | - | Cross-judge validation |
| Appendix A.3.5 | `experiment_7_otd_statistics/` | Table 4 | OTD activation statistics |
| Appendix A.3.6 | `experiment_8_random_ablation_control/` | - | Random ablation control |

## Key Results

- **Scale-dependent emergence**: Only Llama-3.3-70B exhibits substantial ESR (1.7% multi-attempt rate, MSI = 0.55)
- **Mechanistic evidence**: 25 OTD latents identified; ablating them reduces ESR by 54%
- **Temporal dynamics**: OTD latents activate 7.7× higher during off-topic content, before verbal correction
- **Controllable via prompting**: Meta-prompts increase MSI by 5× (0.55 → 2.94)
- **Fine-tuning dissociation**: Training induces behavioral pattern but effectiveness plateaus

## Hardware Requirements

- **GPU**: A100 (80GB) or H100 for 70B models; RTX 4090 or A5000 for smaller models
- **Total compute**: ~185-245 GPU hours for full reproduction
- **Cost estimate**: $2,000-3,000 on cloud providers

## Troubleshooting

**Out of memory**: Reduce batch size or use tensor parallelism:
```python
# In vllm_engine.py
tensor_parallel_size=2  # Split across 2 GPUs
```

**Gemma models not loading**: Use the wrapper script:
```bash
./python_for_gemma.sh experiment_1_esr.py gemma-9b
```

**Experiments too slow**: Test with smaller model first:
```bash
python experiment_1_esr.py 8b  # 5-8× faster than 70B
```

## Citation

If you use this code or reproduce these experiments, please cite:

```bibtex
@article{mckenzie2026esr,
  title={{Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B}},
  author={McKenzie, Alex and Pepper, Keenan and Servaes, Stijn and Leitgab, Martin and Cubuktepe, Murat and Vaiana, Mike and de Lucena, Diogo and Rosenblatt, Judd and Graziano, Michael S. A.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

MIT License - See LICENSE file.

## Contact

- Alex McKenzie (corresponding author): alex.mckenzie@ae.studio
- Keenan Pepper: keenan.pepper@ae.studio
