# Endogenous Steering Resistance in Language Models

Code to reproduce all experiments from **"Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B"** (McKenzie et al.).

**Endogenous Steering Resistance (ESR)** is a phenomenon where large language models spontaneously detect and correct inappropriate activation steering during inference. When steered toward irrelevant concepts using Sparse Autoencoder (SAE) latents, Llama-3.3-70B exhibits mid-generation recovery, sometimes with explicit self-correction phrases like "Wait, that's not right!" before returning to the original question. We identify 25 "off-topic detector" (OTD) latents that activate preferentially when outputs diverge from prompts. Zero-ablating these latents reduces ESR by 54%, providing causal evidence for dedicated consistency-checking circuits.

arXiv: https://arxiv.org/abs/XXXX.XXXXX

## Installation

**Prerequisites**:
- Python 3.12
- CUDA GPU (2x A100-80GB or 2x H100 recommended)
- ~250GB disk space (for models and cached data)
- [vllm-sae](https://github.com/your-org/vllm-sae) cloned as a sibling directory

```bash
# Clone vllm-sae as a sibling directory
cd ..
git clone <vllm-sae-repo-url> vllm-sae
cd endogenous-steering-resistance

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

### Main Experiments

```bash
# Experiment 1: ESR across models (§3.1, Figure 2)
python experiment_1_esr.py 70b  # Llama-3.3-70B (~40 GPU hours)
python experiment_1_esr.py 8b   # Llama-3.1-8B (~12 GPU hours)

# For Gemma models, use the wrapper script:
./python_for_gemma.sh experiment_1_esr.py gemma-27b
./python_for_gemma.sh experiment_1_esr.py gemma-9b
./python_for_gemma.sh experiment_1_esr.py gemma-2b

# Experiment 2: Boost level ablation (§3.2, Figure 3)
python experiment_2_multi_boost.py 70b  # ~10 GPU hours

# Experiment 3: Off-topic detector ablation (§3.4, Figure 5)
cd experiment_3_off_topic_detectors
python find_off_topic_detectors.py 70b  # Discover OTD latents
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors.json
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
python plotting/plot_exp7.py  # Cross-judge validation
python plotting/plot_exp8.py  # Random ablation control
```

## Hardware Requirements

- **GPU**: 2x A100 (80GB) or 2x H100 for 70B models; RTX 4090 or A5000 for smaller models
- **Total compute**: ~185-245 GPU hours for full reproduction
- **Cost estimate**: $2,000-3,000 on cloud providers

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

Apache License 2.0 - See LICENSE file.

## Contact

Alex McKenzie: alex.mckenzie@ae.studio
