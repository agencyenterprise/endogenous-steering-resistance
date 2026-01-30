# Endogenous Steering Resistance

This repository contains the code for the paper:

**"Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B"**

by McKenzie, Pepper, Servaes, Leitgab, Cubuktepe, Vaiana, de Lucena, Rosenblatt, and Graziano.

This research was conducted at [AE Studio](https://ae.studio/).

[Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) | [Project Website](https://example.com/esr)

---

## Setup

### Hardware Requirements

Most experiments were run on **2× H100 GPUs**. Running experiments with Llama-3.3-70B requires at least **90GB of VRAM**.

### Installation

All code was developed and tested on Ubuntu.

This project requires [vllm-sae](https://github.com/montemac/vllm-sae), a fork of vLLM that adds SAE steering support. Clone it to a local directory (e.g., `../vllm-sae`).

> **Note**: The convoluted installation steps below are due to pinned dependencies in vllm-sae and sae_lens that don't play nicely together. We install them in a specific order to avoid conflicts.

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

> **Note**: For Gemma models, set `VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_FLASH_ATTN_VERSION=3` before running.

---

## Experiments

### Experiment 1: ESR Across Model Sizes

Measures ESR incidence across Llama-3.3-70B, Llama-3.1-8B, and Gemma-2 models (2B, 9B, 27B).

**Paper**: Section 3.1, Figure 2

```bash
python -m experiment_01_esr 70b  # or 8b, gemma-2b, gemma-9b, gemma-27b
python plotting/plot_exp1.py
```

---

### Experiment 2: Boost Level Ablation

Sweeps steering strength to characterize the ESR operating regime.

**Paper**: Section 3.2, Figure 3

```bash
python -m experiment_02_multi_boost 70b
python plotting/plot_exp2.py
```

---

### Experiment 3: Off-Topic Detector Ablation

Identifies off-topic detector latents via contrastive search, then ablates them to measure causal contribution to ESR.

**Paper**: Section 2.3, Section 3.4, Figure 5

```bash
cd experiment_03_off_topic_detectors
python find_off_topic_detectors.py 70b
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors_Meta-Llama-3.3-70B-Instruct.json
python ../plotting/plot_exp3.py Meta-Llama-3.3-70B-Instruct
```

---

### Experiment 4: Fine-Tuning

Fine-tunes Llama-3.1-8B on synthetic self-correction examples to test whether ESR can be induced through training.

**Paper**: Section 3.5, Figure 7, Appendix A.3.4

> **Note**: The training step uses a separate virtual environment managed by `experiment_04_finetuning/pyproject.toml`. The ESR evaluation step uses the main environment.

```bash
cd experiment_04_finetuning
uv run python setup_masked_ratio_sweep.py  # Generate datasets
bash train.sh                               # Train models (separate venv)
bash run_esr.sh                             # Evaluate (main venv)
python ../plotting/plot_exp4.py
```

---

### Experiment 5: Meta-Prompting

Tests whether meta-prompts (e.g., "stay on topic") can enhance ESR.

**Paper**: Section 3.3, Figure 4, Appendix A.3.2

```bash
python -m experiment_05_prompt_variants 70b --from-results <exp1_results.json>
python plotting/plot_exp5.py
```

---

### Experiment 6: Sequential Activation Analysis

Analyzes per-token activation patterns of off-topic detector latents during self-correction episodes.

**Paper**: Section 3.6, Figure 6

```bash
python -m experiment_06_sequential_activations
python plotting/plot_exp6.py
```

---

### Experiment 7: Cross-Judge Validation

Re-grades responses with multiple judge models (GPT-5-Mini, Qwen3-32B, Gemini-2.5-Flash) to validate ESR findings.

**Paper**: Appendix A.2.2, Figures A.4–A.6

```bash
python experiment_07_cross_judge/run_cross_judge.py --n-samples 1000
python plotting/plot_exp7.py
```

---

### Experiment 8: No-Steering Baseline

Confirms 0% self-correction rate without steering intervention.

**Paper**: Appendix A.3.1, Figures A.2–A.3

```bash
python -m experiment_08_no_steering_baseline 70b --from-results <exp1_results.json>
python plotting/plot_exp8.py
```

---

### Experiment 9: Activation Statistics

Quantitative analysis of off-topic detector activations across 146 self-correction episodes.

**Paper**: Appendix A.4, Figures A.15–A.16

```bash
python experiment_09_activation_stats/run_activation_stats.py all
python plotting/plot_exp9.py
```

---

### Experiment 10: Random Latent Ablation Control

Control experiment showing that ablating random latents (matched for activation statistics) does not reduce ESR.

**Paper**: Appendix A.3.3, Figure A.14

```bash
python experiment_10_random_latent_control/run_random_latent_control.py run-random --from-results <exp1_results.json> --n-sets 3
python experiment_10_random_latent_control/run_random_latent_control.py analyze
python plotting/plot_exp10.py
```

---

### Generate All Plots

```bash
python plotting/plot_all.py
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mckenzie2026esr,
  title={Endogenous Resistance to Activation Steering in Language Models: Evidence for Internal Consistency Monitoring in Llama-3.3-70B},
  author={McKenzie, Alex and Pepper, Keenan and Servaes, Stijn and Leitgab, Martin and Cubuktepe, Murat and Vaiana, Mike and de Lucena, Diogo and Rosenblatt, Judd and Graziano, Michael S. A.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
