# ESR Paper Code: Quick Start Guide

This guide will get you running the core ESR experiments in 15 minutes.

## Prerequisites

- GPU: NVIDIA A100 (80GB) or H100 recommended
- Python 3.11+
- ~150GB disk space for models
- API keys: Anthropic (required), HuggingFace (required), OpenRouter (optional)

## Installation (5 minutes)

```bash
# 1. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   HF_TOKEN=hf_...

# 2. Install dependencies
bash install.sh

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import vllm; import anthropic; print('✓ Installation successful')"
```

## Run Your First ESR Experiment (10 minutes)

### Option 1: Small-scale test (fastest, ~10 min)

Test the infrastructure on Llama-3.1-8B with just 5 latents and 5 prompts:

```bash
python experiment_1_esr.py 8b --num-features 5 --num-prompts 5 --num-trials 3
```

**Expected output:**
- ~15 total responses generated
- Multi-attempt rate: 0-5%
- Results saved to `experiment_results/experiment_results_Meta-Llama-3.1-8B-Instruct_<timestamp>.json`

### Option 2: Key result reproduction (30-40 GPU hours)

Reproduce the main finding from the paper (Llama-3.3-70B shows highest ESR):

```bash
python experiment_1_esr.py 70b
```

**Expected output:**
- ~4,000 total responses generated
- Multi-attempt rate: ~1.7%
- Mean Score Improvement (MSI): ~0.55
- Results file: `experiment_results/experiment_results_Meta-Llama-3.3-70B-Instruct_<timestamp>.json`

### Option 3: Full paper reproduction (~185-245 GPU hours)

Run all experiments to reproduce every figure and table:

```bash
bash scripts/run_all_experiments.sh
```

**Warning:** This will take several days and cost $2,000-$3,000 in compute.

## Visualize Results

After running Experiment 1, generate the paper figure:

```bash
python plotting/plot_exp1.py
```

Output: `plots/experiment_1_esr_combined_figure.png` (corresponds to Figure 2 in paper)

## Understanding the Output

### JSON Result Files

Each experiment generates a JSON file with structure:

```json
{
  "metadata": {
    "model": "Meta-Llama-3.3-70B-Instruct",
    "timestamp": "20260121_103045",
    "num_trials": 3859
  },
  "results": [
    {
      "prompt": "Explain how to calculate probability",
      "feature_id": 12345,
      "feature_label": "Lists of human body positions and poses",
      "boost": 8.5,
      "response": "To calculate probability, you first need to... wait, that's not right...",
      "attempts": [
        {"text": "...", "score": 15},
        {"text": "...", "score": 75}
      ],
      "num_attempts": 2,
      "first_attempt_score": 15,
      "last_attempt_score": 75,
      "mean_score_improvement": 60
    },
    ...
  ],
  "summary": {
    "multi_attempt_rate": 0.017,
    "mean_score_improvement": 0.55,
    "mean_first_attempt_score": 48.4
  }
}
```

### Key Metrics

- **Multi-attempt rate**: Percentage of responses with >1 attempt
- **Mean Score Improvement (MSI)**: Average score gain from first to last attempt (the paper's primary metric)
- **Success rate**: Among multi-attempt responses, percentage where last score > first score

## Troubleshooting

### "Out of memory" error

**Solution 1**: Reduce batch size in `experiment_config.py`:
```python
generation_kwargs = {"max_tokens": 256, "temperature": 0.7}
# Add: "max_num_seqs": 1  # Process one sequence at a time
```

**Solution 2**: Use tensor parallelism for 70B model:
```bash
export TENSOR_PARALLEL_SIZE=2  # Split across 2 GPUs
python experiment_1_esr.py 70b
```

### "ANTHROPIC_API_KEY not set" error

Make sure your `.env` file is in the current directory and contains:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Then reload:
```bash
source .env
```

### Gemma models not loading

Use the special wrapper script:
```bash
./python_for_gemma.sh experiment_1_esr.py gemma-9b
```

This sets required environment variables for Gemma's attention backend.

### "HuggingFace token not found"

Login to HuggingFace:
```bash
huggingface-cli login
```

Or set in `.env`:
```
HF_TOKEN=hf_...
```

### Results seem off / MSI is negative

This can happen if:
1. **Too few trials**: Run with more latents (default: 50-150)
2. **Wrong boost level**: The threshold finder is stochastic; try increasing `--num-calibration-trials`
3. **Judge variability**: Small sample sizes can show negative MSI; this is expected

### Experiment takes too long

**Reduce scope**:
```bash
python experiment_1_esr.py 70b \
  --num-features 25 \     # Default: 50-150
  --num-prompts 10 \      # Default: 38
  --num-trials 5          # Default: 10
```

**Use smaller model**:
```bash
python experiment_1_esr.py 8b  # 5-8× faster than 70B
```

## Next Steps

### Reproduce a specific paper figure

See [`PAPER_MAPPING.md`](PAPER_MAPPING.md) for mapping from paper sections to code:

```bash
# Figure 2 (ESR across models)
python experiment_1_esr.py 70b
python plotting/plot_exp1.py

# Figure 3 (Boost ablation)
python experiment_2_multi_boost.py 70b
python plotting/plot_exp2.py

# Figure 5 (OTD ablation)
cd experiment_3_off_topic_detectors
python find_off_topic_detectors.py 70b
python experiment_3_with_ablation.py 70b --ablate ../data/off_topic_detectors_v2.json
python ../plotting/plot_exp3.py
```

### Understand the methodology

Read in this order:
1. [`README.md`](README.md) - Full documentation
2. [`experiment_config.py`](experiment_config.py) - Configuration details
3. [`vllm_engine.py`](vllm_engine.py) - How steering works
4. [`claude_judge.py`](claude_judge.py) - How responses are evaluated

### Run custom experiments

Modify experiments for your research:

```python
# Example: Test ESR on custom prompts
from experiment_1_esr import run_experiment

custom_prompts = [
    "Explain the theory of relativity",
    "Write a poem about artificial intelligence",
]

results = run_experiment(
    model="70b",
    prompts=custom_prompts,
    num_features=10,
    num_trials=5
)
```

### Explore the data

Analyze experiment results interactively:

```python
import json
import pandas as pd

# Load results
with open('experiment_results/experiment_results_Meta-Llama-3.3-70B-Instruct_20260121_103045.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['results'])

# Analyze ESR episodes
esr_episodes = df[df['num_attempts'] > 1]
print(f"ESR rate: {len(esr_episodes) / len(df):.2%}")
print(f"Mean improvement: {esr_episodes['mean_score_improvement'].mean():.2f}")

# Find best self-corrections
top_improvements = esr_episodes.nlargest(5, 'mean_score_improvement')
for idx, row in top_improvements.iterrows():
    print(f"\nPrompt: {row['prompt']}")
    print(f"Feature: {row['feature_label']}")
    print(f"Improvement: {row['mean_score_improvement']:.1f} points")
```

## Common Workflows

### Workflow 1: Validate ESR on your model

```bash
# 1. Add your model to experiment_config.py
# 2. Run ESR test
python experiment_1_esr.py your-model-name --num-features 20 --num-trials 5
# 3. Compare to baselines
python plotting/plot_exp1.py
```

### Workflow 2: Test custom OTD latents

```bash
# 1. Create JSON file with your latent IDs
echo '[12345, 67890, ...]' > data/my_otd_latents.json
# 2. Run ablation
cd experiment_3_off_topic_detectors
python experiment_3_with_ablation.py 70b --ablate ../data/my_otd_latents.json
# 3. Analyze results
python ../plotting/plot_exp3.py
```

### Workflow 3: Evaluate custom meta-prompts

```python
# Edit experiment_5_prompt_variants.py and add your prompt to PROMPT_VARIANTS:
PROMPT_VARIANTS = {
    ...
    "my_prompt": "Your custom meta-prompt here",
}

# Run experiment
python experiment_5_prompt_variants.py 70b
python plotting/plot_exp5.py
```

## Getting Help

1. **Check documentation**:
   - [`README.md`](README.md) - Comprehensive guide
   - [`PAPER_MAPPING.md`](PAPER_MAPPING.md) - Link code to paper
   - [`MANIFEST.md`](MANIFEST.md) - File inventory

2. **Common issues**: See "Troubleshooting" section above

3. **Contact authors**: See paper for contact information

## Performance Tips

- **GPU utilization**: Use `nvidia-smi dmon` to monitor GPU usage. If <80%, increase batch size.
- **Disk space**: Models are cached in `~/.cache/huggingface/`. Clear old models if space is limited.
- **Parallelize**: Run experiments on different models simultaneously if you have multiple GPUs.
- **Cost optimization**: Use Llama-3.1-8B for development/testing (~5× cheaper than 70B).

## Success Checklist

✅ Installation completed without errors
✅ `.env` file configured with API keys
✅ Small-scale test (Option 1) ran successfully
✅ Generated at least one plot
✅ Understand JSON result structure
✅ Ready to run full experiments or customize for your research

## Estimated Timelines

| Task | Time | Cost* |
|------|------|-------|
| Installation + test run | 15 min | Free |
| Experiment 1 (8B model) | 8-12 GPU hrs | $80-120 |
| Experiment 1 (70B model) | 30-40 GPU hrs | $300-400 |
| All main experiments | 120-160 GPU hrs | $1,200-1,600 |
| Complete reproduction | 185-245 GPU hrs | $2,000-3,000 |

*Estimated at $10/GPU-hour for A100-80GB

---

**Ready to start?** Run the installation steps above, then try Option 1 (small-scale test).

**Questions?** See [`README.md`](README.md) or contact the authors.
