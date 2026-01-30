# Experiment 10: Random Latent Ablation Control

Control experiment testing whether the effects of ablating off-topic detector (OTD)
latents on ESR are specific to those latents, or if any random set of latents would
produce similar results.

## Question

Are OTD latents specifically responsible for ESR, or would ablating any random set
of latents produce similar results?

## Method

1. Run feature steering with 25 OTD latents ablated (set to zero)
2. Run feature steering with 25 randomly selected latents ablated (excluding dead latents)
3. Compare first-attempt scores and mean improvement (ESR metric)

Uses experiment 1 features and prompts to ensure comparability.

## Usage

```bash
# Run with OTD ablation
python run_random_latent_control.py --ablation-type otd --from-results <exp1_results.json>

# Run with random ablation (generates 3 random sets)
python run_random_latent_control.py --ablation-type random --n-sets 3 --from-results <exp1_results.json>

# Analyze and plot results
python run_random_latent_control.py analyze
```

## Output

Results are saved to:
- `experiment_results/claude_haiku_4_5_20251001_judge/random_latent_control/`

## Key Finding

Both OTD and random ablation produce much higher first-attempt scores (~69% vs 50%).
This suggests ablating ANY latents interferes with steering, causing more on-topic
first attempts regardless of which latents are ablated.

## Source

Adapted from AGI-1652-random-latent-ablation-control.
