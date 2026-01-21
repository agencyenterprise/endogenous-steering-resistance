# Random Latent Ablation Control Experiment

## Question

Are off-topic detector (OTD) latents specifically responsible for error self-repair (ESR), or would ablating any random set of latents produce similar results?

## Method

1. Ran feature steering with 25 separability-based OTD latents ablated (set to zero)
2. Ran feature steering with 25 randomly selected latents ablated (excluding dead latents)
3. Compared first-attempt scores and mean improvement (ESR metric)

## Results

| Condition | First Attempt | Mean Improvement |
|-----------|---------------|------------------|
| Steered baseline (no ablation) | 49.9% | 0.519 |
| OTD Ablation | 68.5% | 0.249 |
| Random Ablation (avg of 3 sets) | 69.2% | 0.224 |

## Key Finding

Both OTD and random ablation produce **much higher first-attempt scores** (~69% vs 50%). This suggests that ablating ANY latents interferes with the steering mechanism, causing the model to generate more on-topic responses on the first attempt.

## Interpretation

The "ESR reduction" observed with OTD ablation may be an artifact of **reduced steering effectiveness**, not the disabling of a specific self-monitoring mechanism. When steering is less effective:
- First attempts are more on-topic (higher scores)
- Less need for self-correction (lower mean improvement)

This is consistent with random ablation producing similar results - any disruption to the model's latent space could reduce steering effectiveness.

## Methodological Note

Previous analysis incorrectly compared random ablation results to a different baseline file (different prompts/seeds), which led to the misleading conclusion that random ablation doesn't affect ESR. When comparing to the correct source file (same prompts/seeds), random ablation shows similar effects to OTD ablation.
