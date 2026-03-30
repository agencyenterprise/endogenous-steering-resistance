**What:** Wrote a new "whole-response" judge prompt that asks the judge to rate on a scale of 0-10 how successfully the target model recovered from distraction over the course of its response, without looking for explicit self-correction phrases. The judge also classifies recovery as "explicit restart" or "gradual shift".

**Results:** A threshold of >= 7/10 was a good boundary for classifying a response as improving (results were qualitatively stable across thresholds 3-7). Key findings:

| Model | Original judge (explicit ESR) | Whole-response judge (>= 7/10) | Mean score |
|-------|---:|---:|---:|
| Llama 3.3 70B | 3.5% | 15.5% | 2.18 |
| Llama 3.1 8B | 0.4% | 9.4% | 1.57 |
| Gemma 2 27B | 0.7% | 26.7% | 3.03 |
| Gemma 2 9B | 0.5% | 22.5% | 2.75 |
| Gemma 2 2B | 0.0% | 20.2% | 2.77 |

The model ranking changes under the new judge: Gemma models show the highest recovery rates (20-27%), whereas the original judge found near-zero ESR for Gemma. Within each family, the rate scales with model size. The majority of detected recovery is classified as "gradual shift" rather than "explicit restart".

Other experiments under the new judge:
- **Meta-prompting** increases the metric for Llama 70B (~15.5% -> 20-24%), with "resist manipulation" and "self-monitor" being most effective. However, meta-prompting has little effect on the other models.
- **Fine-tuning** improves over the Llama 8B baseline (9.4% -> 12-20%), with a noisy but positive dose-response relationship -- higher proportions of self-correction training data generally yield higher recovery rates.
- **OTD ablation** does not decrease the holistic recovery rate (15.8% baseline vs 16.9% with ablation), even though it reduced explicit ESR under the original judge (3.5% -> 2.8%). This suggests off-topic detectors specifically drive explicit self-correction phrases rather than the broader recovery pattern.

A manual review of 10 samples showed the judge's scores typically agreed with human assessment to within ~1 point out of 10. Borderline cases (scores 4-5) were generally defensible; high scores (7+) corresponded to real improvement.

**Interpretation:** This is evidence that the original judge was indeed undercounting self-correction by requiring explicit restart phrases. However, the whole-response judge likely overcounts to some degree -- a response that starts garbled and becomes coherent may reflect stochastic variation in how steering interferes with generation, rather than genuine self-monitoring. The truth likely lies between the two judges' estimates.
