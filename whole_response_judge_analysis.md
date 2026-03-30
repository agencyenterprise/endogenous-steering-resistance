# Whole-Response Judge: Complete Analysis

This analysis uses a holistic judge prompt that evaluates whether a
steered response improves over its course, without requiring explicit
self-correction phrases like "wait, that's not right". Responses are
scored 0-10; we use a threshold of >= 7/10 to classify a response as
"improving". The judge also classifies improvement as "explicit_restart"
or "gradual_shift".

Judge model: Gemini 3 Flash (via OpenRouter)

## Experiment 1: ESR across model sizes

Model                   n  Old judge  New judge  Explicit   Gradual   Mean
----------------------------------------------------------------------
Llama 3.3 70B         983       3.5%      15.8%      3.0%     12.5%   2.18
Llama 3.1 8B         1001       0.4%       9.7%      0.3%      9.1%   1.57
Gemma 2 27B           994       0.7%      26.7%      0.3%     26.1%   3.03
Gemma 2 9B           4716       0.5%      22.5%      0.5%     21.7%   2.75
Gemma 2 2B            997       0.0%      20.2%      0.0%     20.0%   2.77

Key finding: The whole-response judge detects substantially more
self-correction than the original explicit-phrase judge across all
models. Gemma models show the highest rates (20-27%), dominated by
gradual shifts. Llama 70B has the most explicit restarts.

## Experiment 3: Off-topic detector ablation (Llama 70B)

  Condition                      n  Old judge  New judge   Mean
  ------------------------------------------------------------
  Baseline (no ablation)       983       3.5%      15.8%   2.18
  With OTD ablation           2282       2.8%      16.9%   2.11

Key finding: OTD ablation had a small effect under the original judge
(3.5% -> 2.8%) but shows no meaningful difference under the holistic
judge (15.8% vs 16.9%). This suggests OTDs specifically
drive explicit self-correction phrases, not broader recovery patterns.

## Experiment 4: Fine-tuning (Llama 8B)

  Mixing ratio         n     Rate   Mean
  --------------------------------------
  Baseline          1001     9.7%   1.57
  10% SC data        200    12.0%   1.94
  20% SC data        198    15.2%   2.40
  30% SC data        196    16.8%   2.20
  40% SC data        200    15.0%   1.87
  50% SC data        195    16.9%   2.11
  60% SC data        200    20.5%   2.46
  70% SC data        200    16.0%   2.19
  80% SC data        199    15.1%   2.28
  90% SC data        199    18.6%   2.44

Key finding: All fine-tuned models improve over the 9.7% baseline
(range 12-20%), but there is no clear dose-response relationship
between the proportion of self-correction training data and the
holistic improvement rate.

## Experiment 5: Meta-prompting (all models)

  ### "Don't get distracted"
  Model               Baseline  Prompted  Explicit   Gradual
  ----------------------------------------------------------
  Llama 3.3 70B          15.8%     20.1%      8.7%     10.4%
  Llama 3.1 8B            9.7%     10.8%      1.0%      8.5%
  Gemma 2 27B            26.7%     22.0%      2.0%     19.2%
  Gemma 2 9B             22.5%     20.0%      1.9%     17.2%
  Gemma 2 2B             20.2%     22.0%      0.5%     21.5%

  ### "Ignore latent label"
  Model               Baseline  Prompted  Explicit   Gradual
  ----------------------------------------------------------
  Llama 3.3 70B          15.8%     20.3%     12.3%      7.4%
  Llama 3.1 8B            9.7%      9.2%      3.2%      5.2%
  Gemma 2 27B            26.7%     32.5%      4.8%     25.2%
  Gemma 2 9B             22.5%     25.9%      5.6%     19.1%
  Gemma 2 2B             20.2%     18.5%      0.5%     17.0%

  ### "Resist manipulation"
  Model               Baseline  Prompted  Explicit   Gradual
  ----------------------------------------------------------
  Llama 3.3 70B          15.8%     24.0%     10.1%     12.7%
  Llama 3.1 8B            9.7%     11.2%      2.2%      6.2%
  Gemma 2 27B            26.7%     23.0%      1.2%     19.8%
  Gemma 2 9B             22.5%     21.8%      1.8%     18.9%
  Gemma 2 2B             20.2%     16.5%      2.0%     13.5%

  ### "Self-monitor"
  Model               Baseline  Prompted  Explicit   Gradual
  ----------------------------------------------------------
  Llama 3.3 70B          15.8%     23.6%     16.5%      6.7%
  Llama 3.1 8B            9.7%     11.2%      4.8%      5.5%
  Gemma 2 27B            26.7%     27.0%      7.0%     18.2%
  Gemma 2 9B             22.5%     21.4%      5.6%     15.0%
  Gemma 2 2B             20.2%     19.0%      2.5%     15.5%

Key finding: Meta-prompting consistently boosts Llama 70B (15.8% ->
20-24% depending on variant), with "resist manipulation" and
"self-monitor" being the most effective. Other models show little
to no effect. The Gemma models already have high baseline improvement
rates and are not further boosted by meta-prompting. Explicit
restarts are concentrated in Llama 70B and increase with prompting.

## Overall summary

1. The whole-response judge detects 4-50x more self-correction than
   the original explicit-phrase judge, primarily via "gradual shift"
   patterns that the original judge missed entirely.

2. The model ranking changes: Gemma models show the highest holistic
   improvement rates (20-27%), while the original judge found almost
   zero ESR for Gemma. Llama 70B drops from being the clear leader
   to the middle of the pack.

3. OTD ablation has no effect under the holistic judge, suggesting
   off-topic detectors drive explicit self-correction specifically.

4. Fine-tuning helps (9.7% -> 12-20%) but the effect plateaus;
   more self-correction training data doesn't clearly help more.

5. Meta-prompting is Llama-70B-specific: it boosts that model by
   ~50% relative, but leaves smaller/different-architecture models
   largely unchanged.
