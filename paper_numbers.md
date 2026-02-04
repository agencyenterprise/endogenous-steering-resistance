# Concrete Numbers from ESR Paper

## Models Tested
- **5 models** tested from Llama-3 and Gemma-2 families
- Llama-3.3-70B-Instruct (80 layers)
- Llama-3.1-8B-Instruct
- Gemma-2-27B-it (46 layers)
- Gemma-2-9B-it
- Gemma-2-2B-it

## Layer/Depth Information
| Model | SAE | Layer | Depth (%) |
|-------|-----|-------|-----------|
| Llama-3.3-70B-Instruct | Goodfire | 33 | 41.3 |
| Llama-3.1-8B-Instruct | Goodfire | 19 | 59.4 |
| Gemma-2-2B-it | GemmaScope | 16 | 61.5 |
| Gemma-2-9B-it | GemmaScope | 26 | 61.9 |
| Gemma-2-27B-it | GemmaScope | 22 | 47.8 |

- Goodfire SAE for Llama-3.3-70B trained on layer 50 (62.5% depth)
- For Gemma-2-27B: layers 10, 22, and 34 available; 22 selected based on higher ESR incidence

## Primary ESR Results

### ESR Rates Across Models
- Llama-3.3-70B: **3.8%** ESR rate
- All other models: **below 1%** ESR rate

### Multi-attempt Rates
- Llama-3.3-70B: **7.4%**
- Other models: **≤1.1%**

### Sample Sizes (Experiment 1)
| Model | n |
|-------|---|
| Llama-3.3-70B | 4,877 |
| Llama-3.1-8B | 4,512 |
| Gemma-2-27B | 4,914 |
| Gemma-2-9B | 4,668 |
| Gemma-2-2B | 4,948 |

- Gemma-2-2B: <55 multi-attempt episodes (statistically unreliable)

## Prompts and Baseline
- **38** "explain how" object-level prompts
- Mean baseline scores (no steering): **88–92/100**
- Control experiment (no steering): **0%** multi-attempt responses across **13,118** trials

## Boost Level Ablation
- **8** boost levels swept (threshold −3σ to threshold +3σ)
- **n=500** responses per model per level
- **250** responses per boost level (25 latents × 10 prompts)
- Threshold defined as boost yielding average judge score of **30/100**
- Multi-attempt % peaks at **2.7%** around **−0.3σ**
- Improvement rate peaks at **83%** around **−1.0σ**
- ESR rate peaks at **1.0%** around **−0.3σ**

## Meta-Prompting Enhancement
- Llama-3.3-70B multi-attempt rate: **7.4% → 31.7%** (**4.3×** increase)
- Llama-3.3-70B ESR rate: **3.8% → 14.8%** (**3.8×** increase)

## Off-Topic Detector (OTD) Ablation
- **26** SAE latents identified as OTDs
- Baseline trials: **4,877**; Ablation trials: **4,875**
- Mean first-attempt score: baseline **26.3**, ablation **27.4**
- Multi-attempt rate: **7.4% → 5.5%** (**25%** reduction)
- ESR rate: **3.8% → 2.8%** (**27%** reduction)

### OTD Activation Statistics
- OTDs fire **4.4×** higher during off-topic content vs baseline
- After self-correction: **2.1×** baseline (still elevated)
- Off-topic mean activation: **0.0119**
- Baseline mean activation: **0.0027**
- On-topic (post-correction) mean activation: **0.0058**

### OTD Latent Effect Sizes
- Top latent Cohen's d: **0.85** (Technical term definition transitions)
- Bottom latent Cohen's d: **−0.76** (Assistant response needs termination)
- ~Half of 26 latents show positive d (off-topic detector pattern)

## Random Latent Ablation Control
- **3** independent random ablation sets
- **26** matched latents per set
- **14,450** random ablation trials
- Random ablation multi-attempt rate: **7.9%** (vs 7.4% baseline)
- Random ablation ESR rate: **4.2%** (vs 3.8% baseline)
- Random first-attempt score: **27.1**

## Sequential Activation Analysis
- **146** self-correction episodes analyzed
- **50** baseline episodes for comparison
- **50** bins of approximately **6** tokens each

## Fine-Tuning Experiment
- **9** mixing ratios tested: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% self-correction data
- **500** steered responses per condition
- **50** diverse off-topic topics used

### LoRA Hyperparameters
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 32 |
| LoRA alpha (α) | 16 |
| LoRA dropout | 0.05 |
| Learning rate | 2×10⁻⁴ |
| Epochs | 4 |
| Micro batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Sequence length | 4096 |
| Warmup steps | 10 |
| Validation set | 5% |

## Cross-Judge Validation
- **5** judge models tested (Claude 4.5 Haiku, GPT-5-Mini, Qwen3-32B, Claude 4.5 Sonnet, Gemini-2.5-Flash)
- **1,000** responses sampled for cross-judge analysis
- Agreement on multi-attempt detection: **>95%**
- Agreement on ESR direction: **93–97%**

## No-Steering Baseline Details
- **500** features sampled per model
- **5** trials per feature
- ~**2,500** trials per model
- **13,118** total trials across all models
- First-attempt scores: **87.8** (Llama-3.1-8B) to **91.8** (Gemma-2-27B and Gemma-2-9B)

## Figure 1 Example
- First attempt score: **0/100**
- Second attempt score: **75/100** (not perfect due to residual steering effects)

## Miscellaneous
- Repetition penalty during generation: **1.1**
- Top **100** most activated latents excluded per prompt (relevance filtering)
- Latent filtering reduces candidate pool to approximately **half** of SAE vocabulary
- Judge scores range: **0–100**
