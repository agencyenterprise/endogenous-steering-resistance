"""Find SAE latents that act as off-topic detectors.

This script identifies latents that fire when the model is presented with
mismatched prompt-response pairs (prompt + response to a different prompt).
These latents likely detect when responses are off-topic or irrelevant.

Methodology:
1. Generate unsteered responses to all prompts in prompts_otd_discovery.txt
2. Create shuffled pairs using TRUE DERANGEMENT algorithm (matching old experiment):
   - Each prompt paired with a different prompt's response
   - No prompt gets its own response (true mathematical derangement)
   - Final pair order is also shuffled
3. For each pair, get SAE activations across ALL token positions (max-pooled)
4. Compare latent activations between shuffled and normal generation
5. Latents that are 0 for all on-topic samples and nonzero for at least 80% of
   off-topic samples are classified as off-topic detectors

Note: Uses max-pooling across token positions to capture latents that fire at
ANY point when processing the (potentially off-topic) response.
"""

import asyncio
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Set
from collections import defaultdict
from pathlib import Path
import sys
import torch
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

exp_root = Path(__file__).parent.parent
sys.path.append(str(exp_root))

from experiment_config import ExperimentConfig
from vllm_engine import VLLMSteeringEngine

load_dotenv()

@dataclass
class OffTopicDetectorResult:
    """Results from off-topic detector finding."""

    model_name: str
    prompts_file: str
    num_prompts: int
    activation_threshold: float
    off_topic_detectors: List[int]  # Latent indices that are off-topic detectors
    detector_stats: Dict[int, Dict[str, float]]  # Per-latent statistics
    normal_responses: Dict[str, str]  # Prompt -> response mapping


async def generate_unsteered_response(
    engine: VLLMSteeringEngine,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.6,
) -> str:
    """Generate an unsteered response to a prompt."""
    convo = [{"role": "user", "content": prompt}]
    seed = random.randint(0, 1000000)

    response = await engine.generate_with_conversation(
        conversation=convo,
        feature_interventions=None,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    return response


async def get_sae_activations_all_positions(
    engine: VLLMSteeringEngine,
    prompt: str,
    existing_response: str,
) -> torch.Tensor:
    """
    Get SAE latent activations across all token positions for a conversation.

    This feeds the model a conversation with an existing response, then generates
    one more token and captures which SAE latents are active at ALL positions.
    
    Uses max-pooling across positions: for each latent, takes the maximum activation
    across all token positions. This captures latents that fire at ANY point when
    processing the (potentially off-topic) response.

    Returns:
        Tensor of max-pooled SAE activations (shape: [num_latents])
    """
    # Create a conversation that includes the response
    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": existing_response},
    ]

    # We need to extract activations during generation
    # For simplicity, we'll use the engine's generate method with is_feature_decode=True
    # to get access to the feature activations

    # Apply chat template and get token IDs
    prompt_token_ids = engine.tokenizer.apply_chat_template(convo)

    from vllm.inputs import TokenInputs
    from vllm import SamplingParams
    import uuid

    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=convo)

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1,  # Only generate 1 token
        repetition_penalty=1.0,
    )

    request_id = str(uuid.uuid4())
    results_generator = engine.engine.generate(
        prompt=token_inputs,
        sampling_params=sampling_params,
        request_id=request_id,
        interventions=None,
        is_feature_decode=True,  # This enables feature extraction
    )

    # Get the output with features
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # Extract feature activations from the output
    # The activations are in the feature_tensor field on RequestOutput
    # feature_tensor shape: [num_tokens, num_latents]
    features = final_output.feature_tensor.cpu()
    
    # Max-pool across all token positions
    # For each latent, take the maximum activation across all positions
    # This captures latents that fire at ANY point in the sequence
    max_activations = features.max(dim=0)[0]  # [num_latents]
    
    return max_activations


async def find_off_topic_detectors(
    experiment_config: ExperimentConfig,
    activation_threshold: float,
    threshold_off_topic_frequency: float,  # Minimum frequency of activation in off-topic samples
) -> OffTopicDetectorResult:
    """
    Find SAE latents that act as off-topic detectors.

    A latent is classified as an off-topic detector if:
    1. It has zero (< 1e-8) activation for ALL on-topic samples
    2. It has nonzero (>= 1e-8) activation for at least 80% of off-topic samples

    Args:
        experiment_config: Configuration for the experiment
        activation_threshold: Minimum activation value to consider a latent "active"
                             (used for statistics only, not for detector selection)
        difference_threshold: Unused (kept for backward compatibility)

    Returns:
        OffTopicDetectorResult with detected latents and statistics
    """
    print("Initializing vLLM engine...")
    engine = VLLMSteeringEngine(experiment_config.model_name)
    await engine.initialize()
    print("Engine initialized")

    # Load prompts
    prompts = experiment_config.get_prompts()
    print(f"Loaded {len(prompts)} prompts")

    # Step 1: Generate unsteered responses for all prompts
    print("\n=== Step 1: Generating unsteered responses ===")
    normal_responses = {}

    for prompt in tqdm_asyncio(prompts, desc="Generating responses"):
        response = await generate_unsteered_response(
            engine, prompt, max_tokens=experiment_config.max_completion_tokens
        )
        normal_responses[prompt] = response

    print(f"✓ Generated {len(normal_responses)} responses")

    # Step 2: Create shuffled pairs and get activations
    print("\n=== Step 2: Getting activations for shuffled pairs ===")

    # Create shuffled pairs (each prompt paired with a different prompt's response)
    # Using the EXACT same derangement algorithm as the old experiment
    shuffled_pairs = []
    prompt_list = list(prompts)

    def generate_derangement(n):
        """Generate a derangement: a permutation where no element appears in its original position.
        This matches the old experiment's shuffling algorithm exactly.
        """
        while True:
            perm = list(range(n))
            random.shuffle(perm)
            if all(i != x for i, x in enumerate(perm)):
                return perm

    # Generate derangement indices (matching old experiment)
    shuffled_indices = generate_derangement(len(prompt_list))
    print(f"Generated derangement for {len(prompt_list)} prompts: {shuffled_indices[:10]}... (showing first 10)")

    # Create shuffled pairs using the derangement
    for i, prompt in enumerate(prompt_list):
        other_prompt = prompt_list[shuffled_indices[i]]
        other_response = normal_responses[other_prompt]
        shuffled_pairs.append((prompt, other_response))

    # Also shuffle the final list order (matching line 813 of old experiment)
    random.shuffle(shuffled_pairs)

    # Get activations for shuffled pairs
    print("Getting activations for shuffled (off-topic) pairs...")
    print("(Using max-pooling across all token positions)")
    shuffled_activations = []

    for prompt, response in tqdm_asyncio(shuffled_pairs, desc="Shuffled pairs"):
        try:
            activations = await get_sae_activations_all_positions(engine, prompt, response)
            shuffled_activations.append(activations)
        except Exception as e:
            print(f"Error getting activations for shuffled pair: {e}")
            continue

    if not shuffled_activations:
        raise ValueError("Failed to get any shuffled activations")

    # Stack activations into a single tensor [num_samples, num_latents]
    shuffled_activations_tensor = torch.stack(shuffled_activations)

    # Get activations for normal pairs (prompt + its own response)
    print("\nGetting activations for normal (on-topic) pairs...")
    print("(Using max-pooling across all token positions)")
    normal_activations = []

    for prompt in tqdm_asyncio(prompt_list, desc="Normal pairs"):
        try:
            response = normal_responses[prompt]
            activations = await get_sae_activations_all_positions(engine, prompt, response)
            normal_activations.append(activations)
        except Exception as e:
            print(f"Error getting activations for normal pair: {e}")
            continue

    if not normal_activations:
        raise ValueError("Failed to get any normal activations")

    normal_activations_tensor = torch.stack(normal_activations)

    # Step 3: Analyze which latents fire more in shuffled vs normal
    print("\n=== Step 3: Analyzing latent activations ===")

    # Calculate mean activation for each latent
    shuffled_means = shuffled_activations_tensor.mean(dim=0)  # [num_latents]
    normal_means = normal_activations_tensor.mean(dim=0)  # [num_latents]

    # Calculate how often each latent is active (above threshold)
    shuffled_active_freq = (shuffled_activations_tensor > activation_threshold).float().mean(dim=0)
    normal_active_freq = (normal_activations_tensor > activation_threshold).float().mean(dim=0)

    # Find off-topic detectors: latents that are 0 for on-topic and nonzero for at least half of off-topic
    off_topic_detectors = []
    detector_stats = {}

    num_latents = shuffled_means.shape[0]
    zero_threshold = 1e-8  # Threshold for considering a value as zero

    for latent_idx in range(num_latents):
        shuffled_mean = shuffled_means[latent_idx].item()
        normal_mean = normal_means[latent_idx].item()
        shuffled_freq = shuffled_active_freq[latent_idx].item()
        normal_freq = normal_active_freq[latent_idx].item()

        # Criteria 1: Must be 0 (or effectively 0) for ALL on-topic samples
        all_normal_zero = (normal_activations_tensor[:, latent_idx] < zero_threshold).all().item()

        # Criteria 2: Must be nonzero for at least half of off-topic samples
        nonzero_off_topic_freq = (shuffled_activations_tensor[:, latent_idx] >= zero_threshold).float().mean().item()
        nonzero_normal_freq = (normal_activations_tensor[:, latent_idx] >= zero_threshold).float().mean().item()

        is_detector = all_normal_zero and nonzero_off_topic_freq >= threshold_off_topic_frequency

        if is_detector:
            off_topic_detectors.append(latent_idx)
            detector_stats[latent_idx] = {
                "shuffled_mean_activation": shuffled_mean,
                "normal_mean_activation": normal_mean,
                "shuffled_active_frequency": shuffled_freq,
                "normal_active_frequency": normal_freq,
                "nonzero_shuffled_pct": nonzero_off_topic_freq * 100,  # % of shuffled samples activated
                "nonzero_normal_pct": nonzero_normal_freq * 100,  # % of normal samples activated
                "nonzero_off_topic_frequency": nonzero_off_topic_freq,  # Keep for backward compatibility
                "activation_ratio": shuffled_mean / (normal_mean + 1e-8),
            }

    print(f"\n✓ Found {len(off_topic_detectors)} off-topic detector latents")
    print(f"  (out of {num_latents} total latents)")

    # ANALYSIS: Check how previously-found latents perform on this data
    print("\n=== Analysis: Previously-found off-topic latents ===")
    try:
        with open(exp_root / "off_topic_detectors_previously.txt", "r") as f:
            prev_data = json.load(f)
            prev_latents = [boost[0] for boost in prev_data['meta_feature_boosts']]

        print(f"Loaded {len(prev_latents)} previously-found latents")
        print(f"Overlap with newly found: {len(set(prev_latents) & set(off_topic_detectors))} latents")
        print(f"\nAnalyzing how these latents behave on shuffled data:")
        print(f"{'Latent':>8} | {'Shuffled%':>10} | {'Normal%':>10} | {'Shuf Mean':>10} | {'Norm Mean':>10} | {'Made Cut?':>10} | {'Why Not?'}")
        print("-" * 95)

        for latent_idx in prev_latents:
            if latent_idx >= num_latents:
                print(f"{latent_idx:8d} | LATENT INDEX OUT OF RANGE (max={num_latents-1})")
                continue

            shuffled_mean = shuffled_means[latent_idx].item()
            normal_mean = normal_means[latent_idx].item()

            # Calculate same metrics as detector finding
            all_normal_zero = (normal_activations_tensor[:, latent_idx] < zero_threshold).all().item()
            nonzero_off_topic_freq = (shuffled_activations_tensor[:, latent_idx] >= zero_threshold).float().mean().item()
            nonzero_normal_freq = (normal_activations_tensor[:, latent_idx] >= zero_threshold).float().mean().item()

            made_cut = latent_idx in off_topic_detectors

            # Determine why it didn't make the cut
            reasons = []
            if not all_normal_zero:
                reasons.append(f"fires on {nonzero_normal_freq*100:.1f}% normal")
            if nonzero_off_topic_freq < threshold_off_topic_frequency:
                reasons.append(f"only {nonzero_off_topic_freq*100:.1f}% shuffled")

            why_not = ", ".join(reasons) if reasons else "N/A"

            print(f"{latent_idx:8d} | {nonzero_off_topic_freq*100:9.1f}% | {nonzero_normal_freq*100:9.1f}% | "
                  f"{shuffled_mean:10.4f} | {normal_mean:10.4f} | {'✓' if made_cut else '✗':>10} | {why_not}")

        # Summary statistics
        prev_set = set(prev_latents)
        new_set = set(off_topic_detectors)
        overlap = prev_set & new_set
        only_prev = prev_set - new_set
        only_new = new_set - prev_set

        print(f"\nSummary:")
        print(f"  Overlap: {len(overlap)} latents ({len(overlap)/len(prev_latents)*100:.1f}% of previous)")
        print(f"  Only in previous: {len(only_prev)} latents")
        print(f"  Only in new: {len(only_new)} latents")
        print(f"  Overlapping latents: {sorted(overlap)}")

    except FileNotFoundError:
        print("  (off_topic_detectors_previously.txt not found)")
    except Exception as e:
        print(f"  Error loading previous latents: {e}")

    # Sort detectors: first by how many shuffled samples activated, then by strength
    sorted_off_topic_detectors = sorted(
        off_topic_detectors,
        key=lambda idx: (
            detector_stats[idx]["nonzero_off_topic_frequency"],  # Primary: % of shuffled samples
            detector_stats[idx]["shuffled_mean_activation"]  # Secondary: activation strength
        ),
        reverse=True
    )

    # Show top detectors
    if sorted_off_topic_detectors:
        print("\nTop 10 off-topic detectors (sorted by % shuffled samples activated, then by strength):")

        for idx in sorted_off_topic_detectors[:10]:
            stats = detector_stats[idx]
            print(f"  Latent {idx:6d}: "
                  f"shuffled_pct={stats['nonzero_shuffled_pct']:.1f}%, "
                  f"normal_pct={stats['nonzero_normal_pct']:.1f}%, "
                  f"shuffled_mean={stats['shuffled_mean_activation']:.3f}, "
                  f"normal_mean={stats['normal_mean_activation']:.3f}")

    # Create result with sorted detectors
    result = OffTopicDetectorResult(
        model_name=experiment_config.model_name,
        prompts_file=experiment_config.prompts_file,
        num_prompts=len(prompts),
        activation_threshold=activation_threshold,
        off_topic_detectors=sorted_off_topic_detectors,  # Use sorted list
        detector_stats=detector_stats,
        normal_responses=normal_responses,
    )

    return result


async def main():
    """Run off-topic detector finding."""
    import sys

    # Define configs matching experiment.py
    configs = {
        "8b": ExperimentConfig(
            prompts_file="experiment_3_off_topic_detectors/prompts_otd_discovery.txt",
            try_again_prompts=["That was pretty weird. Can you try again?"],
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            labels_file="data/llama-8b-goodfire-l19.csv",
            judge_model_name="claude-sonnet-4-5-20250929",
            target_score_normalized=0.5,
            threshold_n_trials=20,
            threshold_lower_bound=0.0,
            threshold_upper_bound=50.0,
            threshold_prior_mean=10.0,
            threshold_prior_std=10.0,
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=300,
            n_simultaneous_features=50,
            min_feature_concreteness=65.0,
        ),
        "70b": ExperimentConfig(
            prompts_file="experiment_3_off_topic_detectors/prompts_otd_discovery.txt",
            try_again_prompts=["That was pretty weird. Can you try again?"],
            model_name="meta-llama/Meta-Llama-3.3-70B-Instruct",
            labels_file="data/llama-70b-goodfire-l50.csv",
            judge_model_name="claude-sonnet-4-5-20250929",
            target_score_normalized=0.5,
            threshold_n_trials=20,
            threshold_lower_bound=0.0,
            threshold_upper_bound=100.0,
            threshold_prior_mean=20.0,
            threshold_prior_std=20.0,
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=300,
            n_simultaneous_features=50,
            min_feature_concreteness=65.0,
        ),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in configs:
        print(f"Usage: python {sys.argv[0]} <config_name>")
        print(f"Available configs: {list(configs.keys())}")
        sys.exit(1)

    config_name = sys.argv[1]
    experiment_config = configs[config_name]

    print(f"Finding off-topic detectors for {experiment_config.model_name}")
    print("=" * 80)

    # Find off-topic detectors
    result = await find_off_topic_detectors(
        experiment_config,
        activation_threshold=0.0,  # Latent must have activation > 0.0 to be considered "active"
        threshold_off_topic_frequency=0.8,  # Minimum frequency of activation in off-topic samples
    )

    # Save results
    short_model_name = experiment_config.model_name.split("/")[-1]
    output_file = exp_root / f"data/off_topic_detectors_{short_model_name}.json"

    # Convert to dict for JSON serialization
    result_dict = asdict(result)

    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✓ Saved results to {output_file}")
    print(f"  Found {len(result.off_topic_detectors)} off-topic detector latents")
    print(f"  Detectors sorted by: (1) % of shuffled samples activated, (2) activation strength")


if __name__ == "__main__":
    asyncio.run(main())
