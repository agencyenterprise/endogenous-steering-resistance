"""Random latent ablation control experiment.

Research question: Does ablating the 27 off-topic detector latents reduce ESR 
because they specifically enable self-monitoring, or because ablating any 27 
similarly-active latents disrupts the network generally?

This experiment:
1. Computes activation statistics for all latents on baseline (unsteered) generations
2. Samples K sets of 27 random latents with similar activation profiles to detectors
3. Runs the ablation experiment with random latents (replaying exact trials)
4. Compares MSI across conditions
"""

import asyncio
import gc
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN for gated models like Llama-3.3-70B)
load_dotenv()

# Local imports (copied from AGI-1516-esr-with-vllm)
from experiment_config import ExperimentConfig
from vllm_engine import VLLMSteeringEngine
from experiment_dataclasses import FeatureInfo
from run_ablation_experiment import run_experiment, PrecomputedTrial


@dataclass
class LatentActivationStats:
    """Activation statistics for a single latent."""
    latent_idx: int
    activation_frequency: float  # % of tokens with non-zero activation
    mean_magnitude_when_active: float  # Mean activation when active
    

@dataclass
class RandomLatentSet:
    """A set of random latents matched to detector activation profiles."""
    set_id: int
    latent_indices: List[int]
    matching_stats: Dict[int, Dict]  # latent_idx -> stats


async def get_sae_activations_for_response(
    engine: VLLMSteeringEngine,
    prompt: str,
    response: str,
) -> torch.Tensor:
    """
    Get SAE latent activations for a prompt+response pair.
    
    Returns max-pooled activations across all token positions.
    Shape: [num_latents]
    """
    from vllm.inputs import TokenInputs
    from vllm import SamplingParams
    import uuid
    
    # Create a conversation that includes the response
    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    
    # Apply chat template and get token IDs
    prompt_token_ids = engine.tokenizer.apply_chat_template(convo)
    
    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=convo)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1,  # Only generate 1 token to get activations
        repetition_penalty=1.0,
    )
    
    request_id = str(uuid.uuid4())
    results_generator = engine.engine.generate(
        prompt=token_inputs,
        sampling_params=sampling_params,
        request_id=request_id,
        interventions=None,
        is_feature_decode=True,  # Enable feature extraction
    )
    
    # Get the output with features
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    # Extract feature activations: shape [num_tokens, num_latents]
    features = final_output.feature_tensor.cpu()
    
    # Max-pool across all token positions
    max_activations = features.max(dim=0)[0]  # [num_latents]
    
    return max_activations


async def compute_activation_statistics(
    engine: VLLMSteeringEngine,
    baseline_results_file: str,
    cache_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute activation statistics for all latents from baseline generations.
    
    Args:
        engine: Initialized vLLM engine
        baseline_results_file: Path to no_steering_baseline.json
        cache_file: Optional path to cache results
        
    Returns:
        (activation_frequencies, mean_magnitudes) - arrays of shape [num_latents]
    """
    # Check cache
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached activation stats from {cache_file}")
        with open(cache_file, "r") as f:
            cached = json.load(f)
        return np.array(cached["activation_frequencies"]), np.array(cached["mean_magnitudes"])
    
    # Load baseline results
    print(f"Loading baseline results from {baseline_results_file}")
    with open(baseline_results_file, "r") as f:
        baseline_data = json.load(f)
    
    # Extract all prompt-response pairs
    pairs = []
    for feature_result in baseline_data["results_by_feature"]:
        for trial in feature_result.get("trials", []):
            if "prompt" in trial and "response" in trial:
                pairs.append((trial["prompt"], trial["response"]))
    
    print(f"Found {len(pairs)} prompt-response pairs in baseline results")
    
    # Get activations for all pairs
    all_activations = []
    for prompt, response in tqdm(pairs, desc="Computing activations"):
        try:
            activations = await get_sae_activations_for_response(engine, prompt, response)
            all_activations.append(activations)
        except Exception as e:
            print(f"Error getting activations: {e}")
            continue
    
    if not all_activations:
        raise ValueError("Failed to get any activations")
    
    # Stack into tensor [num_samples, num_latents]
    activations_tensor = torch.stack(all_activations)
    print(f"Activations tensor shape: {activations_tensor.shape}")
    
    # Compute statistics
    num_latents = activations_tensor.shape[1]
    zero_threshold = 1e-8
    
    # Activation frequency: % of samples where latent is active
    is_active = (activations_tensor > zero_threshold).float()
    activation_frequencies = is_active.mean(dim=0).numpy()  # [num_latents]
    
    # Mean magnitude when active
    # For each latent, compute mean of non-zero activations
    mean_magnitudes = np.zeros(num_latents)
    for i in range(num_latents):
        active_mask = activations_tensor[:, i] > zero_threshold
        if active_mask.any():
            mean_magnitudes[i] = activations_tensor[active_mask, i].mean().item()
    
    # Cache results
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "activation_frequencies": activation_frequencies.tolist(),
                "mean_magnitudes": mean_magnitudes.tolist(),
                "num_samples": len(pairs),
                "baseline_file": str(baseline_results_file),
            }, f, indent=2)
        print(f"Cached activation stats to {cache_file}")
    
    return activation_frequencies, mean_magnitudes


def sample_matched_random_latents(
    detector_indices: List[int],
    detector_stats: Dict[str, Dict],
    all_activation_freqs: np.ndarray,
    all_mean_magnitudes: np.ndarray,
    num_sets: int = 5,
    seed: int = 42,
) -> List[RandomLatentSet]:
    """
    Sample K sets of truly random latents, only filtering out dead latents.

    Strategy: For each set, sample N latents (same count as detectors) from
    all non-dead latents. No activation profile matching - just random selection.

    Args:
        detector_indices: List of off-topic detector latent indices
        detector_stats: Statistics for each detector from the detector JSON (unused)
        all_activation_freqs: Activation frequencies for all latents [num_latents]
        all_mean_magnitudes: Mean magnitudes for all latents [num_latents]
        num_sets: Number of random sets to sample
        seed: Random seed for reproducibility

    Returns:
        List of RandomLatentSet objects
    """
    random.seed(seed)
    np.random.seed(seed)

    num_latents = len(all_activation_freqs)
    num_to_sample = len(detector_indices)
    detector_set = set(detector_indices)

    # Get detector stats for comparison (informational only)
    detector_freqs = np.array([all_activation_freqs[idx] for idx in detector_indices])
    detector_mags = np.array([all_mean_magnitudes[idx] for idx in detector_indices])

    print(f"\nDetector stats (for reference):")
    print(f"  Count: {num_to_sample}")
    print(f"  Frequency: mean={detector_freqs.mean():.4f}, std={detector_freqs.std():.4f}")
    print(f"  Magnitude: mean={detector_mags.mean():.4f}, std={detector_mags.std():.4f}")

    # Get candidate latents: exclude detectors and dead latents (zero activation)
    min_activation = 0.001  # Filter out completely dead latents
    valid_candidates = [
        i for i in range(num_latents)
        if i not in detector_set and all_activation_freqs[i] > min_activation
    ]

    print(f"\nCandidate pool: {len(valid_candidates)} non-dead latents (excluding {num_to_sample} detectors)")

    # Sample K sets of truly random latents
    random_sets = []
    used_latents = set()

    for set_id in range(num_sets):
        # Get available latents (prefer unused, but allow reuse if needed)
        available = [idx for idx in valid_candidates if idx not in used_latents]

        if len(available) < num_to_sample:
            print(f"Warning: Reusing latents for set {set_id}")
            available = valid_candidates

        # Truly random sample
        selected_latents = random.sample(available, num_to_sample)
        used_latents.update(selected_latents)

        # Compute stats for selected latents (informational)
        selected_freqs = np.array([all_activation_freqs[i] for i in selected_latents])
        selected_mags = np.array([all_mean_magnitudes[i] for i in selected_latents])

        matching_stats = {
            "detector_freq_mean": float(detector_freqs.mean()),
            "selected_freq_mean": float(selected_freqs.mean()),
            "detector_mag_mean": float(detector_mags.mean()),
            "selected_mag_mean": float(selected_mags.mean()),
        }

        random_sets.append(RandomLatentSet(
            set_id=set_id,
            latent_indices=selected_latents,
            matching_stats=matching_stats,
        ))

        print(f"\nRandom set {set_id}:")
        print(f"  Latents: {selected_latents[:5]}... (showing first 5)")
        print(f"  Frequency: mean={selected_freqs.mean():.4f}")
        print(f"  Magnitude: mean={selected_mags.mean():.4f}")
    
    return random_sets


def cleanup_engine(engine: Optional[VLLMSteeringEngine]) -> None:
    """Clean up vLLM engine and free GPU memory."""
    if engine is None:
        return
    
    print("\nCleaning up vLLM engine to free GPU memory...")
    
    # Delete the engine
    try:
        if hasattr(engine, 'engine') and engine.engine is not None:
            del engine.engine
        del engine
    except Exception as e:
        print(f"Warning during engine cleanup: {e}")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Give the system a moment to clean up
    import time
    time.sleep(2)
    
    print("GPU memory freed")


async def main():
    """Run the random latent ablation control experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Random latent ablation control experiment")
    parser.add_argument("--config", type=str, default="70b", choices=["8b", "70b"],
                       help="Model config to use")
    parser.add_argument("--baseline-file", type=str, required=True,
                       help="Path to no_steering_baseline.json for activation stats")
    parser.add_argument("--detector-file", type=str, required=True,
                       help="Path to off_topic_detectors JSON file")
    parser.add_argument("--from-results", type=str, required=True,
                       help="Path to original experiment results to replay trials from")
    parser.add_argument("--num-random-sets", type=int, default=3,
                       help="Number of random latent sets to test (default: 3)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only compute activation stats, don't run experiment")
    parser.add_argument("--max-features", type=int, default=None,
                       help="Limit number of features to test (for quick test runs)")
    args = parser.parse_args()
    
    # Define configs
    configs = {
        "8b": ExperimentConfig(
            prompts_file="prompts.txt",
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
            prompts_file="prompts.txt",
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
    
    config = configs[args.config]
    
    print("=" * 80)
    print("Random Latent Ablation Control Experiment")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Baseline file: {args.baseline_file}")
    print(f"Detector file: {args.detector_file}")
    print(f"Results to replay: {args.from_results}")
    print(f"Number of random sets: {args.num_random_sets}")
    print("=" * 80)
    
    # Load detector info
    print(f"\nLoading off-topic detectors from {args.detector_file}")
    with open(args.detector_file, "r") as f:
        detector_data = json.load(f)
    detector_indices = detector_data["off_topic_detectors"]
    detector_stats = detector_data["detector_stats"]
    print(f"Loaded {len(detector_indices)} off-topic detector latents")
    
    # Step 1: Compute activation statistics (or load from cache)
    print("\n" + "=" * 80)
    print("Step 1: Computing activation statistics on baseline generations")
    print("=" * 80)
    
    cache_file = Path(__file__).parent / "data" / f"activation_stats_{args.config}.json"
    
    # Check if we need to initialize engine for activation stats
    engine = None
    if not cache_file.exists():
        print("\nInitializing vLLM engine for activation computation...")
        engine = VLLMSteeringEngine(config.model_name)
        await engine.initialize()
        print("Engine initialized")
        
        activation_freqs, mean_magnitudes = await compute_activation_statistics(
            engine, args.baseline_file, str(cache_file)
        )
        
        # Clean up engine to free GPU memory before Step 3
        cleanup_engine(engine)
        engine = None
    else:
        # Load from cache without initializing engine
        print(f"Loading cached activation stats from {cache_file}")
        with open(cache_file, "r") as f:
            cached = json.load(f)
        activation_freqs = np.array(cached["activation_frequencies"])
        mean_magnitudes = np.array(cached["mean_magnitudes"])
    
    print(f"Have stats for {len(activation_freqs)} latents")
    
    if args.stats_only:
        print("\n--stats-only flag set, exiting after computing stats")
        return
    
    # Step 2: Sample matched random latents
    print("\n" + "=" * 80)
    print("Step 2: Sampling matched random latents")
    print("=" * 80)
    
    random_sets = sample_matched_random_latents(
        detector_indices=detector_indices,
        detector_stats=detector_stats,
        all_activation_freqs=activation_freqs,
        all_mean_magnitudes=mean_magnitudes,
        num_sets=args.num_random_sets,
        seed=args.random_seed,
    )
    
    # Save random sets for reproducibility
    random_sets_file = Path(__file__).parent / "data" / f"random_latent_sets_{args.config}.json"
    random_sets_file.parent.mkdir(parents=True, exist_ok=True)
    with open(random_sets_file, "w") as f:
        json.dump([asdict(s) for s in random_sets], f, indent=2)
    print(f"\nSaved random latent sets to {random_sets_file}")
    
    # Step 3: Run ablation experiment for each random set
    print("\n" + "=" * 80)
    print("Step 3: Running ablation experiments with random latent sets")
    print("=" * 80)
    
    # Load the original experiment results for replay
    print(f"\nLoading original results from {args.from_results}")
    with open(args.from_results, "r") as f:
        original_results = json.load(f)
    
    # Extract features, thresholds, and trials
    precomputed_features = []
    for result in original_results["results_by_feature"]:
        if not result.get("error") and result.get("threshold") is not None:
            feature = FeatureInfo(
                index_in_sae=result["feature_index_in_sae"],
                label=result["feature_label"],
            )
            threshold = result["threshold"]
            
            trials = []
            for trial in result.get("trials", []):
                if "prompt" in trial and "seed" in trial:
                    trials.append(PrecomputedTrial(
                        prompt=trial["prompt"],
                        seed=trial["seed"]
                    ))
            
            precomputed_features.append((feature, threshold, trials))
    
    print(f"Loaded {len(precomputed_features)} features with {sum(len(t) for _, _, t in precomputed_features)} total trials")

    # Limit features if --max-features is set
    if args.max_features and len(precomputed_features) > args.max_features:
        precomputed_features = precomputed_features[:args.max_features]
        print(f"Limited to {len(precomputed_features)} features (--max-features={args.max_features})")

    # Run experiment for each random set
    for random_set in random_sets:
        print(f"\n{'='*80}")
        print(f"Running ablation with random set {random_set.set_id}")
        print(f"Latents: {random_set.latent_indices}")
        print(f"{'='*80}")
        
        # Update config with output suffix
        config.source_results_file = args.from_results
        
        await run_experiment(
            experiment_config=config,
            ablate_latents=random_set.latent_indices,
            precomputed_features=precomputed_features,
            output_suffix=f"_random_ablation_set{random_set.set_id}",
            output_dir="experiment_results",
        )
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Compute MSI for each condition using the analysis scripts")
    print("2. Compare random ablation MSI vs detector ablation MSI vs baseline")
    print("3. Run statistical tests (bootstrap or SEM)")


if __name__ == "__main__":
    asyncio.run(main())
