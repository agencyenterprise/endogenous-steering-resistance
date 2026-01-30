"""Experiment runner for testing multiple boost values per latent."""

import asyncio
import json
import os
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables (including HF_TOKEN)
load_dotenv()

import numpy as np
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from judge import create_judge, get_judge_folder_name, Judge
from experiment_config import ExperimentConfig
from vllm_engine import VLLMSteeringEngine
from sample_features import sample_filtered_features
from concreteness_filtering import ConcretenessGrader
from experiment_dataclasses import FeatureInfo, TrialResult, FeatureResult, ExperimentResult
from experiment_1_esr import generate_response, get_score, get_feature_threshold


def compute_boost_levels(threshold_cache: dict, n_levels: int = 10) -> tuple[np.ndarray, float, float]:
    """
    Compute boost levels based on threshold cache statistics.

    Args:
        threshold_cache: Dictionary mapping feature indices to threshold values
                        (supports both old format: just numbers, and new format: dicts with "threshold" key)
        n_levels: Number of boost levels to generate (default: 10)

    Returns:
        Tuple of (boost_levels array, mean, std) where boost_levels ranges
        from (mean - 3*std) to (mean + 3*std)
    """
    # Handle both old format (just a number) and new format (dict with threshold key)
    values = np.array([
        v["threshold"] if isinstance(v, dict) else v
        for v in threshold_cache.values()
        if (v["threshold"] if isinstance(v, dict) else v) is not None
    ])
    mean = np.mean(values)
    std = np.std(values)

    # Generate n_levels evenly spaced values from mean-3*std to mean+3*std
    boost_levels = np.linspace(mean - 3*std, mean + 3*std, n_levels)

    return boost_levels, mean, std


async def run_one_feature(
    engine: VLLMSteeringEngine,
    judge: Judge,
    experiment_config: ExperimentConfig,
    feature: FeatureInfo,
    boost_levels: np.ndarray,
    pbar: Optional[tqdm] = None,
) -> FeatureResult:
    """
    Run the experiment for a single feature across multiple boost levels.

    Uses batched generation and scoring:
    1. Generate all responses in parallel (GPU work)
    2. Score all responses in parallel (Claude API work)
    This keeps GPU and Claude API busy independently.
    """
    prompts = experiment_config.get_prompts()
    try:
        # Find baseline threshold
        if pbar:
            pbar.set_description(f"Feature {feature.index_in_sae}: Finding threshold")
        baseline_threshold = await get_feature_threshold(
            engine, judge, feature, prompts, experiment_config
        )
        if pbar:
            pbar.write(f"✓ Feature {feature.index_in_sae}: baseline threshold = {baseline_threshold:.2f}")

        # Sample prompts without replacement for each boost level
        sampled_prompts = random.sample(
            prompts, min(len(prompts), experiment_config.n_trials_per_feature)
        )

        # Process each boost level - store all trials in flat list
        all_trials = []
        for i, boost_value in enumerate(boost_levels):
            # Convert numpy float to Python float for serialization
            boost_value = float(boost_value)

            # BATCH 1: Generate all responses in parallel (GPU work)
            if pbar:
                pbar.set_description(f"Feature {feature.index_in_sae}: Generating responses for boost {i+1}/{len(boost_levels)} (value={boost_value:.2f})")

            generation_results = await asyncio.gather(
                *[
                    generate_response(engine, experiment_config, prompt, feature, boost_value)
                    for prompt in sampled_prompts
                ]
            )

            # BATCH 2: Score all responses in parallel (Claude API work)
            if pbar:
                pbar.set_description(f"Feature {feature.index_in_sae}: Scoring {len(generation_results)} responses for boost {i+1}/{len(boost_levels)}")

            scoring_results = await asyncio.gather(
                *[
                    judge.grade_response(response, prompt, feature.label)
                    for prompt, response, seed in generation_results
                ]
            )

            # Combine results into trials - store boost_value in threshold field
            for (prompt, response, seed), score in zip(generation_results, scoring_results):
                all_trials.append(
                    TrialResult(
                        prompt=prompt,
                        feature_index_in_sae=feature.index_in_sae,
                        feature_label=feature.label,
                        threshold=boost_value,  # Store boost value here
                        seed=seed,
                        response=response,
                        score=score,
                        error=None if "error" not in score else score["error"],
                    )
                )

        return FeatureResult(
            feature_index_in_sae=feature.index_in_sae,
            feature_label=feature.label,
            threshold=baseline_threshold,  # Store baseline threshold here
            trials=all_trials,
        )
    except Exception as e:
        if pbar:
            pbar.write(f"❌ Feature {feature.index_in_sae}: {str(e)}")
        error_msg = f"{type(e).__name__}: {str(e)}"
        return FeatureResult(
            feature_index_in_sae=feature.index_in_sae,
            feature_label=feature.label,
            threshold=None,
            trials=[],
            error=error_msg,
        )


async def run_experiment(
    experiment_config: ExperimentConfig,
    n_boost_levels: int = 10,
    timeout_hours: float = 100,
    base_model_for_sae: Optional[str] = None,
    precomputed_features: Optional[List[tuple]] = None,
):
    """
    Run a multi-boost experiment on a model.

    For each feature:
        First find a baseline threshold for the feature that reduces the model's
        output to an average score of 50/100 across all prompts.
        Then, for each of n_boost_levels boost values (ranging from mean-3*std to mean+3*std
        based on the threshold cache statistics):
            Randomly sample prompts from the dataset, and for each prompt:
                With the feature boosted to the current boost level, generate a response.
                Score the response based on the original prompt (the scoring also takes
                the feature label into account).
    Stores everything nicely in a JSON file and flushes to disk at the end of every
    completed feature.

    Args:
        experiment_config: Configuration for the experiment
        n_boost_levels: Number of boost levels to test per feature (default: 10)
        timeout_hours: If provided, the experiment will be cancelled after this many hours
        base_model_for_sae: If model_name is a local path, specify which HuggingFace model's
                           SAE configuration to use
        precomputed_features: Optional list of (FeatureInfo, threshold, prompts) tuples.
                             If provided, skips feature sampling. Note: thresholds are ignored
                             for multi-boost (we use boost_levels instead), but prompts are used.
    """
    # Initialize engine and judge
    print("Initializing vLLM engine...")
    engine = VLLMSteeringEngine(
        experiment_config.model_name,
        base_model_for_sae=base_model_for_sae
    )
    await engine.initialize()
    print("Engine initialized")

    judge = create_judge(experiment_config.judge_model_name)

    # Load threshold cache to compute boost levels
    print("Loading threshold cache to compute boost levels...")
    cache_file = experiment_config.get_threshold_cache_file()
    if not os.path.exists(cache_file):
        raise ValueError(f"Threshold cache file not found: {cache_file}. Please run the standard experiment first.")

    with open(cache_file, "r") as f:
        threshold_cache = json.load(f)

    boost_levels, mean, std = compute_boost_levels(threshold_cache, n_boost_levels)
    # Clamp boost levels to be non-negative (negative boosts de-activate features, which is unusual)
    boost_levels = np.maximum(boost_levels, 0.0)
    print(f"Computed {n_boost_levels} boost levels based on threshold cache:")
    print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"  Range: [{boost_levels[0]:.2f}, {boost_levels[-1]:.2f}]")
    print(f"  Boost levels: {[f'{b:.2f}' for b in boost_levels]}")

    # Use precomputed features or sample new ones
    if precomputed_features is not None:
        print(f"\n✓ Using {len(precomputed_features)} precomputed features")
        # Extract just the FeatureInfo objects (ignore thresholds since we use boost_levels)
        features = [feature for feature, _, _ in precomputed_features]
    else:
        # Load feature labels
        print("Loading feature labels...")
        feature_labels = experiment_config.get_labels()
        num_sae_features = len(feature_labels)
        print(f"Loaded {num_sae_features} feature labels")

        # Load prompts for relevance filtering
        prompts = experiment_config.get_prompts()

        # Sample features with concreteness and relevance filtering
        print(f"\nSampling {experiment_config.n_features} features...")
        grader = ConcretenessGrader()
        feature_indices = await sample_filtered_features(
            engine=engine,
            prompts=prompts,
            feature_labels=feature_labels,
            n_features=experiment_config.n_features,
            concreteness_threshold=experiment_config.min_feature_concreteness,
            num_sae_features=num_sae_features,
            candidate_multiplier=3,
            grader=grader,
            labels_file=experiment_config.labels_file,
        )

        if len(feature_indices) < experiment_config.n_features:
            print(f"⚠️  Warning: Only found {len(feature_indices)}/{experiment_config.n_features} features")

        # Create FeatureInfo objects
        features = [
            FeatureInfo(
                index_in_sae=idx,
                label=feature_labels.get(idx, f"feature_{idx}"),
            )
            for idx in feature_indices
        ]

    # Set up semaphore for concurrent feature processing
    semaphore = asyncio.Semaphore(experiment_config.n_simultaneous_features)

    # Create progress bar
    pbar = tqdm(
        total=len(features),
        desc="Processing features",
        unit="feature",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        smoothing=0  # No smoothing: due to parallelism + vLLM batching, first task in batch is slow, rest complete rapidly
    )

    async def run_one_feature_with_semaphore(feature):
        async with semaphore:
            return await run_one_feature(engine, judge, experiment_config, feature, boost_levels, pbar)

    experiment_result = ExperimentResult(
        experiment_config=experiment_config.to_dict(),
        results_by_feature=[],
        n_boost_levels=n_boost_levels,
        boost_range_std=3.0,
    )

    short_model_name = experiment_config.model_name.split("/")[-1]
    start_time = time.time()
    completed_count = 0

    async def process_results():
        nonlocal completed_count
        # Create the final filename once at the start
        judge_folder = get_judge_folder_name(experiment_config.judge_model_name)
        results_base_dir = f"experiment_results/{judge_folder}_judge"
        final_filename = f"{results_base_dir}/experiment_multi_boost_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        temp_filename = final_filename + ".tmp"

        Path(final_filename).parent.mkdir(parents=True, exist_ok=True)

        for feature_result in asyncio.as_completed(tasks):
            try:
                result = await feature_result
                experiment_result.results_by_feature.append(result)
                completed_count += 1

                # Calculate timing info
                elapsed = time.time() - start_time
                avg_time_per_feature = elapsed / completed_count
                remaining_features = len(features) - completed_count
                est_remaining = avg_time_per_feature * remaining_features

                # Update progress bar
                pbar.update(1)

                # Log completion with timing
                if result.error:
                    pbar.write(f"❌ [{completed_count}/{len(features)}] Feature {result.feature_index_in_sae}: {result.feature_label[:40]}... - ERROR: {result.error[:50]}")
                else:
                    n_trials = len(result.trials)
                    n_boost_levels_tested = len(set(t.threshold for t in result.trials))
                    baseline_str = f"{result.threshold:.2f}" if result.threshold is not None else "N/A"
                    pbar.write(f"✅ [{completed_count}/{len(features)}] Feature {result.feature_index_in_sae}: {result.feature_label[:40]}... ({n_trials} trials across {n_boost_levels_tested} boost levels, baseline={baseline_str})")

                # Write to temporary file first - include metadata
                result_dict = asdict(experiment_result)
                result_dict["boost_levels"] = [float(b) for b in boost_levels]
                result_dict["threshold_cache_stats"] = {
                    "mean": float(mean),
                    "std": float(std),
                }
                with open(temp_filename, "w") as f:
                    json.dump(result_dict, f, indent=4)

                # Move temp file to final location
                if os.path.exists(final_filename):
                    os.remove(final_filename)
                os.rename(temp_filename, final_filename)

                # Log save every 5 features
                if completed_count % 5 == 0:
                    pbar.write(f"💾 Saved checkpoint: {completed_count}/{len(features)} features completed")

            except Exception as e:
                pbar.write(f"❌ Error processing feature result: {str(e)}")
                pbar.write(f"   Traceback: {traceback.format_exc()}")
                # Clean up temporary file if it exists
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                continue

        pbar.close()
        total_time = time.time() - start_time
        pbar.write(f"\n{'='*80}")
        pbar.write(f"✅ Experiment completed!")
        pbar.write(f"   Total features: {len(features)}")
        pbar.write(f"   Successful: {sum(1 for r in experiment_result.results_by_feature if not r.error)}")
        pbar.write(f"   Failed: {sum(1 for r in experiment_result.results_by_feature if r.error)}")
        pbar.write(f"   Boost levels per feature: {n_boost_levels}")
        pbar.write(f"   Total trials: {sum(len(r.trials) for r in experiment_result.results_by_feature if not r.error)}")
        pbar.write(f"   Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        pbar.write(f"   Avg time per feature: {total_time/len(features):.1f} seconds")
        pbar.write(f"   Final results: {final_filename}")
        pbar.write(f"{'='*80}\n")

    tasks = [run_one_feature_with_semaphore(feature) for feature in features]

    print(f"\n{'='*80}")
    print(f"Starting multi-boost experiment with {len(features)} features")
    print(f"  Model: {experiment_config.model_name}")
    print(f"  Boost levels per feature: {n_boost_levels}")
    print(f"  Trials per boost level: {experiment_config.n_trials_per_feature}")
    print(f"  Concurrent features: {experiment_config.n_simultaneous_features}")
    print(f"  Total expected trials: {len(features) * n_boost_levels * experiment_config.n_trials_per_feature}")
    if precomputed_features is not None:
        print(f"  Using precomputed features from results file")
    print(f"{'='*80}\n")

    try:
        await asyncio.wait_for(process_results(), timeout=timeout_hours * 3600)
    except asyncio.TimeoutError:
        print(f"Experiment timed out after {timeout_hours} hours")
        # Save final results before exiting
        if experiment_result.results_by_feature:
            judge_folder = get_judge_folder_name(experiment_config.judge_model_name)
            results_base_dir = f"experiment_results/{judge_folder}_judge"
            new_filename = f"{results_base_dir}/experiment_multi_boost_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}_TIMEOUT.json"
            result_dict = asdict(experiment_result)
            result_dict["boost_levels"] = [float(b) for b in boost_levels]
            result_dict["threshold_cache_stats"] = {
                "mean": float(mean),
                "std": float(std),
            }
            with open(new_filename, "w") as f:
                json.dump(result_dict, f, indent=4)
            print(f"Saved partial results to {new_filename}")


if __name__ == "__main__":
    import argparse
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
        "8b-finetuned": ExperimentConfig(
            prompts_file="prompts.txt",
            try_again_prompts=["That was pretty weird. Can you try again?"],
            model_name="experiment_4_finetuning/outputs-lora-8b-self-correction/run-1-merged",
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
        "gemma-2-2b": ExperimentConfig(
            prompts_file="prompts.txt",
            try_again_prompts=["That was pretty weird. Can you try again?"],
            model_name="google/gemma-2-2b-it-res-16k-layer-16",  # IT model, Layer 16/25 = 64.0% depth (using PT SAE)
            labels_file=None,  # TODO: Add Gemma labels file if available
            judge_model_name="claude-sonnet-4-5-20250929",
            target_score_normalized=0.5,
            threshold_n_trials=20,
            threshold_lower_bound=0.0,
            threshold_upper_bound=200.0,  # Increased from 30.0 - model needs higher boosts
            threshold_prior_mean=50.0,    # Increased from 5.0
            threshold_prior_std=50.0,     # Increased from 5.0
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=400,
            n_simultaneous_features=20,
            min_feature_concreteness=65.0,
        ),
        "gemma-2-9b": ExperimentConfig(
            prompts_file="prompts.txt",
            try_again_prompts=["That was pretty weird. Can you try again?"],
            model_name="google/gemma-2-9b-it-res-16k-layer-20",  # Instruct-tuned version at Layer 20/42 = 47.6% depth
            labels_file="data/labels/gemma-2-9b-res-16k-layer-26.csv",  # Note: using PT labels, may need IT labels
            judge_model_name="claude-sonnet-4-5-20250929",
            target_score_normalized=0.5,
            threshold_n_trials=20,
            threshold_lower_bound=0.0,
            threshold_upper_bound=560.0,  # Updated from 50.0 based on boost range testing
            threshold_prior_mean=280.0,   # Updated from 10.0 based on boost range testing
            threshold_prior_std=280.0,    # Updated from 10.0 based on boost range testing
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=300,
            n_simultaneous_features=20,
            min_feature_concreteness=65.0,
        ),
        "gemma-2-27b": ExperimentConfig(
            prompts_file="prompts.txt",
            try_again_prompts=["That was pretty weird. Can you try again?"],
            model_name="google/gemma-2-27b-it-res-131k-layer-22",  # IT model, Layer 22/45 = 48.9% depth (using PT SAE)
            labels_file=None,  # TODO: Add Gemma labels file if available
            judge_model_name="claude-sonnet-4-5-20250929",
            target_score_normalized=0.5,
            threshold_n_trials=20,
            threshold_lower_bound=0.0,
            threshold_upper_bound=80.0,
            threshold_prior_mean=15.0,
            threshold_prior_std=15.0,
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=400,
            n_simultaneous_features=20,
            min_feature_concreteness=65.0,
        ),
    }

    # Map config names to their base model for SAE (if needed)
    base_model_for_sae_map = {
        "8b-finetuned": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # For custom 8B fine-tunes
    }

    parser = argparse.ArgumentParser(description="Run multi-boost ESR experiment")
    parser.add_argument("config", choices=list(configs.keys()), help="Base config to use")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (for fine-tuned models). Uses base config's SAE settings.")
    parser.add_argument("--from-results", type=str, default=None,
                        help="Load features from an existing results file. "
                             "Useful for comparing models on exact same features.")
    parser.add_argument("--n-boost-levels", type=int, default=10,
                        help="Number of boost levels to test per feature (default: 10)")
    parser.add_argument("--judge", type=str, default=None,
                        help="Override judge model (e.g., 'haiku', 'sonnet')")
    args = parser.parse_args()

    config_name = args.config
    experiment_config = configs[config_name]

    # Override model path if provided
    if args.model_path:
        print(f"Using custom model path: {args.model_path}")
        experiment_config.model_name = args.model_path
        # For fine-tuned models, we need to specify the base model for SAE loading
        base_model = base_model_for_sae_map.get(config_name, None)
        if base_model is None:
            raise ValueError(f"No base_model_for_sae mapping for config '{config_name}'. "
                           f"Add it to base_model_for_sae_map to use --model-path.")
    else:
        base_model = base_model_for_sae_map.get(config_name, None)

    # Load precomputed features from results file if provided
    precomputed_features = None
    if args.from_results:
        print(f"\nLoading features from {args.from_results}")
        with open(args.from_results, "r") as f:
            results_data = json.load(f)

        # Extract features from results (ignore thresholds since we use boost_levels)
        precomputed_features = []
        for result in results_data["results_by_feature"]:
            if not result.get("error") and result.get("threshold") is not None:
                feature = FeatureInfo(
                    index_in_sae=result["feature_index_in_sae"],
                    label=result["feature_label"],
                )
                # Store None for threshold and prompts since multi-boost doesn't use them
                precomputed_features.append((feature, None, None))

        print(f"✓ Loaded {len(precomputed_features)} features from results file")

        # Override experiment config's n_features to match loaded features
        experiment_config.n_features = len(precomputed_features)

    # Override judge model if provided
    if args.judge:
        from judge import resolve_model_id
        experiment_config.judge_model_name = resolve_model_id(args.judge)
        print(f"Using judge model: {experiment_config.judge_model_name}")

    experiment_config.to_dict()

    asyncio.run(run_experiment(
        experiment_config,
        n_boost_levels=args.n_boost_levels,
        base_model_for_sae=base_model,
        precomputed_features=precomputed_features,
    ))