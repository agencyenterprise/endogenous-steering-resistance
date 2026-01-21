"""Main experiment runner for endogenous steering resistance with vLLM."""

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Load environment variables (including HF_TOKEN)
load_dotenv()

from claude_judge import ClaudeJudge
from experiment_config import ExperimentConfig
from threshold_finder import find_threshold
from vllm_engine import VLLMSteeringEngine
from sample_features import sample_filtered_features
from concreteness_filtering import ConcretenessGrader
from experiment_dataclasses import FeatureInfo, TrialResult, FeatureResult, ExperimentResult


async def generate_response(
    engine: VLLMSteeringEngine,
    experiment_config: ExperimentConfig,
    prompt: str,
    feature: FeatureInfo,
    threshold: float,
) -> tuple[str, str, int]:
    """
    Generate a single response with feature intervention.

    Returns: (prompt, response, seed)
    """
    intervention = None
    if not experiment_config.disable_steering:
        intervention = [{"feature_id": feature.index_in_sae, "value": threshold}]
    convo = [{"role": "user", "content": prompt}]
    seed = random.randint(
        experiment_config.seed_start,
        experiment_config.seed_start + experiment_config.n_possible_seeds,
    )
    response = await engine.generate_with_conversation(
        conversation=convo,
        feature_interventions=intervention,
        max_tokens=experiment_config.max_completion_tokens,
        seed=seed,
    )
    return prompt, response, seed


async def get_score(
    engine: VLLMSteeringEngine,
    judge: ClaudeJudge,
    prompts: List[str],
    feature: FeatureInfo,
    boost: float,
    experiment_config: ExperimentConfig,
    max_retries: int = 3,
) -> float:
    """
    Get a score (between 0 and 1) for a given feature and boost value.
    Uses a random prompt from `prompts`. Retries with different prompts on failure.
    """
    score = None
    retries = 0
    last_error = None

    while score is None:
        prompt = random.choice(prompts)
        intervention = None
        if not experiment_config.disable_steering:
            intervention = [{"feature_id": feature.index_in_sae, "value": boost}]
        convo = [{"role": "user", "content": prompt}]
        seed = random.randint(
            experiment_config.seed_start,
            experiment_config.seed_start + experiment_config.n_possible_seeds,
        )

        response = await engine.generate_with_conversation(
            conversation=convo,
            feature_interventions=intervention,
            max_tokens=experiment_config.max_completion_tokens // 2,
            seed=seed,
        )
        score_obj = await judge.grade_response(response, prompt, feature.label)

        if "error" in score_obj:
            retries += 1
            last_error = score_obj
            if retries >= max_retries:
                # After max retries, raise with debug info
                debug_info = (
                    f"\n{'='*80}\n"
                    f"JUDGE ERROR DURING THRESHOLD FINDING (after {max_retries} retries)\n"
                    f"{'='*80}\n"
                    f"Error: {score_obj['error']}\n"
                    f"\nFeature: {feature.label}\n"
                    f"Boost: {boost}\n"
                    f"\nPrompt:\n{prompt}\n"
                    f"\nGenerated Response:\n{response[:500]}{'...' if len(response) > 500 else ''}\n"
                    f"\nRaw Judge Response:\n{score_obj.get('raw_response', 'None')}\n"
                    f"{'='*80}\n"
                )
                raise Exception(debug_info)
            # Retry with a different prompt
            continue

        score = (
            score_obj["attempts"][0]["score"]
            if "attempts" in score_obj and len(score_obj["attempts"]) > 0
            else None
        )
    return score / 100.0


async def get_feature_threshold(
    engine: VLLMSteeringEngine,
    judge: ClaudeJudge,
    feature: FeatureInfo,
    prompts: List[str],
    experiment_config: ExperimentConfig,
    show_progress: bool = False,
) -> float:
    """
    For a given feature, find an overall threshold that reduces the model's
    output to an average score of 50/100 across the given prompts.
    """
    # Zero-steering baseline: no threshold needed, and we should not touch the cache.
    if experiment_config.disable_steering:
        return 0.0

    # Read existing cache if it exists
    threshold_cache = {}
    cache_file = experiment_config.get_threshold_cache_file()
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            threshold_cache = json.load(f)

    # Return cached value if available (handle both old and new format)
    if str(feature.index_in_sae) in threshold_cache:
        cached_value = threshold_cache[str(feature.index_in_sae)]
        # Handle old format (just a number) vs new format (dict with threshold, config, score)
        if isinstance(cached_value, dict):
            threshold_value = cached_value.get("threshold")
        else:
            threshold_value = cached_value

        # Only use cache if threshold is not None
        if threshold_value is not None:
            if show_progress:
                print(f"Using cached threshold for feature {feature.index_in_sae}")
            return threshold_value

    # Find new threshold if not in cache
    threshold = await find_threshold(
        target_score=experiment_config.target_score_normalized,
        get_score_fn=lambda x: get_score(
            engine, judge, prompts, feature, x, experiment_config
        ),
        prior_mean=experiment_config.threshold_prior_mean,
        prior_std=experiment_config.threshold_prior_std,
        n_trials=experiment_config.threshold_n_trials,
        show_progress=show_progress,
        lower_bound=experiment_config.threshold_lower_bound,
        upper_bound=experiment_config.threshold_upper_bound,
    )

    # Convert numpy float to Python float for serialization
    threshold = float(round(threshold, 2))

    # Verify the achieved score with the found threshold
    achieved_score = await get_score(engine, judge, prompts, feature, threshold, experiment_config)

    # Update cache with new value while preserving existing entries
    # (need to reload in case another coroutine wrote to it during the above)
    threshold_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            threshold_cache = json.load(f)

    # Store threshold with full metadata
    threshold_cache[str(feature.index_in_sae)] = {
        "threshold": float(threshold),
        "achieved_score": float(achieved_score),
        "config": {
            "target_score": experiment_config.target_score_normalized,
            "n_trials": experiment_config.threshold_n_trials,
            "prior_mean": experiment_config.threshold_prior_mean,
            "prior_std": experiment_config.threshold_prior_std,
            "lower_bound": experiment_config.threshold_lower_bound,
            "upper_bound": experiment_config.threshold_upper_bound,
        }
    }

    with open(cache_file, "w") as f:
        json.dump(threshold_cache, f, indent=2)

    return float(threshold)


async def run_one_feature(
    engine: VLLMSteeringEngine,
    judge: ClaudeJudge,
    experiment_config: ExperimentConfig,
    feature: FeatureInfo,
    pbar: Optional[tqdm] = None,
    precomputed_threshold: Optional[float] = None,
    precomputed_prompts: Optional[List[str]] = None,
) -> FeatureResult:
    """
    Run the experiment for a single feature.

    Uses batched generation and scoring:
    1. Generate all responses in parallel (GPU work)
    2. Score all responses in parallel (Claude API work)
    This keeps GPU and Claude API busy independently.

    Args:
        precomputed_threshold: If provided, skip threshold finding and use this value
        precomputed_prompts: If provided, use these exact prompts instead of sampling
    """
    try:

        prompts = experiment_config.get_prompts()

        # Find or use precomputed threshold
        if precomputed_threshold is not None:
            threshold = precomputed_threshold
            if pbar:
                pbar.write(f"✓ Feature {feature.index_in_sae}: using precomputed threshold = {threshold:.2f}")
        else:
            if pbar:
                pbar.set_description(f"Feature {feature.index_in_sae}: Finding threshold")
            threshold = await get_feature_threshold(
                engine, judge, feature, prompts, experiment_config
            )
            if pbar:
                pbar.write(f"✓ Feature {feature.index_in_sae}: threshold = {threshold:.2f}")

        # Use precomputed prompts or sample new ones
        if precomputed_prompts is not None:
            sampled_prompts = precomputed_prompts
        else:
            sampled_prompts = random.sample(
                prompts, min(len(prompts), experiment_config.n_trials_per_feature)
            )

        # BATCH 1: Generate all responses in parallel (GPU work)
        if pbar:
            pbar.set_description(f"Feature {feature.index_in_sae}: Generating {len(sampled_prompts)} responses")

        generation_results = await asyncio.gather(
            *[
                generate_response(engine, experiment_config, prompt, feature, threshold)
                for prompt in sampled_prompts
            ]
        )

        # BATCH 2: Score all responses in parallel (Claude API work)
        if pbar:
            pbar.set_description(f"Feature {feature.index_in_sae}: Scoring {len(generation_results)} responses")

        scoring_results = await asyncio.gather(
            *[
                judge.grade_response(response, prompt, feature.label)
                for prompt, response, seed in generation_results
            ]
        )

        # Combine results into trials
        trials = []
        for (prompt, response, seed), score in zip(generation_results, scoring_results):
            trials.append(
                TrialResult(
                    prompt=prompt,
                    feature_index_in_sae=feature.index_in_sae,
                    feature_label=feature.label,
                    threshold=threshold,
                    seed=seed,
                    response=response,
                    score=score,
                    error=None if "error" not in score else score["error"],
                )
            )

        return FeatureResult(
            feature_index_in_sae=feature.index_in_sae,
            feature_label=feature.label,
            threshold=threshold,
            trials=trials,
        )
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else f"{type(e).__name__}: {traceback.format_exc()}"
        if pbar:
            pbar.write(f"❌ Feature {feature.index_in_sae}: {error_msg[:500]}")
        return FeatureResult(
            feature_index_in_sae=feature.index_in_sae,
            feature_label=feature.label,
            threshold=None,
            trials=[],
            error=error_msg,
        )


async def run_experiment(
    experiment_config: ExperimentConfig,
    timeout_hours: float = 100,
    base_model_for_sae: Optional[str] = None,
    precomputed_features: Optional[List[tuple]] = None,
    n_prompts_limit: Optional[int] = None,
    output_suffix: Optional[str] = None,
):
    """
    Run a whole experiment on a model.

    For each feature:
        First find an overall threshold for the feature that reduces the model's
        output to an average score of 50/100 across all prompts.
        Then, randomly sample prompts from the dataset, and for each prompt:
            With the feature boosted to the threshold, generate a response.
            Then append a try-again prompt to the end of the message history, and
            generate a response again.
            Score both responses based on the original prompt (the scoring also takes
            the feature label into account).
    Stores everything nicely in a JSON file and flushes to disk at the end of every
    completed feature.

    Args:
        experiment_config: Configuration for the experiment
        timeout_hours: If provided, the experiment will be cancelled after this many hours
        base_model_for_sae: If model_name is a local path, specify which HuggingFace model's
                           SAE configuration to use
        precomputed_features: Optional list of (FeatureInfo, threshold, prompts) tuples.
                             If provided, skips feature sampling and threshold finding.
    """
    # Initialize engine and judge
    print("Initializing vLLM engine...")
    engine = VLLMSteeringEngine(
        experiment_config.model_name,
        base_model_for_sae=base_model_for_sae
    )
    await engine.initialize()
    print("Engine initialized")

    judge = ClaudeJudge(model_name=experiment_config.judge_model_name)

    # Use precomputed features or sample new ones
    if precomputed_features is not None:
        print(f"\n✓ Using {len(precomputed_features)} precomputed features with cached thresholds and prompts")
        features_with_data = precomputed_features
    else:
        # Load feature labels
        print("Loading feature labels...")
        feature_labels = experiment_config.get_labels()
        num_sae_features = len(feature_labels)
        print(f"Loaded {num_sae_features} feature labels")

        # Load prompts for relevance filtering
        prompts = experiment_config.get_prompts()
        if n_prompts_limit is not None:
            prompts = prompts[:n_prompts_limit]
            print(f"  Limited to first {n_prompts_limit} prompts")

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

        # Create FeatureInfo objects with no precomputed data
        features_with_data = [
            (FeatureInfo(index_in_sae=idx, label=feature_labels.get(idx, f"feature_{idx}")), None, None)
            for idx in feature_indices
        ]

    # Set up semaphore for concurrent feature processing
    semaphore = asyncio.Semaphore(experiment_config.n_simultaneous_features)

    # Create progress bar
    pbar = tqdm(
        total=len(features_with_data),
        desc="Processing features",
        unit="feature",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        smoothing=0  # No smoothing: due to parallelism + vLLM batching, first task in batch is slow, rest complete rapidly
    )

    async def run_one_feature_with_semaphore(feature_data):
        feature, threshold, prompts = feature_data
        async with semaphore:
            return await run_one_feature(
                engine, judge, experiment_config, feature, pbar,
                precomputed_threshold=threshold,
                precomputed_prompts=prompts,
            )

    experiment_result = ExperimentResult(
        experiment_config=experiment_config.to_dict(), results_by_feature=[]
    )

    short_model_name = experiment_config.model_name.split("/")[-1]
    start_time = time.time()
    completed_count = 0

    async def process_results():
        nonlocal completed_count
        # Create the final filename once at the start
        suffix_part = f"_{output_suffix}" if output_suffix else ""
        final_filename = f"experiment_results/experiment_results_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}{suffix_part}.json"
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
                remaining_features = len(features_with_data) - completed_count
                est_remaining = avg_time_per_feature * remaining_features

                # Update progress bar
                pbar.update(1)

                # Log completion with timing
                if result.error and result.error.strip():
                    pbar.write(f"❌ [{completed_count}/{len(features_with_data)}] Feature {result.feature_index_in_sae}: {result.feature_label[:40]}... - ERROR: {result.error[:50]}")
                else:
                    n_trials = len(result.trials)
                    threshold_str = f"{result.threshold:.2f}" if result.threshold is not None else "N/A"
                    pbar.write(f"✅ [{completed_count}/{len(features_with_data)}] Feature {result.feature_index_in_sae}: {result.feature_label[:40]}... ({n_trials} trials, threshold={threshold_str})")

                # Write to temporary file first
                with open(temp_filename, "w") as f:
                    json.dump(asdict(experiment_result), f, indent=4)

                # Move temp file to final location
                if os.path.exists(final_filename):
                    os.remove(final_filename)
                os.rename(temp_filename, final_filename)

                # Log save every 5 features
                if completed_count % 5 == 0:
                    pbar.write(f"💾 Saved checkpoint: {completed_count}/{len(features_with_data)} features completed")

            except Exception as e:
                pbar.write(f"❌ Error processing feature result: {str(e)}")
                # Clean up temporary file if it exists
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                continue

        pbar.close()
        total_time = time.time() - start_time
        pbar.write(f"\n{'='*80}")
        pbar.write(f"✅ Experiment completed!")
        pbar.write(f"   Total features: {len(features_with_data)}")
        pbar.write(f"   Successful: {sum(1 for r in experiment_result.results_by_feature if not r.error)}")
        pbar.write(f"   Failed: {sum(1 for r in experiment_result.results_by_feature if r.error)}")
        pbar.write(f"   Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        pbar.write(f"   Avg time per feature: {total_time/len(features_with_data):.1f} seconds")
        pbar.write(f"   Final results: {final_filename}")
        pbar.write(f"{'='*80}\n")

    tasks = [run_one_feature_with_semaphore(feature_data) for feature_data in features_with_data]

    print(f"\n{'='*80}")
    print(f"Starting experiment with {len(features_with_data)} features")
    print(f"  Model: {experiment_config.model_name}")
    print(f"  Trials per feature: {experiment_config.n_trials_per_feature}")
    print(f"  Concurrent features: {experiment_config.n_simultaneous_features}")
    print(f"  Total expected trials: {len(features_with_data) * experiment_config.n_trials_per_feature}")
    if precomputed_features is not None:
        print(f"  Using precomputed thresholds and prompts from results file")
    print(f"{'='*80}\n")

    try:
        await asyncio.wait_for(process_results(), timeout=timeout_hours * 3600)
    except asyncio.TimeoutError:
        print(f"Experiment timed out after {timeout_hours} hours")
        # Save final results before exiting
        if experiment_result.results_by_feature:
            suffix_part = f"_{output_suffix}" if output_suffix else ""
            new_filename = f"experiment_results/experiment_results_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}{suffix_part}_TIMEOUT.json"
            with open(new_filename, "w") as f:
                json.dump(asdict(experiment_result), f, indent=4)
            print(f"Saved partial results to {new_filename}")

if __name__ == "__main__":
    import sys
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
            n_features=500,
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
            threshold_n_trials=30,
            threshold_lower_bound=0.0,
            threshold_upper_bound=50.0,
            threshold_prior_mean=10.0,
            threshold_prior_std=8.0,
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=50,
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
            n_features=500,
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
            n_features=500,
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
            n_features=500,
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
            threshold_upper_bound=5000.0,
            threshold_prior_mean=1500.0,
            threshold_prior_std=1500.0,
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=500,
            n_simultaneous_features=20,
            min_feature_concreteness=65.0,
        ),
    }

    # Map config names to their base model for SAE (if needed)
    base_model_for_sae_map = {
        "8b-finetuned": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # For custom 8B fine-tunes
    }

    import argparse
    parser = argparse.ArgumentParser(description="Run ESR experiment")
    parser.add_argument("config", choices=list(configs.keys()), help="Base config to use")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (for fine-tuned models). Uses base config's SAE settings.")
    parser.add_argument("--from-results", type=str, default=None,
                        help="Load features, thresholds, and prompts from an existing results file. "
                             "Useful for comparing models on exact same conditions.")
    parser.add_argument("--no-steering", action="store_true",
                        help="Disable feature interventions entirely (zero-steering baseline). "
                             "Works well with --from-results to keep the same features/prompts.")
    parser.add_argument("--recalibrate-thresholds", action="store_true",
                        help="When used with --from-results, re-find thresholds on the current model "
                             "instead of reusing thresholds from the results file.")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of trials per feature")
    parser.add_argument("--n-features", type=int, default=None,
                        help="Override number of features to sample")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Limit prompts to first N (useful for smaller runs)")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Add suffix to output filename (e.g., 'no_steering_baseline')")
    args = parser.parse_args()

    config_name = args.config
    experiment_config = configs[config_name]
    experiment_config.disable_steering = bool(args.no_steering)

    # Apply CLI overrides
    if args.n_trials is not None:
        experiment_config.n_trials_per_feature = args.n_trials
    if args.n_features is not None:
        experiment_config.n_features = args.n_features

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
        print(f"\nLoading features, thresholds, and prompts from {args.from_results}")
        with open(args.from_results, "r") as f:
            results_data = json.load(f)

        # Extract features, thresholds, and prompts from results
        precomputed_features = []
        for result in results_data["results_by_feature"]:
            if not result.get("error"):
                feature = FeatureInfo(
                    index_in_sae=result["feature_index_in_sae"],
                    label=result["feature_label"],
                )
                threshold = result.get("threshold")
                # Extract prompts from trials
                prompts = [trial["prompt"] for trial in result.get("trials", [])]
                # For zero-steering, the threshold is irrelevant but we keep structure consistent.
                if experiment_config.disable_steering:
                    threshold = 0.0
                # For recalibration, force threshold refit on this model.
                if args.recalibrate_thresholds and not experiment_config.disable_steering:
                    threshold = None
                precomputed_features.append((feature, threshold, prompts))

        print(f"✓ Loaded {len(precomputed_features)} features with cached thresholds and prompts")

        # Override experiment config's n_features to match loaded features
        experiment_config.n_features = len(precomputed_features)
        # Track provenance: which results file was used as source
        experiment_config.source_results_file = args.from_results

    experiment_config.to_dict()

    asyncio.run(run_experiment(
        experiment_config,
        base_model_for_sae=base_model,
        precomputed_features=precomputed_features,
        n_prompts_limit=args.n_prompts,
        output_suffix=args.output_suffix,
    ))