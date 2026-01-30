"""Experiment runner with optional off-topic detector ablation.

This script extends the regular experiment to optionally ablate (zero-clamp)
off-topic detector latents during generation, to test whether these latents
are causally related to Endogenous Steering Resistance.
"""

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional
from pathlib import Path
import sys

from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from judge import create_judge, get_judge_folder_name, Judge
from experiment_config import ExperimentConfig
from threshold_finder import find_threshold
from vllm_engine import VLLMSteeringEngine
from sample_features import sample_filtered_features
from concreteness_filtering import ConcretenessGrader
from experiment_dataclasses import FeatureInfo, TrialResult, FeatureResult, ExperimentResult


@dataclass
class PrecomputedTrial:
    """Represents a precomputed trial with exact prompt and seed to replay."""
    prompt: str
    seed: int


async def generate_response(
    engine: VLLMSteeringEngine,
    experiment_config: ExperimentConfig,
    prompt: str,
    feature: FeatureInfo,
    threshold: float,
    ablate_latents: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> tuple[str, str, int]:
    """
    Generate a single response with feature intervention.

    Args:
        engine: vLLM engine
        experiment_config: Experiment configuration
        prompt: User prompt
        feature: Feature to steer
        threshold: Steering strength
        ablate_latents: Optional list of latent indices to zero-ablate
        seed: Optional seed for generation (if None, generates random seed)

    Returns: (prompt, response, seed)
    """
    # Build intervention list
    intervention = [{"feature_id": feature.index_in_sae, "value": threshold}]

    # Add ablation interventions if specified
    if ablate_latents:
        for latent_id in ablate_latents:
            intervention.append({
                "feature_id": latent_id,
                "value": 0.0,
                "mode": "clamp"
            })

    convo = [{"role": "user", "content": prompt}]
    if seed is None:
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
    judge: Judge,
    prompts: List[str],
    feature: FeatureInfo,
    boost: float,
    experiment_config: ExperimentConfig,
    ablate_latents: Optional[List[int]] = None,
) -> float:
    """
    Get a score (between 0 and 1) for a given feature and boost value.
    Uses a random prompt from `prompts`.

    Args:
        ablate_latents: Optional list of latent indices to zero-ablate
    """
    score = None
    while score is None:
        prompt = random.choice(prompts)

        # Build intervention list
        intervention = [{"feature_id": feature.index_in_sae, "value": boost}]

        # Add ablation interventions if specified
        if ablate_latents:
            for latent_id in ablate_latents:
                intervention.append({
                    "feature_id": latent_id,
                    "value": 0.0,
                    "mode": "clamp"
                })

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
        score_obj = await judge.grade_response(response, prompt, feature.label)

        score = (
            score_obj["attempts"][0]["score"]
            if "attempts" in score_obj and len(score_obj["attempts"]) > 0
            else None
        )
    return score / 100.0


async def get_feature_threshold(
    engine: VLLMSteeringEngine,
    judge: Judge,
    feature: FeatureInfo,
    prompts: List[str],
    experiment_config: ExperimentConfig,
    ablate_latents: Optional[List[int]] = None,
    show_progress: bool = False,
) -> float:
    """
    For a given feature, find an overall threshold that reduces the model's
    output to an average score of 50/100 across the given prompts.

    Args:
        ablate_latents: Optional list of latent indices to zero-ablate
    """
    # Read existing cache if it exists
    threshold_cache = {}
    cache_file = experiment_config.get_threshold_cache_file()

    # Use different cache for ablation experiments
    if ablate_latents:
        cache_file = cache_file.replace(".json", "_with_ablation.json")

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
            engine, judge, prompts, feature, x, experiment_config, ablate_latents
        ),
        prior_mean=experiment_config.threshold_prior_mean,
        prior_std=experiment_config.threshold_prior_std,
        n_trials=experiment_config.threshold_n_trials,
        show_progress=show_progress,
        lower_bound=experiment_config.threshold_lower_bound,
        upper_bound=experiment_config.threshold_upper_bound,
    )

    # vLLM request serialization requires plain Python types (not numpy scalars).
    threshold = float(round(float(threshold), 2))

    # Verify the achieved score with the found threshold
    achieved_score = await get_score(engine, judge, prompts, feature, threshold, experiment_config, ablate_latents)

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
    judge: Judge,
    experiment_config: ExperimentConfig,
    feature: FeatureInfo,
    ablate_latents: Optional[List[int]] = None,
    pbar: Optional[tqdm] = None,
    precomputed_threshold: Optional[float] = None,
    precomputed_trials: Optional[List[PrecomputedTrial]] = None,
) -> FeatureResult:
    """
    Run the experiment for a single feature.

    Uses batched generation and scoring:
    1. Generate all responses in parallel (GPU work)
    2. Score all responses in parallel (Claude API work)
    This keeps GPU and Claude API busy independently.

    Args:
        ablate_latents: Optional list of latent indices to zero-ablate
        precomputed_threshold: Optional precomputed threshold to use instead of finding one
        precomputed_trials: Optional list of exact trials (prompt+seed) to replay
    """
    try:

        prompts = experiment_config.get_prompts()

        # Find or use precomputed threshold
        if precomputed_threshold is not None:
            threshold = precomputed_threshold
            if pbar:
                pbar.write(f"✓ Feature {feature.index_in_sae}: using cached threshold = {threshold:.2f}")
        else:
            if pbar:
                pbar.set_description(f"Feature {feature.index_in_sae}: Finding threshold")
            threshold = await get_feature_threshold(
                engine, judge, feature, prompts, experiment_config, ablate_latents
            )
            if pbar:
                pbar.write(f"✓ Feature {feature.index_in_sae}: threshold = {threshold:.2f}")

        # Use precomputed trials if provided, otherwise sample new prompts
        if precomputed_trials:
            trials_to_run = precomputed_trials
            if pbar:
                pbar.write(f"✓ Feature {feature.index_in_sae}: replaying {len(trials_to_run)} precomputed trials")
        else:
            # Sample prompts without replacement
            sampled_prompts = random.sample(
                prompts, min(len(prompts), experiment_config.n_trials_per_feature)
            )
            trials_to_run = [PrecomputedTrial(prompt=p, seed=None) for p in sampled_prompts]

        # BATCH 1: Generate all responses in parallel (GPU work)
        if pbar:
            pbar.set_description(f"Feature {feature.index_in_sae}: Generating {len(trials_to_run)} responses")

        generation_results = await asyncio.gather(
            *[
                generate_response(engine, experiment_config, trial.prompt, feature, threshold, ablate_latents, trial.seed)
                for trial in trials_to_run
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
        error_msg = str(e).strip() if str(e) else ""
        if not error_msg:
            # Some exceptions (e.g. CancelledError) may stringify to empty.
            error_msg = f"{type(e).__name__}: {traceback.format_exc()}"
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
    ablate_latents: Optional[List[int]] = None,
    timeout_hours: float = 100,
    precomputed_features: Optional[List[tuple[FeatureInfo, float, List[PrecomputedTrial]]]] = None,
    output_folder: Optional[str] = None,
):
    """
    Run a whole experiment on a model, optionally with off-topic detector ablation.

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
        ablate_latents: Optional list of latent indices to zero-ablate during generation
        timeout_hours: If provided, the experiment will be cancelled after this many hours
        precomputed_features: Optional list of (FeatureInfo, threshold, trials) tuples to use
                            instead of sampling and computing thresholds. If trials is provided,
                            those exact trials (prompt+seed) will be replayed.
    """
    # Initialize engine and judge
    print("Initializing vLLM engine...")
    engine = VLLMSteeringEngine(experiment_config.model_name)
    await engine.initialize()
    print("Engine initialized")

    if ablate_latents:
        print(f"\n⚠️  ABLATION MODE: Zero-clamping {len(ablate_latents)} off-topic detector latents")

    judge = create_judge(experiment_config.judge_model_name)

    # Determine features to test
    if precomputed_features is not None:
        # Use precomputed features, thresholds, and trials from existing results
        print(f"\n✓ Using {len(precomputed_features)} precomputed features with cached thresholds and trials")
        features_with_data = precomputed_features
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
        )

        if len(feature_indices) < experiment_config.n_features:
            print(f"⚠️  Warning: Only found {len(feature_indices)}/{experiment_config.n_features} features")

        # Create FeatureInfo objects (thresholds and trials will be computed later)
        features_with_data = [
            (FeatureInfo(
                index_in_sae=idx,
                label=feature_labels.get(idx, f"feature_{idx}"),
            ), None, None)
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
        feature, threshold, trials = feature_data
        async with semaphore:
            return await run_one_feature(engine, judge, experiment_config, feature, ablate_latents, pbar, threshold, trials)

    experiment_result = ExperimentResult(
        experiment_config=experiment_config.to_dict(), results_by_feature=[]
    )

    short_model_name = experiment_config.model_name.split("/")[-1]
    start_time = time.time()
    completed_count = 0

    async def process_results():
        nonlocal completed_count
        # Create the final filename once at the start
        ablation_suffix = "_with_ablation" if ablate_latents else ""
        if output_folder is not None:
            results_base_dir = f"experiment_results/{output_folder}"
        else:
            judge_folder = get_judge_folder_name(experiment_config.judge_model_name)
            results_base_dir = f"experiment_results/{judge_folder}_judge"
        final_filename = f"{results_base_dir}/experiment_results_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}{ablation_suffix}.json"
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
                if result.error and str(result.error).strip():
                    pbar.write(f"❌ [{completed_count}/{len(features_with_data)}] Feature {result.feature_index_in_sae}: {result.feature_label[:40]}... - ERROR: {result.error[:50]}")
                else:
                    n_trials = len(result.trials)
                    threshold_str = f"{result.threshold:.2f}" if result.threshold is not None else "N/A"
                    pbar.write(f"✅ [{completed_count}/{len(features_with_data)}] Feature {result.feature_index_in_sae}: {result.feature_label[:40]}... ({n_trials} trials, threshold={threshold_str})")

                # Write to temporary file first
                result_dict = asdict(experiment_result)
                # Add ablation metadata
                if ablate_latents:
                    result_dict["ablated_latents"] = ablate_latents
                    result_dict["num_ablated_latents"] = len(ablate_latents)

                with open(temp_filename, "w") as f:
                    json.dump(result_dict, f, indent=4)

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
        if ablate_latents:
            pbar.write(f"   Ablated {len(ablate_latents)} latents")
        pbar.write(f"{'='*80}\n")

    tasks = [run_one_feature_with_semaphore(feature_data) for feature_data in features_with_data]

    print(f"\n{'='*80}")
    print(f"Starting experiment with {len(features_with_data)} features")
    print(f"  Model: {experiment_config.model_name}")
    print(f"  Trials per feature: {experiment_config.n_trials_per_feature}")
    print(f"  Concurrent features: {experiment_config.n_simultaneous_features}")
    print(f"  Total expected trials: {len(features_with_data) * experiment_config.n_trials_per_feature}")
    if ablate_latents:
        print(f"  Ablating {len(ablate_latents)} off-topic detector latents")
    if precomputed_features:
        print(f"  Using precomputed features, thresholds, and trials (replaying exact trials)")
    print(f"{'='*80}\n")

    try:
        await asyncio.wait_for(process_results(), timeout=timeout_hours * 3600)
    except asyncio.TimeoutError:
        print(f"Experiment timed out after {timeout_hours} hours")
        # Save final results before exiting
        if experiment_result.results_by_feature:
            ablation_suffix = "_with_ablation" if ablate_latents else ""
            if output_folder is not None:
                results_base_dir = f"experiment_results/{output_folder}"
            else:
                judge_folder = get_judge_folder_name(experiment_config.judge_model_name)
                results_base_dir = f"experiment_results/{judge_folder}_judge"
            new_filename = f"{results_base_dir}/experiment_results_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}{ablation_suffix}_TIMEOUT.json"
            result_dict = asdict(experiment_result)
            if ablate_latents:
                result_dict["ablated_latents"] = ablate_latents
                result_dict["num_ablated_latents"] = len(ablate_latents)
            with open(new_filename, "w") as f:
                json.dump(result_dict, f, indent=4)
            print(f"Saved partial results to {new_filename}")


async def main():
    """Run experiment with optional ablation."""
    import sys

    # Define configs matching experiment.py
    # By default, use heldout prompts to avoid overfitting concerns
    configs = {
        "8b": ExperimentConfig(
            prompts_file="prompts.txt",
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
            model_name="meta-llama/Meta-Llama-3.3-70B-Instruct",
            labels_file="data/llama-70b-goodfire-l50.csv",
            judge_model_name="claude-sonnet-4-5-20250929",
            target_score_normalized=0.3,  # Target ~30 first-attempt score
            threshold_n_trials=20,
            threshold_lower_bound=0.0,
            threshold_upper_bound=100.0,
            threshold_prior_mean=20.0,
            threshold_prior_std=20.0,
            n_possible_seeds=1000000,
            seed_start=0,
            max_completion_tokens=512,
            n_trials_per_feature=5,
            n_features=50,
            n_simultaneous_features=50,
            min_feature_concreteness=65.0,
        ),
    }

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config_name> [--ablate <detector_file>] [--from-results <results_file>] [--judge <model>] [--output-folder <folder>]")
        print(f"Available configs: {list(configs.keys())}")
        print(f"  --ablate: Path to JSON file with off-topic detectors to ablate")
        print(f"  --from-results: Path to existing experiment results to reuse features and thresholds")
        print(f"  --judge: Override judge model (e.g., 'haiku', 'sonnet', 'gemini-3-flash-preview')")
        print(f"  --output-folder: Override output folder (e.g., 'haiku_judge_old_detectors')")
        sys.exit(1)

    config_name = sys.argv[1]
    if config_name not in configs:
        raise ValueError(f"Config must be one of {list(configs.keys())}, not {config_name}")

    experiment_config = configs[config_name]

    # Check for ablation flag
    ablate_latents = None
    if "--ablate" in sys.argv:
        ablate_idx = sys.argv.index("--ablate")
        if ablate_idx + 1 < len(sys.argv):
            detector_file = sys.argv[ablate_idx + 1]
            print(f"Loading off-topic detectors from {detector_file}")
            with open(detector_file, "r") as f:
                detector_data = json.load(f)
                ablate_latents = detector_data["off_topic_detectors"]
            print(f"Loaded {len(ablate_latents)} off-topic detector latents to ablate")
        else:
            raise ValueError("--ablate requires a path to detector JSON file")

    # Check for --from-results flag
    precomputed_features = None
    if "--from-results" in sys.argv:
        results_idx = sys.argv.index("--from-results")
        if results_idx + 1 < len(sys.argv):
            results_file = sys.argv[results_idx + 1]
            print(f"\nLoading features, thresholds, and trials from {results_file}")
            with open(results_file, "r") as f:
                results_data = json.load(f)

            # Extract features, thresholds, and trials from results
            precomputed_features = []
            for result in results_data["results_by_feature"]:
                if not result.get("error") and result.get("threshold") is not None:
                    feature = FeatureInfo(
                        index_in_sae=result["feature_index_in_sae"],
                        label=result["feature_label"],
                    )
                    threshold = result["threshold"]

                    # Extract trials (prompt + seed)
                    trials = []
                    for trial in result.get("trials", []):
                        if "prompt" in trial and "seed" in trial:
                            trials.append(PrecomputedTrial(
                                prompt=trial["prompt"],
                                seed=trial["seed"]
                            ))

                    precomputed_features.append((feature, threshold, trials))

            print(f"✓ Loaded {len(precomputed_features)} features with cached thresholds and trials")
            total_trials = sum(len(trials) for _, _, trials in precomputed_features)
            print(f"✓ Total trials to replay: {total_trials}")

            # Override experiment config's n_features to match loaded features
            experiment_config.n_features = len(precomputed_features)
        else:
            raise ValueError("--from-results requires a path to results JSON file")

    # Check for --judge flag
    if "--judge" in sys.argv:
        judge_idx = sys.argv.index("--judge")
        if judge_idx + 1 < len(sys.argv):
            from judge import resolve_model_id
            judge_model = sys.argv[judge_idx + 1]
            experiment_config.judge_model_name = resolve_model_id(judge_model)
            print(f"Using judge model: {experiment_config.judge_model_name}")
        else:
            raise ValueError("--judge requires a model name")

    # Check for --output-folder flag
    output_folder = None
    if "--output-folder" in sys.argv:
        folder_idx = sys.argv.index("--output-folder")
        if folder_idx + 1 < len(sys.argv):
            output_folder = sys.argv[folder_idx + 1]
            print(f"Using output folder: experiment_results/{output_folder}/")
        else:
            raise ValueError("--output-folder requires a folder name")

    await run_experiment(
        experiment_config,
        ablate_latents=ablate_latents,
        precomputed_features=precomputed_features,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    asyncio.run(main())
