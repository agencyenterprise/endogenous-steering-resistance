"""Wikipedia vector steering ESR experiment.

Replicates the core ESR experiment (experiment 01) but uses pre-computed
Wikipedia contrastive vectors instead of SAE feature interventions.
Vectors are added directly to the residual stream at the extraction layer.
"""

import asyncio
import json
import os
import random
import time
from dataclasses import asdict
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

load_dotenv()

from judge import create_judge, get_judge_folder_name, Judge
from experiment_config import ExperimentConfig
from threshold_finder import find_threshold
from vllm_engine import VLLMSteeringEngine
from experiment_dataclasses import FeatureInfo, TrialResult, FeatureResult, ExperimentResult


EXP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTORS_PATH = EXP_ROOT / "data" / "wikipedia_vectors" / "wikipedia_contrastive_dataset_l48.pt"
DEFAULT_METADATA_PATH = EXP_ROOT / "data" / "wikipedia_vectors" / "wikipedia_metadata_l48.json"


def load_wikipedia_vectors_metadata(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> dict:
    """Load wikipedia vector metadata (titles, layer, model info)."""
    with open(metadata_path) as f:
        return json.load(f)


def sample_wikipedia_vectors(
    metadata: dict,
    n_vectors: int,
) -> list[FeatureInfo]:
    """Sample N wikipedia vectors, using their titles as labels.

    Args:
        metadata: Wikipedia vectors metadata dict with 'titles' key
        n_vectors: Number of vectors to sample

    Returns:
        List of FeatureInfo where index_in_sae is the vector_id
    """
    titles = metadata["titles"]
    num_available = len(titles)

    if n_vectors > num_available:
        print(f"Warning: requested {n_vectors} vectors but only {num_available} available")
        n_vectors = num_available

    indices = random.sample(range(num_available), n_vectors)

    return [
        FeatureInfo(index_in_sae=idx, label=titles[idx])
        for idx in indices
    ]


async def generate_response(
    engine: VLLMSteeringEngine,
    experiment_config: ExperimentConfig,
    prompt: str,
    feature: FeatureInfo,
    threshold: float,
) -> tuple[str, str, int]:
    """Generate a single response with vector steering intervention.

    Returns: (prompt, response, seed)
    """
    intervention = None
    if not experiment_config.disable_steering:
        intervention = [{"vector_id": feature.index_in_sae, "value": threshold}]
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


async def get_score_for_prompt(
    engine: VLLMSteeringEngine,
    judge: Judge,
    prompt: str,
    feature: FeatureInfo,
    boost: float,
    experiment_config: ExperimentConfig,
    max_retries: int = 3,
) -> float:
    """Get a score (0-1) for a specific prompt-vector-boost combination."""
    score = None
    retries = 0

    while score is None:
        intervention = None
        if not experiment_config.disable_steering:
            intervention = [{"vector_id": feature.index_in_sae, "value": boost}]
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
            if retries >= max_retries:
                raise Exception(
                    f"Judge error after {max_retries} retries for vector {feature.index_in_sae} "
                    f"({feature.label}), boost={boost}: {score_obj['error']}"
                )
            continue

        score = (
            score_obj["attempts"][0]["score"]
            if "attempts" in score_obj and len(score_obj["attempts"]) > 0
            else None
        )

    return score / 100.0


async def get_score(
    engine: VLLMSteeringEngine,
    judge: Judge,
    prompts: list[str],
    feature: FeatureInfo,
    boost: float,
    experiment_config: ExperimentConfig,
    max_retries: int = 3,
    n_samples: int = 1,
) -> float:
    """Get averaged score across n_samples prompts. Used for threshold calibration."""
    scores = []
    sampled_prompts = random.sample(prompts, min(n_samples, len(prompts)))

    for prompt in sampled_prompts:
        score = await get_score_for_prompt(
            engine, judge, prompt, feature, boost, experiment_config, max_retries
        )
        scores.append(score)

    return sum(scores) / len(scores)


async def get_vector_threshold(
    engine: VLLMSteeringEngine,
    judge: Judge,
    feature: FeatureInfo,
    prompts: list[str],
    experiment_config: ExperimentConfig,
    show_progress: bool = False,
) -> float:
    """Find threshold for a wikipedia vector that reduces on-topic score to target."""
    if experiment_config.disable_steering:
        return 0.0

    # Read existing cache
    threshold_cache = {}
    cache_file = experiment_config.get_threshold_cache_file()
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            threshold_cache = json.load(f)

    # Return cached value if available
    cache_key = str(feature.index_in_sae)
    if cache_key in threshold_cache:
        cached_value = threshold_cache[cache_key]
        threshold_value = cached_value.get("threshold") if isinstance(cached_value, dict) else cached_value
        if threshold_value is not None:
            if show_progress:
                print(f"Using cached threshold for vector {feature.index_in_sae} ({feature.label})")
            return threshold_value

    # Find new threshold
    n_samples = getattr(experiment_config, 'threshold_samples_per_trial', 1)
    threshold = await find_threshold(
        target_score=experiment_config.target_score_normalized,
        get_score_fn=lambda x: get_score(
            engine, judge, prompts, feature, x, experiment_config, n_samples=n_samples
        ),
        prior_mean=experiment_config.threshold_prior_mean,
        prior_std=experiment_config.threshold_prior_std,
        n_trials=experiment_config.threshold_n_trials,
        show_progress=show_progress,
        lower_bound=experiment_config.threshold_lower_bound,
        upper_bound=experiment_config.threshold_upper_bound,
    )

    threshold = float(round(threshold, 2))

    # Verify achieved score
    achieved_score = await get_score(engine, judge, prompts, feature, threshold, experiment_config)

    # Update cache
    threshold_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            threshold_cache = json.load(f)

    threshold_cache[cache_key] = {
        "threshold": float(threshold),
        "achieved_score": float(achieved_score),
        "vector_label": feature.label,
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


async def run_one_vector(
    engine: VLLMSteeringEngine,
    judge: Judge,
    experiment_config: ExperimentConfig,
    feature: FeatureInfo,
    pbar: tqdm | None = None,
    precomputed_threshold: float | None = None,
    precomputed_prompts: list[str] | None = None,
) -> FeatureResult:
    """Run the experiment for a single wikipedia vector."""
    try:
        prompts = experiment_config.get_prompts()

        if precomputed_prompts is not None:
            sampled_prompts = precomputed_prompts
        else:
            sampled_prompts = random.sample(
                prompts, min(len(prompts), experiment_config.n_trials_per_feature)
            )

        # Find or use precomputed threshold
        if precomputed_threshold is not None:
            threshold = precomputed_threshold
            if pbar:
                pbar.write(f"  Vector {feature.index_in_sae} ({feature.label}): using precomputed threshold = {threshold:.2f}")
        else:
            if pbar:
                pbar.set_description(f"Vector {feature.index_in_sae} ({feature.label[:30]}): Finding threshold")
            threshold = await get_vector_threshold(
                engine, judge, feature, prompts, experiment_config, show_progress=True
            )
            if pbar:
                pbar.write(f"  Vector {feature.index_in_sae} ({feature.label}): threshold = {threshold:.2f}")

        # Generate all responses in parallel
        if pbar:
            pbar.set_description(f"Vector {feature.index_in_sae}: Generating {len(sampled_prompts)} responses")

        generation_results = await asyncio.gather(
            *[
                generate_response(engine, experiment_config, prompt, feature, threshold)
                for prompt in sampled_prompts
            ]
        )

        # Score all responses in parallel
        if pbar:
            pbar.set_description(f"Vector {feature.index_in_sae}: Scoring {len(generation_results)} responses")

        scoring_results = await asyncio.gather(
            *[
                judge.grade_response(response, prompt, feature.label)
                for prompt, response, seed in generation_results
            ]
        )

        # Combine into trials
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
            pbar.write(f"  Vector {feature.index_in_sae}: {error_msg[:500]}")
        return FeatureResult(
            feature_index_in_sae=feature.index_in_sae,
            feature_label=feature.label,
            threshold=None,
            trials=[],
            error=error_msg,
        )


async def run_experiment(
    experiment_config: ExperimentConfig,
    vectors_path: str | Path = DEFAULT_VECTORS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    timeout_hours: float = 100,
    precomputed_features: list[tuple] | None = None,
    output_suffix: str | None = None,
    output_folder: str | None = None,
    repetition_penalty: float | None = None,
):
    """Run the wikipedia vector ESR experiment.

    Same flow as experiment_01 but with wikipedia vectors:
    - Loads pre-computed contrastive vectors instead of SAE features
    - Samples N vectors from 200 available
    - Skips concreteness/relevance filtering
    - Uses vector_id interventions instead of feature_id
    """
    metadata = load_wikipedia_vectors_metadata(metadata_path)
    steering_layer = metadata["layer_idx"]

    print("Initializing vLLM engine with vector steering...")
    engine = VLLMSteeringEngine(
        experiment_config.model_name,
        steering_vectors_path=str(vectors_path),
        steering_layer=steering_layer,
        repetition_penalty=repetition_penalty,
    )
    await engine.initialize()
    print(f"Engine initialized (steering layer: {steering_layer}, vectors: {metadata['num_titles']})")
    if repetition_penalty is not None:
        print(f"Using custom repetition penalty: {repetition_penalty}")

    judge = create_judge(experiment_config.judge_model_name)

    # Determine output folder
    if output_folder is not None:
        results_base_dir = f"data/experiment_results/{output_folder}"
    else:
        judge_folder = get_judge_folder_name(experiment_config.judge_model_name)
        results_base_dir = f"data/experiment_results/{judge_folder}_judge"

    # Use precomputed features or sample new vectors
    if precomputed_features is not None:
        print(f"\n  Using {len(precomputed_features)} precomputed vectors with cached thresholds and prompts")
        features_with_data = precomputed_features
    else:
        print(f"\nSampling {experiment_config.n_features} wikipedia vectors...")
        features = sample_wikipedia_vectors(metadata, experiment_config.n_features)
        features_with_data = [(f, None, None) for f in features]
        print(f"  Sampled {len(features)} vectors")
        for f in features:
            print(f"    [{f.index_in_sae}] {f.label}")

    # Set up concurrent processing
    semaphore = asyncio.Semaphore(experiment_config.n_simultaneous_features)

    pbar = tqdm(
        total=len(features_with_data),
        desc="Processing vectors",
        unit="vector",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        smoothing=0,
    )

    async def run_one_with_semaphore(feature_data):
        feature, threshold, prompts = feature_data
        async with semaphore:
            return await run_one_vector(
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
        suffix_part = f"_{output_suffix}" if output_suffix else ""
        final_filename = f"{results_base_dir}/experiment_wikipedia_esr_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}{suffix_part}.json"
        temp_filename = final_filename + ".tmp"

        Path(final_filename).parent.mkdir(parents=True, exist_ok=True)

        for feature_result in asyncio.as_completed(tasks):
            try:
                result = await feature_result
                experiment_result.results_by_feature.append(result)
                completed_count += 1

                elapsed = time.time() - start_time
                pbar.update(1)

                if result.error and result.error.strip():
                    pbar.write(f"  [{completed_count}/{len(features_with_data)}] Vector {result.feature_index_in_sae}: {result.feature_label[:40]}... - ERROR: {result.error[:50]}")
                else:
                    n_trials = len(result.trials)
                    threshold_str = f"{result.threshold:.2f}" if result.threshold is not None else "N/A"
                    pbar.write(f"  [{completed_count}/{len(features_with_data)}] Vector {result.feature_index_in_sae}: {result.feature_label[:40]}... ({n_trials} trials, threshold={threshold_str})")

                # Write checkpoint
                with open(temp_filename, "w") as f:
                    json.dump(asdict(experiment_result), f, indent=4)
                if os.path.exists(final_filename):
                    os.remove(final_filename)
                os.rename(temp_filename, final_filename)

                if completed_count % 5 == 0:
                    pbar.write(f"  Saved checkpoint: {completed_count}/{len(features_with_data)} vectors completed")

            except Exception as e:
                pbar.write(f"  Error processing vector result: {str(e)}")
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                continue

        pbar.close()
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"  Wikipedia ESR experiment completed!")
        print(f"   Total vectors: {len(features_with_data)}")
        print(f"   Successful: {sum(1 for r in experiment_result.results_by_feature if not r.error)}")
        print(f"   Failed: {sum(1 for r in experiment_result.results_by_feature if r.error)}")
        print(f"   Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"   Avg time per vector: {total_time/len(features_with_data):.1f} seconds")
        print(f"   Final results: {final_filename}")
        print(f"{'='*80}\n")

    tasks = [run_one_with_semaphore(feature_data) for feature_data in features_with_data]

    print(f"\n{'='*80}")
    print(f"Starting Wikipedia vector ESR experiment with {len(features_with_data)} vectors")
    print(f"  Model: {experiment_config.model_name}")
    print(f"  Steering layer: {steering_layer}")
    print(f"  Trials per vector: {experiment_config.n_trials_per_feature}")
    print(f"  Concurrent vectors: {experiment_config.n_simultaneous_features}")
    print(f"  Total expected trials: {len(features_with_data) * experiment_config.n_trials_per_feature}")
    print(f"{'='*80}\n")

    try:
        await asyncio.wait_for(process_results(), timeout=timeout_hours * 3600)
    except asyncio.TimeoutError:
        print(f"Experiment timed out after {timeout_hours} hours")
        if experiment_result.results_by_feature:
            suffix_part = f"_{output_suffix}" if output_suffix else ""
            new_filename = f"{results_base_dir}/experiment_wikipedia_esr_{short_model_name}_{time.strftime('%Y%m%d_%H%M%S')}{suffix_part}_TIMEOUT.json"
            with open(new_filename, "w") as f:
                json.dump(asdict(experiment_result), f, indent=4)
            print(f"Saved partial results to {new_filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Wikipedia vector ESR experiment")
    parser.add_argument("--n-vectors", type=int, default=80,
                        help="Number of wikipedia vectors to test (default: 80, max 200)")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of trials per vector (default: 5)")
    parser.add_argument("--n-simultaneous", type=int, default=10,
                        help="Number of vectors to process concurrently (default: 10)")
    parser.add_argument("--judge", "-j", type=str, default=None,
                        help="Override judge model")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Add suffix to output filename")
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Override output folder name")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Override repetition penalty (default: 1.2 for 70B)")
    parser.add_argument("--vectors-path", type=str, default=None,
                        help="Override path to contrastive vectors .pt file")
    parser.add_argument("--metadata-path", type=str, default=None,
                        help="Override path to metadata JSON file")
    parser.add_argument("--from-results", type=str, default=None,
                        help="Load vectors, thresholds, and prompts from an existing results file")
    parser.add_argument("--no-steering", action="store_true",
                        help="Disable steering (zero-steering baseline)")
    parser.add_argument("--recalibrate-thresholds", action="store_true",
                        help="When used with --from-results, re-find thresholds instead of reusing them")
    args = parser.parse_args()

    vectors_path = Path(args.vectors_path) if args.vectors_path else DEFAULT_VECTORS_PATH
    metadata_path = Path(args.metadata_path) if args.metadata_path else DEFAULT_METADATA_PATH

    # Config tuned for 70B wikipedia vector steering
    # Initial runs showed thresholds ~2-3 still overshoot (scores 0.0 at target 0.3),
    # so prior is centered low with a tight upper bound to search the right range.
    experiment_config = ExperimentConfig(
        prompts_file="prompts.txt",
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        labels_file=None,  # No SAE labels; wikipedia titles used directly
        judge_model_name="claude-sonnet-4-5-20250929",
        target_score_normalized=0.3,
        threshold_n_trials=10,
        threshold_samples_per_trial=5,
        per_prompt_calibration=False,
        threshold_lower_bound=0.0,
        threshold_upper_bound=5.0,
        threshold_prior_mean=1.5,
        threshold_prior_std=1.5,
        n_possible_seeds=1000000,
        seed_start=0,
        max_completion_tokens=512,
        n_trials_per_feature=args.n_trials,
        n_features=args.n_vectors,
        n_simultaneous_features=args.n_simultaneous,
        min_feature_concreteness=0.0,  # Not used; all vectors treated as concrete
    )

    experiment_config.disable_steering = bool(args.no_steering)

    if args.judge:
        from judge import resolve_model_id
        experiment_config.judge_model_name = resolve_model_id(args.judge)
        print(f"Using judge model: {experiment_config.judge_model_name}")

    # Load precomputed features from results file if provided
    precomputed_features = None
    if args.from_results:
        print(f"\nLoading vectors, thresholds, and prompts from {args.from_results}")
        with open(args.from_results) as f:
            results_data = json.load(f)

        precomputed_features = []
        for result in results_data["results_by_feature"]:
            if not result.get("error"):
                feature = FeatureInfo(
                    index_in_sae=result["feature_index_in_sae"],
                    label=result["feature_label"],
                )
                threshold = result.get("threshold")
                prompts = [trial["prompt"] for trial in result.get("trials", [])]
                if experiment_config.disable_steering:
                    threshold = 0.0
                if args.recalibrate_thresholds and not experiment_config.disable_steering:
                    threshold = None
                precomputed_features.append((feature, threshold, prompts))

        print(f"  Loaded {len(precomputed_features)} vectors with cached thresholds and prompts")
        experiment_config.n_features = len(precomputed_features)
        experiment_config.source_results_file = args.from_results

    experiment_config.to_dict()

    asyncio.run(run_experiment(
        experiment_config,
        vectors_path=vectors_path,
        metadata_path=metadata_path,
        precomputed_features=precomputed_features,
        output_suffix=args.output_suffix,
        output_folder=args.output_folder,
        repetition_penalty=args.repetition_penalty,
    ))
