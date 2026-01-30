#!/usr/bin/env python3
"""
Experiment 10: Random Latent Ablation Control

Control experiment testing whether ablating random latents produces similar effects
to ablating off-topic detector (OTD) latents.

Usage:
    # Run random ablation experiments (3 independent random sets)
    python run_random_latent_control.py run-random --n-sets 3 --from-results <exp1_results.json>

    # Analyze and compare results (reads OTD ablation from experiment 3)
    python run_random_latent_control.py analyze
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
HAIKU_RESULTS_DIR = BASE_DIR / "experiment_results" / "claude_haiku_4_5_20251001_judge"
OUTPUT_DIR = HAIKU_RESULTS_DIR / "random_latent_control"

# Off-topic detector latents - loaded from old detectors file
def _load_otd_latents():
    detector_file = BASE_DIR / "data" / "off_topic_detectors_old.json"
    with open(detector_file) as f:
        data = json.load(f)
    return data["off_topic_detectors"]

OTD_LATENTS = _load_otd_latents()

# Number of latents to ablate (match OTD count)
N_ABLATE = len(OTD_LATENTS)


def get_random_latent_set(
    n_latents: int,
    num_sae_features: int = 65536,
    exclude: set[int] = None,
    seed: int = None
) -> list[int]:
    """Generate a random set of latent indices, excluding dead latents and OTDs."""
    if seed is not None:
        random.seed(seed)

    exclude = exclude or set()
    # Always exclude OTD latents
    exclude = exclude | set(OTD_LATENTS)

    available = [i for i in range(num_sae_features) if i not in exclude]
    return random.sample(available, n_latents)


async def run_ablation(
    from_results: str,
    ablate_latents: list[int],
    output_suffix: str,
    judge: str = "haiku",
):
    """Run the ablation experiment using the run_ablation_experiment module."""
    # Import from the original AGI-1652 directory
    source_dir = BASE_DIR.parent / "AGI-1652-random-latent-ablation-control"
    sys.path.insert(0, str(source_dir))

    from run_ablation_experiment import run_experiment, ExperimentConfig, FeatureInfo, PrecomputedTrial
    from judge import resolve_model_id

    print(f"Loading features from: {from_results}")
    with open(from_results, "r") as f:
        results_data = json.load(f)

    # Extract config info
    exp_config = results_data.get("experiment_config", {})
    model_name = exp_config.get("model_name", "meta-llama/Meta-Llama-3.3-70B-Instruct")

    # Create experiment config
    config = ExperimentConfig(
        prompts_file="prompts.txt",
        model_name=model_name,
        labels_file=exp_config.get("labels_file", "data/llama-70b-goodfire-l50.csv"),
        judge_model_name=resolve_model_id(judge),
        target_score_normalized=exp_config.get("target_score_normalized", 0.3),
        threshold_n_trials=exp_config.get("threshold_n_trials", 10),
        threshold_lower_bound=exp_config.get("threshold_lower_bound", 0.0),
        threshold_upper_bound=exp_config.get("threshold_upper_bound", 100.0),
        threshold_prior_mean=exp_config.get("threshold_prior_mean", 7.0),
        threshold_prior_std=exp_config.get("threshold_prior_std", 5.0),
        n_possible_seeds=exp_config.get("n_possible_seeds", 1000000),
        seed_start=exp_config.get("seed_start", 0),
        max_completion_tokens=exp_config.get("max_completion_tokens", 512),
        n_trials_per_feature=exp_config.get("n_trials_per_feature", 5),
        n_features=200,
        n_simultaneous_features=exp_config.get("n_simultaneous_features", 10),
        min_feature_concreteness=exp_config.get("min_feature_concreteness", 65.0),
    )

    # Extract precomputed features
    precomputed_features = []
    for result in results_data["results_by_feature"]:
        if not result.get("error") and result.get("threshold") is not None:
            feature = FeatureInfo(
                index_in_sae=result["feature_index_in_sae"],
                label=result["feature_label"],
            )
            threshold = result["threshold"]

            # Extract trials
            trials = []
            for trial in result.get("trials", []):
                if "prompt" in trial and "seed" in trial:
                    trials.append(PrecomputedTrial(
                        prompt=trial["prompt"],
                        seed=trial["seed"]
                    ))

            precomputed_features.append((feature, threshold, trials))

    print(f"Loaded {len(precomputed_features)} features")
    print(f"Ablating {len(ablate_latents)} latents")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the experiment
    await run_experiment(
        config,
        ablate_latents=ablate_latents,
        precomputed_features=precomputed_features,
        output_suffix=output_suffix,
        output_dir=str(OUTPUT_DIR),
    )


def analyze_results():
    """Analyze and compare OTD vs random ablation results.

    OTD ablation results are read from experiment 3 (main haiku judge directory).
    Random ablation results are read from this experiment's output directory.
    """
    print("=" * 60)
    print("Analyzing Random Latent Ablation Control Results")
    print("=" * 60)

    # Find OTD ablation files from experiment 3 (main haiku judge directory)
    # These are files with "_with_ablation" suffix that have 26 ablated latents (old OTD)
    otd_files = []
    for f in HAIKU_RESULTS_DIR.glob("*_with_ablation.json"):
        with open(f) as fp:
            data = json.load(fp)
        if len(data.get("ablated_latents", [])) == len(OTD_LATENTS):
            otd_files.append(f)

    # Find random ablation files from this experiment's output
    random_files = list(OUTPUT_DIR.glob("*_random_ablation_*.json")) if OUTPUT_DIR.exists() else []

    print(f"Found {len(otd_files)} OTD ablation files (from experiment 3)")
    print(f"Found {len(random_files)} random ablation files")

    if not otd_files and not random_files:
        print("No result files found. Run experiments first.")
        return

    def compute_metrics(result_file: Path) -> dict:
        """Compute ESR metrics from a result file."""
        with open(result_file) as f:
            data = json.load(f)

        first_scores = []
        improvements = []

        for feature in data.get("results_by_feature", []):
            if feature.get("error"):
                continue

            for trial in feature.get("trials", []):
                score = trial.get("score", {})
                attempts = score.get("attempts", [])

                if not attempts:
                    continue

                first_score = attempts[0].get("score", 0)
                last_score = attempts[-1].get("score", 0)
                first_scores.append(first_score)

                if len(attempts) > 1:
                    improvements.append(last_score - first_score)

        return {
            "n_trials": len(first_scores),
            "mean_first_score": sum(first_scores) / len(first_scores) if first_scores else 0,
            "n_multi_attempt": len(improvements),
            "mean_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements) * 100 if improvements else 0,
        }

    # Analyze each condition
    results = {}

    for f in otd_files:
        metrics = compute_metrics(f)
        results["OTD Ablation"] = metrics
        print(f"\nOTD Ablation ({f.name}):")
        print(f"  First attempt score: {metrics['mean_first_score']:.1f}%")
        print(f"  Mean improvement: {metrics['mean_improvement']:.2f}")
        print(f"  Multi-attempt trials: {metrics['n_multi_attempt']}/{metrics['n_trials']}")

    random_metrics = []
    for f in random_files:
        metrics = compute_metrics(f)
        random_metrics.append(metrics)
        print(f"\nRandom Ablation ({f.name}):")
        print(f"  First attempt score: {metrics['mean_first_score']:.1f}%")
        print(f"  Mean improvement: {metrics['mean_improvement']:.2f}")

    if random_metrics:
        avg_first = sum(m["mean_first_score"] for m in random_metrics) / len(random_metrics)
        avg_imp = sum(m["mean_improvement"] for m in random_metrics) / len(random_metrics)
        results["Random Ablation (avg)"] = {
            "mean_first_score": avg_first,
            "mean_improvement": avg_imp,
        }
        print(f"\nRandom Ablation Average:")
        print(f"  First attempt score: {avg_first:.1f}%")
        print(f"  Mean improvement: {avg_imp:.2f}")

    # Save summary
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Random Latent Ablation Control"
    )
    parser.add_argument(
        "command",
        choices=["run-random", "analyze"],
        help="Command to run"
    )
    parser.add_argument(
        "--from-results",
        type=str,
        help="Path to experiment 1 results file (required for run-random)"
    )
    parser.add_argument(
        "--n-sets",
        type=int,
        default=3,
        help="Number of random latent sets to test (default: 3)"
    )
    parser.add_argument(
        "--judge",
        "-j",
        type=str,
        default="haiku",
        help="Judge model to use (default: haiku)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random latent selection"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 10: Random Latent Ablation Control")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    if args.command == "analyze":
        analyze_results()
        return

    # run-random requires --from-results
    if not args.from_results:
        print("Error: --from-results is required for run-random")
        sys.exit(1)

    if args.command == "run-random":
        print(f"Running {args.n_sets} random ablation experiments")
        for i in range(args.n_sets):
            seed = args.seed + i
            random_latents = get_random_latent_set(N_ABLATE, seed=seed)
            print(f"\nRandom set {i} (seed={seed}): {random_latents[:5]}...")

            asyncio.run(run_ablation(
                from_results=args.from_results,
                ablate_latents=random_latents,
                output_suffix=f"_random_ablation_set{i}",
                judge=args.judge,
            ))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
