#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "tqdm",
#   "python-dotenv",
# ]
# ///
"""
Regrade experiment responses using multiple judge models via OpenRouter.

This script samples trials from experiment results, prioritizing self-correction
examples, and regrades them using multiple LLM judges to validate robustness
of ESR findings.

Usage:
    python regrade_cross_judge.py --n-samples 1000
"""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Load environment variables from .env
load_dotenv()

from judge import create_judge, Judge, JUDGE_MODELS


@dataclass
class Trial:
    """A trial to be regraded."""

    source_file: str
    feature_index: int
    feature_label: str
    prompt: str
    response: str
    seed: int
    original_score: dict | None
    has_self_correction: bool
    model_name: str = ""


def extract_model_name(filename: str) -> str:
    """Extract the target model name from a filename."""
    patterns = [
        ("Meta-Llama-3.3-70B-Instruct", "Llama-3.3-70B"),
        ("Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
        ("gemma-2-27b-it", "Gemma-2-27B"),
        ("gemma-2-9b-it", "Gemma-2-9B"),
        ("gemma-2-9b-res", "Gemma-2-9B"),
        ("gemma-2-2b-it", "Gemma-2-2B"),
    ]
    for pattern, name in patterns:
        if pattern in filename:
            return name
    return "Unknown"


def load_all_trials(experiment_results_dir: Path, exclude_patterns: list[str] | None = None) -> list[Trial]:
    """Load all trials from experiment results files.

    Args:
        experiment_results_dir: Directory containing experiment results
        exclude_patterns: List of patterns to exclude from filenames (e.g., ["ratio", "merged"])
    """
    trials = []
    exclude_patterns = exclude_patterns or []

    result_files = list(experiment_results_dir.glob("experiment_results_*.json"))

    # Filter out files matching exclude patterns
    if exclude_patterns:
        original_count = len(result_files)
        result_files = [
            f for f in result_files
            if not any(pattern in f.name for pattern in exclude_patterns)
        ]
        print(f"Filtered out {original_count - len(result_files)} files matching patterns: {exclude_patterns}")

    print(f"Found {len(result_files)} experiment result files")

    for file_path in tqdm(result_files, desc="Loading experiment files"):
        try:
            with open(file_path) as f:
                data = json.load(f)

            results_by_feature = data.get("results_by_feature", [])
            model_name = extract_model_name(file_path.name)

            for feature_result in results_by_feature:
                feature_index = feature_result.get("feature_index_in_sae")
                feature_label = feature_result.get("feature_label", "")

                for trial in feature_result.get("trials", []):
                    if trial.get("error"):
                        continue

                    response = trial.get("response", "")
                    if not response or not response.strip():
                        continue

                    # Check if this trial has self-correction
                    score = trial.get("score", {})
                    attempts = []
                    if isinstance(score, dict):
                        attempts = score.get("attempts", [])

                    has_self_correction = len(attempts) > 1

                    trials.append(
                        Trial(
                            source_file=file_path.name,
                            feature_index=feature_index,
                            feature_label=feature_label,
                            prompt=trial.get("prompt", ""),
                            response=response,
                            seed=trial.get("seed", 0),
                            original_score=score,
                            has_self_correction=has_self_correction,
                            model_name=model_name,
                        )
                    )

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    return trials


def select_samples(
    trials: list[Trial],
    n_samples: int,
    min_per_model: int = 100,
    include_all_multi_attempt: bool = True,
) -> list[Trial]:
    """
    Select a sample of trials for regrading with stratification by model.

    Args:
        trials: All available trials
        n_samples: Target number of samples to select
        min_per_model: Minimum samples per model (will exceed n_samples if needed)
        include_all_multi_attempt: If True, include all multi-attempt samples

    Returns:
        List of selected trials
    """
    # Group trials by model
    trials_by_model: dict[str, list[Trial]] = {}
    for t in trials:
        if t.model_name not in trials_by_model:
            trials_by_model[t.model_name] = []
        trials_by_model[t.model_name].append(t)

    print(f"\nTrials by model:")
    for model, model_trials in sorted(trials_by_model.items()):
        multi = sum(1 for t in model_trials if t.has_self_correction)
        print(f"  {model}: {len(model_trials)} total, {multi} multi-attempt")

    selected = []
    selected_ids = set()  # Track (source_file, seed) to avoid duplicates

    def add_trial(t: Trial) -> bool:
        trial_id = (t.source_file, t.seed, t.feature_index)
        if trial_id not in selected_ids:
            selected_ids.add(trial_id)
            selected.append(t)
            return True
        return False

    # Step 1: Include all multi-attempt samples if requested
    if include_all_multi_attempt:
        for model, model_trials in trials_by_model.items():
            multi_attempt = [t for t in model_trials if t.has_self_correction]
            for t in multi_attempt:
                add_trial(t)
        print(f"\nAfter adding all multi-attempt: {len(selected)} samples")

    # Step 2: Ensure minimum samples per model
    for model, model_trials in trials_by_model.items():
        current_count = sum(1 for t in selected if t.model_name == model)
        needed = min_per_model - current_count

        if needed > 0:
            # Get trials not yet selected
            available = [t for t in model_trials if (t.source_file, t.seed, t.feature_index) not in selected_ids]
            random.shuffle(available)

            for t in available[:needed]:
                add_trial(t)

            final_count = sum(1 for t in selected if t.model_name == model)
            if final_count < min_per_model:
                print(f"  Warning: {model} only has {final_count} samples (wanted {min_per_model})")

    print(f"\nAfter ensuring min per model: {len(selected)} samples")

    # Step 3: If we're under n_samples, add more randomly
    if len(selected) < n_samples:
        remaining_needed = n_samples - len(selected)
        available = [t for t in trials if (t.source_file, t.seed, t.feature_index) not in selected_ids]
        random.shuffle(available)

        for t in available[:remaining_needed]:
            add_trial(t)

    print(f"Final selection: {len(selected)} samples")

    # Print final distribution
    print("\nFinal samples by model:")
    for model in sorted(trials_by_model.keys()):
        count = sum(1 for t in selected if t.model_name == model)
        multi = sum(1 for t in selected if t.model_name == model and t.has_self_correction)
        print(f"  {model}: {count} total, {multi} multi-attempt")

    return selected


async def regrade_with_model(
    judge: Judge,
    trials: list[Trial],
    model_name: str,
) -> list[dict]:
    """Regrade all trials with a single judge model."""

    async with httpx.AsyncClient() as client:
        async def grade_one(trial: Trial) -> dict:
            result = await judge.grade_response(
                response=trial.response,
                prompt=trial.prompt,
                feature_label=trial.feature_label,
                client=client,
            )
            return {
                "source_file": trial.source_file,
                "feature_index": trial.feature_index,
                "feature_label": trial.feature_label,
                "prompt": trial.prompt,
                "seed": trial.seed,
                "has_self_correction": trial.has_self_correction,
                "original_score": trial.original_score,
                "new_score": result,
            }

        tasks = [grade_one(trial) for trial in trials]
        results = await tqdm_asyncio.gather(
            *tasks, desc=f"Grading with {model_name}", total=len(tasks)
        )

    return results


async def main():
    parser = argparse.ArgumentParser(description="Regrade responses with multiple judges")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to regrade (default: 1000)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(JUDGE_MODELS.keys()),
        choices=list(JUDGE_MODELS.keys()),
        help="Judge models to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cross_judge_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiment-results-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent requests per model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show what would be done without making API calls",
    )
    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="+",
        default=["ratio", "merged"],
        help="Patterns to exclude from filenames (default: ratio, merged)",
    )
    parser.add_argument(
        "--min-per-model",
        type=int,
        default=100,
        help="Minimum samples per target model (default: 100)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all trials
    print("\n=== Loading trials ===")
    all_trials = load_all_trials(args.experiment_results_dir, exclude_patterns=args.exclude_patterns)
    print(f"Total trials loaded: {len(all_trials)}")

    # Select samples
    print("\n=== Selecting samples ===")
    samples = select_samples(
        all_trials,
        args.n_samples,
        min_per_model=args.min_per_model,
        include_all_multi_attempt=True,
    )
    print(f"Selected {len(samples)} samples")

    self_correction_count = sum(1 for t in samples if t.has_self_correction)
    print(f"  - Self-correction samples: {self_correction_count}")
    print(f"  - Other samples: {len(samples) - self_correction_count}")

    # Show source file distribution
    source_files = {}
    for t in samples:
        source_files[t.source_file] = source_files.get(t.source_file, 0) + 1
    print("\nSamples per source file:")
    for fname, count in sorted(source_files.items(), key=lambda x: -x[1])[:10]:
        print(f"  {fname}: {count}")
    if len(source_files) > 10:
        print(f"  ... and {len(source_files) - 10} more files")

    if args.dry_run:
        print("\n=== DRY RUN - No API calls will be made ===")
        print(f"Would regrade {len(samples)} samples with models: {args.models}")

        # Estimate costs (rough estimates)
        avg_input_tokens = 1500  # prompt + response + instructions
        avg_output_tokens = 500  # judge response

        print("\nEstimated costs per model (very rough):")
        cost_estimates = {
            "gpt-5-mini": (0.25, 2.00),  # $/M input, $/M output
            "qwen3-32b": (0.08, 0.24),
            "haiku-4.5": (1.00, 5.00),
            "gemini-2.5-flash": (0.30, 2.50),
        }
        total_cost = 0
        for model in args.models:
            if model in cost_estimates:
                input_cost, output_cost = cost_estimates[model]
                model_cost = (
                    len(samples) * avg_input_tokens * input_cost / 1_000_000
                    + len(samples) * avg_output_tokens * output_cost / 1_000_000
                )
                total_cost += model_cost
                print(f"  {model}: ~${model_cost:.2f}")

        print(f"\nTotal estimated cost: ~${total_cost:.2f}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save sample metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_meta = {
        "timestamp": timestamp,
        "n_samples": len(samples),
        "self_correction_count": self_correction_count,
        "models": args.models,
        "seed": args.seed,
        "source_files": dict(source_files),
    }
    with open(args.output_dir / f"sample_metadata_{timestamp}.json", "w") as f:
        json.dump(sample_meta, f, indent=2)

    # Regrade with each model
    for model_name in args.models:
        print(f"\n=== Regrading with {model_name} ===")
        model_id = JUDGE_MODELS[model_name]

        judge = create_judge(
            model_id,
            max_concurrent=args.max_concurrent,
        )

        results = await regrade_with_model(judge, samples, model_name)

        # Count errors and successes
        errors = sum(1 for r in results if r["new_score"].get("error"))
        successes = len(results) - errors
        print(f"Completed: {successes} successes, {errors} errors")

        # Save results
        output_file = args.output_dir / f"regraded_{model_name}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "model_id": model_id,
                    "timestamp": timestamp,
                    "n_samples": len(samples),
                    "n_errors": errors,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Saved to {output_file}")

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
