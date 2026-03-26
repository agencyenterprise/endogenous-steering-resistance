"""
Prefill control experiment: does self-correction happen when the unsteered model
completes a prefix from a steered response?

Approach:
1. Load existing experiment results (steered Llama 70B responses)
2. Sample episodes (prompt + latent pairs)
3. For each episode, take a character-level prefix of the steered response
4. Feed the prefix to the unsteered model as a prefill and let it complete
5. Judge the full completion (prefill + continuation) with the same judge
6. Compare self-correction rates between steered and prefill-completed responses

Prefix modes:
- "char_fraction": Take first X% of the response by character count
- "pre_correction": Take everything up to (but not including) the self-correction point
"""

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

from judge import create_judge, get_judge_folder_name
from vllm_engine import VLLMSteeringEngine


# Patterns that indicate self-correction language
CORRECTION_PATTERNS = re.compile(
    r"(?i)("
    r"wait.*(?:not right|mistake|wrong|error)"
    r"|let me try again"
    r"|let'?s start over"
    r"|actually,?\s*(?:let me|I (?:should|need|made))"
    r"|I'?d like to revise"
    r"|I apologize.*let me"
    r"|let me correct"
    r"|sorry.*let me"
    r"|I made (?:a |an )?(?:mistake|error)"
    r"|that'?s not (?:right|correct|what)"
    r"|I need to (?:start over|redo|fix)"
    r")"
)


@dataclass
class Episode:
    """A single steered episode extracted from existing results."""
    prompt: str
    feature_index: int
    feature_label: str
    threshold: float
    steered_response: str
    steered_attempts: list
    correction_char_pos: int | None  # char position where self-correction starts, or None


@dataclass
class PrefillTrialResult:
    """Result from one prefill completion trial."""
    prompt: str
    feature_index: int
    feature_label: str
    threshold: float
    prefix_mode: str  # "char_fraction" or "pre_correction"
    prefix_length_chars: int
    prefix_fraction: float  # fraction of total steered response
    prefill_text: str
    completion: str  # full text (prefill + continuation)
    score: dict
    steered_response: str
    steered_attempts: list
    steered_n_attempts: int
    completion_n_attempts: int


def find_correction_position(response: str) -> int | None:
    """Find the character position where self-correction language first appears."""
    match = CORRECTION_PATTERNS.search(response)
    if match:
        return match.start()
    return None


def load_episodes(
    results_dir: str,
    model_filter: str = "Meta-Llama-3.3-70B",
) -> list[Episode]:
    """Load episodes from existing experiment result files."""
    episodes = []
    results_path = Path(results_dir)

    for f in sorted(results_path.glob("experiment_results_*.json")):
        name = f.name
        if model_filter not in name:
            continue
        # Skip special runs
        if any(x in name for x in [
            "random", "extra", "fresh", "repro", "overlap",
            "multi_boost", "no_steering", "ablation"
        ]):
            continue

        with open(f) as fh:
            data = json.load(fh)

        for feat in data["results_by_feature"]:
            for trial in feat["trials"]:
                if trial.get("error"):
                    continue
                score = trial.get("score", {})
                attempts = score.get("attempts", [])
                if not attempts:
                    continue

                response = trial["response"]
                correction_pos = find_correction_position(response)

                episodes.append(Episode(
                    prompt=trial["prompt"],
                    feature_index=feat["feature_index_in_sae"],
                    feature_label=feat["feature_label"],
                    threshold=trial["threshold"],
                    steered_response=response,
                    steered_attempts=attempts,
                    correction_char_pos=correction_pos,
                ))

    return episodes


async def run_prefill_experiment(
    prefix_fractions: list[float],
    n_episodes: int = 200,
    include_pre_correction: bool = True,
    results_dir: str = "data/experiment_results/claude_haiku_4_5_20251001_judge",
    judge_model: str = "claude-3-5-haiku-20241022",
    model_name: str = "meta-llama/Meta-Llama-3.3-70B-Instruct",
    max_completion_tokens: int = 512,
    output_dir: str = "data/experiment_results/prefill_control",
):
    """
    Run the prefill control experiment.

    Args:
        prefix_fractions: List of fractions (0.0-1.0) of the steered response to use as prefix.
                         e.g. [0.1, 0.25, 0.5] means try 10%, 25%, 50% prefixes.
        n_episodes: Number of episodes to sample.
        include_pre_correction: If True, also run a "pre_correction" condition that uses
                               the prefix right up to the self-correction point.
        results_dir: Directory containing existing experiment results.
        judge_model: Model to use for judging completions.
        model_name: Model to use for unsteered completion.
        max_completion_tokens: Max tokens for completion.
        output_dir: Where to save results.
    """
    # Load episodes
    print("Loading episodes from existing results...")
    all_episodes = load_episodes(results_dir)
    print(f"Loaded {len(all_episodes)} episodes")

    multi_attempt = [e for e in all_episodes if len(e.steered_attempts) >= 2]
    print(f"  Multi-attempt (self-correction): {len(multi_attempt)}")
    print(f"  Single-attempt: {len(all_episodes) - len(multi_attempt)}")

    # Sample episodes
    if n_episodes < len(all_episodes):
        sampled = random.sample(all_episodes, n_episodes)
    else:
        sampled = all_episodes
        print(f"  Using all {len(sampled)} episodes (requested {n_episodes})")

    sampled_multi = sum(1 for e in sampled if len(e.steered_attempts) >= 2)
    print(f"Sampled {len(sampled)} episodes ({sampled_multi} with self-correction)")

    # Build list of (episode, prefix_mode, prefix_fraction) jobs
    jobs: list[tuple[Episode, str, float]] = []
    for ep in sampled:
        for frac in prefix_fractions:
            jobs.append((ep, "char_fraction", frac))
        if include_pre_correction and ep.correction_char_pos is not None:
            # Use the fraction that corresponds to the correction point
            correction_frac = ep.correction_char_pos / len(ep.steered_response)
            jobs.append((ep, "pre_correction", correction_frac))

    print(f"\nTotal jobs: {len(jobs)}")
    print(f"  Per-fraction: {len(sampled)} episodes x {len(prefix_fractions)} fractions = {len(sampled) * len(prefix_fractions)}")
    if include_pre_correction:
        n_with_correction = sum(1 for e in sampled if e.correction_char_pos is not None)
        print(f"  Pre-correction: {n_with_correction} episodes with detectable correction point")

    # Initialize engine (no steering)
    print("\nInitializing vLLM engine (unsteered)...")
    engine = VLLMSteeringEngine(model_name)
    await engine.initialize()
    print("Engine initialized")

    # Initialize judge
    judge = create_judge(judge_model)

    # Process jobs
    os.makedirs(output_dir, exist_ok=True)
    results: list[PrefillTrialResult] = []

    # Group jobs by prefix mode + fraction for progress reporting
    job_groups: dict[str, list[tuple[Episode, str, float]]] = {}
    for ep, mode, frac in jobs:
        key = f"{mode}_{frac:.2f}"
        job_groups.setdefault(key, []).append((ep, mode, frac))

    for group_key, group_jobs in job_groups.items():
        print(f"\n--- Running group: {group_key} ({len(group_jobs)} episodes) ---")
        group_results = []

        # Process in batches to avoid overwhelming the engine
        batch_size = 20
        for batch_start in range(0, len(group_jobs), batch_size):
            batch = group_jobs[batch_start:batch_start + batch_size]

            # Generate completions
            async def process_one(ep: Episode, mode: str, frac: float) -> PrefillTrialResult | None:
                # Compute prefix
                prefix_chars = int(len(ep.steered_response) * frac)
                prefix_chars = max(1, min(prefix_chars, len(ep.steered_response) - 1))
                prefill_text = ep.steered_response[:prefix_chars]

                # Generate completion with unsteered model
                convo = [{"role": "user", "content": ep.prompt}]
                try:
                    completion = await engine.generate_with_conversation(
                        conversation=convo,
                        feature_interventions=None,  # No steering
                        max_tokens=max_completion_tokens,
                        prefill=prefill_text,
                    )
                except Exception as e:
                    print(f"  Generation error: {e}")
                    return None

                # Judge the completion
                try:
                    score = await judge.grade_response(
                        completion, ep.prompt, ep.feature_label
                    )
                except Exception as e:
                    score = {"error": str(e)}

                completion_attempts = score.get("attempts", [])

                return PrefillTrialResult(
                    prompt=ep.prompt,
                    feature_index=ep.feature_index,
                    feature_label=ep.feature_label,
                    threshold=ep.threshold,
                    prefix_mode=mode,
                    prefix_length_chars=prefix_chars,
                    prefix_fraction=frac,
                    prefill_text=prefill_text,
                    completion=completion,
                    score=score,
                    steered_response=ep.steered_response,
                    steered_attempts=ep.steered_attempts,
                    steered_n_attempts=len(ep.steered_attempts),
                    completion_n_attempts=len(completion_attempts),
                )

            batch_results = await asyncio.gather(
                *[process_one(ep, mode, frac) for ep, mode, frac in batch]
            )

            for r in batch_results:
                if r is not None:
                    group_results.append(r)

            done = batch_start + len(batch)
            print(f"  Processed {done}/{len(group_jobs)}")

        results.extend(group_results)

        # Report intermediate results for this group
        if group_results:
            n_multi = sum(1 for r in group_results if r.completion_n_attempts >= 2)
            steered_multi = sum(1 for r in group_results if r.steered_n_attempts >= 2)
            print(f"\n  Results for {group_key}:")
            print(f"    Episodes: {len(group_results)}")
            print(f"    Steered self-correction rate: {steered_multi}/{len(group_results)} ({100*steered_multi/len(group_results):.1f}%)")
            print(f"    Prefill self-correction rate: {n_multi}/{len(group_results)} ({100*n_multi/len(group_results):.1f}%)")

        # Save incremental results after each group
        _save_results(results, output_dir, prefix_fractions, n_episodes)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    _print_summary(results)

    return results


def _save_results(
    results: list[PrefillTrialResult],
    output_dir: str,
    prefix_fractions: list[float],
    n_episodes: int,
):
    """Save results to JSON."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fracs_str = "_".join(f"{f:.0%}" for f in prefix_fractions)

    output = {
        "config": {
            "prefix_fractions": prefix_fractions,
            "n_episodes": n_episodes,
            "timestamp": timestamp,
        },
        "results": [asdict(r) for r in results],
        "summary": _compute_summary(results),
    }

    outfile = os.path.join(output_dir, f"prefill_control_{timestamp}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved results to {outfile}")


def _compute_summary(results: list[PrefillTrialResult]) -> dict:
    """Compute summary statistics grouped by prefix mode + fraction."""
    groups: dict[str, list[PrefillTrialResult]] = {}
    for r in results:
        key = f"{r.prefix_mode}_{r.prefix_fraction:.2f}"
        groups.setdefault(key, []).append(r)

    summary = {}
    for key, group in sorted(groups.items()):
        n = len(group)
        steered_multi = sum(1 for r in group if r.steered_n_attempts >= 2)
        prefill_multi = sum(1 for r in group if r.completion_n_attempts >= 2)

        # Average scores
        steered_first_scores = []
        prefill_first_scores = []
        for r in group:
            if r.steered_attempts:
                steered_first_scores.append(r.steered_attempts[0]["score"])
            if r.score.get("attempts"):
                prefill_first_scores.append(r.score["attempts"][0]["score"])

        summary[key] = {
            "n_episodes": n,
            "prefix_mode": group[0].prefix_mode,
            "prefix_fraction": group[0].prefix_fraction,
            "steered_self_correction_rate": steered_multi / n if n else 0,
            "prefill_self_correction_rate": prefill_multi / n if n else 0,
            "steered_multi_attempt_count": steered_multi,
            "prefill_multi_attempt_count": prefill_multi,
            "mean_steered_first_score": (
                sum(steered_first_scores) / len(steered_first_scores)
                if steered_first_scores else None
            ),
            "mean_prefill_first_score": (
                sum(prefill_first_scores) / len(prefill_first_scores)
                if prefill_first_scores else None
            ),
        }

    return summary


def _print_summary(results: list[PrefillTrialResult]):
    """Print a formatted summary table."""
    summary = _compute_summary(results)

    print(f"{'Condition':<25} {'N':>5} {'Steered SC':>12} {'Prefill SC':>12} {'Steered 1st':>12} {'Prefill 1st':>12}")
    print("-" * 80)

    for key, stats in sorted(summary.items()):
        mode = stats["prefix_mode"]
        frac = stats["prefix_fraction"]
        if mode == "char_fraction":
            label = f"{frac:.0%} prefix"
        else:
            label = "pre-correction"

        steered_sc = f"{stats['steered_self_correction_rate']:.1%}"
        prefill_sc = f"{stats['prefill_self_correction_rate']:.1%}"
        steered_1st = f"{stats['mean_steered_first_score']:.1f}" if stats["mean_steered_first_score"] is not None else "N/A"
        prefill_1st = f"{stats['mean_prefill_first_score']:.1f}" if stats["mean_prefill_first_score"] is not None else "N/A"

        print(f"{label:<25} {stats['n_episodes']:>5} {steered_sc:>12} {prefill_sc:>12} {steered_1st:>12} {prefill_1st:>12}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prefill control experiment")
    parser.add_argument(
        "--prefix-fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help="Fractions of steered response to use as prefix (default: 0.1 0.25 0.5)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=200,
        help="Number of episodes to sample (default: 200)",
    )
    parser.add_argument(
        "--no-pre-correction", action="store_true",
        help="Skip the pre-correction prefix condition",
    )
    parser.add_argument(
        "--results-dir",
        default="data/experiment_results/claude_haiku_4_5_20251001_judge",
        help="Directory containing existing experiment results",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-3-5-haiku-20241022",
        help="Judge model (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3.3-70B-Instruct",
        help="Model for unsteered completion",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max completion tokens (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiment_results/prefill_control",
        help="Output directory",
    )

    args = parser.parse_args()

    await run_prefill_experiment(
        prefix_fractions=args.prefix_fractions,
        n_episodes=args.n_episodes,
        include_pre_correction=not args.no_pre_correction,
        results_dir=args.results_dir,
        judge_model=args.judge_model,
        model_name=args.model_name,
        max_completion_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )
