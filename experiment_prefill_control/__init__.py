"""
Prefill control experiment: does self-correction happen when the unsteered model
completes a prefix from a steered response?

Approach:
1. Load existing experiment results (steered Llama 70B responses)
2. Sample episodes (prompt + latent pairs)
3. For each episode, take a prefix of N characters from the steered response
4. Feed the prefix to the unsteered model as a prefill and let it complete
5. Judge the full completion (prefill + continuation) with the same judge
6. Compare self-correction rates between steered and prefill-completed responses

Prefix modes:
- "num_chars": Take first N characters of the steered response
- "pre_correction": Take everything up to (but not including) the self-correction point
"""

import asyncio
import json
import os
import random
import re
import statistics
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
    prefix_mode: str  # "num_chars" or "pre_correction"
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


def compute_mean_correction_position(episodes: list[Episode]) -> int:
    """Compute the mean character position of self-correction across multi-attempt episodes."""
    positions = [
        e.correction_char_pos
        for e in episodes
        if e.correction_char_pos is not None and len(e.steered_attempts) >= 2
    ]
    if not positions:
        raise ValueError("No episodes with detectable self-correction found")
    mean_pos = int(statistics.mean(positions))
    print(f"  Self-correction position stats (n={len(positions)}):")
    print(f"    Mean: {mean_pos} chars")
    print(f"    Median: {int(statistics.median(positions))} chars")
    print(f"    Stdev: {int(statistics.stdev(positions))} chars")
    print(f"    Range: {min(positions)}-{max(positions)} chars")
    return mean_pos


async def run_prefill_experiment(
    prefix_lengths: list[int],
    n_episodes: int = 200,
    include_pre_correction: bool = False,
    results_dir: str = "data/experiment_results/claude_haiku_4_5_20251001_judge",
    judge_model: str = "claude-haiku-4-5-20251001",
    model_name: str = "meta-llama/Meta-Llama-3.3-70B-Instruct",
    max_completion_tokens: int = 512,
    output_dir: str = "data/experiment_results/prefill_control",
):
    """
    Run the prefill control experiment.

    Args:
        prefix_lengths: List of character counts to use as prefix lengths.
                       e.g. [500, 1000, 1500] means try 500, 1000, 1500 char prefixes.
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

    # Report correction position stats
    compute_mean_correction_position(all_episodes)

    # Sample episodes
    if n_episodes < len(all_episodes):
        sampled = random.sample(all_episodes, n_episodes)
    else:
        sampled = all_episodes
        print(f"  Using all {len(sampled)} episodes (requested {n_episodes})")

    sampled_multi = sum(1 for e in sampled if len(e.steered_attempts) >= 2)
    print(f"Sampled {len(sampled)} episodes ({sampled_multi} with self-correction)")

    # Build list of (episode, prefix_mode, prefix_length) jobs
    jobs: list[tuple[Episode, str, int]] = []
    for ep in sampled:
        for length in prefix_lengths:
            # Clamp to response length - 1
            clamped = min(length, len(ep.steered_response) - 1)
            if clamped > 0:
                jobs.append((ep, "num_chars", clamped))
        if include_pre_correction and ep.correction_char_pos is not None:
            jobs.append((ep, "pre_correction", ep.correction_char_pos))

    print(f"\nTotal jobs: {len(jobs)}")
    print(f"  Prefix lengths: {prefix_lengths}")
    print(f"  Per-length: ~{len(sampled)} episodes x {len(prefix_lengths)} lengths")
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

    # Group jobs by prefix mode + length for progress reporting
    job_groups: dict[str, list[tuple[Episode, str, int]]] = {}
    for ep, mode, length in jobs:
        key = f"{mode}_{length}"
        job_groups.setdefault(key, []).append((ep, mode, length))

    for group_key, group_jobs in job_groups.items():
        print(f"\n--- Running group: {group_key} ({len(group_jobs)} episodes) ---")
        group_results = []

        # Process in batches to avoid overwhelming the engine
        batch_size = 20
        for batch_start in range(0, len(group_jobs), batch_size):
            batch = group_jobs[batch_start:batch_start + batch_size]

            async def process_one(ep: Episode, mode: str, length: int) -> PrefillTrialResult | None:
                prefill_text = ep.steered_response[:length]

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
                    prefix_length_chars=length,
                    prefix_fraction=length / len(ep.steered_response),
                    prefill_text=prefill_text,
                    completion=completion,
                    score=score,
                    steered_response=ep.steered_response,
                    steered_attempts=ep.steered_attempts,
                    steered_n_attempts=len(ep.steered_attempts),
                    completion_n_attempts=len(completion_attempts),
                )

            batch_results = await asyncio.gather(
                *[process_one(ep, mode, length) for ep, mode, length in batch]
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
        _save_results(results, output_dir, prefix_lengths, n_episodes)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    _print_summary(results)

    return results


def _save_results(
    results: list[PrefillTrialResult],
    output_dir: str,
    prefix_lengths: list[int],
    n_episodes: int,
):
    """Save results to JSON."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output = {
        "config": {
            "prefix_lengths": prefix_lengths,
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
    """Compute summary statistics grouped by prefix mode + length."""
    groups: dict[str, list[PrefillTrialResult]] = {}
    for r in results:
        key = f"{r.prefix_mode}_{r.prefix_length_chars}"
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
            "prefix_length_chars": group[0].prefix_length_chars,
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
        length = stats["prefix_length_chars"]
        if mode == "num_chars":
            label = f"{length} chars"
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
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=[1100],
        help="Number of characters to use as prefix (default: 1100, the mean correction position)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=200,
        help="Number of episodes to sample (default: 200)",
    )
    parser.add_argument(
        "--pre-correction", action="store_true",
        help="Include the pre-correction prefix condition",
    )
    parser.add_argument(
        "--results-dir",
        default="data/experiment_results/claude_haiku_4_5_20251001_judge",
        help="Directory containing existing experiment results",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5-20251001",
        help="Judge model (default: claude-haiku-4-5-20251001)",
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
        prefix_lengths=args.prefix_lengths,
        n_episodes=args.n_episodes,
        include_pre_correction=args.pre_correction,
        results_dir=args.results_dir,
        judge_model=args.judge_model,
        model_name=args.model_name,
        max_completion_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )
