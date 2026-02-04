#!/usr/bin/env python3
"""
Extract all concrete numbers from the ESR paper data.

This script computes all statistics referenced in paper_numbers.md from raw experimental data.
It outputs a structured report that can be compared against the paper for verification.

Usage:
    python plotting/extract_paper_numbers.py [--output-json paper_numbers_computed.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_file_utils import (
    parse_results_filename,
    canonicalize_model_name,
    CanonicalModelInfo,
    ModelFamily,
)
from plotting.plot_utils import (
    is_degraded_output,
    should_ignore_file,
    collect_experiment_1_result_files,
)

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "data" / "experiment_results"
HAIKU_RESULTS_DIR = RESULTS_DIR / "claude_haiku_4_5_20251001_judge"
ACTIVATION_STATS_DIR = HAIKU_RESULTS_DIR / "activation_stats"
RANDOM_CONTROL_DIR = HAIKU_RESULTS_DIR / "random_latent_control"


# ============================================================================
# Data Extraction Utilities
# ============================================================================

def extract_trial_metrics(results: dict, exclude_degraded: bool = False) -> dict:
    """Extract trial-level metrics from experiment results."""
    first_scores = []
    score_deltas = []
    attempt_counts = []
    improved_count = 0
    multi_attempt_count = 0
    total_trials = 0

    for feature_result in results.get("results_by_feature", []):
        if feature_result.get("error"):
            continue

        for trial in feature_result.get("trials", []):
            if trial.get("error"):
                continue

            if exclude_degraded:
                response = trial.get("response", "")
                if is_degraded_output(response):
                    continue

            score_dict = trial.get("score", {})
            attempts = score_dict.get("attempts", [])

            if not attempts:
                continue

            scores = [a.get("score") for a in attempts if a.get("score") is not None]
            if not scores:
                continue

            total_trials += 1
            first_scores.append(scores[0])
            attempt_counts.append(len(scores))

            if len(scores) > 1:
                multi_attempt_count += 1
                delta = scores[-1] - scores[0]
                score_deltas.append(delta)
                if scores[-1] > scores[0]:
                    improved_count += 1

    return {
        "total_trials": total_trials,
        "multi_attempt_count": multi_attempt_count,
        "improved_count": improved_count,
        "first_scores": first_scores,
        "score_deltas": score_deltas,
        "attempt_counts": attempt_counts,
    }


def compute_statistics(metrics: dict) -> dict:
    """Compute summary statistics from trial metrics."""
    total = metrics["total_trials"]
    multi = metrics["multi_attempt_count"]
    improved = metrics["improved_count"]
    first_scores = metrics["first_scores"]
    score_deltas = metrics["score_deltas"]

    if total == 0:
        return {
            "n_trials": 0,
            "multi_attempt_pct": 0,
            "multi_attempt_pct_se": 0,
            "esr_rate": 0,
            "esr_rate_se": 0,
            "improvement_rate": 0,
            "improvement_rate_se": 0,
            "mean_first_score": 0,
            "mean_first_score_se": 0,
            "mean_score_improvement": 0,
            "mean_score_improvement_se": 0,
        }

    # Multi-attempt percentage
    multi_pct = multi / total * 100
    p_multi = multi_pct / 100
    multi_pct_se = np.sqrt(p_multi * (1 - p_multi) / total) * 100

    # ESR rate (improved multi-attempt / total)
    esr_rate = improved / total * 100
    p_esr = esr_rate / 100
    esr_rate_se = np.sqrt(p_esr * (1 - p_esr) / total) * 100

    # Improvement rate (improved / multi-attempt)
    if multi > 0:
        improvement_rate = improved / multi * 100
        p_imp = improvement_rate / 100
        improvement_rate_se = np.sqrt(p_imp * (1 - p_imp) / multi) * 100
    else:
        improvement_rate = 0
        improvement_rate_se = 0

    # Mean first score
    if first_scores:
        mean_first = float(np.mean(first_scores))
        mean_first_se = float(np.std(first_scores, ddof=1) / np.sqrt(len(first_scores))) if len(first_scores) > 1 else 0
    else:
        mean_first = 0
        mean_first_se = 0

    # Mean score improvement (multi-attempt only)
    if score_deltas:
        mean_improvement = float(np.mean(score_deltas))
        mean_improvement_se = float(np.std(score_deltas, ddof=1) / np.sqrt(len(score_deltas))) if len(score_deltas) > 1 else 0
    else:
        mean_improvement = 0
        mean_improvement_se = 0

    return {
        "n_trials": total,
        "n_multi_attempt": multi,
        "n_improved": improved,
        "multi_attempt_pct": round(multi_pct, 2),
        "multi_attempt_pct_se": round(multi_pct_se, 3),
        "esr_rate": round(esr_rate, 2),
        "esr_rate_se": round(esr_rate_se, 3),
        "improvement_rate": round(improvement_rate, 1),
        "improvement_rate_se": round(improvement_rate_se, 2),
        "mean_first_score": round(mean_first, 1),
        "mean_first_score_se": round(mean_first_se, 2),
        "mean_score_improvement": round(mean_improvement, 2),
        "mean_score_improvement_se": round(mean_improvement_se, 2),
    }


# ============================================================================
# Experiment-Specific Extraction Functions
# ============================================================================

def extract_experiment_1_numbers() -> dict:
    """Extract ESR rates across models (Experiment 1)."""
    print("\n--- Experiment 1: ESR Across Models ---")

    selected_files, model_info_map, model_files = collect_experiment_1_result_files(
        BASE_DIR,
        excluded_families={ModelFamily.FINETUNED_8B},
        haiku_only=True,
    )

    results = {}
    for model_name, files in model_files.items():
        all_metrics = {
            "total_trials": 0,
            "multi_attempt_count": 0,
            "improved_count": 0,
            "first_scores": [],
            "score_deltas": [],
            "attempt_counts": [],
        }

        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            metrics = extract_trial_metrics(data)

            all_metrics["total_trials"] += metrics["total_trials"]
            all_metrics["multi_attempt_count"] += metrics["multi_attempt_count"]
            all_metrics["improved_count"] += metrics["improved_count"]
            all_metrics["first_scores"].extend(metrics["first_scores"])
            all_metrics["score_deltas"].extend(metrics["score_deltas"])
            all_metrics["attempt_counts"].extend(metrics["attempt_counts"])

        stats = compute_statistics(all_metrics)
        results[model_name] = stats
        print(f"  {model_name}: n={stats['n_trials']}, ESR={stats['esr_rate']:.1f}%, Multi-att={stats['multi_attempt_pct']:.1f}%")

    return results


def extract_experiment_2_numbers() -> dict:
    """Extract boost level sweep results (Experiment 2)."""
    print("\n--- Experiment 2: Boost Level Sweep ---")

    multi_boost_files = list(HAIKU_RESULTS_DIR.glob("experiment_multi_boost_*.json"))

    if not multi_boost_files:
        print("  Warning: No multi-boost files found")
        return {}

    results_by_model: dict[str, dict] = {}

    for f in multi_boost_files:
        with open(f) as fp:
            data = json.load(fp)

        config = data.get("experiment_config", {}) or {}
        model_name = config.get("model_name", "unknown")
        model_info = canonicalize_model_name(model_name)

        boost_levels = data.get("boost_levels", [])
        threshold_stats = data.get("threshold_cache_stats", {})

        # Count total trials and boost levels
        total_trials = 0
        multi_attempt_trials = 0
        improved_trials = 0

        for feature_result in data.get("results_by_feature", []):
            if feature_result.get("error"):
                continue

            for trial in feature_result.get("trials", []):
                if trial.get("error"):
                    continue

                score_dict = trial.get("score", {})
                attempts = score_dict.get("attempts", [])

                if not attempts:
                    continue

                scores = [a.get("score") for a in attempts if a.get("score") is not None]
                if not scores:
                    continue

                total_trials += 1
                if len(scores) > 1:
                    multi_attempt_trials += 1
                    if scores[-1] > scores[0]:
                        improved_trials += 1

        if model_info.display_name not in results_by_model:
            results_by_model[model_info.display_name] = {
                "n_boost_levels": len(boost_levels),
                "total_trials": 0,
                "multi_attempt_trials": 0,
                "improved_trials": 0,
                "threshold_mean": threshold_stats.get("mean", 0),
                "threshold_std": threshold_stats.get("std", 0),
            }

        results_by_model[model_info.display_name]["total_trials"] += total_trials
        results_by_model[model_info.display_name]["multi_attempt_trials"] += multi_attempt_trials
        results_by_model[model_info.display_name]["improved_trials"] += improved_trials

    for model_name, stats in results_by_model.items():
        print(f"  {model_name}: {stats['n_boost_levels']} boost levels, {stats['total_trials']} trials")

    return results_by_model


def extract_experiment_3_numbers(exp1_results: dict | None = None) -> dict:
    """Extract OTD ablation results (Experiment 3)."""
    print("\n--- Experiment 3: OTD Ablation ---")

    # Target latent counts for valid ablation files
    TARGET_LATENT_COUNTS = {22, 25, 26}

    # Excluded files
    EXCLUDED_FILES = {
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_165238_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_155058_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251229_163636.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251229_171444.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251230_131048_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_180833_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260120_125251_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260120_133023_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260120_134938_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260121_172308_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260121_172840_with_ablation.json",
    }

    # Find ablation files
    ablation_files = list(HAIKU_RESULTS_DIR.glob("*70B*with_ablation*.json"))
    valid_ablation_files = []
    ablated_latents = []

    for f in ablation_files:
        if f.name in EXCLUDED_FILES:
            continue
        with open(f) as fp:
            data = json.load(fp)
        num_ablated = len(data.get("ablated_latents", []))
        if num_ablated in TARGET_LATENT_COUNTS:
            valid_ablation_files.append(f)
            ablated_latents = data.get("ablated_latents", [])

    # Aggregate ablation metrics
    ablation_all = {
        "total_trials": 0,
        "multi_attempt_count": 0,
        "improved_count": 0,
        "first_scores": [],
        "score_deltas": [],
        "attempt_counts": [],
    }

    for f in valid_ablation_files:
        with open(f) as fp:
            data = json.load(fp)
        metrics = extract_trial_metrics(data)
        ablation_all["total_trials"] += metrics["total_trials"]
        ablation_all["multi_attempt_count"] += metrics["multi_attempt_count"]
        ablation_all["improved_count"] += metrics["improved_count"]
        ablation_all["first_scores"].extend(metrics["first_scores"])
        ablation_all["score_deltas"].extend(metrics["score_deltas"])

    ablation_stats = compute_statistics(ablation_all)

    # Get baseline (exp1 Llama 70B)
    if exp1_results is None:
        exp1_results = extract_experiment_1_numbers()
    baseline_stats = exp1_results.get("Llama 3.3 70B", {})

    # Calculate reductions
    baseline_multi = baseline_stats.get("multi_attempt_pct", 0)
    ablation_multi = ablation_stats["multi_attempt_pct"]
    multi_reduction = (baseline_multi - ablation_multi) / baseline_multi * 100 if baseline_multi > 0 else 0

    baseline_esr = baseline_stats.get("esr_rate", 0)
    ablation_esr = ablation_stats["esr_rate"]
    esr_reduction = (baseline_esr - ablation_esr) / baseline_esr * 100 if baseline_esr > 0 else 0

    results = {
        "n_otd_latents": len(ablated_latents),
        "otd_latent_indices": ablated_latents,
        "baseline": baseline_stats,
        "ablation": ablation_stats,
        "multi_attempt_reduction_pct": round(multi_reduction, 1),
        "esr_reduction_pct": round(esr_reduction, 1),
    }

    print(f"  OTD latents ablated: {results['n_otd_latents']}")
    print(f"  Baseline: n={baseline_stats.get('n_trials', 0)}, ESR={baseline_esr:.1f}%, Multi={baseline_multi:.1f}%")
    print(f"  Ablation: n={ablation_stats['n_trials']}, ESR={ablation_esr:.1f}%, Multi={ablation_multi:.1f}%")
    print(f"  Multi-attempt reduction: {multi_reduction:.1f}%")
    print(f"  ESR reduction: {esr_reduction:.1f}%")

    return results


def extract_experiment_5_numbers(exp1_results: dict | None = None) -> dict:
    """Extract meta-prompting results (Experiment 5).

    The baseline comes from experiment 1 results, not exp5 files.
    """
    print("\n--- Experiment 5: Meta-Prompting ---")

    # Find experiment 5 files
    exp5_files = list(HAIKU_RESULTS_DIR.glob("experiment_5_prompt_variants_*.json"))

    results_by_model: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "total_trials": 0,
        "multi_attempt_count": 0,
        "improved_count": 0,
        "first_scores": [],
        "score_deltas": [],
        "attempt_counts": [],
    }))

    for f in exp5_files:
        with open(f) as fp:
            data = json.load(fp)

        config = data.get("experiment_config", {}) or {}
        model_name = config.get("model_name", "unknown")
        variant_id = config.get("prompt_variant_id", "unknown")

        metrics = extract_trial_metrics(data)
        model_info = canonicalize_model_name(model_name)

        # Aggregate metrics for the same model+variant combo
        agg = results_by_model[model_info.display_name][variant_id]
        agg["total_trials"] += metrics["total_trials"]
        agg["multi_attempt_count"] += metrics["multi_attempt_count"]
        agg["improved_count"] += metrics["improved_count"]
        agg["first_scores"].extend(metrics["first_scores"])
        agg["score_deltas"].extend(metrics["score_deltas"])
        agg["attempt_counts"].extend(metrics["attempt_counts"])

    # Convert aggregated metrics to statistics
    stats_by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for model_name, variants in results_by_model.items():
        for variant_id, metrics in variants.items():
            stats_by_model[model_name][variant_id] = compute_statistics(metrics)

    # Get baseline from experiment 1 for Llama 70B
    if exp1_results is None:
        exp1_results = extract_experiment_1_numbers()

    baseline_stats = exp1_results.get("Llama 3.3 70B", {})
    self_monitor_stats = stats_by_model.get("Llama 3.3 70B", {}).get("self_monitor", {})

    if baseline_stats and self_monitor_stats:
        baseline_multi = baseline_stats.get("multi_attempt_pct", 0)
        monitor_multi = self_monitor_stats.get("multi_attempt_pct", 0)
        multi_increase = monitor_multi / baseline_multi if baseline_multi > 0 else 0

        baseline_esr = baseline_stats.get("esr_rate", 0)
        monitor_esr = self_monitor_stats.get("esr_rate", 0)
        esr_increase = monitor_esr / baseline_esr if baseline_esr > 0 else 0

        print(f"  Llama 3.3 70B:")
        print(f"    Baseline (from exp1): Multi={baseline_multi:.1f}%, ESR={baseline_esr:.1f}%")
        print(f"    Self-monitor: n={self_monitor_stats.get('n_trials', 0)}, Multi={monitor_multi:.1f}%, ESR={monitor_esr:.1f}%")
        print(f"    Multi-attempt increase: {multi_increase:.1f}x ({baseline_multi:.1f}% → {monitor_multi:.1f}%)")
        print(f"    ESR increase: {esr_increase:.1f}x ({baseline_esr:.1f}% → {monitor_esr:.1f}%)")

        return {
            "llama_70b_baseline_multi_pct": baseline_multi,
            "llama_70b_self_monitor_multi_pct": monitor_multi,
            "llama_70b_multi_increase_factor": round(multi_increase, 1),
            "llama_70b_baseline_esr": baseline_esr,
            "llama_70b_self_monitor_esr": monitor_esr,
            "llama_70b_esr_increase_factor": round(esr_increase, 1),
            "all_models": dict(stats_by_model),
        }

    return {"all_models": dict(stats_by_model)}


def extract_experiment_8_numbers() -> dict:
    """Extract no-steering baseline results (Experiment 8)."""
    print("\n--- Experiment 8: No-Steering Baseline ---")

    baseline_files = list(HAIKU_RESULTS_DIR.glob("*_no_steering_baseline.json"))

    all_metrics = {
        "total_trials": 0,
        "multi_attempt_count": 0,
        "improved_count": 0,
        "first_scores": [],
        "score_deltas": [],
        "attempt_counts": [],
    }

    for f in baseline_files:
        with open(f) as fp:
            data = json.load(fp)
        metrics = extract_trial_metrics(data)
        all_metrics["total_trials"] += metrics["total_trials"]
        all_metrics["multi_attempt_count"] += metrics["multi_attempt_count"]
        all_metrics["improved_count"] += metrics["improved_count"]
        all_metrics["first_scores"].extend(metrics["first_scores"])
        all_metrics["score_deltas"].extend(metrics["score_deltas"])

    stats = compute_statistics(all_metrics)

    print(f"  Total trials: {stats['n_trials']}")
    print(f"  Multi-attempt: {stats['multi_attempt_pct']:.2f}%")
    print(f"  Mean first-attempt score: {stats['mean_first_score']:.1f}")

    return stats


def extract_experiment_9_numbers() -> dict:
    """Extract activation statistics (Experiment 9)."""
    print("\n--- Experiment 9: Activation Statistics ---")

    results_file = ACTIVATION_STATS_DIR / "analysis_results.json"
    baseline_file = ACTIVATION_STATS_DIR / "baseline_stats.json"
    metadata_file = ACTIVATION_STATS_DIR / "episode_metadata.json"

    if not results_file.exists():
        print("  Warning: Activation stats not found")
        return {}

    with open(results_file) as f:
        results = json.load(f)

    with open(baseline_file) as f:
        baseline = json.load(f)

    with open(metadata_file) as f:
        metadata = json.load(f)

    otd = results.get("otd_group", {})
    n_episodes = results.get("n_episodes", 0)
    n_baseline_episodes = baseline.get("n_episodes", 0)

    # Calculate ratios
    baseline_mean = baseline.get("otd_mean", 0)
    off_topic_mean = otd.get("mean_off_topic", 0)
    on_topic_mean = otd.get("mean_on_topic", 0)

    ratio_off = off_topic_mean / baseline_mean if baseline_mean > 0 else 0
    ratio_on = on_topic_mean / baseline_mean if baseline_mean > 0 else 0

    print(f"  Self-correction episodes: {n_episodes}")
    print(f"  Baseline episodes: {n_baseline_episodes}")
    print(f"  OTD activation ratio (off-topic vs baseline): {ratio_off:.1f}x")
    print(f"  OTD activation ratio (on-topic vs baseline): {ratio_on:.1f}x")

    return {
        "n_self_correction_episodes": n_episodes,
        "n_baseline_episodes": n_baseline_episodes,
        "otd_off_topic_mean": round(off_topic_mean, 4),
        "otd_on_topic_mean": round(on_topic_mean, 4),
        "baseline_otd_mean": round(baseline_mean, 4),
        "otd_ratio_off_topic_vs_baseline": round(ratio_off, 1),
        "otd_ratio_on_topic_vs_baseline": round(ratio_on, 1),
    }


def extract_experiment_10_numbers() -> dict:
    """Extract random ablation control results (Experiment 10)."""
    print("\n--- Experiment 10: Random Ablation Control ---")

    random_files = list(RANDOM_CONTROL_DIR.glob("*_random_ablation_*.json"))

    if not random_files:
        print("  Warning: Random ablation results not found")
        return {}

    all_metrics = {
        "total_trials": 0,
        "multi_attempt_count": 0,
        "improved_count": 0,
        "first_scores": [],
        "score_deltas": [],
        "attempt_counts": [],
    }

    for f in random_files:
        with open(f) as fp:
            data = json.load(fp)
        metrics = extract_trial_metrics(data)
        all_metrics["total_trials"] += metrics["total_trials"]
        all_metrics["multi_attempt_count"] += metrics["multi_attempt_count"]
        all_metrics["improved_count"] += metrics["improved_count"]
        all_metrics["first_scores"].extend(metrics["first_scores"])
        all_metrics["score_deltas"].extend(metrics["score_deltas"])

    stats = compute_statistics(all_metrics)

    print(f"  Random ablation trials: {stats['n_trials']}")
    print(f"  Multi-attempt: {stats['multi_attempt_pct']:.1f}%")
    print(f"  ESR rate: {stats['esr_rate']:.1f}%")

    return {"random_ablation": stats}


def extract_experiment_4_numbers() -> dict:
    """Extract fine-tuning results (Experiment 4)."""
    print("\n--- Experiment 4: Fine-tuning ---")

    MASKED_RATIO_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    results = {}

    # Find base 8B file
    base_files = sorted(HAIKU_RESULTS_DIR.glob("experiment_results_Meta-Llama-3.1-8B-Instruct_*.json"))
    base_files = [f for f in base_files if "fresh_prompts" not in f.name and "no_steering" not in f.name]

    if base_files:
        with open(base_files[-1]) as fp:
            data = json.load(fp)
        metrics = extract_trial_metrics(data)
        stats = compute_statistics(metrics)
        results["base"] = stats
        print(f"  Base: n={stats['n_trials']}, Multi={stats['multi_attempt_pct']:.1f}%, Improvement={stats['improvement_rate']:.1f}%")

    # Find masked-ratio files
    for pct in MASKED_RATIO_PCTS:
        pattern = f"experiment_results_masked-ratio-{pct}pct-merged_*.json"
        files = sorted(HAIKU_RESULTS_DIR.glob(pattern))
        if files:
            with open(files[-1]) as fp:
                data = json.load(fp)
            metrics = extract_trial_metrics(data)
            stats = compute_statistics(metrics)
            results[f"{pct}pct"] = stats
            print(f"  {pct}%: n={stats['n_trials']}, Multi={stats['multi_attempt_pct']:.1f}%, Improvement={stats['improvement_rate']:.1f}%")

    return results


def extract_model_info() -> dict:
    """Extract model and layer information."""
    print("\n--- Model Information ---")

    models = {
        "Llama 3.3 70B": {
            "layer": 33,
            "depth_pct": 41.3,
            "sae": "Goodfire",
            "total_layers": 80,
            "sae_trained_layer": 50,
        },
        "Llama 3.1 8B": {
            "layer": 19,
            "depth_pct": 59.4,
            "sae": "Goodfire",
        },
        "Gemma 2 2B": {
            "layer": 16,
            "depth_pct": 61.5,
            "sae": "GemmaScope",
        },
        "Gemma 2 9B": {
            "layer": 26,
            "depth_pct": 61.9,
            "sae": "GemmaScope",
        },
        "Gemma 2 27B": {
            "layer": 22,
            "depth_pct": 47.8,
            "sae": "GemmaScope",
            "total_layers": 46,
        },
    }

    return models


def extract_prompt_info() -> dict:
    """Count prompts used in experiments."""
    print("\n--- Prompt Information ---")

    # Read from the prompts file if it exists
    prompts_file = BASE_DIR / "prompts.json"
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
        n_prompts = len(prompts)
    else:
        n_prompts = 38  # Default from paper

    print(f"  Number of prompts: {n_prompts}")
    return {"n_prompts": n_prompts}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract paper numbers from ESR experiment data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save paper_numbers_computed.json (relative paths resolved from experiment base dir). Default: plots/",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else (BASE_DIR / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXTRACTING PAPER NUMBERS FROM ESR DATA")
    print("=" * 70)

    all_numbers = {}

    # Extract numbers from each experiment
    all_numbers["models"] = extract_model_info()
    all_numbers["prompts"] = extract_prompt_info()

    # Experiment 1 is used as baseline for experiments 3 and 5, so extract once
    exp1_results = extract_experiment_1_numbers()
    all_numbers["experiment_1_esr_across_models"] = exp1_results

    all_numbers["experiment_2_boost_sweep"] = extract_experiment_2_numbers()
    all_numbers["experiment_3_otd_ablation"] = extract_experiment_3_numbers(exp1_results)
    all_numbers["experiment_4_finetuning"] = extract_experiment_4_numbers()
    all_numbers["experiment_5_meta_prompting"] = extract_experiment_5_numbers(exp1_results)
    all_numbers["experiment_8_no_steering_baseline"] = extract_experiment_8_numbers()
    all_numbers["experiment_9_activation_stats"] = extract_experiment_9_numbers()
    all_numbers["experiment_10_random_ablation"] = extract_experiment_10_numbers()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY NUMBERS")
    print("=" * 70)

    exp1 = all_numbers["experiment_1_esr_across_models"]
    llama70b = exp1.get("Llama 3.3 70B", {})
    print(f"\n5 models tested from Llama-3 and Gemma-2 families")
    print(f"Llama 3.3 70B ESR rate: {llama70b.get('esr_rate', 0):.1f}%")
    print(f"Llama 3.3 70B multi-attempt rate: {llama70b.get('multi_attempt_pct', 0):.1f}%")
    print(f"Llama 3.3 70B n_trials: {llama70b.get('n_trials', 0)}")

    exp3 = all_numbers["experiment_3_otd_ablation"]
    print(f"\nOTD latents identified: {exp3.get('n_otd_latents', 0)}")
    print(f"Multi-attempt reduction from ablation: {exp3.get('multi_attempt_reduction_pct', 0):.0f}%")
    print(f"ESR reduction from ablation: {exp3.get('esr_reduction_pct', 0):.0f}%")

    exp5 = all_numbers["experiment_5_meta_prompting"]
    print(f"\nMeta-prompting multi-attempt increase: {exp5.get('llama_70b_multi_increase_factor', 0):.1f}x")
    print(f"Meta-prompting: {exp5.get('llama_70b_baseline_multi_pct', 0):.1f}% → {exp5.get('llama_70b_self_monitor_multi_pct', 0):.1f}%")

    exp8 = all_numbers["experiment_8_no_steering_baseline"]
    print(f"\nNo-steering baseline trials: {exp8.get('n_trials', 0)}")
    print(f"No-steering multi-attempt rate: {exp8.get('multi_attempt_pct', 0):.2f}%")

    exp9 = all_numbers["experiment_9_activation_stats"]
    print(f"\nSelf-correction episodes analyzed: {exp9.get('n_self_correction_episodes', 0)}")
    print(f"OTD activation ratio (off-topic vs baseline): {exp9.get('otd_ratio_off_topic_vs_baseline', 0):.1f}x")

    # Save to JSON
    output_path = output_dir / "paper_numbers_computed.json"
    with open(output_path, "w") as f:
        json.dump(all_numbers, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
