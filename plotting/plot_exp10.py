#!/usr/bin/env python3
"""
Experiment 10: Random Latent Ablation Control Plots

Creates comparison plot showing ESR effects across:
1. Steered baseline (no ablation)
2. Detector ablation (OTD latents ablated)
3. Random ablation (matched random latents ablated)

Results are loaded from data/experiment_results/claude_haiku_4_5_20251001_judge/random_latent_control/

Adapted from AGI-1652-random-latent-ablation-control/create_plot.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent
HAIKU_RESULTS_DIR = BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge"
RANDOM_CONTROL_DIR = HAIKU_RESULTS_DIR / "random_latent_control"


def is_degraded_output(response: str, min_repeats: int = 5) -> bool:
    """Check if a response is degraded (contains repetitive patterns)."""
    words = response.split()
    if len(words) < min_repeats:
        return False

    max_repeat = 1
    current_repeat = 1

    for i in range(1, len(words)):
        if words[i] == words[i - 1] and len(words[i]) > 1:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1

    return max_repeat >= min_repeats


def extract_metrics(results: Dict, exclude_degraded: bool = False) -> Dict:
    """
    Extract metrics from experiment results.

    Returns dict with:
    - total_trials, multi_attempt_trials, improved_multi_attempts
    - pct_multi_attempt, pct_multi_attempt_se
    - pct_improved, pct_improved_se
    - mean_improvement, mean_improvement_se (multi-attempt trials only)
    - first_attempt_mean, first_attempt_se
    """
    multi_attempt_improvements = []
    first_scores = []
    total_trials = 0
    improved_multi_attempts = 0

    for feature_result in results.get("results_by_feature", []):
        if feature_result.get("error"):
            continue

        for trial in feature_result.get("trials", []):
            if trial.get("error"):
                continue

            # Skip degraded outputs if requested
            if exclude_degraded:
                response = trial.get("response", "")
                if is_degraded_output(response):
                    continue

            score_data = trial.get("score", {})
            attempts = score_data.get("attempts", [])

            if not attempts:
                continue

            total_trials += 1
            first_score = attempts[0].get("score", 0)
            first_scores.append(first_score)

            if len(attempts) >= 2:
                last_score = attempts[-1].get("score", 0)
                improvement = last_score - first_score
                multi_attempt_improvements.append(improvement)
                if last_score > first_score:
                    improved_multi_attempts += 1

    multi_attempt_trials = len(multi_attempt_improvements)

    if total_trials == 0:
        return {
            "total_trials": 0,
            "multi_attempt_trials": 0,
            "improved_multi_attempts": 0,
            "pct_multi_attempt": 0,
            "pct_multi_attempt_se": 0,
            "pct_improved": 0,
            "pct_improved_se": 0,
            "mean_improvement": 0,
            "mean_improvement_se": 0,
            "first_attempt_mean": 0,
            "first_attempt_se": 0,
        }

    # Calculate percentages
    pct_multi_attempt = (multi_attempt_trials / total_trials * 100)
    pct_improved = (improved_multi_attempts / multi_attempt_trials * 100) if multi_attempt_trials > 0 else 0
    mean_improvement = np.mean(multi_attempt_improvements) if multi_attempt_improvements else 0
    first_attempt_mean = np.mean(first_scores)

    # Calculate standard errors
    p_multi = pct_multi_attempt / 100
    pct_multi_attempt_se = np.sqrt(p_multi * (1 - p_multi) / total_trials) * 100

    p_improved = pct_improved / 100
    pct_improved_se = np.sqrt(p_improved * (1 - p_improved) / multi_attempt_trials) * 100 if multi_attempt_trials > 0 else 0

    mean_improvement_se = np.std(multi_attempt_improvements, ddof=1) / np.sqrt(len(multi_attempt_improvements)) if len(multi_attempt_improvements) > 1 else 0
    first_attempt_se = np.std(first_scores, ddof=1) / np.sqrt(len(first_scores)) if len(first_scores) > 1 else 0

    # ESR Rate (of ALL responses) - responses with multi-attempt AND improvement
    pct_improved_of_all = (improved_multi_attempts / total_trials * 100)
    p_improved_all = pct_improved_of_all / 100
    pct_improved_of_all_se = np.sqrt(p_improved_all * (1 - p_improved_all) / total_trials) * 100

    return {
        "total_trials": total_trials,
        "multi_attempt_trials": multi_attempt_trials,
        "improved_multi_attempts": improved_multi_attempts,
        "pct_multi_attempt": pct_multi_attempt,
        "pct_multi_attempt_se": pct_multi_attempt_se,
        "pct_improved": pct_improved,
        "pct_improved_se": pct_improved_se,
        "pct_improved_of_all": pct_improved_of_all,
        "pct_improved_of_all_se": pct_improved_of_all_se,
        "mean_improvement": mean_improvement,
        "mean_improvement_se": mean_improvement_se,
        "first_attempt_mean": first_attempt_mean,
        "first_attempt_se": first_attempt_se,
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics from multiple experiment runs."""
    if not metrics_list:
        return {}

    total_trials = sum(m["total_trials"] for m in metrics_list)
    multi_attempt_trials = sum(m["multi_attempt_trials"] for m in metrics_list)

    if total_trials == 0:
        return metrics_list[0]

    # Weighted average for mean improvement
    mean_improvements = [m["mean_improvement"] for m in metrics_list]
    multi_weights = [m["multi_attempt_trials"] for m in metrics_list]
    mean_improvement = np.average(mean_improvements, weights=multi_weights) if sum(multi_weights) > 0 else 0
    mean_improvement_se = np.std(mean_improvements) / np.sqrt(len(mean_improvements)) if len(mean_improvements) > 1 else 0

    # Weighted average for first attempt scores
    first_means = [m["first_attempt_mean"] for m in metrics_list]
    total_weights = [m["total_trials"] for m in metrics_list]
    first_attempt_mean = np.average(first_means, weights=total_weights)
    first_attempt_se = np.std(first_means) / np.sqrt(len(first_means)) if len(first_means) > 1 else 0

    # Aggregate counts
    improved_multi_attempts = sum(m["improved_multi_attempts"] for m in metrics_list)

    pct_multi_attempt = (multi_attempt_trials / total_trials * 100)
    pct_improved = (improved_multi_attempts / multi_attempt_trials * 100) if multi_attempt_trials > 0 else 0

    # SE for aggregated proportions
    p_multi = pct_multi_attempt / 100
    pct_multi_attempt_se = np.sqrt(p_multi * (1 - p_multi) / total_trials) * 100

    p_improved = pct_improved / 100
    pct_improved_se = np.sqrt(p_improved * (1 - p_improved) / multi_attempt_trials) * 100 if multi_attempt_trials > 0 else 0

    # ESR Rate (of ALL responses)
    pct_improved_of_all = (improved_multi_attempts / total_trials * 100)
    p_improved_all = pct_improved_of_all / 100
    pct_improved_of_all_se = np.sqrt(p_improved_all * (1 - p_improved_all) / total_trials) * 100

    return {
        "total_trials": total_trials,
        "multi_attempt_trials": multi_attempt_trials,
        "improved_multi_attempts": improved_multi_attempts,
        "pct_multi_attempt": pct_multi_attempt,
        "pct_multi_attempt_se": pct_multi_attempt_se,
        "pct_improved": pct_improved,
        "pct_improved_se": pct_improved_se,
        "pct_improved_of_all": pct_improved_of_all,
        "pct_improved_of_all_se": pct_improved_of_all_se,
        "mean_improvement": mean_improvement,
        "mean_improvement_se": mean_improvement_se,
        "first_attempt_mean": first_attempt_mean,
        "first_attempt_se": first_attempt_se,
    }


def create_comparison_plot(
    baseline_metrics: Dict,
    detector_metrics: Dict,
    random_metrics: Dict,
    output_dir: Path,
) -> None:
    """
    Create a 3-panel bar chart comparing key metrics across all conditions.

    Panels:
    1. Mean First-Attempt Score
    2. Multi-Attempt %
    3. Mean Score Improvement (Multi-Attempt Trials)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Color scheme
    colors = ['#6A0DAD', '#17A2B8', '#FFA500']  # Purple, Teal, Orange
    labels = ['Steered\nBaseline', 'Detector\nAblation', 'Random\nAblation']

    z_score = 1.96

    # Panel 1: Mean First-Attempt Score
    ax1 = axes[0]
    x1 = [0, 1, 2]
    values1 = [
        baseline_metrics['first_attempt_mean'],
        detector_metrics['first_attempt_mean'],
        random_metrics['first_attempt_mean'],
    ]
    errors1 = [
        z_score * baseline_metrics['first_attempt_se'],
        z_score * detector_metrics['first_attempt_se'],
        z_score * random_metrics['first_attempt_se'],
    ]
    bars1 = ax1.bar(x1, values1, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors1, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})

    ax1.set_ylabel('Mean First-Attempt Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    min_val = min(values1) - max(errors1) - 5
    max_val = max(values1) + max(errors1) + 10
    ax1.set_ylim(max(0, min_val), min(100, max_val))

    baseline_first = baseline_metrics['first_attempt_mean']
    for i, (bar, value, err) in enumerate(zip(bars1, values1, errors1)):
        height = bar.get_height() + err
        if i == 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            change = value - baseline_first
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}\n({change:+.1f})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: Multi-Attempt %
    ax2 = axes[1]
    x2 = [0, 1, 2]
    values2 = [
        baseline_metrics['pct_multi_attempt'],
        detector_metrics['pct_multi_attempt'],
        random_metrics['pct_multi_attempt'],
    ]
    errors2 = [
        z_score * baseline_metrics['pct_multi_attempt_se'],
        z_score * detector_metrics['pct_multi_attempt_se'],
        z_score * random_metrics['pct_multi_attempt_se'],
    ]
    bars2 = ax2.bar(x2, values2, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors2, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})

    ax2.set_ylabel('Multi-Attempt %', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    max_with_error2 = max(v + e for v, e in zip(values2, errors2))
    ax2.set_ylim(0, max_with_error2 * 1.4)

    for i, (bar, value, err) in enumerate(zip(bars2, values2, errors2)):
        height = bar.get_height() + err
        ax2.text(bar.get_x() + bar.get_width()/2., height + max_with_error2 * 0.02,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 3: ESR Rate (of ALL responses)
    ax3 = axes[2]
    x3 = [0, 1, 2]
    values3 = [
        baseline_metrics['pct_improved_of_all'],
        detector_metrics['pct_improved_of_all'],
        random_metrics['pct_improved_of_all'],
    ]
    errors3 = [
        z_score * baseline_metrics['pct_improved_of_all_se'],
        z_score * detector_metrics['pct_improved_of_all_se'],
        z_score * random_metrics['pct_improved_of_all_se'],
    ]
    bars3 = ax3.bar(x3, values3, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors3, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})

    ax3.set_ylabel('ESR Rate', fontsize=12, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 10)

    all_metrics = [baseline_metrics, detector_metrics, random_metrics]
    for i, (bar, value, err, m) in enumerate(zip(bars3, values3, errors3, all_metrics)):
        height = bar.get_height() + err
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add sample counts below bars
        ax3.text(bar.get_x() + bar.get_width()/2., -0.5,
                f'({m["improved_multi_attempts"]}/{m["total_trials"]})',
                ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()

    png_file = output_dir / "experiment_10_random_ablation_control.png"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {png_file}")
    plt.close()

    # Save data JSON
    json_file = output_dir / "experiment_10_random_ablation_control.json"
    plot_data = {
        "baseline": baseline_metrics,
        "detector_ablation": detector_metrics,
        "random_ablation": random_metrics,
    }
    with open(json_file, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"  Saved: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 10 plots (random latent ablation control)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots (relative paths resolved from experiment base dir). Default: plots/"
    )
    parser.add_argument(
        "--haiku-only",
        action="store_true",
        help="Only use experiment results from the haiku judge folder (default behavior)"
    )
    parser.add_argument(
        "--exclude-degraded",
        action="store_true",
        help="Filter out degraded (repetitive) outputs"
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else (BASE_DIR / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 10: Random Latent Ablation Control Plots")
    print("=" * 60)
    print(f"Data directory: {RANDOM_CONTROL_DIR}")
    print(f"Output directory: {output_dir}")

    if not RANDOM_CONTROL_DIR.exists():
        print(f"\nWarning: Random control directory not found: {RANDOM_CONTROL_DIR}")
        print("Run experiment_10_random_latent_control first.")
        return

    # Excluded files (same as experiment 3 for consistency)
    EXCLUDED_FILES = {
        # Old ablation runs (before separability + normal_pct filtering)
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_165238_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_155058_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251229_163636.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251229_171444.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251230_131048_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251219_180833_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260120_125251_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260120_133023_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260120_134938_with_ablation.json",
        # Incomplete runs (small file size)
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260121_172308_with_ablation.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20260121_172840_with_ablation.json",
    }

    # Target latent counts for valid ablation files (same as experiment 3)
    # 26 latents = old OTD set (off_topic_detectors_old.json)
    # 25 latents = new separability-based OTD set
    # 22 latents = legacy
    TARGET_LATENT_COUNTS = {22, 25, 26}

    # Find result files
    baseline_files = list(HAIKU_RESULTS_DIR.glob("experiment_results_*70B*.json"))
    baseline_files = [f for f in baseline_files if "ablation" not in f.name and "baseline" not in f.name]
    # Apply exclusion filter to baseline files
    baseline_files = [f for f in baseline_files if f.name not in EXCLUDED_FILES]

    # Load OTD ablation from experiment 3's with_ablation files (not random_latent_control)
    detector_files = list(HAIKU_RESULTS_DIR.glob("*70B*with_ablation*.json"))
    random_files = list(RANDOM_CONTROL_DIR.glob("*_random_ablation_*.json"))

    print(f"\nFound files:")
    print(f"  Steered baseline: {len(baseline_files)}")
    print(f"  Detector ablation (before filtering): {len(detector_files)}")
    print(f"  Random ablation: {len(random_files)}")

    # Filter detector ablation files (same logic as experiment 3)
    valid_detector_files = []
    for f in detector_files:
        if f.name in EXCLUDED_FILES:
            print(f"  Excluding: {f.name}")
            continue
        with open(f, "r") as fp:
            results = json.load(fp)
        num_ablated = len(results.get("ablated_latents", []))
        if num_ablated in TARGET_LATENT_COUNTS:
            valid_detector_files.append(f)
            print(f"  Including: {f.name} ({num_ablated} latents)")
        else:
            print(f"  Skipping: {f.name} ({num_ablated} latents, not in {TARGET_LATENT_COUNTS})")

    detector_files = valid_detector_files
    print(f"  Detector ablation (after filtering): {len(detector_files)}")

    if not baseline_files:
        print("\nWarning: No baseline files found. Need experiment 1 results for 70B.")
        return

    if not detector_files and not random_files:
        print("\nWarning: No ablation results found. Run experiment 10 first.")
        return

    # Extract metrics
    print("\nExtracting metrics...")

    # Baseline (use all files, no limit)
    baseline_metrics_list = []
    for f in baseline_files:
        with open(f, "r") as fp:
            results = json.load(fp)
        metrics = extract_metrics(results, exclude_degraded=args.exclude_degraded)
        baseline_metrics_list.append(metrics)
        print(f"  Baseline {f.name}: {metrics['total_trials']} trials")

    baseline_metrics = aggregate_metrics(baseline_metrics_list) if baseline_metrics_list else {
        "total_trials": 0, "multi_attempt_trials": 0, "improved_multi_attempts": 0,
        "pct_multi_attempt": 0, "pct_multi_attempt_se": 0, "pct_improved": 0,
        "pct_improved_se": 0, "mean_improvement": 0, "mean_improvement_se": 0,
        "first_attempt_mean": 50, "first_attempt_se": 0,
    }

    # Detector ablation (already filtered)
    detector_metrics_list = []
    for f in detector_files:
        with open(f, "r") as fp:
            results = json.load(fp)
        metrics = extract_metrics(results, exclude_degraded=args.exclude_degraded)
        detector_metrics_list.append(metrics)
        print(f"  Detector {f.name}: {metrics['total_trials']} trials")

    detector_metrics = aggregate_metrics(detector_metrics_list) if detector_metrics_list else baseline_metrics.copy()

    # Random ablation
    random_metrics_list = []
    for f in random_files:
        with open(f, "r") as fp:
            results = json.load(fp)
        metrics = extract_metrics(results, exclude_degraded=args.exclude_degraded)
        random_metrics_list.append(metrics)
        print(f"  Random {f.name}: {metrics['total_trials']} trials")

    random_metrics = aggregate_metrics(random_metrics_list) if random_metrics_list else baseline_metrics.copy()

    # Create plot
    print("\nCreating comparison plot...")
    create_comparison_plot(baseline_metrics, detector_metrics, random_metrics, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nMean Score Improvement:")
    print(f"  Baseline: {baseline_metrics['mean_improvement']:.4f}")
    print(f"  Detector ablation: {detector_metrics['mean_improvement']:.4f}")
    print(f"  Random ablation: {random_metrics['mean_improvement']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
