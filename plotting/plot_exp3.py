#!/usr/bin/env python
"""
Plotting functions for ablation experiment results.

Creates visualizations comparing baseline and ablation conditions, including:
1. Histogram comparison figure with 2 facets
2. Bar chart comparing key metrics (% multi-attempt, % improved, mean improvement)

Usage:
    python plot_exp3.py <model_identifier>

The script will find all experiment_results_*.json files matching the model identifier
and aggregate baseline (non-ablation) vs ablation runs for comparison.
"""

import json
import argparse
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from plotting.plot_utils import is_degraded_output


BASE_DIR = Path(__file__).parent.parent


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def load_results(filepath: str) -> dict:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_experiment_files(
    model_identifier: str,
    results_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Find all experiment result files matching the model identifier.

    Returns:
        Tuple of (baseline_files, ablation_files)
    """
    baseline_files = []
    ablation_files = []

    # Match exp1 files with various suffixes:
    # - Basic: experiment_results_{model}_YYYYMMDD_HHMMSS.json
    # - With suffix: experiment_results_{model}_YYYYMMDD_HHMMSS_suffix.json
    # - Date only: experiment_results_{model}_YYYYMMDD_suffix.json (e.g., overlapping_features)
    # Exclude: _with_ablation, _no_steering, _random_ablation, _otd_ablation
    pattern = re.compile(
        rf"experiment_results_{re.escape(model_identifier)}_\d{{8}}(_\d{{6}})?(_.+)?\.json"
    )

    excluded_suffixes = ['_with_ablation', '_no_steering', '_random_ablation', '_otd_ablation']

    for filepath in results_dir.glob("experiment_results_*.json"):
        if pattern.match(filepath.name):
            # Check for ablation files
            if "_with_ablation" in filepath.name:
                ablation_files.append(filepath)
            # Skip other non-baseline files
            elif any(suffix in filepath.name for suffix in excluded_suffixes):
                continue
            else:
                baseline_files.append(filepath)

    # Sort by filename (which includes timestamp)
    baseline_files.sort()
    ablation_files.sort()

    return baseline_files, ablation_files


def aggregate_trial_data_from_files(filepaths: list[Path], label: str = "", exclude_degraded: bool = False) -> list[dict]:
    """
    Load and aggregate trial data from multiple experiment result files.

    Args:
        filepaths: List of result file paths to load
        label: Label for logging (e.g., "baseline" or "ablation")

    Returns:
        Combined list of trial data from all files
    """
    all_trial_data = []
    total_seen = 0
    total_degraded = 0
    
    for filepath in filepaths:
        results = load_results(str(filepath))
        trial_data, seen, degraded = extract_trial_data(results, exclude_degraded)
        all_trial_data.extend(trial_data)
        total_seen += seen
        total_degraded += degraded
    
    if total_seen > 0:
        pct_degraded = total_degraded / total_seen * 100
        print(f"  {label}: {len(all_trial_data)}/{total_seen} trials kept "
              f"({total_degraded} degraded filtered out, {pct_degraded:.1f}%)")
    
    return all_trial_data


def get_ablation_info_from_files(filepaths: list[Path]) -> tuple[int, list[int]]:
    """
    Extract ablation information from ablation experiment files.

    Returns:
        Tuple of (num_ablated_latents, list_of_ablated_latent_indices)
    """
    all_ablated = set()
    for filepath in filepaths:
        results = load_results(str(filepath))
        ablated = results.get("ablated_latents", [])
        all_ablated.update(ablated)
    return len(all_ablated), sorted(all_ablated)


def extract_trial_data(results: dict, exclude_degraded: bool = False) -> tuple[list[dict], int, int]:
    """Extract all trial data from results dict.

    Returns:
        Tuple of (trial_data, total_trials_seen, degraded_count)
    """
    trial_data = []
    total_seen = 0
    degraded_count = 0

    for feature_result in results['results_by_feature']:
        if feature_result.get('error'):
            continue

        feature_idx = feature_result['feature_index_in_sae']
        feature_label = feature_result['feature_label']
        threshold = feature_result.get('threshold')

        for trial in feature_result['trials']:
            total_seen += 1

            # Skip degraded outputs (repetitive patterns) if exclude_degraded
            response = trial.get('response', '')
            if is_degraded_output(response):
                degraded_count += 1
                if exclude_degraded:
                    continue

            score_obj = trial.get('score')
            if not score_obj or not isinstance(score_obj, dict):
                continue

            attempts = score_obj.get('attempts', [])
            if not attempts:
                continue

            # Extract scores
            first_score = attempts[0]['score']
            last_score = attempts[-1]['score']
            num_attempts = len(attempts)
            has_multiple = num_attempts > 1
            improvement = last_score - first_score if has_multiple else 0

            trial_data.append({
                'feature_idx': feature_idx,
                'feature_label': feature_label,
                'threshold': threshold,
                'first_score': first_score,
                'last_score': last_score,
                'num_attempts': num_attempts,
                'has_multiple_attempts': has_multiple,
                'improvement': improvement,
                'prompt': trial.get('prompt', ''),
            })

    return trial_data, total_seen, degraded_count


def calculate_metrics(trial_data: list[dict]) -> dict:
    """
    Calculate key metrics from trial data.

    Returns dictionary with:
    - total_trials
    - multi_attempt_trials
    - improved_multi_attempts
    - mean_first_score, mean_first_score_se
    - pct_multi_attempt, pct_multi_attempt_se
    - pct_improved, pct_improved_se
    - mean_improvement, mean_improvement_se
    """
    total_trials = len(trial_data)
    multi_attempt_trials = sum(1 for t in trial_data if t['has_multiple_attempts'])

    # Calculate first-attempt scores (all trials)
    first_scores = [t['first_score'] for t in trial_data]
    mean_first_score = np.mean(first_scores) if first_scores else 0

    # Calculate improvements for multi-attempt trials only
    multi_attempt_improvements = [t['improvement'] for t in trial_data if t['has_multiple_attempts']]
    improved_multi_attempts = sum(1 for t in trial_data if t['has_multiple_attempts'] and t['improvement'] > 0)

    # Calculate percentages
    pct_multi_attempt = (multi_attempt_trials / total_trials * 100) if total_trials > 0 else 0
    pct_improved = (improved_multi_attempts / multi_attempt_trials * 100) if multi_attempt_trials > 0 else 0
    mean_improvement = np.mean(multi_attempt_improvements) if multi_attempt_improvements else 0

    # Calculate standard errors
    # SE for first score mean: std / sqrt(n)
    mean_first_score_se = np.std(first_scores, ddof=1) / np.sqrt(len(first_scores)) if len(first_scores) > 1 else 0

    # SE for binomial proportion: sqrt(p*(1-p)/n) * 100 (in percentage points)
    p_multi = pct_multi_attempt / 100
    pct_multi_attempt_se = np.sqrt(p_multi * (1 - p_multi) / total_trials) * 100 if total_trials > 0 else 0

    p_improved = pct_improved / 100
    pct_improved_se = np.sqrt(p_improved * (1 - p_improved) / multi_attempt_trials) * 100 if multi_attempt_trials > 0 else 0

    # SE for mean improvement: std / sqrt(n)
    mean_improvement_se = np.std(multi_attempt_improvements, ddof=1) / np.sqrt(len(multi_attempt_improvements)) if len(multi_attempt_improvements) > 1 else 0

    # ESR Rate (of ALL responses) - responses with multi-attempt AND improvement
    pct_improved_of_all = (improved_multi_attempts / total_trials * 100) if total_trials > 0 else 0
    p_improved_all = pct_improved_of_all / 100
    pct_improved_of_all_se = np.sqrt(p_improved_all * (1 - p_improved_all) / total_trials) * 100 if total_trials > 0 else 0

    return {
        'total_trials': total_trials,
        'multi_attempt_trials': multi_attempt_trials,
        'improved_multi_attempts': improved_multi_attempts,
        'mean_first_score': mean_first_score,
        'mean_first_score_se': mean_first_score_se,
        'pct_multi_attempt': pct_multi_attempt,
        'pct_multi_attempt_se': pct_multi_attempt_se,
        'pct_improved': pct_improved,
        'pct_improved_se': pct_improved_se,
        'pct_improved_of_all': pct_improved_of_all,
        'pct_improved_of_all_se': pct_improved_of_all_se,
        'mean_improvement': mean_improvement,
        'mean_improvement_se': mean_improvement_se,
    }


def create_histogram_comparison(
    baseline_data: list[dict],
    ablation_data: list[dict],
    output_path: str,
) -> None:
    """
    Create a 2-facet histogram comparison figure.

    Top: Distribution of number of attempts
    Bottom: Distribution of score improvements (multi-attempt trials only)
    """
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Color scheme (colorblind-friendly purple/teal)
    baseline_color = "#6A0DAD"  # Deep purple
    ablation_color = "#17A2B8"  # Teal

    # Top: Number of attempts distribution
    ax = axes[0]
    baseline_attempts = [t['num_attempts'] for t in baseline_data]
    ablation_attempts = [t['num_attempts'] for t in ablation_data]

    max_attempts = max(max(baseline_attempts), max(ablation_attempts))
    bins = np.arange(1, max_attempts + 2) - 0.5

    ax.hist(baseline_attempts, bins=bins, alpha=0.6, label='Baseline (no ablation)',
            color=baseline_color, edgecolor='black', linewidth=0.5)
    ax.hist(ablation_attempts, bins=bins, alpha=0.6, label='Ablation (off-topic detectors ablated)',
            color=ablation_color, edgecolor='black', linewidth=0.5)

    baseline_multi_pct = sum(1 for t in baseline_data if t['has_multiple_attempts']) / len(baseline_data) * 100
    ablation_multi_pct = sum(1 for t in ablation_data if t['has_multiple_attempts']) / len(ablation_data) * 100

    ax.set_xlabel('Number of Judge Attempts', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max_attempts + 1))

    # Bottom: Score improvement distribution (multi-attempt only)
    ax = axes[1]
    baseline_improvements = [t['improvement'] for t in baseline_data if t['has_multiple_attempts']]
    ablation_improvements = [t['improvement'] for t in ablation_data if t['has_multiple_attempts']]

    if baseline_improvements and ablation_improvements:
        all_improvements = baseline_improvements + ablation_improvements
        bins = np.linspace(min(all_improvements), max(all_improvements), 21)

        ax.hist(baseline_improvements, bins=bins, alpha=0.6, label='Baseline (no ablation)',
                color=baseline_color, edgecolor='black', linewidth=0.5)
        ax.hist(ablation_improvements, bins=bins, alpha=0.6, label='Ablation (off-topic detectors ablated)',
                color=ablation_color, edgecolor='black', linewidth=0.5)

        ax.axvline(np.mean(baseline_improvements), color=baseline_color, linestyle='--',
                   linewidth=2, label=f'Baseline mean: {np.mean(baseline_improvements):.1f}')
        ax.axvline(np.mean(ablation_improvements), color=ablation_color, linestyle='--',
                   linewidth=2, label=f'Ablation mean: {np.mean(ablation_improvements):.1f}')
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Score Improvement (Last - First Attempt)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram comparison saved to {output_path}")
    plt.close()

    # Save data JSON alongside the plot
    data_output_path = Path(output_path).with_suffix('.json')
    plot_data = {
        "baseline": {
            "attempts_distribution": baseline_attempts,
            "improvements_distribution": baseline_improvements,
            "multi_attempt_pct": baseline_multi_pct,
        },
        "ablation": {
            "attempts_distribution": ablation_attempts,
            "improvements_distribution": ablation_improvements,
            "multi_attempt_pct": ablation_multi_pct,
        }
    }
    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Histogram comparison data saved to {data_output_path}")


def create_boost_comparison(
    baseline_data: list[dict],
    ablation_data: list[dict],
    output_path: str,
) -> None:
    """
    Create a 3-facet line plot showing metrics vs normalized steering strength.

    Top: Average number of attempts vs normalized threshold
    Middle: Score improvement (multi-attempt only) vs normalized threshold
    Bottom: Final score vs normalized threshold
    """
    # Compute normalization statistics (across both conditions)
    all_thresholds = [t['threshold'] for t in baseline_data + ablation_data if t['threshold'] is not None]
    threshold_mean = np.mean(all_thresholds)
    threshold_std = np.std(all_thresholds)

    print(f"Threshold statistics: mean={threshold_mean:.2f}, std={threshold_std:.2f}")

    # Normalize thresholds
    for t in baseline_data:
        if t['threshold'] is not None:
            t['normalized_threshold'] = (t['threshold'] - threshold_mean) / threshold_std

    for t in ablation_data:
        if t['threshold'] is not None:
            t['normalized_threshold'] = (t['threshold'] - threshold_mean) / threshold_std

    # Aggregate by normalized threshold bins
    def aggregate_by_threshold(data, bin_size=0.2):
        """Aggregate trials into normalized threshold bins."""
        bins = defaultdict(list)
        for trial in data:
            if 'normalized_threshold' not in trial:
                continue
            bin_center = round(trial['normalized_threshold'] / bin_size) * bin_size
            bins[bin_center].append(trial)
        return bins

    baseline_bins = aggregate_by_threshold(baseline_data)
    ablation_bins = aggregate_by_threshold(ablation_data)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Color scheme (colorblind-friendly purple/teal)
    baseline_color = "#6A0DAD"  # Deep purple
    ablation_color = "#17A2B8"  # Teal

    # Helper function to compute statistics for each bin
    def compute_bin_stats(bins):
        """Compute statistics for each threshold bin."""
        thresholds = []
        avg_attempts = []
        avg_improvements = []
        avg_scores = []

        for threshold, trials in sorted(bins.items()):
            thresholds.append(threshold)

            # Average number of attempts
            avg_attempts.append(np.mean([t['num_attempts'] for t in trials]))

            # Average improvement (multi-attempt only)
            multi_attempt_trials = [t for t in trials if t['has_multiple_attempts']]
            if multi_attempt_trials:
                avg_improvements.append(np.mean([t['improvement'] for t in multi_attempt_trials]))
            else:
                avg_improvements.append(np.nan)

            # Average final score
            avg_scores.append(np.mean([t['last_score'] for t in trials]))

        return thresholds, avg_attempts, avg_improvements, avg_scores

    baseline_stats = compute_bin_stats(baseline_bins)
    ablation_stats = compute_bin_stats(ablation_bins)

    # Top: Average number of attempts
    ax = axes[0]
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, zorder=1)

    ax.plot(baseline_stats[0], baseline_stats[1], 'o-', color=baseline_color,
            linewidth=3, markersize=6, label='Baseline (no ablation)', alpha=0.8)
    ax.plot(ablation_stats[0], ablation_stats[1], 's-', color=ablation_color,
            linewidth=3, markersize=6, label='Ablation (off-topic detectors ablated)', alpha=0.8)

    ax.set_xlabel('Normalized Steering Strength (Std Deviations from Mean)', fontsize=12)
    ax.set_ylabel('Avg Judge Attempts', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1.0)

    # Middle: Score improvement (multi-attempt only)
    ax = axes[1]
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, zorder=1)

    # Filter out NaN values for plotting
    baseline_valid = [(t, s) for t, s in zip(baseline_stats[0], baseline_stats[2]) if not np.isnan(s)]
    ablation_valid = [(t, s) for t, s in zip(ablation_stats[0], ablation_stats[2]) if not np.isnan(s)]

    if baseline_valid:
        b_thresh, b_imp = zip(*baseline_valid)
        ax.plot(b_thresh, b_imp, 'o-', color=baseline_color,
                linewidth=3, markersize=6, label='Baseline (no ablation)', alpha=0.8)

    if ablation_valid:
        a_thresh, a_imp = zip(*ablation_valid)
        ax.plot(a_thresh, a_imp, 's-', color=ablation_color,
                linewidth=3, markersize=6, label='Ablation (off-topic detectors ablated)', alpha=0.8)

    ax.set_xlabel('Normalized Steering Strength (Std Deviations from Mean)', fontsize=12)
    ax.set_ylabel('Score Improvement', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom: Final scores
    ax = axes[2]
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, zorder=1, label='Target (50)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, zorder=1)

    ax.plot(baseline_stats[0], baseline_stats[3], 'o-', color=baseline_color,
            linewidth=3, markersize=6, label='Baseline (no ablation)', alpha=0.8)
    ax.plot(ablation_stats[0], ablation_stats[3], 's-', color=ablation_color,
            linewidth=3, markersize=6, label='Ablation (off-topic detectors ablated)', alpha=0.8)

    ax.set_xlabel('Normalized Steering Strength (Std Deviations from Mean)', fontsize=12)
    ax.set_ylabel('Average Final Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Boost comparison saved to {output_path}")
    plt.close()

    # Save data JSON alongside the plot
    data_output_path = Path(output_path).with_suffix('.json')
    plot_data = {
        "threshold_stats": {
            "mean": threshold_mean,
            "std": threshold_std,
        },
        "baseline": {
            "thresholds": baseline_stats[0],
            "avg_attempts": baseline_stats[1],
            "avg_improvements": [x if not np.isnan(x) else None for x in baseline_stats[2]],
            "avg_scores": baseline_stats[3],
        },
        "ablation": {
            "thresholds": ablation_stats[0],
            "avg_attempts": ablation_stats[1],
            "avg_improvements": [x if not np.isnan(x) else None for x in ablation_stats[2]],
            "avg_scores": ablation_stats[3],
        }
    }
    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Boost comparison data saved to {data_output_path}")


def create_metrics_bar_chart(
    baseline_data: list[dict],
    ablation_data: list[dict],
    output_path: str,
) -> None:
    """
    Create a 3-panel bar chart comparing key metrics between baseline and ablation.

    Panels:
    1. Mean first-attempt score
    2. Multi-Attempt %
    3. Mean Score Improvement (Multi-Attempt Trials)
    """
    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline_data)
    ablation_metrics = calculate_metrics(ablation_data)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Color scheme (colorblind-friendly purple/teal)
    colors = ['#6A0DAD', '#17A2B8']  # Purple for baseline, Teal for ablation

    # 95% CI multiplier
    z_score = 1.96

    # Panel 1: Mean First-Attempt Score with error bars
    ax1 = axes[0]
    x1 = [0, 1]
    values1 = [baseline_metrics['mean_first_score'], ablation_metrics['mean_first_score']]
    errors1 = [z_score * baseline_metrics['mean_first_score_se'], z_score * ablation_metrics['mean_first_score_se']]
    bars1 = ax1.bar(x1, values1, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors1, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})

    ax1.set_ylabel('Mean First-Attempt Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(['Baseline', 'Ablation'], fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)

    # Add value labels (offset for error bars)
    for i, (bar, value, err) in enumerate(zip(bars1, values1, errors1)):
        height = bar.get_height() + err
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 2: Multi-Attempt % with error bars
    ax2 = axes[1]
    x2 = [0, 1]
    values2 = [baseline_metrics['pct_multi_attempt'], ablation_metrics['pct_multi_attempt']]
    errors2 = [z_score * baseline_metrics['pct_multi_attempt_se'], z_score * ablation_metrics['pct_multi_attempt_se']]
    bars2 = ax2.bar(x2, values2, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors2, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})

    ax2.set_ylabel('Multi-Attempt %', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Baseline', 'Ablation'], fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    max_with_error2 = max(v + e for v, e in zip(values2, errors2))
    ax2.set_ylim(0, max_with_error2 * 1.3)

    # Add value labels (offset for error bars)
    for i, (bar, value, err) in enumerate(zip(bars2, values2, errors2)):
        height = bar.get_height() + err
        ax2.text(bar.get_x() + bar.get_width()/2., height + max_with_error2 * 0.02,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add sample counts below bars
        if i == 0:
            count = baseline_metrics['multi_attempt_trials']
            total = baseline_metrics['total_trials']
        else:
            count = ablation_metrics['multi_attempt_trials']
            total = ablation_metrics['total_trials']
        ax2.text(bar.get_x() + bar.get_width()/2., -max_with_error2 * 0.1,
                f'({count}/{total})',
                ha='center', va='top', fontsize=9, style='italic')

    # Panel 3: ESR Rate (of ALL responses) with error bars
    ax3 = axes[2]
    x3 = [0, 1]
    values3 = [baseline_metrics['pct_improved_of_all'], ablation_metrics['pct_improved_of_all']]
    errors3 = [z_score * baseline_metrics['pct_improved_of_all_se'], z_score * ablation_metrics['pct_improved_of_all_se']]
    bars3 = ax3.bar(x3, values3, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors3, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})

    ax3.set_ylabel('ESR Rate', fontsize=12, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(['Baseline', 'Ablation'], fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 10)

    # Add value labels (offset for error bars)
    for i, (bar, value, err) in enumerate(zip(bars3, values3, errors3)):
        height = bar.get_height() + err
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add sample counts below bars
        if i == 0:
            count = baseline_metrics['improved_multi_attempts']
            total = baseline_metrics['total_trials']
        else:
            count = ablation_metrics['improved_multi_attempts']
            total = ablation_metrics['total_trials']
        ax3.text(bar.get_x() + bar.get_width()/2., -0.5,
                f'({count}/{total})',
                ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics bar chart saved to {output_path}")
    plt.close()

    # Save data JSON alongside the plot
    data_output_path = Path(output_path).with_suffix('.json')

    def _ci95(value: float, se: float) -> list[float]:
        """Return [lower, upper] bounds of a 95% confidence interval."""
        return [value - 1.96 * se, value + 1.96 * se]

    plot_data = {
        "baseline": {
            "total_trials": baseline_metrics['total_trials'],
            "multi_attempt_trials": baseline_metrics['multi_attempt_trials'],
            "improved_multi_attempts": baseline_metrics['improved_multi_attempts'],
            "pct_multi_attempt": baseline_metrics['pct_multi_attempt'],
            "pct_multi_attempt_se": baseline_metrics['pct_multi_attempt_se'],
            "pct_multi_attempt_ci95": _ci95(baseline_metrics['pct_multi_attempt'], baseline_metrics['pct_multi_attempt_se']),
            "pct_improved_of_all": baseline_metrics['pct_improved_of_all'],
            "pct_improved_of_all_se": baseline_metrics['pct_improved_of_all_se'],
            "pct_improved_of_all_ci95": _ci95(baseline_metrics['pct_improved_of_all'], baseline_metrics['pct_improved_of_all_se']),
            "mean_first_score": baseline_metrics['mean_first_score'],
            "mean_first_score_se": baseline_metrics['mean_first_score_se'],
            "mean_first_score_ci95": _ci95(baseline_metrics['mean_first_score'], baseline_metrics['mean_first_score_se']),
        },
        "ablation": {
            "total_trials": ablation_metrics['total_trials'],
            "multi_attempt_trials": ablation_metrics['multi_attempt_trials'],
            "improved_multi_attempts": ablation_metrics['improved_multi_attempts'],
            "pct_multi_attempt": ablation_metrics['pct_multi_attempt'],
            "pct_multi_attempt_se": ablation_metrics['pct_multi_attempt_se'],
            "pct_multi_attempt_ci95": _ci95(ablation_metrics['pct_multi_attempt'], ablation_metrics['pct_multi_attempt_se']),
            "pct_improved_of_all": ablation_metrics['pct_improved_of_all'],
            "pct_improved_of_all_se": ablation_metrics['pct_improved_of_all_se'],
            "pct_improved_of_all_ci95": _ci95(ablation_metrics['pct_improved_of_all'], ablation_metrics['pct_improved_of_all_se']),
            "mean_first_score": ablation_metrics['mean_first_score'],
            "mean_first_score_se": ablation_metrics['mean_first_score_se'],
            "mean_first_score_ci95": _ci95(ablation_metrics['mean_first_score'], ablation_metrics['mean_first_score_se']),
        }
    }
    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Metrics bar chart data saved to {data_output_path}")


def print_paper_numbers(
    baseline_data: list[dict],
    ablation_data: list[dict],
    num_ablated: int,
    ablated_latents: list[int],
) -> None:
    """
    Print all key numbers needed for the paper.
    """
    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline_data)
    ablation_metrics = calculate_metrics(ablation_data)

    # Calculate ESR reduction
    baseline_msi = baseline_metrics['mean_improvement']
    ablation_msi = ablation_metrics['mean_improvement']

    if baseline_msi != 0:
        esr_reduction_pct = (baseline_msi - ablation_msi) / baseline_msi * 100
    else:
        esr_reduction_pct = 0

    # Calculate first attempt score change
    baseline_first_scores = [t['first_score'] for t in baseline_data]
    ablation_first_scores = [t['first_score'] for t in ablation_data]
    baseline_first_mean = np.mean(baseline_first_scores)
    ablation_first_mean = np.mean(ablation_first_scores)
    first_score_change = ablation_first_mean - baseline_first_mean

    # Calculate multi-attempt reduction
    baseline_multi_pct = baseline_metrics['pct_multi_attempt']
    ablation_multi_pct = ablation_metrics['pct_multi_attempt']
    if baseline_multi_pct != 0:
        multi_attempt_reduction_pct = (baseline_multi_pct - ablation_multi_pct) / baseline_multi_pct * 100
    else:
        multi_attempt_reduction_pct = 0

    print("\n" + "=" * 70)
    print("NUMBERS FOR THE PAPER")
    print("=" * 70)

    print("\n--- Off-Topic Detector Information ---")
    print(f"Number of off-topic detector latents ablated: {num_ablated}")
    if ablated_latents:
        print(f"Latent indices: {sorted(ablated_latents)}")

    print("\n--- Sample Sizes ---")
    print(f"Baseline total trials: {baseline_metrics['total_trials']}")
    print(f"Ablation total trials: {ablation_metrics['total_trials']}")

    print("\n--- Multi-Attempt Statistics ---")
    print(f"Baseline % multi-attempt: {baseline_multi_pct:.2f}%")
    print(f"Ablation % multi-attempt: {ablation_multi_pct:.2f}%")
    print(f"  → Reduction in multi-attempt rate: {multi_attempt_reduction_pct:.1f}%")

    print("\n--- Mean Score Improvement (ESR Metric) ---")
    print(f"Baseline Mean Score Improvement: {baseline_msi:.4f}")
    print(f"Ablation Mean Score Improvement: {ablation_msi:.4f}")
    print(f"  → ESR reduction: {esr_reduction_pct:.1f}%")

    print("\n--- First Attempt Scores (Topical Coherence) ---")
    print(f"Baseline mean first attempt score: {baseline_first_mean:.2f}%")
    print(f"Ablation mean first attempt score: {ablation_first_mean:.2f}%")
    print(f"  → Change in first attempt score: {first_score_change:+.2f} percentage points")

    print("\n--- Success Rate (Multi-Attempt Trials) ---")
    print(f"Baseline % improved (of multi-attempt): {baseline_metrics['pct_improved']:.1f}%")
    print(f"Ablation % improved (of multi-attempt): {ablation_metrics['pct_improved']:.1f}%")

    print("\n--- Suggested Paper Text ---")
    print(f"\"We identify {num_ablated} 'off-topic detector' latents...\"")
    print(f"\"Ablating the off-topic detector latents reduced ESR by {esr_reduction_pct:.0f}%,\"")
    print(f"\"while topical coherence in first attempts changed by {abs(first_score_change):.1f} percentage points\"")
    print(f"\"(from {baseline_first_mean:.2f}% to {ablation_first_mean:.2f}% average score).\"")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot Experiment 3 ablation results (baseline vs ablation runs)")
    parser.add_argument("model_identifier", help="Model identifier substring to match in experiment result filenames")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots (relative paths are resolved from the experiment base dir). Default: plots/",
    )
    parser.add_argument(
        "--haiku-only",
        action="store_true",
        help="Only use experiment results from the haiku judge folder",
    )
    parser.add_argument(
        "--exclude-degraded",
        action="store_true",
        help="Filter out degraded (repetitive) outputs instead of including them",
    )
    args = parser.parse_args()

    model_identifier = args.model_identifier

    # Determine directories
    if args.haiku_only:
        results_dir = BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge"
    else:
        results_dir = BASE_DIR / "data" / "experiment_results"
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find experiment files
    baseline_files, ablation_files = find_experiment_files(model_identifier, results_dir)

    if not baseline_files:
        print(f"Error: No baseline files found for model '{model_identifier}'")
        print(f"Searched in: {results_dir}")
        sys.exit(1)

    if not ablation_files:
        print(f"Error: No ablation files found for model '{model_identifier}'")
        print(f"Searched in: {results_dir}")
        sys.exit(1)

    print(f"Model: {model_identifier}")
    print(f"\nFound {len(baseline_files)} baseline file(s):")
    for f in baseline_files:
        print(f"  - {f.name}")

    print(f"\nFound {len(ablation_files)} ablation file(s):")
    for f in ablation_files:
        print(f"  - {f.name}")

    # Only use the new 22-latent ablation with separability + normal activation filtering
    # Exclude all other ablation runs
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

    # Filter to only ablation files with expected latent counts
    # 26 latents = old OTD set (off_topic_detectors_old.json)
    # 25 latents = new separability-based OTD set
    # 22 latents = legacy
    TARGET_LATENT_COUNTS = {22, 25, 26}
    valid_ablation_files = []
    for ablation_file in ablation_files:
        if ablation_file.name in EXCLUDED_FILES:
            print(f"  Excluding: {ablation_file.name}")
            continue
        results = load_results(str(ablation_file))
        num_ablated = len(results.get("ablated_latents", []))
        if num_ablated in TARGET_LATENT_COUNTS:
            valid_ablation_files.append(ablation_file)
            print(f"  Including: {ablation_file.name} ({num_ablated} latents)")
        else:
            print(f"  Skipping: {ablation_file.name} ({num_ablated} latents, not in {TARGET_LATENT_COUNTS})")

    if not valid_ablation_files:
        print(f"Error: No valid ablation files found (expected latent counts: {TARGET_LATENT_COUNTS})!")
        sys.exit(1)

    # Filter baseline files using the same exclusion set
    valid_baseline_files = [f for f in baseline_files if f.name not in EXCLUDED_FILES]
    excluded_baseline = [f for f in baseline_files if f.name in EXCLUDED_FILES]
    if excluded_baseline:
        print(f"\nExcluded {len(excluded_baseline)} baseline file(s):")
        for f in excluded_baseline:
            print(f"  Excluding: {f.name}")

    # Aggregate baseline data from valid baseline files
    filter_msg = " (filtering degraded outputs)" if args.exclude_degraded else ""
    print(f"\nAggregating baseline data{filter_msg}...")
    baseline_data = aggregate_trial_data_from_files(valid_baseline_files, label="Baseline", exclude_degraded=args.exclude_degraded)

    # Aggregate ablation data from all valid ablation files (combined)
    print(f"\nAggregating ablation data from {len(valid_ablation_files)} file(s){filter_msg}...")
    ablation_data = aggregate_trial_data_from_files(valid_ablation_files, label="Ablation", exclude_degraded=args.exclude_degraded)

    # Get ablation info from the first file (they should all have the same latents)
    first_results = load_results(str(valid_ablation_files[0]))
    ablated_latents = first_results.get("ablated_latents", [])
    num_ablated = len(ablated_latents)

    print(f"\n{'='*70}")
    print(f"Generating combined plot for {num_ablated}-latent ablation")
    print(f"  Files combined: {[f.name for f in valid_ablation_files]}")
    print("=" * 70)

    # Generate single combined plot
    output_suffix = f"combined_{num_ablated}latents"

    print("  Generating metrics bar chart...")
    create_metrics_bar_chart(
        baseline_data,
        ablation_data,
        str(output_dir / f"experiment_3_ablation_metrics_bar_chart_{output_suffix}.png"),
    )

    # Print numbers for the paper
    print_paper_numbers(baseline_data, ablation_data, num_ablated, ablated_latents)

    print(f"\n{'='*70}")
    print(f"Done! Plot saved to {output_dir}/experiment_3_ablation_metrics_bar_chart_{output_suffix}.png")


if __name__ == "__main__":
    main()
