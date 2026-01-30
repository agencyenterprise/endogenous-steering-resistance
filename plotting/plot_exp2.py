#!/usr/bin/env python3
"""
Plot multi-boost experiment results showing how boost level affects self-correction behavior.

For each model, produces a single figure with 3 vertically stacked graphs:
1. Multi-attempt rate (% of responses with multiple attempts)
2. Multi-Attempt Improvement Rate (% of multi-attempt trials that improved)
3. ESR Rate (% of ALL responses with multi-attempt AND improvement)
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_file_utils import (
    parse_results_filename,
    canonicalize_model_name,
    get_model_color,
    CanonicalModelInfo,
)
from plotting.plot_utils import is_degraded_output

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def load_multi_boost_data(result_file: Path) -> dict | None:
    """
    Load and validate multi-boost experiment data from a JSON file.

    Returns dict with boost_levels, threshold_cache_stats, and results_by_feature,
    or None if the file is invalid.
    """
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Validate required fields
        if 'boost_levels' not in data:
            print(f"Warning: {result_file.name} missing boost_levels, skipping")
            return None
        if 'threshold_cache_stats' not in data:
            print(f"Warning: {result_file.name} missing threshold_cache_stats, skipping")
            return None
        if 'results_by_feature' not in data:
            print(f"Warning: {result_file.name} missing results_by_feature, skipping")
            return None

        return data
    except Exception as e:
        print(f"Error loading {result_file.name}: {e}")
        return None


def extract_trial_stats_by_boost(
    data: dict,
    exclude_degraded: bool = False,
) -> tuple[dict[float, list[int]], dict[float, list[bool]], dict[float, list[float]], dict[float, list[float]], float, float]:
    """
    Extract trial statistics grouped by boost level.

    Returns:
        - attempts_by_boost: dict mapping boost level to list of attempt counts
        - success_by_boost: dict mapping boost level to list of success flags
                           (True if last > first for multi-attempt trials)
        - improvement_by_boost: dict mapping boost level to list of score deltas
                               (last_score - first_score for multi-attempt trials)
        - first_scores_by_boost: dict mapping boost level to list of first-attempt scores
        - mean: threshold cache mean
        - std: threshold cache std
    """
    mean = data['threshold_cache_stats']['mean']
    std = data['threshold_cache_stats']['std']

    attempts_by_boost: dict[float, list[int]] = defaultdict(list)
    success_by_boost: dict[float, list[bool]] = defaultdict(list)
    improvement_by_boost: dict[float, list[float]] = defaultdict(list)
    first_scores_by_boost: dict[float, list[float]] = defaultdict(list)

    for feature_result in data.get('results_by_feature', []):
        if feature_result.get('error'):
            continue

        for trial in feature_result.get('trials', []):
            if trial.get('error'):
                continue

            # Skip degraded outputs (repetitive patterns) if exclude_degraded
            if exclude_degraded:
                response = trial.get('response', '')
                if is_degraded_output(response):
                    continue

            boost_level = trial.get('threshold')
            if boost_level is None:
                continue

            # Extract attempts from score dict
            score_dict = trial.get('score', {})
            attempts = score_dict.get('attempts', [])

            if not attempts:
                continue

            # Extract scores from attempts
            scores = [att.get('score') for att in attempts if att.get('score') is not None]

            if not scores:
                continue

            num_attempts = len(scores)
            attempts_by_boost[boost_level].append(num_attempts)
            first_scores_by_boost[boost_level].append(float(scores[0]))

            # For success calculation, only consider multi-attempt trials
            if num_attempts > 1:
                first_score = scores[0]
                last_score = scores[-1]
                success = last_score > first_score
                success_by_boost[boost_level].append(success)
                improvement_by_boost[boost_level].append(float(last_score - first_score))

    return attempts_by_boost, success_by_boost, improvement_by_boost, first_scores_by_boost, mean, std


def compute_metrics_by_normalized_boost(
    attempts_by_boost: dict[float, list[int]],
    success_by_boost: dict[float, list[bool]],
    improvement_by_boost: dict[float, list[float]],
    first_scores_by_boost: dict[float, list[float]],
    mean: float,
    std: float,
    n_bins: int = 50,
    smooth_sigma: float = 0.5,
) -> dict:
    """
    Compute smoothed metrics as a function of normalized boost level.

    Args:
        attempts_by_boost: dict mapping boost level to list of attempt counts
        success_by_boost: dict mapping boost level to list of success flags
        improvement_by_boost: dict mapping boost level to list of score improvements
        first_scores_by_boost: dict mapping boost level to list of first-attempt scores
        mean: threshold cache mean
        std: threshold cache std
        n_bins: number of bins for smoothing
        smooth_sigma: gaussian smoothing sigma

    Returns:
        Dict with:
        - normalized_boosts: array of normalized boost values
        - first_score_mean, first_score_se: mean first-attempt score and standard error
        - multi_attempt_rate, multi_attempt_rate_se: rate and standard error
        - mean_improvement, mean_improvement_se: mean and standard error
    """
    if std == 0:
        std = 1.0  # Avoid division by zero

    # Normalize boost levels
    all_boosts = sorted(attempts_by_boost.keys())
    if not all_boosts:
        return {
            "normalized_boosts": np.array([]),
            "first_score_mean": np.array([]),
            "first_score_se": np.array([]),
            "multi_attempt_rate": np.array([]),
            "multi_attempt_rate_se": np.array([]),
            "mean_improvement": np.array([]),
            "mean_improvement_se": np.array([]),
        }

    normalized_boosts = [(b - mean) / std for b in all_boosts]

    # Compute metrics at each boost level
    first_score_mean_raw = []
    first_score_se_raw = []
    multi_attempt_rate_raw = []
    multi_attempt_rate_se_raw = []
    mean_improvement_raw = []
    mean_improvement_se_raw = []
    success_rate_raw = []
    success_rate_se_raw = []
    pct_improved_of_all_raw = []
    pct_improved_of_all_se_raw = []

    for boost in all_boosts:
        attempts_list = attempts_by_boost[boost]
        first_scores_list = first_scores_by_boost.get(boost, [])
        improvement_list = improvement_by_boost.get(boost, [])
        n_total = len(attempts_list) if attempts_list else 1

        # First-attempt score mean with SE
        if first_scores_list:
            mean_first = np.mean(first_scores_list)
            std_first = np.std(first_scores_list, ddof=1) if len(first_scores_list) > 1 else 0
            se_first = std_first / np.sqrt(len(first_scores_list))
        else:
            mean_first = 0.0
            se_first = 0.0
        first_score_mean_raw.append(mean_first)
        first_score_se_raw.append(se_first)

        # Multi-attempt rate with SE (binomial proportion)
        if attempts_list:
            multi_count = sum(1 for a in attempts_list if a > 1)
            multi_rate = multi_count / n_total
            # SE for binomial proportion: sqrt(p*(1-p)/n)
            se_multi = np.sqrt(multi_rate * (1 - multi_rate) / n_total) if n_total > 0 else 0
        else:
            multi_rate = 0.0
            se_multi = 0.0
        multi_attempt_rate_raw.append(multi_rate)
        multi_attempt_rate_se_raw.append(se_multi)

        # Mean score improvement with SE (among multi-attempt trials only)
        if improvement_list:
            mean_impr = float(np.mean(improvement_list))
            std_impr = float(np.std(improvement_list, ddof=1)) if len(improvement_list) > 1 else 0.0
            se_impr = std_impr / np.sqrt(len(improvement_list))
        else:
            mean_impr = 0.0
            se_impr = 0.0
        mean_improvement_raw.append(mean_impr)
        mean_improvement_se_raw.append(se_impr)

        # Multi-attempt improvement rate (% of multi-attempt trials that improved)
        success_list = success_by_boost.get(boost, [])
        n_multi = len(success_list) if success_list else 0
        if n_multi > 0:
            success_rate = sum(success_list) / n_multi
            se_success = np.sqrt(success_rate * (1 - success_rate) / n_multi)
        else:
            success_rate = 0.0
            se_success = 0.0
        success_rate_raw.append(success_rate)
        success_rate_se_raw.append(se_success)

        # ESR Rate (of ALL responses - multi-attempt AND improved)
        n_improved = sum(success_list) if success_list else 0
        if n_total > 0:
            pct_improved_of_all = n_improved / n_total
            se_improved_of_all = np.sqrt(pct_improved_of_all * (1 - pct_improved_of_all) / n_total)
        else:
            pct_improved_of_all = 0.0
            se_improved_of_all = 0.0
        pct_improved_of_all_raw.append(pct_improved_of_all)
        pct_improved_of_all_se_raw.append(se_improved_of_all)

    # Convert to arrays
    normalized_boosts = np.array(normalized_boosts)
    first_score_mean_raw = np.array(first_score_mean_raw)
    first_score_se_raw = np.array(first_score_se_raw)
    multi_attempt_rate_raw = np.array(multi_attempt_rate_raw)
    multi_attempt_rate_se_raw = np.array(multi_attempt_rate_se_raw)
    mean_improvement_raw = np.array(mean_improvement_raw)
    mean_improvement_se_raw = np.array(mean_improvement_se_raw)
    success_rate_raw = np.array(success_rate_raw)
    success_rate_se_raw = np.array(success_rate_se_raw)
    pct_improved_of_all_raw = np.array(pct_improved_of_all_raw)
    pct_improved_of_all_se_raw = np.array(pct_improved_of_all_se_raw)

    # Apply gaussian smoothing if we have enough points
    if len(normalized_boosts) >= 3:
        first_score_mean = gaussian_filter1d(first_score_mean_raw, sigma=smooth_sigma)
        first_score_se = gaussian_filter1d(first_score_se_raw, sigma=smooth_sigma)
        multi_attempt_rate = gaussian_filter1d(multi_attempt_rate_raw, sigma=smooth_sigma)
        multi_attempt_rate_se = gaussian_filter1d(multi_attempt_rate_se_raw, sigma=smooth_sigma)
        mean_improvement = gaussian_filter1d(mean_improvement_raw, sigma=smooth_sigma)
        mean_improvement_se = gaussian_filter1d(mean_improvement_se_raw, sigma=smooth_sigma)
        success_rate = gaussian_filter1d(success_rate_raw, sigma=smooth_sigma)
        success_rate_se = gaussian_filter1d(success_rate_se_raw, sigma=smooth_sigma)
        pct_improved_of_all = gaussian_filter1d(pct_improved_of_all_raw, sigma=smooth_sigma)
        pct_improved_of_all_se = gaussian_filter1d(pct_improved_of_all_se_raw, sigma=smooth_sigma)
    else:
        first_score_mean = first_score_mean_raw
        first_score_se = first_score_se_raw
        multi_attempt_rate = multi_attempt_rate_raw
        multi_attempt_rate_se = multi_attempt_rate_se_raw
        mean_improvement = mean_improvement_raw
        mean_improvement_se = mean_improvement_se_raw
        success_rate = success_rate_raw
        success_rate_se = success_rate_se_raw
        pct_improved_of_all = pct_improved_of_all_raw
        pct_improved_of_all_se = pct_improved_of_all_se_raw

    return {
        "normalized_boosts": normalized_boosts,
        "first_score_mean": first_score_mean,
        "first_score_se": first_score_se,
        "multi_attempt_rate": multi_attempt_rate,
        "multi_attempt_rate_se": multi_attempt_rate_se,
        "mean_improvement": mean_improvement,
        "mean_improvement_se": mean_improvement_se,
        "success_rate": success_rate,
        "success_rate_se": success_rate_se,
        "pct_improved_of_all": pct_improved_of_all,
        "pct_improved_of_all_se": pct_improved_of_all_se,
    }


def create_model_figure(
    metrics: dict,
    model_info: CanonicalModelInfo,
) -> plt.Figure:
    """
    Create a figure with 3 vertically stacked graphs for a single model.

    Args:
        metrics: Dict from compute_metrics_by_normalized_boost containing
                 normalized_boosts, multi_attempt_rate, success_rate, pct_improved_of_all,
                 and their corresponding _se (standard error) arrays.
        model_info: Model information for coloring.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    color = get_model_color(model_info, scheme="boost_analysis")
    linewidth = 4.0
    z_score = 1.96  # 95% CI

    normalized_boosts = metrics["normalized_boosts"]
    multi_attempt_rate = metrics["multi_attempt_rate"]
    multi_attempt_rate_se = metrics["multi_attempt_rate_se"]
    success_rate = metrics["success_rate"]
    success_rate_se = metrics["success_rate_se"]
    pct_improved_of_all = metrics["pct_improved_of_all"]
    pct_improved_of_all_se = metrics["pct_improved_of_all_se"]

    # Plot 1: Multi-attempt rate
    ax1 = axes[0]
    ax1.plot(normalized_boosts, multi_attempt_rate * 100, color=color, linewidth=linewidth)
    # SE band (convert to percentage)
    ax1.fill_between(
        normalized_boosts,
        (multi_attempt_rate - z_score * multi_attempt_rate_se) * 100,
        (multi_attempt_rate + z_score * multi_attempt_rate_se) * 100,
        color=color, alpha=0.2
    )
    ax1.set_ylabel('Multi-Attempt %', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # Plot 2: Multi-Attempt Improvement Rate (% of multi-attempt trials that improved)
    ax2 = axes[1]
    ax2.plot(normalized_boosts, success_rate * 100, color=color, linewidth=linewidth)
    # SE band (convert to percentage)
    ax2.fill_between(
        normalized_boosts,
        (success_rate - z_score * success_rate_se) * 100,
        (success_rate + z_score * success_rate_se) * 100,
        color=color, alpha=0.2
    )
    ax2.set_ylabel('Multi-Attempt\nImprovement Rate', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)

    # Plot 3: ESR Rate (of ALL responses)
    ax3 = axes[2]
    ax3.plot(normalized_boosts, pct_improved_of_all * 100, color=color, linewidth=linewidth)
    # SE band (convert to percentage)
    ax3.fill_between(
        normalized_boosts,
        (pct_improved_of_all - z_score * pct_improved_of_all_se) * 100,
        (pct_improved_of_all + z_score * pct_improved_of_all_se) * 100,
        color=color, alpha=0.2
    )
    ax3.set_ylabel('ESR Rate', fontsize=16)
    ax3.set_xlabel('Normalized Boost Level (standard deviations from mean)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.set_ylim(0, 5)

    # Add vertical line at 0 (mean boost)
    for ax in axes:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    return fig


def main():
    """Main function to generate multi-boost experiment plots."""
    parser = argparse.ArgumentParser(description="Plot Experiment 2 multi-boost sweep results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots/data (relative paths are resolved from the experiment base dir). Default: plots/",
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

    print("Multi-Boost Experiment Visualization")
    print("=" * 60)

    # Find all multi-boost result files
    result_dir = BASE_DIR / 'experiment_results'
    if args.haiku_only:
        result_dir = result_dir / 'claude_haiku_4_5_20251001_judge'
        print(f"Using haiku judge folder: {result_dir}")
    multi_boost_files = list(result_dir.glob('experiment_multi_boost_*.json'))

    if not multi_boost_files:
        print("No multi-boost result files found!")
        return

    print(f"Found {len(multi_boost_files)} multi-boost result file(s)")

    # Group files by model
    model_files: dict[str, list[Path]] = defaultdict(list)
    model_info_map: dict[str, CanonicalModelInfo] = {}

    for result_file in multi_boost_files:
        parsed = parse_results_filename(result_file)
        if parsed is None:
            print(f"Warning: Could not parse filename: {result_file.name}")
            continue

        model_info = canonicalize_model_name(parsed.model_name)
        model_files[model_info.display_name].append(result_file)
        model_info_map[model_info.display_name] = model_info

    if not model_files:
        print("No valid result files found!")
        return

    # Sort files by modification time within each model (most recent first)
    for model_name in model_files:
        model_files[model_name] = sorted(
            model_files[model_name],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

    # Print which files are being used for each model
    for model_name, files in sorted(model_files.items()):
        print(f"\n{model_name}: {len(files)} file(s)")
        for f in files:
            print(f"  - {f.name}")

    plots_dir = _resolve_output_dir(args.output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Process each model
    for model_name, files in sorted(model_files.items()):
        print(f"\nProcessing {model_name}...")

        # Combine data from all files for this model
        all_attempts_by_boost: dict[float, list[int]] = defaultdict(list)
        all_success_by_boost: dict[float, list[bool]] = defaultdict(list)
        all_improvement_by_boost: dict[float, list[float]] = defaultdict(list)
        all_first_scores_by_boost: dict[float, list[float]] = defaultdict(list)
        mean_sum, std_sum, n_files = 0.0, 0.0, 0

        for result_file in files:
            data = load_multi_boost_data(result_file)
            if data is None:
                continue

            attempts_by_boost, success_by_boost, improvement_by_boost, first_scores_by_boost, mean, std = extract_trial_stats_by_boost(data, args.exclude_degraded)

            # Merge data
            for boost, attempts in attempts_by_boost.items():
                all_attempts_by_boost[boost].extend(attempts)
            for boost, successes in success_by_boost.items():
                all_success_by_boost[boost].extend(successes)
            for boost, improvements in improvement_by_boost.items():
                all_improvement_by_boost[boost].extend(improvements)
            for boost, first_scores in first_scores_by_boost.items():
                all_first_scores_by_boost[boost].extend(first_scores)

            mean_sum += mean
            std_sum += std
            n_files += 1

        if n_files == 0:
            print(f"  No valid data for {model_name}, skipping")
            continue

        # Average the mean and std across files
        avg_mean = mean_sum / n_files
        avg_std = std_sum / n_files

        # Compute metrics
        metrics = compute_metrics_by_normalized_boost(
            all_attempts_by_boost, all_success_by_boost, all_improvement_by_boost, all_first_scores_by_boost, avg_mean, avg_std
        )

        normalized_boosts = metrics["normalized_boosts"]
        if len(normalized_boosts) == 0:
            print(f"  No data points for {model_name}, skipping")
            continue

        # Print summary stats
        total_trials = sum(len(v) for v in all_attempts_by_boost.values())
        total_multi = sum(len(v) for v in all_success_by_boost.values())
        print(f"  Total trials: {total_trials}")
        print(f"  Multi-attempt trials: {total_multi}")
        print(f"  Boost levels: {len(normalized_boosts)}")
        print(f"  Normalized boost range: [{normalized_boosts.min():.2f}, {normalized_boosts.max():.2f}]")

        # Create figure
        model_info = model_info_map[model_name]
        fig = create_model_figure(metrics, model_info)

        # Save figure
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        output_path = plots_dir / f"experiment_2_multi_boost_{safe_model_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)

        # Save data JSON alongside the plot
        data_output_path = output_path.with_suffix('.json')
        plot_data = {
            "model_name": model_name,
            "normalized_boosts": metrics["normalized_boosts"].tolist(),
            "multi_attempt_rate": metrics["multi_attempt_rate"].tolist(),
            "multi_attempt_rate_se": metrics["multi_attempt_rate_se"].tolist(),
            "success_rate": metrics["success_rate"].tolist(),
            "success_rate_se": metrics["success_rate_se"].tolist(),
            "pct_improved_of_all": metrics["pct_improved_of_all"].tolist(),
            "pct_improved_of_all_se": metrics["pct_improved_of_all_se"].tolist(),
            "threshold_stats": {
                "mean": avg_mean,
                "std": avg_std,
            },
            "summary": {
                "total_trials": total_trials,
                "multi_attempt_trials": total_multi,
                "boost_levels": len(normalized_boosts),
            }
        }
        with open(data_output_path, 'w') as f:
            json.dump(plot_data, f, indent=2)
        print(f"  Saved data: {data_output_path}")

    print(f"\nDone! Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
