#!/usr/bin/env python3
"""
Plot ESR (Endogenous Steering Resistance) results for non-ablation experiments.
Reproduces the combined figure visualization from ICLR-2025 analysis.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Increase font sizes for all text elements
plt.rcParams.update({
    'font.size': 16,           # Base font size (default is 10)
    'axes.titlesize': 20,      # Title font size
    'axes.labelsize': 16,      # Axis label font size
    'xtick.labelsize': 14,     # X-axis tick label size
    'ytick.labelsize': 14,     # Y-axis tick label size
    'legend.fontsize': 14,     # Legend font size
    'figure.titlesize': 22,    # Figure title font size
})

from result_file_utils import (
    parse_results_filename,
    canonicalize_model_name,
    get_model_color,
    CanonicalModelInfo,
    ModelFamily,
)
from plotting.plot_utils import is_degraded_output, collect_experiment_1_result_files

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def extract_score_deltas_and_attempts(
    result_files: list[Path],
    model_info_map: dict[Path, CanonicalModelInfo]
) -> tuple[dict[str, list[float]], dict[str, list[int]], dict[str, CanonicalModelInfo]]:
    """
    Extract score deltas (first to last attempt) and attempt counts by model.

    Args:
        result_files: List of result file paths
        model_info_map: Mapping from filepath to CanonicalModelInfo

    Returns:
        - model_deltas: Dict mapping display names to lists of score deltas (only multi-attempt trials)
        - model_attempts: Dict mapping display names to lists of attempt counts (all trials)
        - model_infos: Dict mapping display names to CanonicalModelInfo (for sorting/coloring)
    """
    model_deltas = defaultdict(list)
    model_attempts = defaultdict(list)
    model_infos: dict[str, CanonicalModelInfo] = {}

    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Get canonical model info from the pre-computed map
            model_info = model_info_map.get(result_file)
            if model_info is None:
                print(f"Warning: No model info for {result_file.name}, skipping")
                continue

            model_name = model_info.display_name
            model_infos[model_name] = model_info

            # Process trials
            for feature_result in data.get('results_by_feature', []):
                if feature_result.get('error'):
                    continue

                for trial in feature_result.get('trials', []):
                    if trial.get('error'):
                        continue

                    # Skip degraded outputs (repetitive patterns)
                    response = trial.get('response', '')
                    if is_degraded_output(response):
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
                    model_attempts[model_name].append(num_attempts)

                    # Only include trials with multiple attempts for delta calculation
                    if num_attempts > 1:
                        first_score = scores[0]
                        last_score = scores[-1]
                        delta = last_score - first_score
                        model_deltas[model_name].append(delta)

        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
            continue

    # Debug: print all models found
    print(f"\n[DEBUG] Models found in model_deltas: {list(model_deltas.keys())}")
    print(f"[DEBUG] Models found in model_attempts: {list(model_attempts.keys())}")
    for model in model_attempts.keys():
        info = model_infos.get(model)
        finetuned_str = " (finetuned)" if info and info.is_finetuned else ""
        print(f"[DEBUG]   {model}{finetuned_str}: {len(model_deltas.get(model, []))} deltas, {len(model_attempts.get(model, []))} attempts")

    # Sort models by parameter count (descending)
    # Use model_attempts.keys() to include models with no multi-attempt trials
    sorted_models = sorted(
        model_attempts.keys(),
        key=lambda x: model_infos.get(x, CanonicalModelInfo("", ModelFamily.UNKNOWN, False)).param_count_b,
        reverse=True
    )

    # Return data in sorted order
    # Use .get() for model_deltas since models with no multi-attempt trials won't have entries
    sorted_deltas = {model: model_deltas.get(model, []) for model in sorted_models}
    sorted_attempts = {model: model_attempts[model] for model in sorted_models}
    sorted_infos = {model: model_infos[model] for model in sorted_models if model in model_infos}

    return sorted_deltas, sorted_attempts, sorted_infos


def calculate_mean_improvements(
    result_files: list[Path],
    model_info_map: dict[Path, CanonicalModelInfo]
) -> dict[str, list[float]]:
    """Calculate mean improvements across multi-attempt trials only."""
    model_improvements = defaultdict(list)

    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Get canonical model info from the pre-computed map
            model_info = model_info_map.get(result_file)
            if model_info is None:
                continue

            model_name = model_info.display_name

            # Process trials
            for feature_result in data.get('results_by_feature', []):
                if feature_result.get('error'):
                    continue

                for trial in feature_result.get('trials', []):
                    if trial.get('error'):
                        continue

                    # Skip degraded outputs (repetitive patterns)
                    response = trial.get('response', '')
                    if is_degraded_output(response):
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

                    # Only include multi-attempt trials
                    if len(scores) > 1:
                        first_score = scores[0]
                        last_score = scores[-1]
                        improvement = last_score - first_score
                        model_improvements[model_name].append(improvement)

        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
            continue

    return model_improvements


def create_combined_figure(
    model_deltas: dict[str, list[float]],
    model_attempts: dict[str, list[int]],
    model_improvements: dict[str, list[float]],
    model_infos: dict[str, CanonicalModelInfo]
) -> plt.Figure:
    """Create combined histograms with aligned horizontal bar charts."""

    # Include all models with attempts (even those with no multi-attempt deltas)
    models_to_plot = {k: v for k, v in model_deltas.items() if k in model_attempts and len(model_attempts[k]) > 0}
    filtered_attempts = {k: v for k, v in model_attempts.items() if len(v) > 0}

    if not models_to_plot:
        print("No models with data to plot!")
        return None

    # Set up the figure with custom layout
    n_models = len(models_to_plot)
    fig_height = max(4, n_models * 0.75)
    fig = plt.figure(figsize=(16, fig_height))

    # Get colors from model_infos
    colors = [
        get_model_color(model_infos[model]) if model in model_infos else "#888888"
        for model in models_to_plot.keys()
    ]

    # Create histogram subplots (left side)
    hist_axes = []
    for i in range(n_models):
        ax = fig.add_subplot(n_models, 1, i+1)
        # Position: [left, bottom, width, height]
        ax.set_position([0.08, 0.95 - (i+1) * (0.85/n_models), 0.55, 0.8/n_models])
        hist_axes.append(ax)

    # Common bin range for all histograms
    bin_range = (-100, 100)
    n_bins = 30
    bin_width = (bin_range[1] - bin_range[0]) / n_bins
    # Offset bins by half width so 0 is centered in a bin, not at an edge
    half_width = bin_width / 2
    bins = np.arange(bin_range[0] - half_width, bin_range[1] + half_width + 1e-9, bin_width)

    # Calculate mean improvements and standard errors for each model
    mean_improvements = []
    mean_improvement_ses = []
    for model in models_to_plot.keys():
        if model in model_improvements and len(model_improvements[model]) > 0:
            improvements = model_improvements[model]
            mean_improvement = np.mean(improvements)
            # SE = std / sqrt(n)
            se = np.std(improvements, ddof=1) / np.sqrt(len(improvements)) if len(improvements) > 1 else 0
            mean_improvements.append(mean_improvement)
            mean_improvement_ses.append(se)
        else:
            mean_improvements.append(0)
            mean_improvement_ses.append(0)

    # Create histogram for each model
    multi_attempt_percentages = []
    multi_attempt_ses = []

    for i, (model, deltas) in enumerate(models_to_plot.items()):
        ax = hist_axes[i]

        # Create histogram only if there are deltas (multi-attempt trials)
        if deltas:
            ax.hist(deltas, bins=bins, alpha=0.7, color=colors[i], density=True,
                    edgecolor='black', linewidth=0.5, label=model)

        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)

        # Labels and formatting
        # Only show "Density" ylabel on the middle histogram
        if i == n_models // 2:
            ax.set_ylabel("Density", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(bin_range)

        # Hide y-axis tick labels for models with no data
        if not deltas:
            ax.set_yticklabels([])

        # Add model name on the right side
        ax.text(0.85, 0.95, model, transform=ax.transAxes, fontsize=16,
               verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')

        # Add statistics as text on the left
        if deltas:
            improved_pct = sum(1 for d in deltas if d > 0) / len(deltas) * 100
            stats_text = f"{improved_pct:.0f}% improved"
        else:
            stats_text = "No self-correction"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Calculate multi-attempt percentage and SE for bar chart
        attempts = filtered_attempts.get(model, [])
        if attempts:
            n = len(attempts)
            multi_pct = sum(1 for a in attempts if a > 1) / n * 100
            # SE for binomial proportion (in percentage points): sqrt(p*(1-p)/n) * 100
            p = multi_pct / 100
            se = np.sqrt(p * (1 - p) / n) * 100 if n > 0 else 0
            multi_attempt_percentages.append(multi_pct)
            multi_attempt_ses.append(se)
        else:
            multi_attempt_percentages.append(0)
            multi_attempt_ses.append(0)

        # Remove x-axis labels for all but bottom
        if i < n_models - 1:
            ax.set_xticklabels([])

    # Set common x-label only on bottom subplot
    hist_axes[-1].set_xlabel("Score Delta (Last Attempt - First Attempt)", fontsize=16)

    # Create first horizontal bar chart (multi-attempt percentage)
    bar_ax1 = fig.add_subplot(1, 1, 1)
    bar_ax1.set_position([0.68, 0.1, 0.13, 0.85])

    # Create second horizontal bar chart (mean score improvement)
    bar_ax2 = fig.add_subplot(1, 1, 1)
    bar_ax2.set_position([0.83, 0.1, 0.13, 0.85])

    # Calculate y positions aligned with histogram rows
    y_positions = []
    for i in range(n_models):
        y_pos = 0.95 - (i+0.5) * (0.85/n_models)
        y_positions.append(y_pos)

    # Normalize y positions to 0-1 range for bar chart
    if n_models == 1:
        y_bar_positions = [0]
    else:
        y_normalized = [(y - min(y_positions)) / (max(y_positions) - min(y_positions)) for y in y_positions]
        y_bar_positions = [y * (n_models - 1) for y in y_normalized]

    # First bar chart: Multi-attempt percentages with 95% CI error bars
    z_score = 1.96  # 95% CI
    multi_attempt_errors = [z_score * se for se in multi_attempt_ses]
    bars1 = bar_ax1.barh(y_bar_positions, multi_attempt_percentages, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1, height=0.6,
                         xerr=multi_attempt_errors, error_kw={'capsize': 3, 'capthick': 1, 'elinewidth': 1})

    # Format first bar chart
    bar_ax1.set_xlabel("% Multi-Attempt", fontsize=16)
    bar_ax1.set_yticks(y_bar_positions)
    bar_ax1.set_yticklabels([])
    bar_ax1.grid(True, alpha=0.3, axis='x')
    max_with_error = max(p + e for p, e in zip(multi_attempt_percentages, multi_attempt_errors)) if multi_attempt_percentages else 10
    bar_ax1.set_xlim(0, max_with_error * 1.3)

    # Add value labels on first bar chart (offset to account for error bars)
    for bar, value, err in zip(bars1, multi_attempt_percentages, multi_attempt_errors):
        width = bar.get_width() + err
        bar_ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{value:.1f}%', ha='left', va='center', fontsize=14, fontweight='bold')

    # Second bar chart: Mean score improvements with SE error bars
    mean_improvement_errors = [z_score * se for se in mean_improvement_ses]
    bars2 = bar_ax2.barh(y_bar_positions, mean_improvements, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1, height=0.6,
                         xerr=mean_improvement_errors, error_kw={'capsize': 3, 'capthick': 1, 'elinewidth': 1})

    # Format second bar chart
    bar_ax2.set_xlabel("Mean Score Improvement", fontsize=16)
    bar_ax2.set_yticks(y_bar_positions)
    bar_ax2.set_yticklabels([])
    bar_ax2.grid(True, alpha=0.3, axis='x')

    # Set x-axis limits to accommodate both positive and negative values plus error bars
    if mean_improvements:
        max_abs_with_error = max(abs(val) + err for val, err in zip(mean_improvements, mean_improvement_errors))
        bar_ax2.set_xlim(-max_abs_with_error * 1.15, max_abs_with_error * 1.15)
    else:
        bar_ax2.set_xlim(-1, 1)

    # Add vertical line at zero
    bar_ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)

    # Add value labels on second bar chart (offset to account for error bars)
    for bar, value, err in zip(bars2, mean_improvements, mean_improvement_errors):
        width = bar.get_width()
        if mean_improvements and max(abs(val) for val in mean_improvements) > 0:
            offset = max(abs(val) + e for val, e in zip(mean_improvements, mean_improvement_errors)) * 0.02
        else:
            offset = 0.01

        if value >= 0:
            bar_ax2.text(width + err + offset, bar.get_y() + bar.get_height()/2,
                         f'{value:.1f}', ha='left', va='center', fontsize=14, fontweight='bold')
        else:
            bar_ax2.text(width - err - offset, bar.get_y() + bar.get_height()/2,
                         f'{value:.1f}', ha='right', va='center', fontsize=14, fontweight='bold')

    return fig


def print_summary_stats(model_deltas: dict[str, list[float]], model_attempts: dict[str, list[int]]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("ESR ANALYSIS SUMMARY (Non-Ablation Experiments)")
    print("="*60)

    for model in model_deltas.keys():
        deltas = model_deltas[model]
        attempts = model_attempts[model]

        print(f"\n{model}:")
        print(f"  Total trials: {len(attempts)}")

        if len(attempts) > 0:
            multi_attempts = sum(1 for a in attempts if a > 1)
            print(f"  Trials with multiple attempts: {multi_attempts} ({multi_attempts/len(attempts)*100:.1f}%)")
            print(f"  Mean attempts per trial: {np.mean(attempts):.2f}")

            if deltas:
                print(f"  Score delta stats (first→last attempt, multi-attempt only):")
                print(f"    Mean: {np.mean(deltas):+.2f}")
                print(f"    Median: {np.median(deltas):+.2f}")
                print(f"    Std: {np.std(deltas):.2f}")
                print(f"    Improved (delta > 0): {sum(1 for d in deltas if d > 0)}/{len(deltas)} ({sum(1 for d in deltas if d > 0)/len(deltas)*100:.1f}%)")
            else:
                print(f"  No trials with multiple attempts")
        else:
            print(f"  No trials found")


def main():
    """Main function to generate ESR plots."""
    parser = argparse.ArgumentParser(description="Plot ESR results for Experiment 1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots/data (relative paths are resolved from the experiment base dir). Default: plots/",
    )
    args = parser.parse_args()

    print("ESR Visualization for Non-Ablation Experiments")
    print("=" * 60)

    # Model families to exclude (using ModelFamily enum values)
    EXCLUDED_FAMILIES = {
        ModelFamily.FINETUNED_8B,  # Exclude all finetuned models
    }

    selected_files, model_info_map, model_files = collect_experiment_1_result_files(
        BASE_DIR,
        excluded_families=EXCLUDED_FAMILIES,
    )

    if not model_files:
        print("No valid result files found!")
        return

    # Print which files are being used for each model
    for model_name, files in sorted(model_files.items()):
        model_info = model_info_map.get(files[0]) if files else None
        family_str = f" [{model_info.family}]" if model_info else ""
        print(f"\n{model_name}{family_str}: {len(files)} file(s)")
        for f in files:
            print(f"  - {f.name}")

    print(f"\nProcessing {len(model_files)} models with {sum(len(files) for files in model_files.values())} total files...")

    # Extract data from all files
    model_deltas, model_attempts, model_infos = extract_score_deltas_and_attempts(selected_files, model_info_map)
    model_improvements = calculate_mean_improvements(selected_files, model_info_map)

    if not model_deltas:
        print("No data found to analyze!")
        return

    # Print summary statistics
    print_summary_stats(model_deltas, model_attempts)

    # Create output data structure
    output_data = {}
    for model_name in model_deltas.keys():
        deltas = model_deltas[model_name]
        attempts = model_attempts[model_name]
        improvements = model_improvements.get(model_name, [])

        # Calculate statistics
        n_attempts = len(attempts)
        multi_attempt_pct = (sum(1 for a in attempts if a > 1) / n_attempts * 100) if attempts else 0
        # SE for binomial proportion
        p = multi_attempt_pct / 100
        multi_attempt_se = np.sqrt(p * (1 - p) / n_attempts) * 100 if n_attempts > 0 else 0

        mean_score_improvement = float(np.mean(improvements)) if improvements else 0
        # SE for mean
        mean_improvement_se = float(np.std(improvements, ddof=1) / np.sqrt(len(improvements))) if len(improvements) > 1 else 0

        pct_improved = (sum(1 for d in deltas if d > 0) / len(deltas) * 100) if deltas else 0

        output_data[model_name] = {
            "score_deltas": [float(d) for d in deltas],  # Multi-attempt only
            "multi_attempt_pct": float(multi_attempt_pct),
            "multi_attempt_se": float(multi_attempt_se),
            "mean_score_improvement": float(mean_score_improvement),  # Multi-attempt only
            "mean_score_improvement_se": float(mean_improvement_se),
            "pct_of_multi_attempts_improved": float(pct_improved),
            "n_trials": n_attempts,
        }

    plots_dir = _resolve_output_dir(args.output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save the data JSON
    data_output_path = plots_dir / "experiment_1_esr_data.json"
    with open(data_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 ESR data saved to: {data_output_path}")

    # Create the combined figure
    fig = create_combined_figure(model_deltas, model_attempts, model_improvements, model_infos)

    if fig is None:
        print("\nNo data available for plotting.")
        return

    # Save the plot
    plot_output_path = plots_dir / "experiment_1_esr_combined_figure.png"
    fig.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"📊 ESR plot saved to: {plot_output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
