#!/usr/bin/env python3
"""
Plot no-steering baseline results.

This script analyzes baseline experiments (no feature steering) and produces:
1. Same plots as steering experiments (score diff histogram, multi-attempt rate, mean improvement)
2. First attempt score distribution for each model (for appendix)

Only loads files with 'no_steering_baseline' suffix.
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
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
})

from result_file_utils import (
    parse_results_filename,
    canonicalize_model_name,
    get_model_color,
    CanonicalModelInfo,
    ModelFamily,
)
from plotting.plot_utils import is_degraded_output

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def extract_trial_data(result_files: list[Path], model_info_map: dict[Path, CanonicalModelInfo]) -> dict:
    """
    Extract comprehensive trial data from result files.
    
    Returns dict with:
        - first_attempt_scores: list of first attempt scores per model
        - score_deltas: list of (last - first) for multi-attempt trials
        - attempt_counts: list of attempt counts for all trials
    """
    data = defaultdict(lambda: {
        'first_attempt_scores': [],
        'score_deltas': [],
        'attempt_counts': [],
        'model_info': None,
    })
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            model_info = model_info_map.get(result_file)
            if model_info is None:
                print(f"Warning: No model info for {result_file.name}, skipping")
                continue
            
            model_name = model_info.display_name
            data[model_name]['model_info'] = model_info
            
            for feature_result in results.get('results_by_feature', []):
                if feature_result.get('error'):
                    continue
                
                for trial in feature_result.get('trials', []):
                    if trial.get('error'):
                        continue
                    
                    # Skip degraded outputs (repetitive patterns)
                    response = trial.get('response', '')
                    if is_degraded_output(response):
                        continue
                    
                    score_dict = trial.get('score', {})
                    attempts = score_dict.get('attempts', [])
                    
                    if not attempts:
                        continue
                    
                    scores = [att.get('score') for att in attempts if att.get('score') is not None]
                    
                    if not scores:
                        continue
                    
                    # First attempt score
                    first_score = scores[0]
                    data[model_name]['first_attempt_scores'].append(first_score)
                    
                    # Attempt count
                    num_attempts = len(scores)
                    data[model_name]['attempt_counts'].append(num_attempts)
                    
                    # Score delta (only for multi-attempt)
                    if num_attempts > 1:
                        delta = scores[-1] - scores[0]
                        data[model_name]['score_deltas'].append(delta)
        
        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
            continue
    
    return dict(data)


def create_main_figure(trial_data: dict) -> plt.Figure:
    """
    Create the main comparison figure with:
    - Score delta histograms (for multi-attempt trials)
    - Multi-attempt percentage bar
    - Mean score improvement bar
    """
    # Sort models by parameter count
    sorted_models = sorted(
        trial_data.keys(),
        key=lambda x: trial_data[x]['model_info'].param_count_b if trial_data[x]['model_info'] else 0,
        reverse=True
    )
    
    n_models = len(sorted_models)
    if n_models == 0:
        print("No models with data to plot!")
        return None
    
    fig_height = max(4, n_models * 0.75)
    fig = plt.figure(figsize=(16, fig_height))
    fig.suptitle("No-Steering Baseline Results", fontsize=22, fontweight='bold')
    
    # Get colors
    colors = [
        get_model_color(trial_data[model]['model_info'])
        for model in sorted_models
    ]
    
    # Create histogram subplots
    hist_axes = []
    for i in range(n_models):
        ax = fig.add_subplot(n_models, 1, i+1)
        ax.set_position([0.08, 0.90 - (i+1) * (0.80/n_models), 0.55, 0.75/n_models])
        hist_axes.append(ax)
    
    # Common bin range
    bin_range = (-100, 100)
    n_bins = 30
    bin_width = (bin_range[1] - bin_range[0]) / n_bins
    half_width = bin_width / 2
    bins = np.arange(bin_range[0] - half_width, bin_range[1] + half_width + 1e-9, bin_width)
    
    multi_attempt_pcts = []
    multi_attempt_ses = []
    mean_improvements = []
    mean_improvement_ses = []
    
    for i, model in enumerate(sorted_models):
        ax = hist_axes[i]
        model_data = trial_data[model]
        deltas = model_data['score_deltas']
        attempts = model_data['attempt_counts']
        
        # Histogram
        if deltas:
            ax.hist(deltas, bins=bins, alpha=0.7, color=colors[i], density=True,
                    edgecolor='black', linewidth=0.5)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        if i == n_models // 2:
            ax.set_ylabel("Density", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(bin_range)
        
        if not deltas:
            ax.set_yticklabels([])
        
        # Model name label
        ax.text(0.85, 0.95, model, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
        
        # Stats text
        if deltas:
            improved_pct = sum(1 for d in deltas if d > 0) / len(deltas) * 100
            stats_text = f"{improved_pct:.0f}% improved"
        else:
            stats_text = "No self-correction"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Calculate metrics
        n = len(attempts)
        if n > 0:
            multi_pct = sum(1 for a in attempts if a > 1) / n * 100
            p = multi_pct / 100
            se = np.sqrt(p * (1 - p) / n) * 100
            multi_attempt_pcts.append(multi_pct)
            multi_attempt_ses.append(se)
        else:
            multi_attempt_pcts.append(0)
            multi_attempt_ses.append(0)
        
        # Mean improvement (all trials, 0 for single-attempt)
        all_improvements = []
        for j, count in enumerate(attempts):
            if count > 1 and j < len(deltas):
                all_improvements.append(deltas[j] if j < len(deltas) else 0)
            else:
                all_improvements.append(0)
        
        # Actually use deltas for multi-attempt, 0 for single
        first_scores = model_data['first_attempt_scores']
        improvements = []
        delta_idx = 0
        for count in attempts:
            if count > 1:
                improvements.append(deltas[delta_idx] if delta_idx < len(deltas) else 0)
                delta_idx += 1
            else:
                improvements.append(0)
        
        if improvements:
            mean_imp = np.mean(improvements)
            se_imp = np.std(improvements, ddof=1) / np.sqrt(len(improvements)) if len(improvements) > 1 else 0
        else:
            mean_imp = 0
            se_imp = 0
        mean_improvements.append(mean_imp)
        mean_improvement_ses.append(se_imp)
        
        if i < n_models - 1:
            ax.set_xticklabels([])
    
    hist_axes[-1].set_xlabel("Score Delta (Last - First Attempt)", fontsize=16)
    
    # Bar charts
    bar_ax1 = fig.add_subplot(1, 1, 1)
    bar_ax1.set_position([0.68, 0.1, 0.13, 0.80])
    
    bar_ax2 = fig.add_subplot(1, 1, 1)
    bar_ax2.set_position([0.83, 0.1, 0.13, 0.80])
    
    y_positions = list(range(n_models))[::-1]
    
    # Multi-attempt bar
    z_score = 1.96
    errors1 = [z_score * se for se in multi_attempt_ses]
    bars1 = bar_ax1.barh(y_positions, multi_attempt_pcts, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1, height=0.6,
                         xerr=errors1, error_kw={'capsize': 3})
    
    bar_ax1.set_xlabel("% Multi-Attempt", fontsize=14)
    bar_ax1.set_yticks(y_positions)
    bar_ax1.set_yticklabels([])
    bar_ax1.grid(True, alpha=0.3, axis='x')
    max_val = max(p + e for p, e in zip(multi_attempt_pcts, errors1)) if multi_attempt_pcts else 1
    bar_ax1.set_xlim(0, max(max_val * 1.5, 5))
    
    for bar, value, err in zip(bars1, multi_attempt_pcts, errors1):
        bar_ax1.text(bar.get_width() + err + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{value:.1f}%', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Mean improvement bar
    errors2 = [z_score * se for se in mean_improvement_ses]
    bars2 = bar_ax2.barh(y_positions, mean_improvements, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1, height=0.6,
                         xerr=errors2, error_kw={'capsize': 3})
    
    bar_ax2.set_xlabel("Mean Improvement", fontsize=14)
    bar_ax2.set_yticks(y_positions)
    bar_ax2.set_yticklabels([])
    bar_ax2.grid(True, alpha=0.3, axis='x')
    bar_ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    if mean_improvements:
        max_abs = max(abs(v) + e for v, e in zip(mean_improvements, errors2))
        bar_ax2.set_xlim(-max(max_abs * 1.5, 1), max(max_abs * 1.5, 1))
    
    for bar, value, err in zip(bars2, mean_improvements, errors2):
        offset = 0.05
        if value >= 0:
            bar_ax2.text(bar.get_width() + err + offset, bar.get_y() + bar.get_height()/2,
                         f'{value:.2f}', ha='left', va='center', fontsize=12, fontweight='bold')
        else:
            bar_ax2.text(bar.get_width() - err - offset, bar.get_y() + bar.get_height()/2,
                         f'{value:.2f}', ha='right', va='center', fontsize=12, fontweight='bold')
    
    return fig


def create_first_attempt_figure(trial_data: dict) -> plt.Figure:
    """
    Create a figure showing first attempt score distribution for each model.
    This is for the appendix - shows baseline quality without steering.
    """
    sorted_models = sorted(
        trial_data.keys(),
        key=lambda x: trial_data[x]['model_info'].param_count_b if trial_data[x]['model_info'] else 0,
        reverse=True
    )
    
    n_models = len(sorted_models)
    if n_models == 0:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_models * 0.6)))
    fig.suptitle("First Attempt Scores (No-Steering Baseline)", fontsize=20, fontweight='bold')
    
    colors = [
        get_model_color(trial_data[model]['model_info'])
        for model in sorted_models
    ]
    
    # Left: Histogram of first attempt scores (stacked/overlaid)
    ax1 = axes[0]
    bins = np.linspace(0, 100, 21)
    
    for i, model in enumerate(sorted_models):
        scores = trial_data[model]['first_attempt_scores']
        if scores:
            ax1.hist(scores, bins=bins, alpha=0.5, color=colors[i], label=model,
                     edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel("First Attempt Score", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.set_title("Score Distribution", fontsize=16)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Bar chart of mean first attempt scores with SE
    ax2 = axes[1]
    
    means = []
    ses = []
    for model in sorted_models:
        scores = trial_data[model]['first_attempt_scores']
        if scores:
            means.append(np.mean(scores))
            ses.append(np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0)
        else:
            means.append(0)
            ses.append(0)
    
    y_positions = list(range(n_models))[::-1]
    z_score = 1.96
    errors = [z_score * se for se in ses]
    
    bars = ax2.barh(y_positions, means, color=colors, alpha=0.7,
                    edgecolor='black', linewidth=1, height=0.6,
                    xerr=errors, error_kw={'capsize': 3})
    
    ax2.set_xlabel("Mean First Attempt Score", fontsize=14)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(sorted_models, fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.set_title("Mean Score by Model", fontsize=16)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, value in zip(bars, means):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 f'{value:.1f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def print_summary(trial_data: dict):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("NO-STEERING BASELINE SUMMARY")
    print("="*70)
    
    sorted_models = sorted(
        trial_data.keys(),
        key=lambda x: trial_data[x]['model_info'].param_count_b if trial_data[x]['model_info'] else 0,
        reverse=True
    )
    
    for model in sorted_models:
        data = trial_data[model]
        first_scores = data['first_attempt_scores']
        deltas = data['score_deltas']
        attempts = data['attempt_counts']
        
        print(f"\n{model}:")
        print(f"  Total trials: {len(attempts)}")
        
        if first_scores:
            print(f"  First attempt score: {np.mean(first_scores):.1f} ± {np.std(first_scores):.1f}")
        
        if attempts:
            multi = sum(1 for a in attempts if a > 1)
            print(f"  Multi-attempt trials: {multi} ({multi/len(attempts)*100:.2f}%)")
        
        if deltas:
            print(f"  Score delta (multi-attempt): {np.mean(deltas):+.1f} ± {np.std(deltas):.1f}")
            improved = sum(1 for d in deltas if d > 0)
            print(f"  Improved: {improved}/{len(deltas)} ({improved/len(deltas)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 8 no-steering baseline results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots/data (relative paths are resolved from the experiment base dir). Default: plots/",
    )
    args = parser.parse_args()

    print("No-Steering Baseline Analysis")
    print("=" * 60)
    
    result_dir = BASE_DIR / 'experiment_results'
    
    # Find baseline files only
    baseline_files = list(result_dir.glob('*_no_steering_baseline.json'))
    
    if not baseline_files:
        print("No baseline result files found!")
        print("Looking for files matching: *_no_steering_baseline.json")
        return
    
    print(f"Found {len(baseline_files)} baseline result files:")
    
    # Parse and build model info map
    model_info_map: dict[Path, CanonicalModelInfo] = {}
    model_files: dict[str, list[Path]] = defaultdict(list)
    
    for result_file in baseline_files:
        # Parse the filename (strip the suffix first for parsing)
        # Filename: experiment_results_MODEL_DATE_TIME_no_steering_baseline.json
        base_name = result_file.name.replace('_no_steering_baseline.json', '.json')
        temp_path = result_file.parent / base_name
        
        parsed = parse_results_filename(temp_path)
        if parsed is None:
            print(f"  Warning: Could not parse {result_file.name}")
            continue
        
        model_info = canonicalize_model_name(parsed.model_name)
        model_info_map[result_file] = model_info
        model_files[model_info.display_name].append(result_file)
        print(f"  - {result_file.name} → {model_info.display_name}")
    
    if not model_info_map:
        print("No valid baseline files found!")
        return
    
    # Extract trial data
    all_files = list(model_info_map.keys())
    trial_data = extract_trial_data(all_files, model_info_map)
    
    # Print summary
    print_summary(trial_data)
    
    # Create output directory
    plots_dir = _resolve_output_dir(args.output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data JSON
    output_data = {}
    for model, data in trial_data.items():
        n = len(data['attempt_counts'])
        multi_pct = sum(1 for a in data['attempt_counts'] if a > 1) / n * 100 if n > 0 else 0
        
        output_data[model] = {
            'n_trials': n,
            'mean_first_attempt_score': float(np.mean(data['first_attempt_scores'])) if data['first_attempt_scores'] else 0,
            'std_first_attempt_score': float(np.std(data['first_attempt_scores'])) if data['first_attempt_scores'] else 0,
            'multi_attempt_pct': float(multi_pct),
            'n_multi_attempt': sum(1 for a in data['attempt_counts'] if a > 1),
            'mean_score_delta': float(np.mean(data['score_deltas'])) if data['score_deltas'] else 0,
            'pct_improved': float(sum(1 for d in data['score_deltas'] if d > 0) / len(data['score_deltas']) * 100) if data['score_deltas'] else 0,
        }
    
    data_path = plots_dir / "experiment_8_baseline_data.json"
    with open(data_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 Data saved to: {data_path}")
    
    # Create main figure
    fig1 = create_main_figure(trial_data)
    if fig1:
        path1 = plots_dir / "experiment_8_baseline_main_figure.png"
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        print(f"📊 Main figure saved to: {path1}")
        plt.close(fig1)
    
    # Create first attempt figure
    fig2 = create_first_attempt_figure(trial_data)
    if fig2:
        path2 = plots_dir / "experiment_8_baseline_first_attempt.png"
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        print(f"📊 First attempt figure saved to: {path2}")
        plt.close(fig2)


if __name__ == "__main__":
    main()

