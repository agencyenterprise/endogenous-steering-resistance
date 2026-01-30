#!/usr/bin/env python3
"""
Experiment 9: Self-Correction Activation Statistics Plots

Generates plots for self-correction activation analysis:
1. Aligned overlay plot - all episodes aligned at correction point
2. Baseline comparison - self-correction vs non-self-correction episodes

Results are loaded from experiment_results/claude_haiku_4_5_20251001_judge/activation_stats/

Adapted from AGI-1635-esr-appendices/self-correction-activation-statistics/plot_activations.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'otd': '#e74c3c',           # Red for OTDs
    'backtracking': '#3498db',  # Blue for backtracking
    'off_topic': '#e74c3c',
    'correction': '#f39c12',
    'on_topic': '#27ae60',
}

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent
HAIKU_RESULTS_DIR = BASE_DIR / "experiment_results" / "claude_haiku_4_5_20251001_judge"
ACTIVATION_STATS_DIR = HAIKU_RESULTS_DIR / "activation_stats"


def generate_aligned_overlay_data(
    metadata_file: Path,
    activations_dir: Path,
    window_before: int = 150,
    window_after: int = 150,
) -> dict:
    """Generate data for aligned_overlay plot."""

    with open(metadata_file) as f:
        data = json.load(f)

    episodes = data['episodes']
    otd_indices = data.get('off_topic_detectors', [])
    bt_indices = data.get('backtracking_latents', [])

    otd_aligned = []
    bt_aligned = []
    correction_lengths = []

    for ep in episodes:
        episode_id = ep['episode_id']
        act_file = activations_dir / f"{episode_id}.npz"
        if not act_file.exists():
            continue

        loaded = np.load(act_file)
        correction_token = ep['correction_start_token']
        if correction_token < 0:
            continue

        # Track correction region length
        on_topic_token = ep.get('on_topic_start_token', 0)
        if on_topic_token > correction_token:
            correction_lengths.append(on_topic_token - correction_token)

        # Mean OTD activation per token
        otd_acts = []
        for idx in otd_indices:
            if str(idx) in loaded.files:
                otd_acts.append(loaded[str(idx)])

        if not otd_acts:
            continue

        mean_otd = np.mean(otd_acts, axis=0)

        # Mean backtracking activation per token
        bt_acts = []
        for idx in bt_indices:
            if str(idx) in loaded.files:
                bt_acts.append(loaded[str(idx)])

        mean_bt = np.mean(bt_acts, axis=0) if bt_acts else np.zeros_like(mean_otd)

        # Extract window around correction point
        start = correction_token - window_before
        end = correction_token + window_after

        padded_otd = np.zeros(window_before + window_after)
        padded_bt = np.zeros(window_before + window_after)

        src_start = max(0, -start)
        src_end = min(len(mean_otd) - start, window_before + window_after)
        dst_start = max(0, start)
        dst_end = min(len(mean_otd), end)

        if dst_end > dst_start and src_end > src_start:
            actual_len = min(src_end - src_start, dst_end - dst_start)
            padded_otd[src_start:src_start + actual_len] = mean_otd[dst_start:dst_start + actual_len]
            padded_bt[src_start:src_start + actual_len] = mean_bt[dst_start:dst_start + actual_len]

        otd_aligned.append(padded_otd)
        bt_aligned.append(padded_bt)

    if not otd_aligned:
        return {"error": "No valid episodes"}

    otd_matrix = np.array(otd_aligned)
    bt_matrix = np.array(bt_aligned)

    median_correction_len = int(np.median(correction_lengths)) if correction_lengths else 20

    otd_mean = np.mean(otd_matrix, axis=0)
    otd_std = np.std(otd_matrix, axis=0)
    otd_sem = otd_std / np.sqrt(len(otd_aligned))
    bt_mean = np.mean(bt_matrix, axis=0)
    bt_std = np.std(bt_matrix, axis=0)
    bt_sem = bt_std / np.sqrt(len(bt_aligned))

    def smooth(arr, window=5):
        return np.convolve(arr, np.ones(window)/window, mode='same')

    x_values = list(range(-window_before, window_after))

    return {
        "description": "Aligned activation patterns - all episodes aligned at correction point",
        "x_axis": {
            "label": "Tokens Relative to Correction Start",
            "values": x_values
        },
        "y_axis": {
            "label": "Mean Activation"
        },
        "n_episodes": len(otd_aligned),
        "window_before": window_before,
        "window_after": window_after,
        "regions": {
            "off_topic": {"start": -window_before, "end": 0},
            "correction": {"start": 0, "end": median_correction_len},
            "on_topic": {"start": median_correction_len, "end": window_after}
        },
        "median_correction_length": median_correction_len,
        "correction_lengths": correction_lengths,
        "series": [
            {
                "name": "Off-Topic Detectors",
                "mean": [float(x) for x in otd_mean],
                "mean_smoothed": [float(x) for x in smooth(otd_mean)],
                "std": [float(x) for x in otd_std],
                "sem": [float(x) for x in otd_sem]
            },
            {
                "name": "Backtracking Latents",
                "mean": [float(x) for x in bt_mean],
                "mean_smoothed": [float(x) for x in smooth(bt_mean)],
                "std": [float(x) for x in bt_std],
                "sem": [float(x) for x in bt_sem]
            }
        ]
    }


def plot_aligned_overlay(data: dict, output_dir: Path):
    """Create aligned overlay plot from JSON data."""
    if "error" in data:
        print(f"  Warning: {data['error']}")
        return

    x = np.array(data["x_axis"]["values"])
    otd_series = data["series"][0]
    bt_series = data["series"][1]
    correction_lengths = data["correction_lengths"]
    median_correction_len = data["median_correction_length"]
    window_before = data["window_before"]
    window_after = data["window_after"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # OTD
    ax.plot(x, otd_series["mean_smoothed"], color=COLORS['otd'], linewidth=2, label='Off-Topic Detectors')
    otd_sem = np.array(otd_series["sem"])
    otd_mean = np.array(otd_series["mean_smoothed"])
    ax.fill_between(x, otd_mean - otd_sem, otd_mean + otd_sem, color=COLORS['otd'], alpha=0.3)

    # Backtracking
    ax.plot(x, bt_series["mean_smoothed"], color=COLORS['backtracking'], linewidth=2, label='Backtracking Latents')
    bt_sem = np.array(bt_series["sem"])
    bt_mean = np.array(bt_series["mean_smoothed"])
    ax.fill_between(x, bt_mean - bt_sem, bt_mean + bt_sem, color=COLORS['backtracking'], alpha=0.3)

    # Shade off-topic region
    ax.axvspan(-window_before, 0, alpha=0.1, color=COLORS['off_topic'])

    # Shade correction regions
    for corr_len in correction_lengths:
        ax.axvspan(0, min(corr_len, window_after), alpha=0.015, color=COLORS['correction'])

    # Mark correction start
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Tokens Relative to Correction Start', fontsize=12)
    ax.set_ylabel('Mean Activation', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-window_before, window_after)

    # Add region labels
    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.92
    ax.text(-window_before * 0.5, label_y, 'Off-topic', ha='center', fontsize=10, color='gray')
    ax.text(median_correction_len * 1.5, label_y, 'Correction\n(fading)', ha='center', fontsize=10, color='gray')
    ax.text(window_after * 0.6, label_y, 'On-topic', ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    png_file = output_dir / "experiment_9_aligned_overlay.png"
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_file}")


def generate_baseline_comparison_data(results: dict, baseline_file: Path) -> dict:
    """Generate data for baseline_comparison plot."""
    if not baseline_file.exists():
        return {"error": f"Baseline file not found: {baseline_file}"}

    with open(baseline_file) as f:
        baseline = json.load(f)

    otd = results['otd_group']

    ratio_off = otd['mean_off_topic'] / baseline['otd_mean'] if baseline['otd_mean'] > 0 else None
    ratio_on = otd['mean_on_topic'] / baseline['otd_mean'] if baseline['otd_mean'] > 0 else None

    return {
        "description": "Off-Topic Detector activation comparison: self-correction vs baseline episodes",
        "x_axis": {
            "label": "Condition",
            "categories": [
                "Self-Correction (Off-topic)",
                "Self-Correction (On-topic)",
                "No Self-Correction (Baseline)"
            ]
        },
        "y_axis": {
            "label": "Mean OTD Activation"
        },
        "bars": [
            {
                "category": "Self-Correction (Off-topic)",
                "mean": otd['mean_off_topic'],
                "std": otd['std_off_topic'],
                "n_episodes": results['n_episodes']
            },
            {
                "category": "Self-Correction (On-topic)",
                "mean": otd['mean_on_topic'],
                "std": otd['std_on_topic'],
                "n_episodes": results['n_episodes']
            },
            {
                "category": "No Self-Correction (Baseline)",
                "mean": baseline['otd_mean'],
                "std": baseline['otd_std'],
                "n_episodes": baseline['n_episodes']
            }
        ],
        "statistics": {
            "self_correction_p_value": otd['wilcoxon_p'],
            "self_correction_cohens_d": otd['cohens_d'],
            "ratio_off_topic_vs_baseline": ratio_off,
            "ratio_on_topic_vs_baseline": ratio_on
        }
    }


def plot_baseline_comparison(data: dict, output_dir: Path):
    """Create baseline comparison plot from JSON data."""
    if "error" in data:
        print(f"  Warning: {data['error']}")
        return

    categories = ['Self-Correction\n(Off-topic)', 'Self-Correction\n(On-topic)', 'No Self-Correction\n(Baseline)']
    means = [bar["mean"] for bar in data["bars"]]
    stds = [bar["std"] for bar in data["bars"]]
    colors = [COLORS['off_topic'], COLORS['on_topic'], '#95a5a6']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Mean OTD Activation', fontsize=12)

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f'{mean:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    png_file = output_dir / "experiment_9_baseline_comparison.png"
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 9 plots (activation statistics)")
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
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else (BASE_DIR / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 9: Activation Statistics Plots")
    print("=" * 60)
    print(f"Data directory: {ACTIVATION_STATS_DIR}")
    print(f"Output directory: {output_dir}")

    # Check if data exists
    results_file = ACTIVATION_STATS_DIR / "analysis_results.json"
    metadata_file = ACTIVATION_STATS_DIR / "episode_metadata.json"
    activations_dir = ACTIVATION_STATS_DIR / "activations"
    baseline_file = ACTIVATION_STATS_DIR / "baseline_stats.json"

    if not ACTIVATION_STATS_DIR.exists():
        print(f"\nWarning: Activation stats directory not found: {ACTIVATION_STATS_DIR}")
        print("Run experiment_9_activation_stats first.")
        return

    # Generate aligned overlay plot
    if metadata_file.exists() and activations_dir.exists():
        print("\nGenerating aligned overlay plot...")
        data = generate_aligned_overlay_data(metadata_file, activations_dir)

        # Save JSON
        json_file = output_dir / "experiment_9_aligned_overlay.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {json_file}")

        # Generate PNG
        plot_aligned_overlay(data, output_dir)
    else:
        print(f"  Skipping aligned overlay - missing metadata or activations")

    # Generate baseline comparison plot
    if results_file.exists():
        print("\nGenerating baseline comparison plot...")
        with open(results_file) as f:
            results = json.load(f)

        data = generate_baseline_comparison_data(results, baseline_file)

        # Save JSON
        json_file = output_dir / "experiment_9_baseline_comparison.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {json_file}")

        # Generate PNG
        plot_baseline_comparison(data, output_dir)
    else:
        print(f"  Skipping baseline comparison - missing {results_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
