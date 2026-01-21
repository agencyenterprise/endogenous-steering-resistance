"""Plot results for experiment 4: Masked ratio fine-tuning experiments.

Plots ESR metrics for models fine-tuned on varying ratios of masked
self-correction data to normal response data.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional

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


BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "experiment_results"


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_metrics(results: Dict) -> Optional[Dict]:
    """Extract ESR metrics from experiment results."""
    features = results.get("results_by_feature", [])

    multi_attempt = 0
    improved = 0
    first_scores = []
    last_scores = []
    multi_attempt_improvements = []  # Only for multi-attempt trials
    attempt_counts = []

    for r in features:
        if r.get("error"):
            continue
        for t in r.get("trials", []):
            attempts = t.get("score", {}).get("attempts", [])
            if len(attempts) > 0:
                first_score = attempts[0].get("score", 0)
                last_score = attempts[-1].get("score", 0)
                first_scores.append(first_score)
                last_scores.append(last_score)
                attempt_counts.append(len(attempts))
                if len(attempts) > 1:
                    multi_attempt += 1
                    multi_attempt_improvements.append(last_score - first_score)
                    if last_score > first_score:
                        improved += 1

    total = len(first_scores)
    if total == 0:
        return None

    # Calculate means (MSI only for multi-attempt trials)
    mean_improvement = np.mean(multi_attempt_improvements) if multi_attempt_improvements else 0
    mean_attempts = np.mean(attempt_counts) if attempt_counts else 1.0

    # Success rate among multi-attempt trials (not total trials)
    success_rate = 100 * improved / multi_attempt if multi_attempt > 0 else 0

    # Calculate standard errors for 95% CI
    # SE for success rate (binomial among multi-attempt trials)
    p_success = success_rate / 100
    success_rate_se = np.sqrt(p_success * (1 - p_success) / multi_attempt) * 100 if multi_attempt > 0 else 0

    # SE for means: std / sqrt(n)
    mean_improvement_se = np.std(multi_attempt_improvements, ddof=1) / np.sqrt(len(multi_attempt_improvements)) if len(multi_attempt_improvements) > 1 else 0
    mean_attempts_se = np.std(attempt_counts, ddof=1) / np.sqrt(len(attempt_counts)) if len(attempt_counts) > 1 else 0

    return {
        "total_trials": total,
        "multi_attempt_count": multi_attempt,
        "success_rate": success_rate,  # % improved among multi-attempt trials
        "success_rate_se": success_rate_se,
        "mean_improvement": mean_improvement,  # Multi-attempt trials only
        "mean_improvement_se": mean_improvement_se,
        "mean_attempts": mean_attempts,
        "mean_attempts_se": mean_attempts_se,
    }


# Results files for masked-ratio sweep (masked self-correction + normal data)
# The percentage indicates the portion of masked self-correction data
# (the rest is normal response data)
MASKED_RATIO_FILES = {
    "Base": "experiment_results_Meta-Llama-3.1-8B-Instruct_20251030_153434.json",
    "10%": "experiment_results_masked-ratio-10pct-merged_20251224_113713.json",
    "20%": "experiment_results_masked-ratio-20pct-merged_20251224_120122.json",
    "30%": "experiment_results_masked-ratio-30pct-merged_20251224_113713.json",
    "40%": "experiment_results_masked-ratio-40pct-merged_20251224_120050.json",
    "50%": "experiment_results_masked-ratio-50pct-merged_20251224_113718.json",
    "60%": "experiment_results_masked-ratio-60pct-merged_20251224_115617.json",
    "70%": "experiment_results_masked-ratio-70pct-merged_20251224_113756.json",
    "80%": "experiment_results_masked-ratio-80pct-merged_20251224_120109.json",
    "90%": "experiment_results_masked-ratio-90pct-merged_20251224_125704.json",
}


def load_masked_ratio_sweep_data():
    """Load metrics for all masked-ratio sweep models."""
    data = []
    for label, filename in MASKED_RATIO_FILES.items():
        if filename is None:
            print(f"Warning: {label} masked-ratio experiment not yet run, skipping")
            continue

        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping")
            continue

        results = load_results(filepath)
        metrics = extract_metrics(results)

        if metrics is None:
            print(f"Warning: {filename} has no valid trials, skipping")
            continue

        data.append({"label": label, **metrics})

    return data


def plot_masked_ratio_sweep(output_dir: Path):
    """Create a 3-panel line plot showing ESR metrics across different masked-ratio mixes."""
    data = load_masked_ratio_sweep_data()

    if not data:
        print("No valid masked-ratio sweep results found!")
        return

    if len(data) < 2:
        print("Need at least 2 models (base + 1 fine-tuned) to create comparison plot")
        return

    # Extract x-axis values (percentages) - skip Base for the line
    # We'll plot Base as a horizontal reference line
    base_data = next((d for d in data if d["label"] == "Base"), None)
    finetuned_data = [d for d in data if d["label"] != "Base"]

    if not finetuned_data:
        print("No fine-tuned models to plot!")
        return

    # Extract numeric x values from labels like "10%", "20%", etc.
    x_values = []
    for d in finetuned_data:
        label = d["label"]
        try:
            x_values.append(int(label.replace("%", "")))
        except ValueError:
            print(f"Warning: Could not parse label {label}")
            continue

    # Sort by x value
    sorted_pairs = sorted(zip(x_values, finetuned_data), key=lambda p: p[0])
    x_values = [p[0] for p in sorted_pairs]
    finetuned_data = [p[1] for p in sorted_pairs]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    color = "#e67e22"  # Orange for the line
    linewidth = 3.0
    z_score = 1.96  # 95% CI

    # Plot 1: Mean Attempts per Response
    ax1 = axes[0]
    mean_attempts = np.array([d["mean_attempts"] for d in finetuned_data])
    mean_attempts_se = np.array([d["mean_attempts_se"] for d in finetuned_data])

    ax1.plot(x_values, mean_attempts, color=color, linewidth=linewidth, marker='o', markersize=8)
    ax1.fill_between(
        x_values,
        mean_attempts - z_score * mean_attempts_se,
        mean_attempts + z_score * mean_attempts_se,
        color=color, alpha=0.2
    )
    ax1.set_ylabel('Mean Attempts\nper Response', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # Add horizontal reference line for base model
    if base_data:
        ax1.axhline(y=base_data["mean_attempts"], color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax1.legend(fontsize=12, loc='upper left')

    # Plot 2: Multi-Attempt Success Rate
    ax2 = axes[1]
    success_rates = np.array([d["success_rate"] for d in finetuned_data])
    success_rate_se = np.array([d["success_rate_se"] for d in finetuned_data])

    ax2.plot(x_values, success_rates, color=color, linewidth=linewidth, marker='o', markersize=8)
    ax2.fill_between(
        x_values,
        success_rates - z_score * success_rate_se,
        success_rates + z_score * success_rate_se,
        color=color, alpha=0.2
    )
    ax2.set_ylabel('Multi-Attempt\nSuccess Rate (%)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='both', labelsize=14)

    # Add horizontal reference line for base model
    if base_data and base_data["multi_attempt_count"] > 0:
        ax2.axhline(y=base_data["success_rate"], color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax2.legend(fontsize=12, loc='upper left')

    # Plot 3: Mean Score Improvement
    ax3 = axes[2]
    mean_improvements = np.array([d["mean_improvement"] for d in finetuned_data])
    mean_improvement_se = np.array([d["mean_improvement_se"] for d in finetuned_data])

    ax3.plot(x_values, mean_improvements, color=color, linewidth=linewidth, marker='o', markersize=8)
    ax3.fill_between(
        x_values,
        mean_improvements - z_score * mean_improvement_se,
        mean_improvements + z_score * mean_improvement_se,
        color=color, alpha=0.2
    )
    ax3.set_ylabel('Mean Score\nImprovement', fontsize=16)
    ax3.set_xlabel('% Masked Self-Correction Data (rest is Normal Responses)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=14)

    # Add horizontal reference line for base model
    if base_data:
        ax3.axhline(y=base_data["mean_improvement"], color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax3.legend(fontsize=12, loc='upper left')

    # Set x-axis ticks
    ax3.set_xticks(x_values)
    ax3.set_xticklabels([f"{x}%" for x in x_values])

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_4_masked_ratio_sweep.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")
    plt.close()

    # Save data JSON alongside the plot
    data_output_path = output_path.with_suffix('.json')
    plot_data = {
        "models": [
            {
                "label": d["label"],
                "total_trials": d["total_trials"],
                "multi_attempt_count": d["multi_attempt_count"],
                "mean_attempts": d["mean_attempts"],
                "mean_attempts_se": d["mean_attempts_se"],
                "success_rate": d["success_rate"],
                "success_rate_se": d["success_rate_se"],
                "mean_improvement": d["mean_improvement"],
                "mean_improvement_se": d["mean_improvement_se"],
            }
            for d in data
        ]
    }
    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Saved data to {data_output_path}")

    # Print summary
    print("\n" + "="*90)
    print("MASKED-RATIO SWEEP RESULTS")
    print("="*90)
    print(f"{'Model':<10} {'Mean Att':>12} {'Success Rate':>14} {'Mean Improv':>14}")
    print("-"*90)
    for d in data:
        print(f"{d['label']:<10} {d['mean_attempts']:>12.2f} {d['success_rate']:>13.1f}% {d['mean_improvement']:>14.2f}")
    print("="*90)


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 4: MASKED RATIO FINE-TUNING RESULTS")
    print("=" * 70)
    parser = argparse.ArgumentParser(description="Plot Experiment 4 masked-ratio sweep results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots/data (relative paths are resolved from the experiment base dir). Default: plots/",
    )
    args = parser.parse_args()
    plot_masked_ratio_sweep(output_dir=_resolve_output_dir(args.output_dir))
