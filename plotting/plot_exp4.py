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
RESULTS_DIR = BASE_DIR / "data" / "experiment_results"
HAIKU_RESULTS_DIR = BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge"


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

    # ESR Rate (of ALL responses) - responses with multi-attempt AND improvement
    pct_improved_of_all = 100 * improved / total if total > 0 else 0
    p_improved_all = pct_improved_of_all / 100
    pct_improved_of_all_se = np.sqrt(p_improved_all * (1 - p_improved_all) / total) * 100 if total > 0 else 0

    return {
        "total_trials": total,
        "multi_attempt_count": multi_attempt,
        "improved_count": improved,
        "success_rate": success_rate,  # % improved among multi-attempt trials
        "success_rate_se": success_rate_se,
        "pct_improved_of_all": pct_improved_of_all,
        "pct_improved_of_all_se": pct_improved_of_all_se,
        "mean_improvement": mean_improvement,  # Multi-attempt trials only
        "mean_improvement_se": mean_improvement_se,
        "mean_attempts": mean_attempts,
        "mean_attempts_se": mean_attempts_se,
    }


# Percentages to look for in masked-ratio sweep
MASKED_RATIO_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def load_masked_ratio_sweep_data(results_dir: Path = RESULTS_DIR):
    """Load metrics for all masked-ratio sweep models by globbing for files."""
    data = []

    # Find base 8B file (non-finetuned)
    base_files = sorted(results_dir.glob("experiment_results_Meta-Llama-3.1-8B-Instruct_*.json"))
    # Exclude finetuned, fresh_prompts, and other variant files
    base_files = [f for f in base_files if "fresh_prompts" not in f.name and "no_steering" not in f.name]
    if base_files:
        # Use the most recent one
        base_file = base_files[-1]
        results = load_results(base_file)
        metrics = extract_metrics(results)
        if metrics:
            data.append({"label": "Base", **metrics})
            print(f"Loaded Base: {base_file.name}")
    else:
        print("Warning: Base 8B file not found")

    # Find masked-ratio files for each percentage
    for pct in MASKED_RATIO_PCTS:
        pattern = f"experiment_results_masked-ratio-{pct}pct-merged_*.json"
        files = sorted(results_dir.glob(pattern))
        if not files:
            print(f"Warning: {pct}% masked-ratio file not found, skipping")
            continue

        # Use the most recent one
        filepath = files[-1]
        results = load_results(filepath)
        metrics = extract_metrics(results)

        if metrics is None:
            print(f"Warning: {filepath.name} has no valid trials, skipping")
            continue

        data.append({"label": f"{pct}%", **metrics})
        print(f"Loaded {pct}%: {filepath.name}")

    return data


def plot_masked_ratio_sweep(output_dir: Path, results_dir: Path = RESULTS_DIR):
    """Create a 3-panel line plot showing ESR metrics across different masked-ratio mixes."""
    data = load_masked_ratio_sweep_data(results_dir)

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

    # Plot 1: % Multi-Attempt
    ax1 = axes[0]
    # Calculate % multi-attempt for each model
    multi_pct = np.array([100 * d["multi_attempt_count"] / d["total_trials"] for d in finetuned_data])
    # SE for binomial proportion
    multi_pct_se = np.array([
        100 * np.sqrt((d["multi_attempt_count"] / d["total_trials"]) * (1 - d["multi_attempt_count"] / d["total_trials"]) / d["total_trials"])
        for d in finetuned_data
    ])

    ax1.plot(x_values, multi_pct, color=color, linewidth=linewidth, marker='o', markersize=8)
    ax1.fill_between(
        x_values,
        multi_pct - z_score * multi_pct_se,
        multi_pct + z_score * multi_pct_se,
        color=color, alpha=0.2
    )
    ax1.set_ylabel('Multi-Attempt %', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # Add horizontal reference line for base model
    if base_data:
        base_multi_pct = 100 * base_data["multi_attempt_count"] / base_data["total_trials"]
        ax1.axhline(y=base_multi_pct, color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax1.legend(fontsize=12, loc='upper left')

    # Plot 2: Multi-Attempt Improvement Rate (% of multi-attempt that improved)
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
    ax2.set_ylabel('Multi-Attempt\nImprovement Rate', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)

    # Add horizontal reference line for base model
    if base_data:
        ax2.axhline(y=base_data["success_rate"], color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax2.legend(fontsize=12, loc='upper left')

    # Plot 3: ESR Rate (of ALL responses)
    ax3 = axes[2]
    pct_improved_of_all = np.array([d["pct_improved_of_all"] for d in finetuned_data])
    pct_improved_of_all_se = np.array([d["pct_improved_of_all_se"] for d in finetuned_data])

    ax3.plot(x_values, pct_improved_of_all, color=color, linewidth=linewidth, marker='o', markersize=8)
    ax3.fill_between(
        x_values,
        pct_improved_of_all - z_score * pct_improved_of_all_se,
        pct_improved_of_all + z_score * pct_improved_of_all_se,
        color=color, alpha=0.2
    )
    ax3.set_ylabel('ESR Rate', fontsize=16)
    ax3.set_xlabel('% Masked Self-Correction Data (rest is Normal Responses)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=14)
    max_improved = max(pct_improved_of_all) + z_score * max(pct_improved_of_all_se)
    if base_data:
        max_improved = max(max_improved, base_data["pct_improved_of_all"])
    ax3.set_ylim(0, max_improved * 1.2)

    # Add horizontal reference line for base model
    if base_data:
        ax3.axhline(y=base_data["pct_improved_of_all"], color='#7f8c8d', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
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
                "improved_count": d["improved_count"],
                "success_rate": d["success_rate"],
                "success_rate_se": d["success_rate_se"],
                "pct_improved_of_all": d["pct_improved_of_all"],
                "pct_improved_of_all_se": d["pct_improved_of_all_se"],
            }
            for d in data
        ]
    }
    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Saved data to {data_output_path}")

    # Print summary
    print("\n" + "="*70)
    print("MASKED-RATIO SWEEP RESULTS")
    print("="*70)
    print(f"{'Model':<10} {'% Multi-Att':>14} {'Improv Rate':>14} {'% Improved':>14}")
    print("-"*70)
    for d in data:
        multi_pct = 100 * d["multi_attempt_count"] / d["total_trials"]
        print(f"{d['label']:<10} {multi_pct:>13.1f}% {d['success_rate']:>13.1f}% {d['pct_improved_of_all']:>13.2f}%")
    print("="*70)


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
    parser.add_argument(
        "--haiku-only",
        action="store_true",
        help="Only use experiment results from the haiku judge folder",
    )
    args = parser.parse_args()
    results_dir = HAIKU_RESULTS_DIR if args.haiku_only else RESULTS_DIR
    if args.haiku_only:
        print(f"Using haiku judge folder: {results_dir}")
    plot_masked_ratio_sweep(output_dir=_resolve_output_dir(args.output_dir), results_dir=results_dir)
