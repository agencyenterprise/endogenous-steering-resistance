#!/usr/bin/env python
"""
Create comparison plot for random latent ablation control experiment.

Similar to plot_exp3.py but includes three conditions:
1. Steered baseline (no ablation)
2. Detector ablation (off-topic detectors ablated)
3. Random ablation (matched random latents ablated)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def extract_metrics(results: Dict) -> Dict:
    """
    Extract metrics from experiment results, filtering degraded outputs.
    
    Returns dict with:
    - total_trials, multi_attempt_trials, improved_multi_attempts
    - pct_multi_attempt, pct_multi_attempt_se
    - pct_improved, pct_improved_se
    - mean_improvement, mean_improvement_se
    - first_attempt_mean, first_attempt_se
    """
    improvements = []
    first_scores = []
    multi_attempt_trials = 0
    improved_multi_attempts = 0
    
    for feature_result in results.get("results_by_feature", []):
        if feature_result.get("error"):
            continue
            
        for trial in feature_result.get("trials", []):
            if trial.get("error"):
                continue
            
            # Skip degraded outputs
            response = trial.get("response", "")
            if is_degraded_output(response):
                continue
                
            score_data = trial.get("score", {})
            attempts = score_data.get("attempts", [])
            
            if not attempts:
                continue
            
            first_score = attempts[0].get("score", 0)
            first_scores.append(first_score)
            
            if len(attempts) >= 2:
                multi_attempt_trials += 1
                last_score = attempts[-1].get("score", 0)
                improvement = last_score - first_score
                if last_score > first_score:
                    improved_multi_attempts += 1
            else:
                improvement = 0
            
            improvements.append(improvement)
    
    total_trials = len(improvements)
    
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
    mean_improvement = np.mean(improvements)
    first_attempt_mean = np.mean(first_scores)
    
    # Calculate standard errors
    p_multi = pct_multi_attempt / 100
    pct_multi_attempt_se = np.sqrt(p_multi * (1 - p_multi) / total_trials) * 100
    
    p_improved = pct_improved / 100
    pct_improved_se = np.sqrt(p_improved * (1 - p_improved) / multi_attempt_trials) * 100 if multi_attempt_trials > 0 else 0
    
    mean_improvement_se = np.std(improvements, ddof=1) / np.sqrt(len(improvements)) if len(improvements) > 1 else 0
    first_attempt_se = np.std(first_scores, ddof=1) / np.sqrt(len(first_scores)) if len(first_scores) > 1 else 0
    
    return {
        "total_trials": total_trials,
        "multi_attempt_trials": multi_attempt_trials,
        "improved_multi_attempts": improved_multi_attempts,
        "pct_multi_attempt": pct_multi_attempt,
        "pct_multi_attempt_se": pct_multi_attempt_se,
        "pct_improved": pct_improved,
        "pct_improved_se": pct_improved_se,
        "mean_improvement": mean_improvement,
        "mean_improvement_se": mean_improvement_se,
        "first_attempt_mean": first_attempt_mean,
        "first_attempt_se": first_attempt_se,
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics from multiple experiment runs."""
    if not metrics_list:
        return {}
    
    # Use weighted averages based on trial counts
    total_trials = sum(m["total_trials"] for m in metrics_list)
    
    if total_trials == 0:
        return metrics_list[0]
    
    # Weighted average for mean improvement
    mean_improvements = [m["mean_improvement"] for m in metrics_list]
    weights = [m["total_trials"] for m in metrics_list]
    mean_improvement = np.average(mean_improvements, weights=weights)
    mean_improvement_se = np.std(mean_improvements) / np.sqrt(len(mean_improvements)) if len(mean_improvements) > 1 else 0
    
    # Weighted average for first attempt scores
    first_means = [m["first_attempt_mean"] for m in metrics_list]
    first_attempt_mean = np.average(first_means, weights=weights)
    first_attempt_se = np.std(first_means) / np.sqrt(len(first_means)) if len(first_means) > 1 else 0
    
    # Aggregate counts
    multi_attempt_trials = sum(m["multi_attempt_trials"] for m in metrics_list)
    improved_multi_attempts = sum(m["improved_multi_attempts"] for m in metrics_list)
    
    pct_multi_attempt = (multi_attempt_trials / total_trials * 100)
    pct_improved = (improved_multi_attempts / multi_attempt_trials * 100) if multi_attempt_trials > 0 else 0
    
    # SE for aggregated proportions
    p_multi = pct_multi_attempt / 100
    pct_multi_attempt_se = np.sqrt(p_multi * (1 - p_multi) / total_trials) * 100
    
    p_improved = pct_improved / 100
    pct_improved_se = np.sqrt(p_improved * (1 - p_improved) / multi_attempt_trials) * 100 if multi_attempt_trials > 0 else 0
    
    return {
        "total_trials": total_trials,
        "multi_attempt_trials": multi_attempt_trials,
        "improved_multi_attempts": improved_multi_attempts,
        "pct_multi_attempt": pct_multi_attempt,
        "pct_multi_attempt_se": pct_multi_attempt_se,
        "pct_improved": pct_improved,
        "pct_improved_se": pct_improved_se,
        "mean_improvement": mean_improvement,
        "mean_improvement_se": mean_improvement_se,
        "first_attempt_mean": first_attempt_mean,
        "first_attempt_se": first_attempt_se,
    }


def create_comparison_plot(
    baseline_metrics: Dict,
    detector_metrics: Dict,
    random_metrics: Dict,
    output_path: str,
) -> None:
    """
    Create a 3-panel bar chart comparing key metrics across all conditions.
    
    Panels:
    1. Mean Score Improvement (ESR metric)
    2. % of samples with multiple attempts
    3. First-attempt score (topical coherence)
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Color scheme
    colors = ['#6A0DAD', '#17A2B8', '#FFA500']  # Purple, Teal, Orange
    labels = ['Steered\nBaseline', 'Detector\nAblation', 'Random\nAblation']
    
    # 95% CI multiplier
    z_score = 1.96
    
    # Panel 1: Mean Score Improvement (main ESR metric)
    ax1 = axes[0]
    x1 = [0, 1, 2]
    values1 = [
        baseline_metrics['mean_improvement'],
        detector_metrics['mean_improvement'],
        random_metrics['mean_improvement'],
    ]
    errors1 = [
        z_score * baseline_metrics['mean_improvement_se'],
        z_score * detector_metrics['mean_improvement_se'],
        z_score * random_metrics['mean_improvement_se'],
    ]
    bars1 = ax1.bar(x1, values1, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors1, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})
    
    ax1.set_ylabel('Mean Score Improvement', fontsize=12, fontweight='bold')
    ax1.set_title('ESR Metric\n(Mean Score Improvement)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Calculate ESR reductions
    baseline_val = baseline_metrics['mean_improvement']
    detector_reduction = (baseline_val - detector_metrics['mean_improvement']) / baseline_val * 100 if baseline_val > 0 else 0
    random_reduction = (baseline_val - random_metrics['mean_improvement']) / baseline_val * 100 if baseline_val > 0 else 0
    
    max_with_error = max(v + e for v, e in zip(values1, errors1))
    ax1.set_ylim(0, max_with_error * 1.4)
    
    # Add value labels with reduction %
    for i, (bar, value, err) in enumerate(zip(bars1, values1, errors1)):
        height = bar.get_height() + err
        if i == 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max_with_error * 0.02,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        elif i == 1:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max_with_error * 0.02,
                    f'{value:.3f}\n({detector_reduction:.0f}% ↓)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#17A2B8')
        else:
            sign = "↓" if random_reduction > 0 else "↑"
            ax1.text(bar.get_x() + bar.get_width()/2., height + max_with_error * 0.02,
                    f'{value:.3f}\n({abs(random_reduction):.0f}% {sign})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FFA500')
    
    # Panel 2: % Multi-Attempt
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
    
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('% of Samples with\nMultiple Attempts', fontsize=13, fontweight='bold', pad=15)
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
        
        # Add sample counts
        if i == 0:
            count = baseline_metrics['multi_attempt_trials']
            total = baseline_metrics['total_trials']
        elif i == 1:
            count = detector_metrics['multi_attempt_trials']
            total = detector_metrics['total_trials']
        else:
            count = random_metrics['multi_attempt_trials']
            total = random_metrics['total_trials']
        ax2.text(bar.get_x() + bar.get_width()/2., -max_with_error2 * 0.08,
                f'({count}/{total})',
                ha='center', va='top', fontsize=8, style='italic')
    
    # Panel 3: First-Attempt Score
    ax3 = axes[2]
    x3 = [0, 1, 2]
    values3 = [
        baseline_metrics['first_attempt_mean'],
        detector_metrics['first_attempt_mean'],
        random_metrics['first_attempt_mean'],
    ]
    errors3 = [
        z_score * baseline_metrics['first_attempt_se'],
        z_score * detector_metrics['first_attempt_se'],
        z_score * random_metrics['first_attempt_se'],
    ]
    bars3 = ax3.bar(x3, values3, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                    yerr=errors3, error_kw={'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5})
    
    ax3.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax3.set_title('First-Attempt Score\n(Topical Coherence)', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Set reasonable y-limits for scores
    min_val = min(values3) - max(errors3) - 5
    max_val = max(values3) + max(errors3) + 10
    ax3.set_ylim(max(0, min_val), min(100, max_val))
    
    baseline_first = baseline_metrics['first_attempt_mean']
    for i, (bar, value, err) in enumerate(zip(bars3, values3, errors3)):
        height = bar.get_height() + err
        if i == 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            change = value - baseline_first
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}\n({change:+.1f})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle('Random Latent Ablation Control: Comparing ESR Effects',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()
    
    # Save data JSON
    data_output_path = Path(output_path).with_suffix('.json')
    plot_data = {
        "baseline": baseline_metrics,
        "detector_ablation": detector_metrics,
        "random_ablation": random_metrics,
    }
    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Data saved to {data_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create comparison plot for random ablation control")
    parser.add_argument("--prior-results-dir", type=str, default="data/prior_experiment_results",
                       help="Directory containing prior experiment results")
    parser.add_argument("--results-dir", type=str, default="experiment_results",
                       help="Directory containing random ablation results")
    parser.add_argument("--output", type=str, default="plots/random_ablation_control_comparison.png",
                       help="Output plot path")
    parser.add_argument("--model", type=str, default="70B",
                       help="Model name filter")
    args = parser.parse_args()
    
    prior_results_dir = Path(args.prior_results_dir)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find files
    def find_files(directory: Path, pattern: str, exclude_patterns: List[str] = None) -> List[Path]:
        files = list(directory.glob(pattern))
        if args.model:
            files = [f for f in files if args.model in f.name]
        if exclude_patterns:
            for excl in exclude_patterns:
                files = [f for f in files if excl not in f.name]
        return sorted(files)
    
    # Steered baseline
    baseline_pattern = "experiment_results_*[0-9][0-9][0-9][0-9][0-9][0-9].json"
    baseline_files = find_files(
        prior_results_dir, 
        baseline_pattern,
        exclude_patterns=["_no_steering_baseline", "_with_ablation", "_random_ablation"]
    )
    
    # Detector ablation
    detector_files = find_files(prior_results_dir, "*_with_ablation.json")
    
    # Random ablation
    random_files = find_files(results_dir, "*_random_ablation*.json")
    
    print(f"Found files:")
    print(f"  Steered baseline: {len(baseline_files)}")
    print(f"  Detector ablation: {len(detector_files)}")
    print(f"  Random ablation: {len(random_files)}")
    
    if not baseline_files or not detector_files or not random_files:
        print("Error: Missing required files!")
        return
    
    # Extract metrics
    print("\nExtracting metrics (filtering degraded outputs)...")
    
    # Baseline
    with open(baseline_files[0], "r") as f:
        results = json.load(f)
    baseline_metrics = extract_metrics(results)
    print(f"  Baseline: {baseline_metrics['total_trials']} trials")
    
    # Detector ablation
    with open(detector_files[0], "r") as f:
        results = json.load(f)
    detector_metrics = extract_metrics(results)
    print(f"  Detector ablation: {detector_metrics['total_trials']} trials")
    
    # Random ablation (aggregate multiple sets)
    random_metrics_list = []
    for f in random_files:
        with open(f, "r") as fp:
            results = json.load(fp)
        metrics = extract_metrics(results)
        random_metrics_list.append(metrics)
        print(f"  Random set {f.name}: {metrics['total_trials']} trials")
    
    random_metrics = aggregate_metrics(random_metrics_list)
    print(f"  Random ablation (aggregated): {random_metrics['total_trials']} trials")
    
    # Create plot
    print("\nCreating comparison plot...")
    create_comparison_plot(
        baseline_metrics,
        detector_metrics,
        random_metrics,
        str(output_path),
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    baseline_imp = baseline_metrics['mean_improvement']
    detector_imp = detector_metrics['mean_improvement']
    random_imp = random_metrics['mean_improvement']
    
    detector_reduction = (baseline_imp - detector_imp) / baseline_imp * 100 if baseline_imp > 0 else 0
    random_reduction = (baseline_imp - random_imp) / baseline_imp * 100 if baseline_imp > 0 else 0
    
    print(f"\nMean Score Improvement (ESR metric):")
    print(f"  Steered baseline: {baseline_imp:.4f}")
    print(f"  Detector ablation: {detector_imp:.4f} ({detector_reduction:.1f}% reduction)")
    print(f"  Random ablation: {random_imp:.4f} ({random_reduction:.1f}% reduction)")
    
    print(f"\nFirst-Attempt Score (Topical Coherence):")
    print(f"  Steered baseline: {baseline_metrics['first_attempt_mean']:.2f}")
    print(f"  Detector ablation: {detector_metrics['first_attempt_mean']:.2f} ({detector_metrics['first_attempt_mean'] - baseline_metrics['first_attempt_mean']:+.2f})")
    print(f"  Random ablation: {random_metrics['first_attempt_mean']:.2f} ({random_metrics['first_attempt_mean'] - baseline_metrics['first_attempt_mean']:+.2f})")


if __name__ == "__main__":
    main()

