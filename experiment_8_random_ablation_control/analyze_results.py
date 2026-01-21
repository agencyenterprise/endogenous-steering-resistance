"""Analyze results from random latent ablation control experiment.

Computes MSI (Multi-attempt Success Index) for each condition and runs
statistical comparisons.
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats


def is_degraded_output(response: str, min_repeats: int = 5) -> bool:
    """
    Check if a response is degraded (contains repetitive patterns).
    Same logic as plot_exp3.py / plot_utils.py.

    Args:
        response: The generated response text
        min_repeats: Minimum consecutive repetitions to count as degraded

    Returns:
        True if the response appears degraded
    """
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


def compute_mean_improvement(results: Dict) -> Tuple[float, float, int, int]:
    """
    Compute mean score improvement across all trials (same as plot_exp3.py).
    Filters out degraded outputs.
    
    For single-attempt trials, improvement is 0.
    For multi-attempt trials, improvement is last_score - first_score.
    
    Returns:
        (mean_improvement, std, n_trials, n_degraded)
    """
    improvements = []
    n_degraded = 0
    
    for feature_result in results.get("results_by_feature", []):
        if feature_result.get("error"):
            continue
            
        for trial in feature_result.get("trials", []):
            if trial.get("error"):
                continue
            
            # Skip degraded outputs
            response = trial.get("response", "")
            if is_degraded_output(response):
                n_degraded += 1
                continue
                
            score_data = trial.get("score", {})
            attempts = score_data.get("attempts", [])
            
            if len(attempts) >= 2:
                first_score = attempts[0].get("score", 0)
                last_score = attempts[-1].get("score", 0)
                improvement = last_score - first_score
            else:
                improvement = 0
            
            improvements.append(improvement)
    
    if not improvements:
        return 0.0, 0.0, 0, n_degraded
    
    return np.mean(improvements), np.std(improvements), len(improvements), n_degraded


def compute_first_attempt_scores(results: Dict) -> Tuple[float, float, int]:
    """
    Compute mean and std of first-attempt scores from experiment results.
    Filters out degraded outputs.
    
    First-attempt score indicates how well the model initially follows steering.
    Lower scores after ablation suggest general network disruption.
    
    Returns:
        (mean_score, std_score, n_trials)
    """
    first_scores = []
    
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
            
            if attempts:
                first_score = attempts[0].get("score", 0)
                first_scores.append(first_score)
    
    if not first_scores:
        return 0.0, 0.0, 0
    
    return np.mean(first_scores), np.std(first_scores), len(first_scores)


def compute_msi(results: Dict) -> Tuple[float, float, int, int]:
    """
    Compute MSI (Multi-attempt Success Index) from experiment results.
    Filters out degraded outputs.
    
    MSI = (multi_attempt_rate) * (success_rate_on_second_attempt)
    
    Where:
    - multi_attempt_rate = fraction of trials where model made multiple attempts
    - success_rate_on_second_attempt = fraction of multi-attempt trials where
      the second attempt scored higher than first
    
    Returns:
        (msi, multi_attempt_rate, num_multi_attempt, total_trials)
    """
    total_trials = 0
    multi_attempt_trials = 0
    successful_corrections = 0
    
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
            
            total_trials += 1
            
            if len(attempts) >= 2:
                multi_attempt_trials += 1
                # Check if second attempt is better than first
                first_score = attempts[0].get("score", 0)
                second_score = attempts[1].get("score", 0)
                if second_score > first_score:
                    successful_corrections += 1
    
    if total_trials == 0:
        return 0.0, 0.0, 0, 0
    
    multi_attempt_rate = multi_attempt_trials / total_trials
    
    if multi_attempt_trials == 0:
        success_rate = 0.0
    else:
        success_rate = successful_corrections / multi_attempt_trials
    
    msi = multi_attempt_rate * success_rate
    
    return msi, multi_attempt_rate, multi_attempt_trials, total_trials


def bootstrap_msi(results: Dict, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for MSI.
    Filters out degraded outputs.
    
    Returns:
        (msi, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    
    # Extract all trial data (filtering degraded outputs)
    trial_data = []
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
            trial_data.append(attempts)
    
    if not trial_data:
        return 0.0, 0.0, 0.0
    
    n_trials = len(trial_data)
    
    # Compute original MSI
    msi, _, _, _ = compute_msi(results)
    
    # Bootstrap
    bootstrap_msis = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_trials, size=n_trials, replace=True)
        
        multi_attempts = 0
        successful = 0
        
        for idx in indices:
            attempts = trial_data[idx]
            if len(attempts) >= 2:
                multi_attempts += 1
                if attempts[1].get("score", 0) > attempts[0].get("score", 0):
                    successful += 1
        
        if multi_attempts > 0:
            mar = multi_attempts / n_trials
            sr = successful / multi_attempts
            bootstrap_msis.append(mar * sr)
        else:
            bootstrap_msis.append(0.0)
    
    ci_lower = np.percentile(bootstrap_msis, 2.5)
    ci_upper = np.percentile(bootstrap_msis, 97.5)
    
    return msi, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description="Analyze random latent ablation results")
    parser.add_argument("--prior-results-dir", type=str, default="data/prior_experiment_results",
                       help="Directory containing prior experiment results (baseline, detector ablation)")
    parser.add_argument("--results-dir", type=str, default="experiment_results",
                       help="Directory containing new experiment results (random ablation)")
    parser.add_argument("--steered-baseline-pattern", type=str, 
                       default="experiment_results_*[0-9][0-9][0-9][0-9][0-9][0-9].json",
                       help="Pattern for steered baseline result files (no special suffix)")
    parser.add_argument("--detector-ablation-pattern", type=str, default="*_with_ablation.json",
                       help="Pattern for detector ablation result files")
    parser.add_argument("--random-ablation-pattern", type=str, default="*_random_ablation*.json",
                       help="Pattern for random ablation result files")
    parser.add_argument("--model", type=str, default="70B",
                       help="Model name filter (e.g., '70B')")
    args = parser.parse_args()
    
    prior_results_dir = Path(args.prior_results_dir)
    results_dir = Path(args.results_dir)
    
    print("=" * 80)
    print("Random Latent Ablation Control Analysis")
    print("=" * 80)
    
    # Find result files
    def find_files(directory: Path, pattern: str, exclude_patterns: List[str] = None) -> List[Path]:
        files = list(directory.glob(pattern))
        if args.model:
            files = [f for f in files if args.model in f.name]
        if exclude_patterns:
            for excl in exclude_patterns:
                files = [f for f in files if excl not in f.name]
        return sorted(files)
    
    # Prior results (baseline, detector ablation) are in prior_results_dir
    # Steered baseline: files without _no_steering_baseline, _with_ablation, or _random_ablation
    steered_baseline_files = find_files(
        prior_results_dir, 
        args.steered_baseline_pattern,
        exclude_patterns=["_no_steering_baseline", "_with_ablation", "_random_ablation"]
    )
    detector_files = find_files(prior_results_dir, args.detector_ablation_pattern)
    # Random ablation results are in results_dir
    random_files = find_files(results_dir, args.random_ablation_pattern)
    
    print(f"\nFound files:")
    print(f"  Steered baseline (no ablation): {len(steered_baseline_files)}")
    print(f"  Detector ablation: {len(detector_files)}")
    print(f"  Random ablation: {len(random_files)}")
    
    # Analyze each condition
    conditions = {}
    
    # Steered baseline (steering applied, no ablation)
    if steered_baseline_files:
        print(f"\n--- Steered baseline (steering, no ablation) ---")
        for f in steered_baseline_files[:1]:  # Use first matching file
            with open(f, "r") as fp:
                results = json.load(fp)
            msi, ci_lower, ci_upper = bootstrap_msi(results)
            _, mar, n_multi, n_total = compute_msi(results)
            first_mean, first_std, _ = compute_first_attempt_scores(results)
            mean_imp, mean_imp_std, n_valid, n_degraded = compute_mean_improvement(results)
            conditions["steered_baseline"] = {
                "msi": msi, "ci_lower": ci_lower, "ci_upper": ci_upper,
                "multi_attempt_rate": mar, "n_multi": n_multi, "n_total": n_total,
                "first_attempt_mean": first_mean, "first_attempt_std": first_std,
                "mean_improvement": mean_imp, "mean_improvement_std": mean_imp_std,
                "n_valid": n_valid, "n_degraded": n_degraded,
                "file": str(f)
            }
            print(f"  File: {f.name}")
            print(f"  Trials: {n_valid} valid, {n_degraded} degraded filtered out")
            print(f"  MSI: {msi:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            print(f"  Mean improvement (plot_exp3 metric): {mean_imp:.4f}")
            print(f"  Multi-attempt rate: {mar:.4f} ({n_multi}/{n_total})")
            print(f"  First-attempt score: {first_mean:.2f} ± {first_std:.2f}")
    
    # Detector ablation
    if detector_files:
        print(f"\n--- Detector ablation ---")
        for f in detector_files[:1]:
            with open(f, "r") as fp:
                results = json.load(fp)
            msi, ci_lower, ci_upper = bootstrap_msi(results)
            _, mar, n_multi, n_total = compute_msi(results)
            first_mean, first_std, _ = compute_first_attempt_scores(results)
            mean_imp, mean_imp_std, n_valid, n_degraded = compute_mean_improvement(results)
            conditions["detector_ablation"] = {
                "msi": msi, "ci_lower": ci_lower, "ci_upper": ci_upper,
                "multi_attempt_rate": mar, "n_multi": n_multi, "n_total": n_total,
                "first_attempt_mean": first_mean, "first_attempt_std": first_std,
                "mean_improvement": mean_imp, "mean_improvement_std": mean_imp_std,
                "n_valid": n_valid, "n_degraded": n_degraded,
                "file": str(f)
            }
            print(f"  File: {f.name}")
            print(f"  Trials: {n_valid} valid, {n_degraded} degraded filtered out")
            print(f"  MSI: {msi:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            print(f"  Mean improvement (plot_exp3 metric): {mean_imp:.4f}")
            print(f"  Multi-attempt rate: {mar:.4f} ({n_multi}/{n_total})")
            print(f"  First-attempt score: {first_mean:.2f} ± {first_std:.2f}")
    
    # Random ablation (multiple sets)
    if random_files:
        print(f"\n--- Random ablation ---")
        random_msis = []
        random_first_scores = []
        random_mean_imps = []
        for f in random_files:
            with open(f, "r") as fp:
                results = json.load(fp)
            msi, ci_lower, ci_upper = bootstrap_msi(results)
            _, mar, n_multi, n_total = compute_msi(results)
            first_mean, first_std, _ = compute_first_attempt_scores(results)
            mean_imp, mean_imp_std, n_valid, n_degraded = compute_mean_improvement(results)
            random_msis.append(msi)
            random_first_scores.append(first_mean)
            random_mean_imps.append(mean_imp)
            print(f"  {f.name}:")
            print(f"    Trials: {n_valid} valid, {n_degraded} degraded filtered out")
            print(f"    MSI: {msi:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            print(f"    Mean improvement: {mean_imp:.4f}")
            print(f"    Multi-attempt rate: {mar:.4f} ({n_multi}/{n_total})")
            print(f"    First-attempt score: {first_mean:.2f} ± {first_std:.2f}")
        
        if random_msis:
            mean_msi = np.mean(random_msis)
            std_msi = np.std(random_msis)
            mean_first = np.mean(random_first_scores)
            std_first = np.std(random_first_scores)
            mean_imp_avg = np.mean(random_mean_imps)
            mean_imp_std = np.std(random_mean_imps)
            conditions["random_ablation"] = {
                "msi_mean": mean_msi, "msi_std": std_msi,
                "msi_values": random_msis,
                "first_attempt_mean": mean_first, "first_attempt_std": std_first,
                "first_attempt_values": random_first_scores,
                "mean_improvement_mean": mean_imp_avg, "mean_improvement_std": mean_imp_std,
                "mean_improvement_values": random_mean_imps,
                "files": [str(f) for f in random_files]
            }
            print(f"\n  Summary:")
            print(f"    Mean MSI: {mean_msi:.4f} ± {std_msi:.4f}")
            print(f"    Mean improvement (plot_exp3): {mean_imp_avg:.4f} ± {mean_imp_std:.4f}")
            print(f"    Mean first-attempt score: {mean_first:.2f} ± {std_first:.2f}")
    
    # Statistical comparisons
    print("\n" + "=" * 80)
    print("Statistical Comparisons (using Mean Improvement - same as plot_exp3.py)")
    print("=" * 80)
    
    if "steered_baseline" in conditions and "detector_ablation" in conditions:
        baseline_imp = conditions["steered_baseline"]["mean_improvement"]
        detector_imp = conditions["detector_ablation"]["mean_improvement"]
        reduction = (baseline_imp - detector_imp) / baseline_imp * 100 if baseline_imp > 0 else 0
        
        print(f"\n1. Detector ablation vs Steered baseline:")
        print(f"   Steered baseline mean improvement: {baseline_imp:.4f}")
        print(f"   Detector ablation mean improvement: {detector_imp:.4f}")
        print(f"   ESR Reduction: {reduction:.1f}%")
    
    if "steered_baseline" in conditions and "random_ablation" in conditions:
        baseline_imp = conditions["steered_baseline"]["mean_improvement"]
        random_imp_mean = conditions["random_ablation"]["mean_improvement_mean"]
        random_imp_std = conditions["random_ablation"]["mean_improvement_std"]
        reduction = (baseline_imp - random_imp_mean) / baseline_imp * 100 if baseline_imp > 0 else 0
        
        print(f"\n2. Random ablation vs Steered baseline:")
        print(f"   Steered baseline mean improvement: {baseline_imp:.4f}")
        print(f"   Random ablation mean improvement: {random_imp_mean:.4f} ± {random_imp_std:.4f}")
        print(f"   ESR Reduction: {reduction:.1f}%")
        
        # One-sample t-test (is random ablation mean different from baseline?)
        if len(conditions["random_ablation"]["mean_improvement_values"]) > 1:
            t_stat, p_val = stats.ttest_1samp(
                conditions["random_ablation"]["mean_improvement_values"],
                baseline_imp
            )
            print(f"   One-sample t-test vs baseline: t={t_stat:.3f}, p={p_val:.4f}")
    
    if "detector_ablation" in conditions and "random_ablation" in conditions:
        detector_imp = conditions["detector_ablation"]["mean_improvement"]
        random_imp_mean = conditions["random_ablation"]["mean_improvement_mean"]
        
        print(f"\n3. Detector ablation vs Random ablation:")
        print(f"   Detector ablation mean improvement: {detector_imp:.4f}")
        print(f"   Random ablation mean improvement: {random_imp_mean:.4f}")
        
        if len(conditions["random_ablation"]["mean_improvement_values"]) > 1:
            t_stat, p_val = stats.ttest_1samp(
                conditions["random_ablation"]["mean_improvement_values"],
                detector_imp
            )
            print(f"   One-sample t-test vs detector: t={t_stat:.3f}, p={p_val:.4f}")
    
    # First-attempt score comparisons
    print("\n" + "=" * 80)
    print("First-Attempt Score Comparisons (Topical Coherence)")
    print("=" * 80)
    
    if "steered_baseline" in conditions:
        baseline_first = conditions["steered_baseline"]["first_attempt_mean"]
        print(f"\nSteered baseline first-attempt score: {baseline_first:.2f}")
        
        if "detector_ablation" in conditions:
            detector_first = conditions["detector_ablation"]["first_attempt_mean"]
            change = detector_first - baseline_first
            print(f"Detector ablation first-attempt score: {detector_first:.2f} ({change:+.2f})")
        
        if "random_ablation" in conditions:
            random_first = conditions["random_ablation"]["first_attempt_mean"]
            random_first_std = conditions["random_ablation"]["first_attempt_std"]
            change = random_first - baseline_first
            print(f"Random ablation first-attempt score: {random_first:.2f} ± {random_first_std:.2f} ({change:+.2f})")
            
            # T-test for first-attempt scores
            if len(conditions["random_ablation"]["first_attempt_values"]) > 1:
                t_stat, p_val = stats.ttest_1samp(
                    conditions["random_ablation"]["first_attempt_values"],
                    baseline_first
                )
                print(f"\nOne-sample t-test (random vs baseline): t={t_stat:.3f}, p={p_val:.4f}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)
    
    if all(k in conditions for k in ["steered_baseline", "detector_ablation", "random_ablation"]):
        baseline_imp = conditions["steered_baseline"]["mean_improvement"]
        detector_imp = conditions["detector_ablation"]["mean_improvement"]
        random_imp = conditions["random_ablation"]["mean_improvement_mean"]
        
        baseline_first = conditions["steered_baseline"]["first_attempt_mean"]
        detector_first = conditions["detector_ablation"]["first_attempt_mean"]
        random_first = conditions["random_ablation"]["first_attempt_mean"]
        
        # Calculate reductions relative to steered baseline
        detector_reduction = (baseline_imp - detector_imp) / baseline_imp * 100 if baseline_imp > 0 else 0
        random_reduction = (baseline_imp - random_imp) / baseline_imp * 100 if baseline_imp > 0 else 0
        
        print(f"\n--- ESR (Mean Improvement - same as plot_exp3.py) ---")
        print(f"  Steered baseline: {baseline_imp:.4f}")
        print(f"  Detector ablation: {detector_imp:.4f} ({detector_reduction:.1f}% reduction)")
        print(f"  Random ablation: {random_imp:.4f} ({random_reduction:.1f}% reduction)")
        
        print(f"\n--- Topical Coherence (First-Attempt Scores) ---")
        print(f"  Steered baseline: {baseline_first:.2f}")
        print(f"  Detector ablation: {detector_first:.2f} ({detector_first - baseline_first:+.2f})")
        print(f"  Random ablation: {random_first:.2f} ({random_first - baseline_first:+.2f})")
        
        print("\n--- Conclusion ---")
        if detector_reduction > random_reduction + 20:  # Detector ablation has much larger effect
            print("✓ Detector ablation shows substantially larger ESR reduction than random ablation")
            print(f"  ({detector_reduction:.1f}% vs {random_reduction:.1f}%).")
            if abs(detector_first - baseline_first) < abs(random_first - baseline_first):
                print("  Additionally, detector ablation has LESS impact on first-attempt scores")
                print(f"  ({detector_first - baseline_first:+.2f} vs {random_first - baseline_first:+.2f}).")
                print("  This supports the hypothesis that off-topic detectors are SPECIFICALLY")
                print("  involved in ESR, not just general network disruption.")
            else:
                print("  However, detector ablation also impacts first-attempt scores more.")
                print("  Results require further investigation.")
        elif abs(random_reduction - detector_reduction) < 15:  # Similar reduction
            print("⚠ Random ablation shows similar ESR reduction to detector ablation.")
            print("  This suggests the ESR reduction may be due to GENERAL network disruption")
            print("  rather than specific suppression of off-topic detection circuitry.")
        else:
            print("? Results show intermediate effect.")
            print(f"  Detector ablation: {detector_reduction:.1f}% reduction, {detector_first - baseline_first:+.2f} first-score change")
            print(f"  Random ablation: {random_reduction:.1f}% reduction, {random_first - baseline_first:+.2f} first-score change")
    
    # Save summary
    summary_file = Path(__file__).parent / "analysis_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(conditions, f, indent=2, default=str)
    print(f"\nSaved analysis summary to {summary_file}")


if __name__ == "__main__":
    main()

