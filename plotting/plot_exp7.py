#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "seaborn",
#   "scipy",
# ]
# ///
"""
Analyze cross-judge regrading results.

Generates plots comparing how different judge models evaluate ESR:
1. Mean score improvement by judge and target model
2. Multi-attempt detection rates by judge
3. Inter-judge agreement metrics

Usage:
    python plot_exp7.py --results-dir cross_judge_results
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent


def extract_target_model(source_file: str) -> str | None:
    """Extract target model name from source filename."""
    patterns = [
        (r"Meta-Llama-3\.3-70B-Instruct", "Llama-3.3-70B"),
        (r"Meta-Llama-3\.1-8B-Instruct", "Llama-3.1-8B"),
        (r"gemma-2-27b-it", "Gemma-2-27B"),
        (r"gemma-2-9b-it", "Gemma-2-9B"),
        (r"gemma-2-2b-it", "Gemma-2-2B"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, source_file, re.IGNORECASE):
            return name
    return None


def load_regraded_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all regraded results by judge model."""
    judge_results = {}

    for result_file in results_dir.glob("regraded_*.json"):
        with open(result_file) as f:
            data = json.load(f)

        model_name = data["model_name"]
        if model_name not in judge_results:
            judge_results[model_name] = []

        judge_results[model_name].extend(data["results"])

    return judge_results


def extract_scores(result: dict) -> tuple[list[int], list[int]] | None:
    """Extract first and last scores from a result, returning (original, new) scores."""
    # Original score
    orig_score = result.get("original_score", {})
    orig_attempts = orig_score.get("attempts", []) if isinstance(orig_score, dict) else []

    # New score
    new_score = result.get("new_score", {})
    if new_score.get("error"):
        return None
    new_attempts = new_score.get("attempts", [])

    if not new_attempts:
        return None

    # Extract scores (convert to float in case they're stored as strings)
    def safe_float(val, default=0):
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    orig_scores = [safe_float(a.get("score", 0)) for a in orig_attempts] if orig_attempts else [0]
    new_scores = [safe_float(a.get("score", 0)) for a in new_attempts]

    return orig_scores, new_scores


def create_score_dataframe(judge_results: dict[str, list[dict]], include_original: bool = True) -> pd.DataFrame:
    """Create a DataFrame with all scores for analysis.

    Args:
        judge_results: Dict mapping judge name to list of results
        include_original: If True, also include original Claude 4.5 Sonnet scores as a separate judge
    """
    rows = []
    seen_trials = set()  # Track trials we've already added original scores for

    for judge_name, results in judge_results.items():
        for result in results:
            scores = extract_scores(result)
            if scores is None:
                continue

            orig_scores, new_scores = scores
            target_model = extract_target_model(result["source_file"])
            # IMPORTANT: feature_label is free text and can collide; feature_index is the stable identifier.
            trial_id = f"{result['source_file']}_{result['seed']}_{result['feature_index']}"

            # Add row for this cross-judge model
            rows.append({
                "judge": judge_name,
                "target_model": target_model,
                "source_file": result["source_file"],
                "feature_index": result["feature_index"],
                "feature_label": result["feature_label"],
                "prompt": result.get("prompt"),
                "seed": result["seed"],
                "new_first_score": new_scores[0],
                "new_last_score": new_scores[-1],
                "new_n_attempts": len(new_scores),
                "new_improvement": new_scores[-1] - new_scores[0] if len(new_scores) > 1 else 0,
                "new_has_multi": len(new_scores) > 1,
            })

            # Add row for original Claude 4.5 Sonnet judge (once per trial)
            if include_original and trial_id not in seen_trials and orig_scores:
                seen_trials.add(trial_id)
                rows.append({
                    "judge": "Sonnet-4.5",
                    "target_model": target_model,
                    "source_file": result["source_file"],
                    "feature_index": result["feature_index"],
                    "feature_label": result["feature_label"],
                    "prompt": result.get("prompt"),
                    "seed": result["seed"],
                    "new_first_score": orig_scores[0],
                    "new_last_score": orig_scores[-1],
                    "new_n_attempts": len(orig_scores),
                    "new_improvement": (orig_scores[-1] - orig_scores[0]) if len(orig_scores) > 1 else 0,
                    "new_has_multi": len(orig_scores) > 1,
                })

    return pd.DataFrame(rows)


def plot_msi_by_judge(df: pd.DataFrame, output_dir: Path):
    """Create bar chart of Mean Score Improvement by judge and target model."""
    # Model order and colors
    model_order = ["Gemma-2-2B", "Gemma-2-9B", "Gemma-2-27B", "Llama-3.1-8B", "Llama-3.3-70B"]
    model_colors = {
        "Gemma-2-2B": "#FAD7A0",
        "Gemma-2-9B": "#F39C12",
        "Gemma-2-27B": "#D35400",
        "Llama-3.1-8B": "#85C1E9",
        "Llama-3.3-70B": "#1A5276",
    }

    judges = sorted(df["judge"].unique())
    n_judges = len(judges)

    fig, axes = plt.subplots(1, n_judges, figsize=(4 * n_judges, 5), sharey=True)
    if n_judges == 1:
        axes = [axes]

    for ax, judge in zip(axes, judges):
        judge_df = df[df["judge"] == judge]

        means = []
        colors = []
        labels = []

        for model in model_order:
            model_df = judge_df[judge_df["target_model"] == model]
            if len(model_df) > 0:
                # Only consider multi-attempt trials for MSI calculation
                multi_attempt_df = model_df[model_df["new_has_multi"]]
                mean_imp = multi_attempt_df["new_improvement"].mean() if len(multi_attempt_df) > 0 else 0
                means.append(mean_imp)
                colors.append(model_colors.get(model, "#888888"))
                labels.append(model)

        if means:
            bars = ax.bar(range(len(means)), means, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            # Add value labels
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_ylabel("Mean Score Improvement", fontsize=10)
    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_msi.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_esr_rate_by_judge(df: pd.DataFrame, output_dir: Path):
    """Create bar chart of ESR rate (% where last > first) by judge and target model."""
    # Model order and colors
    model_order = ["Gemma-2-2B", "Gemma-2-9B", "Gemma-2-27B", "Llama-3.1-8B", "Llama-3.3-70B"]
    model_colors = {
        "Gemma-2-2B": "#FAD7A0",
        "Gemma-2-9B": "#F39C12",
        "Gemma-2-27B": "#D35400",
        "Llama-3.1-8B": "#85C1E9",
        "Llama-3.3-70B": "#1A5276",
    }

    judges = sorted(df["judge"].unique())
    n_judges = len(judges)

    fig, axes = plt.subplots(1, n_judges, figsize=(4 * n_judges, 5), sharey=True)
    if n_judges == 1:
        axes = [axes]

    for ax, judge in zip(axes, judges):
        judge_df = df[df["judge"] == judge]

        rates = []
        colors = []
        labels = []

        for model in model_order:
            model_df = judge_df[judge_df["target_model"] == model]
            if len(model_df) > 0:
                # ESR % = percentage where improvement > 0 (multi-attempt trials only)
                multi_attempt_df = model_df[model_df["new_has_multi"]]
                esr_rate = (multi_attempt_df["new_improvement"] > 0).mean() * 100 if len(multi_attempt_df) > 0 else 0
                rates.append(esr_rate)
                colors.append(model_colors.get(model, "#888888"))
                labels.append(model)

        if rates:
            bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

            # Add value labels
            for bar, val in zip(bars, rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_ylabel("ESR Rate (%)", fontsize=10)
    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_esr_rate.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_multi_attempt_rates(df: pd.DataFrame, output_dir: Path):
    """Create bar chart of multi-attempt detection rates by judge."""
    model_order = ["Gemma-2-2B", "Gemma-2-9B", "Gemma-2-27B", "Llama-3.1-8B", "Llama-3.3-70B"]
    model_colors = {
        "Gemma-2-2B": "#FAD7A0",
        "Gemma-2-9B": "#F39C12",
        "Gemma-2-27B": "#D35400",
        "Llama-3.1-8B": "#85C1E9",
        "Llama-3.3-70B": "#1A5276",
    }

    judges = sorted(df["judge"].unique())
    n_judges = len(judges)

    fig, axes = plt.subplots(1, n_judges, figsize=(4 * n_judges, 5), sharey=True)
    if n_judges == 1:
        axes = [axes]

    for ax, judge in zip(axes, judges):
        judge_df = df[df["judge"] == judge]

        rates = []
        colors = []
        labels = []

        for model in model_order:
            model_df = judge_df[judge_df["target_model"] == model]
            if len(model_df) > 0:
                multi_rate = model_df["new_has_multi"].mean() * 100
                rates.append(multi_rate)
                colors.append(model_colors.get(model, "#888888"))
                labels.append(model)

        if rates:
            bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

            # Add value labels
            for bar, val in zip(bars, rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_ylabel("Multi-Attempt Detection Rate (%)", fontsize=10)
    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_multi_attempt.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_esr_rate_all_trials_by_judge(df: pd.DataFrame, output_dir: Path):
    """Create bar chart of ESR Rate (% of ALL trials with multi-attempt AND improvement) by judge and target model."""
    model_order = ["Gemma-2-2B", "Gemma-2-9B", "Gemma-2-27B", "Llama-3.1-8B", "Llama-3.3-70B"]
    model_colors = {
        "Gemma-2-2B": "#FAD7A0",
        "Gemma-2-9B": "#F39C12",
        "Gemma-2-27B": "#D35400",
        "Llama-3.1-8B": "#85C1E9",
        "Llama-3.3-70B": "#1A5276",
    }

    judges = sorted(df["judge"].unique())
    n_judges = len(judges)

    fig, axes = plt.subplots(1, n_judges, figsize=(4 * n_judges, 5), sharey=True)
    if n_judges == 1:
        axes = [axes]

    for ax, judge in zip(axes, judges):
        judge_df = df[df["judge"] == judge]

        rates = []
        colors = []
        labels = []

        for model in model_order:
            model_df = judge_df[judge_df["target_model"] == model]
            if len(model_df) > 0:
                # % Improved = % of ALL responses with multi-attempt AND improvement > 0
                n_improved = ((model_df["new_has_multi"]) & (model_df["new_improvement"] > 0)).sum()
                pct_improved = (n_improved / len(model_df)) * 100
                rates.append(pct_improved)
                colors.append(model_colors.get(model, "#888888"))
                labels.append(model)

        if rates:
            bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

            # Add value labels
            for bar, val in zip(bars, rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_ylabel("ESR Rate", fontsize=10)
    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_esr_rate.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create a heatmap of pairwise score correlations between judges."""
    judges = sorted(df["judge"].unique())
    n_judges = len(judges)

    # Create a unique trial ID for matching across judges
    df = df.copy()
    df["trial_id"] = df["source_file"] + "_" + df["seed"].astype(str) + "_" + df["feature_index"].astype(str)

    # Build correlation matrix
    corr_matrix = np.ones((n_judges, n_judges))

    for i, judge1 in enumerate(judges):
        for j, judge2 in enumerate(judges):
            if i == j:
                continue

            df1 = df[df["judge"] == judge1].set_index("trial_id")
            df2 = df[df["judge"] == judge2].set_index("trial_id")

            # Get unique common trials (handle duplicates by taking first occurrence)
            common_trials = df1.index.intersection(df2.index).unique()
            if len(common_trials) < 10:
                corr_matrix[i, j] = np.nan
                continue

            # Get scores, handling duplicates by taking first occurrence
            scores1 = df1.loc[common_trials, "new_first_score"]
            scores2 = df2.loc[common_trials, "new_first_score"]
            
            # If there are duplicates in the index, take first occurrence
            if scores1.index.duplicated().any():
                scores1 = scores1[~scores1.index.duplicated(keep='first')]
            if scores2.index.duplicated().any():
                scores2 = scores2[~scores2.index.duplicated(keep='first')]
            
            # Align to common index (only trials present in both)
            common_idx = scores1.index.intersection(scores2.index)
            scores1_aligned = scores1.loc[common_idx]
            scores2_aligned = scores2.loc[common_idx]
            
            # Create boolean mask and filter
            valid = scores1_aligned.notna() & scores2_aligned.notna()
            if valid.sum() > 10:
                # Convert to numpy arrays for pearsonr
                s1_valid = scores1_aligned[valid].values
                s2_valid = scores2_aligned[valid].values
                r, _ = pearsonr(s1_valid, s2_valid)
                corr_matrix[i, j] = r
            else:
                corr_matrix[i, j] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(7, 6))

    mask = np.eye(n_judges, dtype=bool)  # Mask diagonal
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        xticklabels=judges,
        yticklabels=judges,
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )

    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_correlation_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_score_scatter(df: pd.DataFrame, output_dir: Path):
    """Create scatter plots comparing scores between judge pairs."""
    judges = sorted(df["judge"].unique())

    # Create a unique trial ID for matching across judges
    df = df.copy()
    df["trial_id"] = df["source_file"] + "_" + df["seed"].astype(str) + "_" + df["feature_index"].astype(str)

    # Select representative pairs (first vs others)
    # Use all unique pairs for a grid
    pairs = [(judges[0], judges[1]), (judges[0], judges[2])]
    if len(judges) > 3:
        pairs.append((judges[1], judges[3]))

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (judge1, judge2) in zip(axes, pairs):
        df1 = df[df["judge"] == judge1].set_index("trial_id")
        df2 = df[df["judge"] == judge2].set_index("trial_id")

        # Get unique common trials (handle duplicates by taking first occurrence)
        common_trials = df1.index.intersection(df2.index).unique()
        if len(common_trials) < 10:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Get scores, handling duplicates by taking first occurrence
        scores1 = df1.loc[common_trials, "new_first_score"]
        scores2 = df2.loc[common_trials, "new_first_score"]
        
        # If there are duplicates in the index, take first occurrence
        if scores1.index.duplicated().any():
            scores1 = scores1[~scores1.index.duplicated(keep='first')]
        if scores2.index.duplicated().any():
            scores2 = scores2[~scores2.index.duplicated(keep='first')]
        
        # Align to common index (only trials present in both)
        common_idx = scores1.index.intersection(scores2.index)
        scores1_aligned = scores1.loc[common_idx]
        scores2_aligned = scores2.loc[common_idx]
        
        # Create boolean mask and filter
        valid = scores1_aligned.notna() & scores2_aligned.notna()
        x = scores1_aligned[valid].values
        y = scores2_aligned[valid].values

        # Scatter plot with alpha for density
        ax.scatter(x, y, alpha=0.3, s=20, edgecolors="none")

        # Add diagonal line (perfect agreement)
        lims = [0, 100]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect agreement")

        # Add regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(lims, p(lims), "r-", alpha=0.7, linewidth=1.5, label=f"Fit (slope={z[0]:.2f})")

        # Correlation
        r, _ = pearsonr(x, y)
        ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(x)}", transform=ax.transAxes,
                va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel(f"{judge1} score", fontsize=10)
        ax.set_ylabel(f"{judge2} score", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_scatter.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_attempts_scatter(df: pd.DataFrame, output_dir: Path):
    """Create a facet grid of scatter plots comparing number of attempts between all judge pairs."""
    judges = sorted(df["judge"].unique())
    n_judges = len(judges)

    # Create a unique trial ID for matching across judges
    df = df.copy()
    df["trial_id"] = df["source_file"] + "_" + df["seed"].astype(str) + "_" + df["feature_index"].astype(str)

    # Build a wide dataframe with attempts per judge as columns
    pivot_data = []
    all_trial_ids = df["trial_id"].unique()

    for trial_id in all_trial_ids:
        row = {"trial_id": trial_id}
        trial_data = df[df["trial_id"] == trial_id]
        for judge in judges:
            judge_row = trial_data[trial_data["judge"] == judge]
            if len(judge_row) > 0:
                row[judge] = judge_row["new_n_attempts"].iloc[0]
            else:
                row[judge] = np.nan
        pivot_data.append(row)

    wide_df = pd.DataFrame(pivot_data)

    # Find global max attempts for consistent axes (capped at 10 for readability)
    max_attempts = min(int(wide_df[judges].max().max()) + 1, 10)

    fig, axes = plt.subplots(n_judges, n_judges, figsize=(3 * n_judges, 3 * n_judges))

    for i, judge_y in enumerate(judges):
        for j, judge_x in enumerate(judges):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram of this judge's attempt counts
                data = wide_df[judge_x].dropna()
                ax.hist(data, bins=np.arange(0.5, max_attempts + 1.5, 1),
                        color="steelblue", edgecolor="white", alpha=0.7)
                ax.set_xlim(0.5, max_attempts + 0.5)
                ax.set_xticks(range(1, max_attempts + 1))
                if i == n_judges - 1:
                    ax.set_xlabel(judge_x, fontsize=9)
                if j == 0:
                    ax.set_ylabel(judge_y, fontsize=9)
            else:
                # Off-diagonal: scatter plot
                valid = wide_df[[judge_x, judge_y]].dropna()
                x = valid[judge_x]
                y = valid[judge_y]

                if len(x) > 0:
                    # Add jitter
                    jitter = 0.15
                    x_jit = x + np.random.uniform(-jitter, jitter, len(x))
                    y_jit = y + np.random.uniform(-jitter, jitter, len(y))

                    ax.scatter(x_jit, y_jit, alpha=0.25, s=15, edgecolors="none", c="steelblue")

                    # Diagonal line
                    ax.plot([0.5, max_attempts + 0.5], [0.5, max_attempts + 0.5],
                            "k--", alpha=0.4, linewidth=1)

                    # Agreement rate
                    agreement = (x == y).mean()
                    ax.text(0.95, 0.05, f"{agreement:.0%}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

                ax.set_xlim(0.5, max_attempts + 0.5)
                ax.set_ylim(0.5, max_attempts + 0.5)
                ax.set_xticks(range(1, max_attempts + 1))
                ax.set_yticks(range(1, max_attempts + 1))
                ax.set_aspect("equal")

                # Labels on edges only
                if i == n_judges - 1:
                    ax.set_xlabel(judge_x, fontsize=9)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(judge_y, fontsize=9)
                else:
                    ax.set_yticklabels([])

    plt.tight_layout()

    output_path = output_dir / "experiment_7_cross_judge_attempts_scatter.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def analyze_judge_agreement(df: pd.DataFrame, output_dir: Path):
    """Analyze agreement between judges on various metrics."""
    judges = sorted(df["judge"].unique())

    print("\n=== Inter-Judge Agreement Analysis ===\n")

    # Create a unique trial ID for matching across judges
    df = df.copy()
    df["trial_id"] = df["source_file"] + "_" + df["seed"].astype(str) + "_" + df["feature_index"].astype(str)

    agreement_results = []

    for i, judge1 in enumerate(judges):
        for j, judge2 in enumerate(judges):
            if i >= j:
                continue

            df1 = df[df["judge"] == judge1].set_index("trial_id")
            df2 = df[df["judge"] == judge2].set_index("trial_id")

            # Handle potential duplicate trial_ids within a judge (e.g., multiple runs merged)
            if df1.index.duplicated().any():
                df1 = df1[~df1.index.duplicated(keep="first")]
            if df2.index.duplicated().any():
                df2 = df2[~df2.index.duplicated(keep="first")]

            common_trials = df1.index.intersection(df2.index)
            if len(common_trials) < 10:
                print(f"{judge1} vs {judge2}: Not enough common trials ({len(common_trials)})")
                continue

            print(f"\n--- {judge1} vs {judge2} ({len(common_trials)} common trials) ---")

            # First score correlation
            scores1 = df1.loc[common_trials, "new_first_score"]
            scores2 = df2.loc[common_trials, "new_first_score"]

            # Align + filter with an index-consistent boolean mask
            common_idx = scores1.index.intersection(scores2.index)
            scores1 = scores1.loc[common_idx]
            scores2 = scores2.loc[common_idx]
            valid = scores1.notna() & scores2.notna()
            if valid.sum() > 10:
                s1 = scores1[valid].values
                s2 = scores2[valid].values

                pearson_r, pearson_p = pearsonr(s1, s2)
                spearman_r, spearman_p = spearmanr(s1, s2)

                print(f"  First score correlation (n={len(scores1)}):")
                print(f"    Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})")
                print(f"    Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})")

                agreement_results.append({
                    "judge1": judge1,
                    "judge2": judge2,
                    "metric": "first_score_pearson",
                    "value": pearson_r,
                })

            # Multi-attempt agreement
            multi1 = df1.loc[common_trials, "new_has_multi"]
            multi2 = df2.loc[common_trials, "new_has_multi"]
            common_idx = multi1.index.intersection(multi2.index)
            multi1 = multi1.loc[common_idx]
            multi2 = multi2.loc[common_idx]
            valid_m = multi1.notna() & multi2.notna()
            if valid_m.any():
                multi_agreement = (multi1[valid_m].astype(bool).values == multi2[valid_m].astype(bool).values).mean()
                print(f"  Multi-attempt detection agreement: {multi_agreement:.3f}")
            else:
                multi_agreement = np.nan
                print("  Multi-attempt detection agreement: n/a")

            agreement_results.append({
                "judge1": judge1,
                "judge2": judge2,
                "metric": "multi_attempt_agreement",
                "value": multi_agreement,
            })

            # ESR direction agreement (for trials both identified as multi-attempt)
            both_multi = (multi1.astype(bool) & multi2.astype(bool)) & valid_m
            if both_multi.sum() > 5:
                # ESR = improvement > 0
                imp1 = df1.loc[both_multi.index, "new_improvement"].loc[both_multi]
                imp2 = df2.loc[both_multi.index, "new_improvement"].loc[both_multi]
                common_idx = imp1.index.intersection(imp2.index)
                esr1 = imp1.loc[common_idx] > 0
                esr2 = imp2.loc[common_idx] > 0
                esr_agreement = (esr1 == esr2).mean()
                print(f"  ESR direction agreement (multi-attempt only, n={both_multi.sum()}): {esr_agreement:.3f}")

                agreement_results.append({
                    "judge1": judge1,
                    "judge2": judge2,
                    "metric": "esr_direction_agreement",
                    "value": esr_agreement,
                })

    # Save agreement results
    if agreement_results:
        agreement_df = pd.DataFrame(agreement_results)
        agreement_path = output_dir / "experiment_7_judge_agreement_metrics.csv"
        agreement_df.to_csv(agreement_path, index=False)
        print(f"\nSaved agreement metrics to: {agreement_path}")


def save_experiment_7_sidecars(df: pd.DataFrame, results_dir: Path, output_dir: Path) -> None:
    """
    Save sidecar data files for Experiment 7 (cross-judge analysis).

    - Raw per-trial/per-judge table as CSV
    - Compact summary JSON with the main pivots/rollups used for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    # Provide a stable trial_id for cross-referencing outside the plotting code
    if "feature_index" in df_out.columns:
        df_out["trial_id"] = (
            df_out["source_file"]
            + "_"
            + df_out["seed"].astype(str)
            + "_"
            + df_out["feature_index"].astype(str)
        )

    trials_csv_path = output_dir / "experiment_7_cross_judge_trials.csv"
    df_out.to_csv(trials_csv_path, index=False)
    print(f"Saved: {trials_csv_path}")

    judges = sorted(df_out["judge"].dropna().unique().tolist())
    target_models = sorted(df_out["target_model"].dropna().unique().tolist())

    # Core pivots that back the plots
    # Only include multi-attempt trials for MSI calculation
    df_multi = df_out[df_out["new_has_multi"]]
    mean_improvement_pivot = (
        df_multi.pivot_table(
            values="new_improvement",
            index="target_model",
            columns="judge",
            aggfunc="mean",
        )
        .reindex(index=target_models, columns=judges)
    )
    multi_attempt_rate_pivot = (
        df_out.pivot_table(
            values="new_has_multi",
            index="target_model",
            columns="judge",
            aggfunc="mean",
        )
        .reindex(index=target_models, columns=judges)
        * 100.0
    )

    summary = {
        "results_dir": str(results_dir),
        "n_rows": int(len(df_out)),
        "judges": judges,
        "target_models": target_models,
        "samples_per_judge": {j: int((df_out["judge"] == j).sum()) for j in judges},
        "samples_per_target_model": {m: int((df_out["target_model"] == m).sum()) for m in target_models},
        "mean_score_improvement_by_judge_and_model": mean_improvement_pivot.round(6).to_dict(),
        "multi_attempt_detection_rate_pct_by_judge_and_model": multi_attempt_rate_pivot.round(6).to_dict(),
    }

    summary_path = output_dir / "experiment_7_cross_judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n=== Summary Statistics ===\n")

    judges = sorted(df["judge"].unique())
    target_models = sorted(df["target_model"].dropna().unique())

    print("Samples per judge:")
    for judge in judges:
        n = len(df[df["judge"] == judge])
        print(f"  {judge}: {n}")

    print("\nSamples per target model (across all judges):")
    for model in target_models:
        n = len(df[df["target_model"] == model]) // len(judges) if judges else 0
        print(f"  {model}: {n}")

    print("\nMean score improvement by judge and model (multi-attempt trials only):")
    df_multi = df[df["new_has_multi"]]
    pivot = df_multi.pivot_table(
        values="new_improvement",
        index="target_model",
        columns="judge",
        aggfunc="mean",
    )
    print(pivot.round(2).to_string())

    print("\nMulti-attempt detection rate (%) by judge and model:")
    pivot = df.pivot_table(
        values="new_has_multi",
        index="target_model",
        columns="judge",
        aggfunc="mean",
    ) * 100
    print(pivot.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-judge regrading results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE_DIR / "cross_judge_results",
        help="Directory containing regraded results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (defaults to plots/)",
    )
    parser.add_argument(
        "--haiku-only",
        action="store_true",
        help="Only use experiment results from the haiku judge folder",
    )
    args = parser.parse_args()

    # Override results-dir if haiku-only
    if args.haiku_only:
        args.results_dir = BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "cross_judge_results"
        print(f"Using haiku judge folder: {args.results_dir}")

    output_dir = args.output_dir if args.output_dir.is_absolute() else (BASE_DIR / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    judge_results = load_regraded_results(args.results_dir)

    if not judge_results:
        print("No results found!")
        return

    print(f"Found results for judges: {list(judge_results.keys())}")

    # Create DataFrame
    df = create_score_dataframe(judge_results)
    print(f"Total rows in DataFrame: {len(df)}")

    # Save sidecar data (raw + summary) like other plotting scripts
    save_experiment_7_sidecars(df=df, results_dir=args.results_dir, output_dir=output_dir)

    # Print summary
    print_summary_stats(df)

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_msi_by_judge(df, output_dir)
    plot_multi_attempt_rates(df, output_dir)
    plot_esr_rate_all_trials_by_judge(df, output_dir)
    # plot_correlation_heatmap(df, output_dir)  # disabled per request
    # plot_score_scatter(df, args.output_dir)
    plot_attempts_scatter(df, output_dir)

    # Analyze agreement
    analyze_judge_agreement(df, output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
