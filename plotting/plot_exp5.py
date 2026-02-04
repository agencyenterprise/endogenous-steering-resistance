#!/usr/bin/env python3
"""
Plot Experiment 5 (prompt-variant sweep) results.

Outputs a single figure containing 3 bar charts:
1) First-attempt score for each variant compared to baseline (delta vs baseline if available)
2) Multi-Attempt % (% of responses with multiple attempts)
3) ESR Rate (% of ALL responses with multi-attempt AND improvement)

This script expects Experiment 5 output JSONs produced by `experiment_05_prompt_variants`.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_file_utils import canonicalize_model_name, get_model_color
from plotting.plot_utils import is_degraded_output

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent


# Increase font sizes for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
})


@dataclass(frozen=True)
class VariantMetrics:
    variant_id: str
    n_trials: int
    mean_first_score: float
    se_first_score: float
    pct_multi_attempt: float
    se_pct_multi_attempt: float
    mean_score_improvement: float
    se_score_improvement: float
    pct_improved_of_all: float
    se_pct_improved_of_all: float


PREFERRED_ORDER = [
    "baseline",
    "dont_get_distracted",
    "ignore_latent_label",
    "self_monitor",
    "resist_manipulation",
]

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "dont_get_distracted": "Don't get distracted",
    "ignore_latent_label": "Ignore {latent_label}",
    "self_monitor": "Meta-prompted",
    "resist_manipulation": "Resist manipulation",
}

# Color scheme inspired by experiment 2 (sweep analysis)
# This avoids overlapping with the model-specific colors from experiment 1.
VARIANT_COLORS = {
    "baseline": "#7f7f7f",                  # Neutral Gray
    "dont_get_distracted": "#6A0DAD",       # Deep Purple
    "ignore_latent_label": "#0E6655",       # Deep Teal
    "self_monitor": "#E67E22",              # Orange
    "resist_manipulation": "#C0392B",       # Dark Red
}


def _get_variant_id_from_data(data: dict, fallback_filename: str) -> str:
    # Preferred: experiment_config.prompt_variant_id (set by experiment_5_prompt_variants.py)
    cfg = data.get("experiment_config", {}) or {}
    vid = cfg.get("prompt_variant_id")
    if vid:
        return str(vid)

    # Fallback: experiment_metadata.prompt_variant_id (also set by our runner)
    meta = data.get("experiment_metadata", {}) or {}
    vid = meta.get("prompt_variant_id")
    if vid:
        return str(vid)

    # Last resort: parse from filename: experiment_5_prompt_variants_<model>_<variant>_<ts>.json
    parts = Path(fallback_filename).name.split("_")
    # Try to locate "...prompt_variants_<model>_<variant>_<YYYYMMDD>_<HHMMSS>.json"
    # We take the third-from-last element as variant id in the common case.
    if len(parts) >= 5:
        # e.g. [..., "<variant>", "20251219", "163817.json"] (but time has .json attached if split on "_")
        # Actually filename is ..._{variant_id}_{YYYYMMDD}_{HHMMSS}.json -> split gives ... variant, YYYYMMDD, HHMMSS.json
        return parts[-3]

    return "unknown"


def _iter_trial_rows(data: dict, variant_id: str, exclude_degraded: bool = False) -> Iterable[Tuple[str, float, int, float]]:
    """
    Yield (variant_id, first_score, n_attempts, score_improvement).
    Skips errored trials, degraded outputs (if exclude_degraded), and trials with no attempts.
    """
    for feature_result in data.get("results_by_feature", []):
        if feature_result.get("error"):
            continue
        for trial in feature_result.get("trials", []):
            if trial.get("error"):
                continue

            # Skip degraded outputs (repetitive patterns) if exclude_degraded
            if exclude_degraded:
                response = trial.get("response", "")
                if is_degraded_output(response):
                    continue

            score_dict = trial.get("score", {}) or {}
            attempts = score_dict.get("attempts", []) or []
            if not attempts:
                continue

            scores = [a.get("score") for a in attempts if a.get("score") is not None]
            if not scores:
                continue

            first = float(scores[0])
            last = float(scores[-1])
            n_attempts = int(len(scores))
            improvement = last - first if n_attempts > 1 else 0.0
            yield (variant_id, first, n_attempts, float(improvement))


def _load_files(files: List[Path], exclude_degraded: bool = False) -> Tuple[List[Tuple[str, float, int, float]], Dict[str, str]]:
    """Return trial rows and variant_id->model_name mapping."""
    rows: List[Tuple[str, float, int, float]] = []
    variant_to_model: Dict[str, str] = {}

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        variant_id = _get_variant_id_from_data(data, f.name)
        model_name = (data.get("experiment_config", {}) or {}).get("model_name", "unknown")
        variant_to_model[variant_id] = str(model_name)

        rows.extend(list(_iter_trial_rows(data, variant_id, exclude_degraded)))

    return rows, variant_to_model


def _load_files_by_model(files: List[Path], exclude_degraded: bool = False) -> Dict[str, List[Tuple[str, float, int, float]]]:
    """Return a dict mapping model_name -> list of trial rows."""
    rows_by_model: Dict[str, List[Tuple[str, float, int, float]]] = {}

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        variant_id = _get_variant_id_from_data(data, f.name)
        model_name = (data.get("experiment_config", {}) or {}).get("model_name", "unknown")

        if model_name not in rows_by_model:
            rows_by_model[model_name] = []

        rows_by_model[model_name].extend(list(_iter_trial_rows(data, variant_id, exclude_degraded)))

    return rows_by_model


def _aggregate(rows: List[Tuple[str, float, int, float]]) -> Dict[str, VariantMetrics]:
    by_variant: Dict[str, List[Tuple[float, int, float]]] = {}
    for variant_id, first, n_attempts, improvement in rows:
        by_variant.setdefault(variant_id, []).append((first, n_attempts, improvement))

    out: Dict[str, VariantMetrics] = {}
    for variant_id, xs in by_variant.items():
        firsts = np.array([x[0] for x in xs], dtype=float)
        attempts = np.array([x[1] for x in xs], dtype=float)

        # Only include multi-attempt trials for MSI calculation
        multi_attempt_imps = np.array([x[2] for x in xs if x[1] > 1], dtype=float)

        n = len(xs)
        is_multi = (attempts > 1).astype(float)
        n_multi = len(multi_attempt_imps)

        # Standard error of the mean (SE). If n<=1, SE=0.
        se_first = float(np.std(firsts, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        # SE for percentage: SE = sqrt(p * (1-p) / n)
        p = np.mean(is_multi)
        se_pct = float(np.sqrt(p * (1 - p) / n) * 100) if n > 1 else 0.0

        # SE for MSI (multi-attempt only)
        se_imps = float(np.std(multi_attempt_imps, ddof=1) / np.sqrt(n_multi)) if n_multi > 1 else 0.0

        # ESR Rate (of ALL responses) - responses with multi-attempt AND improvement
        n_improved = sum(1 for x in xs if x[1] > 1 and x[2] > 0)
        pct_improved_of_all = (n_improved / n * 100) if n > 0 else 0.0
        p_improved = pct_improved_of_all / 100
        se_pct_improved_of_all = float(np.sqrt(p_improved * (1 - p_improved) / n) * 100) if n > 1 else 0.0

        out[variant_id] = VariantMetrics(
            variant_id=variant_id,
            n_trials=n,
            mean_first_score=float(np.mean(firsts)),
            se_first_score=se_first,
            pct_multi_attempt=float(p * 100),
            se_pct_multi_attempt=se_pct,
            mean_score_improvement=float(np.mean(multi_attempt_imps)) if n_multi > 0 else 0.0,
            se_score_improvement=se_imps,
            pct_improved_of_all=pct_improved_of_all,
            se_pct_improved_of_all=se_pct_improved_of_all,
        )

    return out


def _sort_variants(variant_ids: List[str]) -> List[str]:
    order_index = {v: i for i, v in enumerate(PREFERRED_ORDER)}
    return sorted(variant_ids, key=lambda v: (order_index.get(v, 10_000), v))


def _display_name(variant_id: str) -> str:
    return DISPLAY_NAMES.get(variant_id, variant_id)


# Short model names for the combined plot
MODEL_SHORT_NAMES = {
    "google/gemma-2-2b-it-res-16k-layer-16": "Gemma 2 2B",
    "google/gemma-2-9b-res-16k-layer-26": "Gemma 2 9B",
    "google/gemma-2-27b-it-res-131k-layer-22": "Gemma 2 27B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "meta-llama/Meta-Llama-3.3-70B-Instruct": "Llama 3.3 70B",
}

# Preferred order for models in combined plot (smallest to largest)
MODEL_ORDER = [
    "google/gemma-2-2b-it-res-16k-layer-16",
    "google/gemma-2-9b-res-16k-layer-26",
    "google/gemma-2-27b-it-res-131k-layer-22",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
]


def _short_model_name(model_name: str) -> str:
    return MODEL_SHORT_NAMES.get(model_name, model_name.split("/")[-1])


def _plot_bars(
    metrics: Dict[str, VariantMetrics],
    model_name: str,
    baseline_variant_id: str,
    output_dir: Path,
    attempts_ymin: float,
) -> Path:
    variant_ids = _sort_variants(list(metrics.keys()))
    if not variant_ids:
        raise ValueError("No trials found to plot (all trials may have been skipped due to errors/missing attempts).")

    baseline = metrics.get(baseline_variant_id)
    has_baseline = baseline is not None

    labels = [_display_name(v) for v in variant_ids]
    colors = [VARIANT_COLORS.get(v, "#888888") for v in variant_ids]
    x = np.arange(len(variant_ids))

    z = 1.96  # 95% CI

    # Chart 1: First attempt score vs baseline (delta if baseline exists, else absolute)
    first_means = np.array([metrics[v].mean_first_score for v in variant_ids], dtype=float)
    first_ses = np.array([metrics[v].se_first_score for v in variant_ids], dtype=float)
    if has_baseline:
        first_values = first_means - baseline.mean_first_score
        # Propagate SE assuming independence: SE_delta = sqrt(SE_v^2 + SE_base^2)
        first_ses = np.sqrt(first_ses**2 + (baseline.se_first_score**2))
        first_ylabel = "Δ First-attempt score (pts)"
    else:
        first_values = first_means
        first_ylabel = "Mean first-attempt score (0–100)"

    # Chart 2: % multi-attempt responses
    pct_multi_means = np.array([metrics[v].pct_multi_attempt for v in variant_ids], dtype=float)
    pct_multi_ses = np.array([metrics[v].se_pct_multi_attempt for v in variant_ids], dtype=float)

    # Chart 3: ESR Rate (of ALL responses)
    pct_improved_means = np.array([metrics[v].pct_improved_of_all for v in variant_ids], dtype=float)
    pct_improved_ses = np.array([metrics[v].se_pct_improved_of_all for v in variant_ids], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_first, ax_multi, ax_improve = axes

    # First chart
    ax = ax_first
    ax.bar(x, first_values, yerr=z * first_ses, capsize=4, color=colors, alpha=0.8)
    if has_baseline:
        ax.axhline(0.0, color="black", linewidth=1)
        # Set symmetric y-limits around zero for delta
        limit = max(8.0, float(np.max(np.abs(first_values + z * first_ses))) * 1.2)
        ax.set_ylim(-limit, limit)
    else:
        ax.set_ylim(0, 100)
    ax.set_ylabel(first_ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Mean First-Attempt Score")
    ax.grid(True, axis="y", alpha=0.3)

    # Second chart: Multi-Attempt %
    ax = ax_multi
    ax.bar(x, pct_multi_means, yerr=z * pct_multi_ses, capsize=4, color=colors, alpha=0.8)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Multi-Attempt %")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(12, float(np.max(pct_multi_means + z * pct_multi_ses)) * 1.2))

    # Third chart: ESR Rate (of ALL responses)
    ax = ax_improve
    ax.bar(x, pct_improved_means, yerr=z * pct_improved_ses, capsize=4, color=colors, alpha=0.8)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("ESR Rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(12, float(np.max(pct_multi_means + z * pct_multi_ses)) * 1.2))

    baseline_note = f" (baseline='{baseline_variant_id}')" if has_baseline else " (no baseline found)"
    fig.suptitle(f"Experiment 5 prompt variants — {model_name}{baseline_note}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.split("/")[-1].replace(" ", "_").replace("/", "_")
    base_name = f"experiment_5_prompt_variant_bars_{safe_model}"
    
    # Save plot
    out_path = output_dir / f"{base_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save sidecar data
    data_path = output_dir / f"{base_name}.json"
    sidecar_data = {
        "model_name": model_name,
        "baseline_variant_id": baseline_variant_id,
        "timestamp": ts,
        "metrics": {vid: asdict(m) for vid, m in metrics.items()}
    }
    with open(data_path, "w") as f:
        json.dump(sidecar_data, f, indent=4)

    return out_path, data_path


def _plot_combined_baseline_vs_resistance(
    metrics_by_model: Dict[str, Dict[str, VariantMetrics]],
    baseline_variant_id: str,
    resistance_variant_id: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Create a combined figure comparing baseline vs resistance prompt across all models.
    
    3 facets (First Attempt Score, % Multi-Attempt, Mean Score Improvement),
    each with grouped bar charts where each group is a model.
    """
    # Sort models by preferred order
    model_names = sorted(
        metrics_by_model.keys(),
        key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 999, m)
    )
    
    # Filter to models that have both baseline and resistance variants
    valid_models = []
    for m in model_names:
        metrics = metrics_by_model[m]
        if baseline_variant_id in metrics and resistance_variant_id in metrics:
            valid_models.append(m)
        else:
            print(f"  ⚠️  Skipping model '{m}' in combined plot (missing baseline or resistance variant)")
    
    if not valid_models:
        raise ValueError("No models have both baseline and resistance variants")
    
    n_models = len(valid_models)
    short_names = [_short_model_name(m) for m in valid_models]
    
    # Extract metrics for each model
    baseline_first = []
    baseline_first_se = []
    resistance_first = []
    resistance_first_se = []
    
    baseline_multi = []
    baseline_multi_se = []
    resistance_multi = []
    resistance_multi_se = []
    
    baseline_improve = []
    baseline_improve_se = []
    resistance_improve = []
    resistance_improve_se = []

    for m in valid_models:
        b = metrics_by_model[m][baseline_variant_id]
        r = metrics_by_model[m][resistance_variant_id]

        baseline_first.append(b.mean_first_score)
        baseline_first_se.append(b.se_first_score)
        resistance_first.append(r.mean_first_score)
        resistance_first_se.append(r.se_first_score)

        baseline_multi.append(b.pct_multi_attempt)
        baseline_multi_se.append(b.se_pct_multi_attempt)
        resistance_multi.append(r.pct_multi_attempt)
        resistance_multi_se.append(r.se_pct_multi_attempt)

        baseline_improve.append(b.pct_improved_of_all)
        baseline_improve_se.append(b.se_pct_improved_of_all)
        resistance_improve.append(r.pct_improved_of_all)
        resistance_improve_se.append(r.se_pct_improved_of_all)
    
    # Convert to numpy arrays
    baseline_first = np.array(baseline_first)
    baseline_first_se = np.array(baseline_first_se)
    resistance_first = np.array(resistance_first)
    resistance_first_se = np.array(resistance_first_se)
    
    baseline_multi = np.array(baseline_multi)
    baseline_multi_se = np.array(baseline_multi_se)
    resistance_multi = np.array(resistance_multi)
    resistance_multi_se = np.array(resistance_multi_se)
    
    baseline_improve = np.array(baseline_improve)
    baseline_improve_se = np.array(baseline_improve_se)
    resistance_improve = np.array(resistance_improve)
    resistance_improve_se = np.array(resistance_improve_se)
    
    z = 1.96  # 95% CI
    bar_width = 0.35
    x = np.arange(n_models)
    
    # Get model colors (same as experiment 1)
    model_colors = []
    for m in valid_models:
        model_info = canonicalize_model_name(m)
        model_colors.append(get_model_color(model_info, scheme="target_models"))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_first, ax_multi, ax_improve = axes
    
    # Panel 1: First Attempt Score
    ax = ax_first
    # Draw bars for each model with their own color
    for i, (m, color) in enumerate(zip(valid_models, model_colors)):
        # Baseline: hatched pattern
        ax.bar(x[i] - bar_width/2, baseline_first[i], bar_width,
               yerr=z * baseline_first_se[i], capsize=3,
               color=color, alpha=0.4, edgecolor=color, linewidth=1.5,
               hatch='///')
        # Resistance: solid fill
        ax.bar(x[i] + bar_width/2, resistance_first[i], bar_width,
               yerr=z * resistance_first_se[i], capsize=3,
               color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Mean First-Attempt Score")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_title("Mean First-Attempt Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(70, float(np.max(resistance_first + z * resistance_first_se)) * 1.15))

    # Panel 2: Multi-Attempt %
    ax = ax_multi
    for i, (m, color) in enumerate(zip(valid_models, model_colors)):
        ax.bar(x[i] - bar_width/2, baseline_multi[i], bar_width,
               yerr=z * baseline_multi_se[i], capsize=3,
               color=color, alpha=0.4, edgecolor=color, linewidth=1.5,
               hatch='///')
        ax.bar(x[i] + bar_width/2, resistance_multi[i], bar_width,
               yerr=z * resistance_multi_se[i], capsize=3,
               color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_title("Multi-Attempt %")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(10, float(np.max(resistance_multi + z * resistance_multi_se)) * 1.3))

    # Panel 3: ESR Rate (of ALL responses)
    ax = ax_improve
    for i, (m, color) in enumerate(zip(valid_models, model_colors)):
        ax.bar(x[i] - bar_width/2, baseline_improve[i], bar_width,
               yerr=z * baseline_improve_se[i], capsize=3,
               color=color, alpha=0.4, edgecolor=color, linewidth=1.5,
               hatch='///')
        ax.bar(x[i] + bar_width/2, resistance_improve[i], bar_width,
               yerr=z * resistance_improve_se[i], capsize=3,
               color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("ESR Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_title("ESR Rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(10, float(np.max(resistance_multi + z * resistance_multi_se)) * 1.3))

    # Create a single figure-level legend with proper styling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.4, edgecolor='gray', linewidth=1.5,
              hatch='///', label=_display_name(baseline_variant_id)),
        Patch(facecolor='gray', alpha=0.9, edgecolor='black', linewidth=0.5,
              label=_display_name(resistance_variant_id)),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               bbox_to_anchor=(0.98, 0.88), framealpha=0.9)
    
    fig.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"experiment_5_combined_baseline_{baseline_variant_id}_vs_{resistance_variant_id}"
    
    # Save plot
    out_path = output_dir / f"{base_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Save sidecar data
    data_path = output_dir / f"{base_name}.json"
    sidecar_data = {
        "baseline_variant_id": baseline_variant_id,
        "resistance_variant_id": resistance_variant_id,
        "timestamp": ts,
        "models": valid_models,
        "metrics_by_model": {
            m: {
                "baseline": asdict(metrics_by_model[m][baseline_variant_id]),
                "resistance": asdict(metrics_by_model[m][resistance_variant_id]),
            }
            for m in valid_models
        }
    }
    with open(data_path, "w") as f:
        json.dump(sidecar_data, f, indent=4)
    
    return out_path, data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 5 prompt-variant sweep results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(BASE_DIR / "data/experiment_results"),
        help="Directory containing Experiment 5 result JSONs (default: data/experiment_results/).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="experiment_5_prompt_variants_*.json",
        help="Glob for selecting result files inside --results-dir.",
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of result JSON files and/or directories. "
            "If a directory is provided, it will be expanded using --glob. "
            "If provided, --results-dir/--glob are ignored except for directory expansion."
        ),
    )
    parser.add_argument(
        "--baseline-variant-id",
        type=str,
        default="baseline",
        help="Variant id to treat as baseline (default: baseline).",
    )
    parser.add_argument(
        "--baseline-results",
        nargs="*",
        default=None,
        help=(
            "Path(s) to Experiment 1 results JSON file(s) to use as baseline. "
            "Each file will be loaded with variant_id='baseline'. "
            "Multiple files for the same model will be aggregated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save plots (default: plots/). Prefer this over --plots-dir.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(BASE_DIR / "plots"),
        help="(Deprecated) Where to save plots (default: plots/). Prefer --output-dir.",
    )
    parser.add_argument(
        "--attempts-ymin",
        type=float,
        default=0.0,
        help="Optional y-axis minimum for the 'attempts per response' panel (default: 0.0).",
    )
    parser.add_argument(
        "--resistance-variant-id",
        type=str,
        default="self_monitor",
        help="Variant id to use as the 'resistance' condition in combined plots (default: self_monitor).",
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

    # Override results-dir if haiku-only
    if args.haiku_only and args.results is None:
        args.results_dir = str(BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge")
        print(f"Using haiku judge folder: {args.results_dir}")

    output_dir_arg = args.output_dir if args.output_dir is not None else args.plots_dir
    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir

    if args.results:
        files: List[Path] = []
        for p in args.results:
            path = Path(p)
            if path.is_dir():
                files.extend(sorted(path.glob(args.glob)))
            else:
                files.append(path)
    else:
        files = sorted(Path(args.results_dir).glob(args.glob))

    if not files and not args.baseline_results:
        raise SystemExit("No result files found. Check --results-dir/--glob or pass --results ...")

    # Load files grouped by model
    rows_by_model = _load_files_by_model(files, args.exclude_degraded) if files else {}

    # Load baseline from Experiment 1 results if provided
    if args.baseline_results:
        for baseline_file in args.baseline_results:
            baseline_path = Path(baseline_file)
            if not baseline_path.exists():
                raise SystemExit(f"--baseline-results file not found: {baseline_path}")
            
            with open(baseline_path, "r") as fp:
                baseline_data = json.load(fp)
            
            model_name = (baseline_data.get("experiment_config", {}) or {}).get("model_name", "unknown")
            baseline_rows = list(_iter_trial_rows(baseline_data, "baseline", args.exclude_degraded))
            
            if model_name not in rows_by_model:
                rows_by_model[model_name] = []
            rows_by_model[model_name].extend(baseline_rows)
            
            print(f"Loaded {len(baseline_rows)} baseline trials for {model_name} from {baseline_path.name}")
    if not rows_by_model:
        raise SystemExit("No usable trial rows found (all trials may have errors or missing 'attempts').")

    model_names = sorted(rows_by_model.keys())
    print(f"Found {len(model_names)} model(s): {model_names}")

    # Collect metrics for each model (for combined plot later)
    metrics_by_model: Dict[str, Dict[str, VariantMetrics]] = {}

    # Generate one plot per model
    for model_name in model_names:
        rows = rows_by_model[model_name]
        if not rows:
            print(f"\n⚠️  No usable trial rows for model '{model_name}', skipping...")
            continue

        metrics = _aggregate(rows)
        metrics_by_model[model_name] = metrics
        variant_ids = _sort_variants(list(metrics.keys()))

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print("Per-variant summary (means across all trials):")
        for vid in variant_ids:
            m = metrics[vid]
            print(
                f"- {vid:24s} n={m.n_trials:6d}  "
                f"first={m.mean_first_score:6.2f}  "
                f"multi_att={m.pct_multi_attempt:5.1f}%  "
                f"improvement={m.mean_score_improvement:6.2f}"
            )

        out_path, data_path = _plot_bars(
            metrics=metrics,
            model_name=model_name,
            baseline_variant_id=args.baseline_variant_id,
            output_dir=output_dir,
            attempts_ymin=float(args.attempts_ymin),
        )
        print(f"\n📊 Saved plot to: {out_path}")
        print(f"📄 Saved sidecar data to: {data_path}")

    # Generate combined baseline vs resistance plot across all models
    if len(metrics_by_model) >= 2:
        print(f"\n{'='*60}")
        print("Generating combined baseline vs. resistance plot...")
        print(f"{'='*60}")
        try:
            combined_path, combined_data_path = _plot_combined_baseline_vs_resistance(
                metrics_by_model=metrics_by_model,
                baseline_variant_id=args.baseline_variant_id,
                resistance_variant_id=args.resistance_variant_id,
                output_dir=output_dir,
            )
            print(f"\n📊 Saved combined plot to: {combined_path}")
            print(f"📄 Saved combined sidecar data to: {combined_data_path}")
        except ValueError as e:
            print(f"  ⚠️  Could not generate combined plot: {e}")


if __name__ == "__main__":
    main()


