#!/usr/bin/env python3
"""
Combined plot comparing prompting vs fine-tuning approaches for improving ESR.

Creates a 2-panel figure:
1. Left: Bar chart showing MSI improvement from prompting across all models
2. Right: Line chart showing MSI across fine-tuning ratios for Llama 8B
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_file_utils import canonicalize_model_name, get_model_color, ModelFamily
from plotting.plot_utils import is_degraded_output, collect_experiment_1_result_files

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "experiment_results"

# Increase font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PromptingMetrics:
    """Metrics for a single model + prompt variant combination."""
    model_name: str
    variant_id: str
    n_trials: int
    mean_score_improvement: float
    se_score_improvement: float


@dataclass
class FinetuningMetrics:
    """Metrics for a single fine-tuning ratio."""
    label: str
    ratio_pct: Optional[int]  # None for baseline
    n_trials: int
    mean_improvement: float
    mean_improvement_se: float


# ============================================================================
# Prompting data loading (from Experiment 5)
# ============================================================================

# Model display names and order
MODEL_SHORT_NAMES = {
    "google/gemma-2-2b-it-res-16k-layer-16": "Gemma 2B",
    "google/gemma-2-9b-it-res-16k-layer-20": "Gemma 9B",
    "google/gemma-2-27b-it-res-131k-layer-22": "Gemma 27B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 8B",
    "meta-llama/Meta-Llama-3.3-70B-Instruct": "Llama 70B",
}

MODEL_ORDER = [
    "google/gemma-2-2b-it-res-16k-layer-16",
    "google/gemma-2-9b-it-res-16k-layer-20",
    "google/gemma-2-27b-it-res-131k-layer-22",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
]

PROMPT_VARIANT_DISPLAY = {
    "baseline": "Baseline",
    "self_monitor": "Self-monitor",
    "dont_get_distracted": "Don't get distracted",
    "ignore_latent_label": "Ignore latent label",
    "resist_manipulation": "Resist manipulation",
}


def _get_variant_id_from_data(data: dict, fallback_filename: str) -> str:
    """Extract variant ID from experiment data."""
    cfg = data.get("experiment_config", {}) or {}
    vid = cfg.get("prompt_variant_id")
    if vid:
        return str(vid)

    meta = data.get("experiment_metadata", {}) or {}
    vid = meta.get("prompt_variant_id")
    if vid:
        return str(vid)

    # Fallback: parse from filename
    parts = Path(fallback_filename).name.split("_")
    if len(parts) >= 5:
        return parts[-3]
    return "unknown"


def _iter_trial_improvements(data: dict) -> List[float]:
    """
    Extract score improvements from all valid trials.
    Returns list of (last_score - first_score) for each trial.
    """
    improvements = []
    
    for feature_result in data.get("results_by_feature", []):
        if feature_result.get("error"):
            continue
        for trial in feature_result.get("trials", []):
            if trial.get("error"):
                continue

            # Skip degraded outputs
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
            improvement = last - first if len(scores) > 1 else 0.0
            improvements.append(improvement)

    return improvements


def load_prompting_data() -> Dict[str, Dict[str, PromptingMetrics]]:
    """
    Load Experiment 5 prompting data.
    
    Returns dict: model_name -> variant_id -> PromptingMetrics
    """
    pattern = "experiment_5_prompt_variants_*.json"
    files = sorted(RESULTS_DIR.glob(pattern))
    
    results: Dict[str, Dict[str, PromptingMetrics]] = {}
    
    # Load experiment 5 variant files
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        variant_id = _get_variant_id_from_data(data, f.name)
        model_name = (data.get("experiment_config", {}) or {}).get("model_name", "unknown")
        
        improvements = _iter_trial_improvements(data)
        if not improvements:
            continue
            
        n = len(improvements)
        mean_imp = float(np.mean(improvements))
        se_imp = float(np.std(improvements, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        
        if model_name not in results:
            results[model_name] = {}
        
        results[model_name][variant_id] = PromptingMetrics(
            model_name=model_name,
            variant_id=variant_id,
            n_trials=n,
            mean_score_improvement=mean_imp,
            se_score_improvement=se_imp,
        )
    
    # Load baseline from Experiment 1 files using the proper collection utility
    # This automatically finds the correct files for each model
    _, model_info_map, model_files = collect_experiment_1_result_files(
        BASE_DIR,
        excluded_families={ModelFamily.FINETUNED_8B},
        haiku_only=args.haiku_only,
    )
    
    # Build a mapping from canonical model name back to full model path
    # (the model_files keys are display names, we need the full paths)
    for display_name, filepaths in model_files.items():
        if not filepaths:
            continue
            
        # Aggregate improvements across all baseline files for this model
        all_improvements = []
        model_name = None
        
        for filepath in filepaths:
            with open(filepath, "r") as fp:
                data = json.load(fp)
            
            # Get the full model name from the config
            cfg_model_name = (data.get("experiment_config", {}) or {}).get("model_name")
            if cfg_model_name:
                model_name = cfg_model_name
            
            improvements = _iter_trial_improvements(data)
            all_improvements.extend(improvements)
        
        if not all_improvements or not model_name:
            continue
            
        n = len(all_improvements)
        mean_imp = float(np.mean(all_improvements))
        se_imp = float(np.std(all_improvements, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        
        if model_name not in results:
            results[model_name] = {}
        
        results[model_name]["baseline"] = PromptingMetrics(
            model_name=model_name,
            variant_id="baseline",
            n_trials=n,
            mean_score_improvement=mean_imp,
            se_score_improvement=se_imp,
        )
    
    return results


# ============================================================================
# Fine-tuning data loading (from Experiment 4)
# ============================================================================

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


def load_finetuning_data() -> List[FinetuningMetrics]:
    """
    Load Experiment 4 fine-tuning data.
    
    Returns list of FinetuningMetrics sorted by ratio.
    """
    results = []
    
    for label, filename in MASKED_RATIO_FILES.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"Warning: Fine-tuning file not found: {filename}")
            continue
            
        with open(filepath, "r") as fp:
            data = json.load(fp)
        
        improvements = _iter_trial_improvements(data)
        if not improvements:
            continue
            
        n = len(improvements)
        mean_imp = float(np.mean(improvements))
        se_imp = float(np.std(improvements, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        
        # Parse ratio from label
        if label == "Base":
            ratio_pct = None
        else:
            ratio_pct = int(label.replace("%", ""))
        
        results.append(FinetuningMetrics(
            label=label,
            ratio_pct=ratio_pct,
            n_trials=n,
            mean_improvement=mean_imp,
            mean_improvement_se=se_imp,
        ))
    
    # Sort: baseline first, then by ratio
    results.sort(key=lambda x: (x.ratio_pct is not None, x.ratio_pct or 0))
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def create_combined_figure(
    prompting_data: Dict[str, Dict[str, PromptingMetrics]],
    finetuning_data: List[FinetuningMetrics],
    best_variant: str = "self_monitor",
    output_dir: Path = None,
) -> Tuple[plt.Figure, Path]:
    """
    Create a single bar chart comparing baseline vs prompted vs fine-tuned for Llama 8B.
    """
    
    # Target model for the comparison
    target_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Get prompting data for target model
    if target_model not in prompting_data:
        raise ValueError(f"Target model {target_model} not found in prompting data")
    
    model_variants = prompting_data[target_model]
    if "baseline" not in model_variants:
        raise ValueError(f"Baseline not found for {target_model}")
    if best_variant not in model_variants:
        raise ValueError(f"{best_variant} not found for {target_model}")
    
    baseline = model_variants["baseline"]
    prompted = model_variants[best_variant]
    
    # Get fine-tuning data at 50% ratio
    finetuned_50 = next((d for d in finetuning_data if d.ratio_pct == 50), None)
    if finetuned_50 is None:
        raise ValueError("50% fine-tuning ratio not found")
    
    z_score = 1.96  # 95% CI
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data for the 3 bars
    labels = ["Baseline", "Meta-Prompted", "Fine-Tuned"]
    values = [
        baseline.mean_score_improvement,
        prompted.mean_score_improvement,
        finetuned_50.mean_improvement,
    ]
    errors = [
        z_score * baseline.se_score_improvement,
        z_score * prompted.se_score_improvement,
        z_score * finetuned_50.mean_improvement_se,
    ]
    
    # Colors: gradient from light to dark blue-teal
    colors = ["#bdc3c7", "#3498db", "#1abc9c"]
    
    x = np.arange(len(labels))
    bar_width = 0.6
    
    bars = ax.bar(x, values, bar_width, yerr=errors, capsize=6,
                  color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    
    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + err + 0.15,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Mean Score Improvement", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    
    # Set y-axis limits
    y_max = max(values) + max(errors) + 1.0
    y_min = min(0, min(values) - 0.5)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = BASE_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "combined_prompting_vs_finetuning.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Saved plot to: {output_path}")
    
    # Save sidecar data
    data_path = output_path.with_suffix('.json')
    sidecar_data = {
        "model": target_model,
        "prompt_variant": best_variant,
        "finetuning_ratio": 50,
        "conditions": {
            "baseline": {
                "msi": baseline.mean_score_improvement,
                "se": baseline.se_score_improvement,
                "n_trials": baseline.n_trials,
            },
            "prompted": {
                "msi": prompted.mean_score_improvement,
                "se": prompted.se_score_improvement,
                "n_trials": prompted.n_trials,
            },
            "finetuned": {
                "msi": finetuned_50.mean_improvement,
                "se": finetuned_50.mean_improvement_se,
                "n_trials": finetuned_50.n_trials,
            },
        }
    }
    with open(data_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)
    print(f"📄 Saved sidecar data to: {data_path}")
    
    return fig, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create combined prompting vs fine-tuning comparison plot"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots/data (relative to experiment base dir). Default: plots/",
    )
    parser.add_argument(
        "--best-variant",
        type=str,
        default="self_monitor",
        help="Which prompt variant to compare against baseline. Default: self_monitor",
    )
    parser.add_argument(
        "--haiku-only",
        action="store_true",
        help="Only use experiment results from the haiku judge folder",
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    
    print("=" * 70)
    print("COMBINED PROMPTING VS FINE-TUNING COMPARISON")
    print("=" * 70)
    
    # Load data
    print("\nLoading prompting data (Experiment 5)...")
    prompting_data = load_prompting_data()
    print(f"  Found {len(prompting_data)} models with prompting data")
    for model, variants in prompting_data.items():
        print(f"    {MODEL_SHORT_NAMES.get(model, model)}: {list(variants.keys())}")
    
    print("\nLoading fine-tuning data (Experiment 4)...")
    finetuning_data = load_finetuning_data()
    print(f"  Found {len(finetuning_data)} fine-tuning configurations")
    
    # Create figure
    print("\nCreating combined figure...")
    fig, output_path = create_combined_figure(
        prompting_data,
        finetuning_data,
        best_variant=args.best_variant,
        output_dir=output_dir,
    )
    
    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()

