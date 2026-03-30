#!/usr/bin/env python3
"""Analyze whole-response judge results for experiments 3, 4, 5."""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THRESHOLD = 7


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_scores(trials: list[dict]) -> list[int]:
    return [t["improvement_score"] for t in trials if t.get("improvement_score") is not None]


def get_valid_trials(trials: list[dict]) -> list[dict]:
    return [t for t in trials if t.get("improvement_score") is not None]


def improvement_rate(scores: list[int], threshold: int = THRESHOLD) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores) * 100


def explicit_rate(trials: list[dict], threshold: int = THRESHOLD) -> float:
    valid = [t for t in trials if t.get("improvement_score") is not None]
    if not valid:
        return 0.0
    return sum(1 for t in valid if t["improvement_score"] >= threshold
               and t.get("improvement_type") == "explicit_restart") / len(valid) * 100


def gradual_rate(trials: list[dict], threshold: int = THRESHOLD) -> float:
    valid = [t for t in trials if t.get("improvement_score") is not None]
    if not valid:
        return 0.0
    return sum(1 for t in valid if t["improvement_score"] >= threshold
               and t.get("improvement_type") == "gradual_shift") / len(valid) * 100


def mean_score(scores: list[int]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# Experiment 3: OTD ablation
# Compare with_ablation vs exp1 baseline (no ablation)
# =============================================================================

def analyze_exp3():
    print("=" * 60)
    print("EXPERIMENT 3: OTD Ablation (Llama 70B)")
    print("=" * 60)

    # Load ablation results
    exp3 = load_results("whole_response_judge_exp3_results.json")
    ablation_scores = []
    for trials in exp3.values():
        ablation_scores.extend(get_scores(trials))

    # Load exp1 baseline for Llama 70B
    exp1 = load_results("whole_response_judge_exp1_results.json")
    baseline_scores = []
    for fpath, trials in exp1.items():
        if "Meta-Llama-3.3-70B" in fpath:
            baseline_scores.extend(get_scores(trials))

    print(f"  Baseline (no ablation): {len(baseline_scores)} trials, "
          f"rate={improvement_rate(baseline_scores):.1f}%, mean={mean_score(baseline_scores):.2f}")
    print(f"  With OTD ablation:      {len(ablation_scores)} trials, "
          f"rate={improvement_rate(ablation_scores):.1f}%, mean={mean_score(ablation_scores):.2f}")

    # Original judge comparison
    orig_baseline_esr = 0
    orig_baseline_total = 0
    for fpath, trials in exp1.items():
        if "Meta-Llama-3.3-70B" in fpath:
            for t in trials:
                attempts = t.get("original_score", {}).get("attempts", [])
                orig_baseline_total += 1
                if len(attempts) >= 2 and attempts[-1].get("score", 0) > attempts[0].get("score", 0):
                    orig_baseline_esr += 1

    orig_ablation_esr = 0
    orig_ablation_total = 0
    for trials in exp3.values():
        for t in trials:
            attempts = t.get("original_score", {}).get("attempts", [])
            orig_ablation_total += 1
            if len(attempts) >= 2 and attempts[-1].get("score", 0) > attempts[0].get("score", 0):
                orig_ablation_esr += 1

    print(f"\n  Original judge baseline ESR: {orig_baseline_esr}/{orig_baseline_total} = "
          f"{orig_baseline_esr/orig_baseline_total*100:.1f}%")
    print(f"  Original judge ablation ESR: {orig_ablation_esr}/{orig_ablation_total} = "
          f"{orig_ablation_esr/orig_ablation_total*100:.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: rate comparison
    labels = ["No ablation\n(baseline)", "OTD ablation"]
    rates = [improvement_rate(baseline_scores), improvement_rate(ablation_scores)]
    means = [mean_score(baseline_scores), mean_score(ablation_scores)]
    colors = ["#3498DB", "#E74C3C"]

    bars = ax1.bar(labels, rates, color=colors, width=0.5)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.5, f"{rate:.1f}%",
                ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Responses showing improvement (%)")
    ax1.set_title(f"Exp 3: OTD ablation effect\n(threshold >= {THRESHOLD}/10)")

    bars = ax2.bar(labels, means, color=colors, width=0.5)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, m + 0.05, f"{m:.2f}",
                ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Mean improvement score (0-10)")
    ax2.set_title("Exp 3: Mean improvement score")

    plt.tight_layout()
    plt.savefig("plot_exp3_whole_response.png", dpi=150)
    print("Saved plot_exp3_whole_response.png")


# =============================================================================
# Experiment 4: Fine-tuning
# =============================================================================

def analyze_exp4():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Fine-tuning (Llama 8B)")
    print("=" * 60)

    exp4 = load_results("whole_response_judge_exp4_results.json")

    ratios = []
    rates = []
    means = []
    for fpath in sorted(exp4.keys()):
        fname = Path(fpath).name
        # Extract ratio from filename like "masked-ratio-10pct-merged"
        ratio = int(fname.split("masked-ratio-")[1].split("pct")[0])
        scores = get_scores(exp4[fpath])
        ratios.append(ratio)
        rates.append(improvement_rate(scores))
        means.append(mean_score(scores))
        print(f"  {ratio}% SC data: {len(scores)} trials, "
              f"rate={improvement_rate(scores):.1f}%, mean={mean_score(scores):.2f}")

    # Also get Llama 8B baseline from exp1
    exp1 = load_results("whole_response_judge_exp1_results.json")
    baseline_scores = []
    for fpath, trials in exp1.items():
        if "Meta-Llama-3.1-8B" in fpath:
            baseline_scores.extend(get_scores(trials))
    print(f"\n  Baseline (no fine-tuning): {len(baseline_scores)} trials, "
          f"rate={improvement_rate(baseline_scores):.1f}%, mean={mean_score(baseline_scores):.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ratios, rates, "o-", color="#3498DB", linewidth=2, markersize=8)
    ax1.axhline(y=improvement_rate(baseline_scores), color="gray", linestyle="--",
                label=f"Llama 8B baseline ({improvement_rate(baseline_scores):.1f}%)")
    ax1.set_xlabel("% self-correction data in fine-tuning mix")
    ax1.set_ylabel("Responses showing improvement (%)")
    ax1.set_title(f"Exp 4: Fine-tuning effect\n(threshold >= {THRESHOLD}/10)")
    ax1.legend()

    ax2.plot(ratios, means, "o-", color="#E67E22", linewidth=2, markersize=8)
    ax2.axhline(y=mean_score(baseline_scores), color="gray", linestyle="--",
                label=f"Llama 8B baseline ({mean_score(baseline_scores):.2f})")
    ax2.set_xlabel("% self-correction data in fine-tuning mix")
    ax2.set_ylabel("Mean improvement score (0-10)")
    ax2.set_title("Exp 4: Mean improvement score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("plot_exp4_whole_response.png", dpi=150)
    print("Saved plot_exp4_whole_response.png")


# =============================================================================
# Experiment 5: Meta-prompting
# =============================================================================

def analyze_exp5():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Meta-prompting (all models, all variants)")
    print("=" * 60)

    MODEL_MAP = {
        'Meta-Llama-3.3-70B': 'Llama 3.3 70B',
        'Meta-Llama-3.1-8B': 'Llama 3.1 8B',
        'gemma-2-27b': 'Gemma 2 27B',
        'gemma-2-2b': 'Gemma 2 2B',
        'gemma-2-9b': 'Gemma 2 9B',
    }
    MODEL_ORDER = ['Llama 3.3 70B', 'Llama 3.1 8B', 'Gemma 2 27B', 'Gemma 2 9B', 'Gemma 2 2B']
    VARIANTS = ['dont_get_distracted', 'ignore_latent_label', 'resist_manipulation', 'self_monitor']
    VARIANT_TITLES = {
        'dont_get_distracted': "Don't get distracted",
        'ignore_latent_label': "Ignore latent label",
        'resist_manipulation': "Resist manipulation",
        'self_monitor': "Self-monitor",
    }

    def get_model(fpath: str) -> str | None:
        for key, name in MODEL_MAP.items():
            if key in fpath:
                return name
        return None

    # Baselines from exp1 (full trials for improvement_type access)
    exp1 = load_results("whole_response_judge_exp1_results.json")
    baseline_trials_by_model: dict[str, list[dict]] = defaultdict(list)
    for fpath, trials in exp1.items():
        model = get_model(fpath)
        if model:
            baseline_trials_by_model[model].extend(get_valid_trials(trials))

    # Load prompted results per variant per model (full trials)
    prompted_trials: dict[str, dict[str, list[dict]]] = {v: defaultdict(list) for v in VARIANTS}

    # Llama 70B from main exp5 file
    exp5_path = Path("whole_response_judge_exp5_results.json")
    if exp5_path.exists():
        exp5 = load_results(str(exp5_path))
        for fpath, trials in exp5.items():
            for variant in VARIANTS:
                if variant in fpath and "Meta-Llama-3.3-70B" in fpath:
                    prompted_trials[variant]["Llama 3.3 70B"].extend(get_valid_trials(trials))

    # Other models from per-variant files
    for variant in VARIANTS:
        other_path = Path(f"whole_response_judge_exp5_{variant}_other_models.json")
        if other_path.exists():
            other = load_results(str(other_path))
            for fpath, trials in other.items():
                model = get_model(fpath)
                if model:
                    prompted_trials[variant][model].extend(get_valid_trials(trials))

    # Convenience score lists
    baseline_by_model = {m: [t["improvement_score"] for t in trials] for m, trials in baseline_trials_by_model.items()}
    prompted = {v: {m: [t["improvement_score"] for t in trials] for m, trials in model_trials.items()}
                for v, model_trials in prompted_trials.items()}

    # Print results
    for variant in VARIANTS:
        print(f"\n  --- {VARIANT_TITLES[variant]} ---")
        for model in MODEL_ORDER:
            b = baseline_by_model.get(model, [])
            p = prompted[variant].get(model, [])
            if p:
                print(f"  {model:<18} baseline={improvement_rate(b):.1f}%  prompted={improvement_rate(p):.1f}%  "
                      f"(n={len(p)})")

    # 4-panel plot: one per variant, each showing baseline vs meta-prompted per model
    available_variants = [v for v in VARIANTS if any(prompted[v].get(m) for m in MODEL_ORDER)]
    if not available_variants:
        print("  No meta-prompted data available yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, variant in zip(axes, VARIANTS):
        available = [m for m in MODEL_ORDER if m in prompted[variant] and prompted[variant][m]]
        if not available:
            ax.set_visible(False)
            continue

        x = np.arange(len(available))
        width = 0.35

        # Baseline: gradual + explicit stacked
        b_gradual = [gradual_rate(baseline_trials_by_model[m]) for m in available]
        b_explicit = [explicit_rate(baseline_trials_by_model[m]) for m in available]
        b_total = [g + e for g, e in zip(b_gradual, b_explicit)]

        # Prompted: gradual + explicit stacked
        p_gradual = [gradual_rate(prompted_trials[variant][m]) for m in available]
        p_explicit = [explicit_rate(prompted_trials[variant][m]) for m in available]
        p_total = [g + e for g, e in zip(p_gradual, p_explicit)]

        ax.bar(x - width/2, b_gradual, width, label="Baseline (gradual)", color="#C0C0C0")
        ax.bar(x - width/2, b_explicit, width, bottom=b_gradual, label="Baseline (explicit)", color="#808080")
        ax.bar(x + width/2, p_gradual, width, label="Prompted (gradual)", color="#5B9BD5")
        ax.bar(x + width/2, p_explicit, width, bottom=p_gradual, label="Prompted (explicit)", color="#ED7D31")

        for i, (b, p) in enumerate(zip(b_total, p_total)):
            ax.text(i - width/2, b + 0.5, f"{b:.1f}%", ha="center", va="bottom", fontsize=7)
            ax.text(i + width/2, p + 0.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel("Improvement rate (%)")
        ax.set_title(f'"{VARIANT_TITLES[variant]}"')
        ax.set_xticks(x)
        ax.set_xticklabels(available, rotation=15, ha="right", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ymax = max(max(b_total), max(p_total)) * 1.2
        ax.set_ylim(0, ymax)

    fig.suptitle(f"Exp 5: Baseline vs meta-prompted by model and prompt variant\n(threshold >= {THRESHOLD}/10)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("plot_exp5_whole_response.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot_exp5_whole_response.png")


if __name__ == "__main__":
    analyze_exp3()
    analyze_exp4()
    analyze_exp5()
