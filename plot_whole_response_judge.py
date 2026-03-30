#!/usr/bin/env python3
"""Visualizations for whole-response judge results."""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_MAP = {
    'Meta-Llama-3.3-70B': 'Llama 3.3 70B',
    'Meta-Llama-3.1-8B': 'Llama 3.1 8B',
    'gemma-2-27b': 'Gemma 2 27B',
    'gemma-2-2b': 'Gemma 2 2B',
    'gemma-2-9b': 'Gemma 2 9B',
}

MODEL_ORDER = ['Llama 3.3 70B', 'Llama 3.1 8B', 'Gemma 2 27B', 'Gemma 2 9B', 'Gemma 2 2B']

THRESHOLD = 7


def load_data():
    with open('whole_response_judge_exp1_results.json') as f:
        data = json.load(f)

    by_model = defaultdict(list)
    for fpath, trials in data.items():
        for key, name in MODEL_MAP.items():
            if key in fpath:
                by_model[name].extend(trials)
                break
    return by_model


def plot_improvement_rates(by_model):
    """Bar chart: improvement rate by model, split by explicit vs gradual."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = MODEL_ORDER
    explicit_rates = []
    gradual_rates = []

    for model in models:
        trials = by_model[model]
        valid = [t for t in trials if t.get('improvement_score') is not None]
        n = len(valid)
        explicit = sum(1 for t in valid if t['improvement_score'] >= THRESHOLD
                       and t.get('improvement_type') == 'explicit_restart')
        gradual = sum(1 for t in valid if t['improvement_score'] >= THRESHOLD
                      and t.get('improvement_type') == 'gradual_shift')
        explicit_rates.append(explicit / n * 100 if n else 0)
        gradual_rates.append(gradual / n * 100 if n else 0)

    x = np.arange(len(models))
    width = 0.6

    bars_gradual = ax.bar(x, gradual_rates, width, label='Gradual shift', color='#5B9BD5')
    bars_explicit = ax.bar(x, explicit_rates, width, bottom=gradual_rates,
                           label='Explicit restart', color='#ED7D31')

    # Add total rate labels on top
    for i, (g, e) in enumerate(zip(gradual_rates, explicit_rates)):
        total = g + e
        ax.text(i, total + 0.8, f'{total:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Responses showing improvement (%)')
    ax.set_title(f'Response improvement rate by model\n(whole-response judge, threshold >= {THRESHOLD}/10)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, max(g + e for g, e in zip(gradual_rates, explicit_rates)) * 1.15)

    plt.tight_layout()
    plt.savefig('plot_whole_response_improvement_rates.png', dpi=150)
    print('Saved plot_whole_response_improvement_rates.png')


def plot_score_distributions(by_model):
    """Score distribution histograms per model."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

    for ax, model in zip(axes, MODEL_ORDER):
        trials = by_model[model]
        scores = [t['improvement_score'] for t in trials if t.get('improvement_score') is not None]
        counts = [scores.count(s) for s in range(11)]
        total = len(scores)
        pcts = [c / total * 100 for c in counts]

        colors = ['#C0392B' if s < THRESHOLD else '#27AE60' for s in range(11)]
        ax.bar(range(11), pcts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Improvement score')
        ax.set_title(model, fontsize=11)
        ax.set_xticks(range(0, 11, 2))
        ax.axvline(x=THRESHOLD - 0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)

    axes[0].set_ylabel('% of responses')
    fig.suptitle('Distribution of improvement scores by model', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('plot_whole_response_score_distributions.png', dpi=150, bbox_inches='tight')
    print('Saved plot_whole_response_score_distributions.png')


def plot_comparison_with_original(by_model):
    """Compare original explicit-phrase judge with whole-response judge."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = MODEL_ORDER
    original_rates = []
    new_rates = []

    for model in models:
        trials = by_model[model]
        n = len(trials)

        # Original judge: multi-attempt with improving scores
        orig_esr = sum(1 for t in trials
                       if len(t.get('original_score', {}).get('attempts', [])) >= 2
                       and t['original_score']['attempts'][-1].get('score', 0) >
                           t['original_score']['attempts'][0].get('score', 0))
        original_rates.append(orig_esr / n * 100 if n else 0)

        # New judge: improvement_score >= threshold
        valid = [t for t in trials if t.get('improvement_score') is not None]
        improving = sum(1 for t in valid if t['improvement_score'] >= THRESHOLD)
        new_rates.append(improving / len(valid) * 100 if valid else 0)

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, original_rates, width, label='Original judge\n(explicit self-correction phrases)',
           color='#95A5A6')
    ax.bar(x + width/2, new_rates, width, label='Whole-response judge\n(holistic improvement)',
           color='#3498DB')

    for i, (o, n) in enumerate(zip(original_rates, new_rates)):
        ax.text(i - width/2, o + 0.3, f'{o:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, n + 0.3, f'{n:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Responses showing self-correction (%)')
    ax.set_title('Original judge vs. whole-response judge:\nself-correction detection rate by model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(max(original_rates), max(new_rates)) * 1.25)

    plt.tight_layout()
    plt.savefig('plot_whole_response_vs_original.png', dpi=150)
    print('Saved plot_whole_response_vs_original.png')


def plot_mean_scores(by_model):
    """Bar chart of mean improvement score by model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    means = []
    for model in MODEL_ORDER:
        scores = [t['improvement_score'] for t in by_model[model]
                  if t.get('improvement_score') is not None]
        means.append(sum(scores) / len(scores))

    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(x, means, 0.6, color='#3498DB')
    for i, m in enumerate(means):
        ax.text(i, m + 0.05, f'{m:.2f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Mean improvement score (0-10)')
    ax.set_title('Mean improvement score by model\n(whole-response judge)')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha='right')
    ax.set_ylim(0, max(means) * 1.25)

    plt.tight_layout()
    plt.savefig('plot_whole_response_mean_scores.png', dpi=150)
    print('Saved plot_whole_response_mean_scores.png')


if __name__ == '__main__':
    by_model = load_data()
    plot_improvement_rates(by_model)
    plot_score_distributions(by_model)
    plot_comparison_with_original(by_model)
    plot_mean_scores(by_model)
