#!/usr/bin/env python3
"""Generate a concise summary of whole-response judge results."""
# /// script
# dependencies = []
# ///

import json
from collections import defaultdict
from pathlib import Path

THRESHOLD = 7

MODEL_MAP = {
    'Meta-Llama-3.3-70B': 'Llama 3.3 70B',
    'Meta-Llama-3.1-8B': 'Llama 3.1 8B',
    'gemma-2-27b': 'Gemma 2 27B',
    'gemma-2-2b': 'Gemma 2 2B',
    'gemma-2-9b': 'Gemma 2 9B',
}
MODEL_ORDER = ['Llama 3.3 70B', 'Llama 3.1 8B', 'Gemma 2 27B', 'Gemma 2 9B', 'Gemma 2 2B']
VARIANTS = ['dont_get_distracted', 'ignore_latent_label', 'resist_manipulation', 'self_monitor']
VARIANT_NAMES = {
    'dont_get_distracted': "Don't get distracted",
    'ignore_latent_label': 'Ignore latent label',
    'resist_manipulation': 'Resist manipulation',
    'self_monitor': 'Self-monitor',
}


def load(path):
    with open(path) as f:
        return json.load(f)


def get_model(fpath):
    for key, name in MODEL_MAP.items():
        if key in fpath:
            return name
    return None


def valid(trials):
    return [t for t in trials if t.get('improvement_score') is not None]


def rate(trials, threshold=THRESHOLD):
    v = valid(trials)
    return sum(1 for t in v if t['improvement_score'] >= threshold) / len(v) * 100 if v else 0


def explicit(trials, threshold=THRESHOLD):
    v = valid(trials)
    return sum(1 for t in v if t['improvement_score'] >= threshold
               and t.get('improvement_type') == 'explicit_restart') / len(v) * 100 if v else 0


def gradual(trials, threshold=THRESHOLD):
    v = valid(trials)
    return sum(1 for t in v if t['improvement_score'] >= threshold
               and t.get('improvement_type') == 'gradual_shift') / len(v) * 100 if v else 0


def mean(trials):
    v = valid(trials)
    scores = [t['improvement_score'] for t in v]
    return sum(scores) / len(scores) if scores else 0


def orig_esr(trials):
    n = len(trials)
    esr = sum(1 for t in trials
              if len(t.get('original_score', {}).get('attempts', [])) >= 2
              and t['original_score']['attempts'][-1].get('score', 0) >
                  t['original_score']['attempts'][0].get('score', 0))
    return esr / n * 100 if n else 0


def main():
    lines = []
    def p(s=''):
        lines.append(s)
        print(s)

    p('# Whole-Response Judge: Complete Analysis')
    p()
    p('This analysis uses a holistic judge prompt that evaluates whether a')
    p('steered response improves over its course, without requiring explicit')
    p('self-correction phrases like "wait, that\'s not right". Responses are')
    p('scored 0-10; we use a threshold of >= 7/10 to classify a response as')
    p('"improving". The judge also classifies improvement as "explicit_restart"')
    p('or "gradual_shift".')
    p()
    p('Judge model: Gemini 3 Flash (via OpenRouter)')
    p()

    # =========================================================================
    # Experiment 1
    # =========================================================================
    exp1 = load('whole_response_judge_exp1_results.json')
    by_model = defaultdict(list)
    for fpath, trials in exp1.items():
        m = get_model(fpath)
        if m:
            by_model[m].extend(trials)

    p('## Experiment 1: ESR across model sizes')
    p()
    p(f'{"Model":<18} {"n":>6} {"Old judge":>10} {"New judge":>10} '
      f'{"Explicit":>9} {"Gradual":>9} {"Mean":>6}')
    p('-' * 70)
    for m in MODEL_ORDER:
        t = by_model[m]
        p(f'{m:<18} {len(valid(t)):>6} {orig_esr(t):>9.1f}% {rate(t):>9.1f}% '
          f'{explicit(t):>8.1f}% {gradual(t):>8.1f}% {mean(t):>6.2f}')
    p()
    p('Key finding: The whole-response judge detects substantially more')
    p('self-correction than the original explicit-phrase judge across all')
    p('models. Gemma models show the highest rates (20-27%), dominated by')
    p('gradual shifts. Llama 70B has the most explicit restarts.')
    p()

    # =========================================================================
    # Experiment 3
    # =========================================================================
    exp3 = load('whole_response_judge_exp3_results.json')
    ablation_trials = []
    for trials in exp3.values():
        ablation_trials.extend(trials)
    baseline_70b = by_model['Llama 3.3 70B']

    p('## Experiment 3: Off-topic detector ablation (Llama 70B)')
    p()
    p(f'  {"Condition":<25} {"n":>6} {"Old judge":>10} {"New judge":>10} {"Mean":>6}')
    p('  ' + '-' * 60)
    p(f'  {"Baseline (no ablation)":<25} {len(valid(baseline_70b)):>6} '
      f'{orig_esr(baseline_70b):>9.1f}% {rate(baseline_70b):>9.1f}% {mean(baseline_70b):>6.2f}')
    p(f'  {"With OTD ablation":<25} {len(valid(ablation_trials)):>6} '
      f'{orig_esr(ablation_trials):>9.1f}% {rate(ablation_trials):>9.1f}% {mean(ablation_trials):>6.2f}')
    p()
    p('Key finding: OTD ablation had a small effect under the original judge')
    p(f'(3.5% -> 2.8%) but shows no meaningful difference under the holistic')
    p(f'judge ({rate(baseline_70b):.1f}% vs {rate(ablation_trials):.1f}%). '
      f'This suggests OTDs specifically')
    p('drive explicit self-correction phrases, not broader recovery patterns.')
    p()

    # =========================================================================
    # Experiment 4
    # =========================================================================
    exp4 = load('whole_response_judge_exp4_results.json')
    baseline_8b = by_model['Llama 3.1 8B']

    p('## Experiment 4: Fine-tuning (Llama 8B)')
    p()
    p(f'  {"Mixing ratio":<15} {"n":>6} {"Rate":>8} {"Mean":>6}')
    p('  ' + '-' * 38)
    p(f'  {"Baseline":<15} {len(valid(baseline_8b)):>6} {rate(baseline_8b):>7.1f}% {mean(baseline_8b):>6.2f}')
    for fpath in sorted(exp4.keys()):
        ratio = fpath.split('masked-ratio-')[1].split('pct')[0]
        t = exp4[fpath]
        p(f'  {ratio + "% SC data":<15} {len(valid(t)):>6} {rate(t):>7.1f}% {mean(t):>6.2f}')
    p()
    p('Key finding: All fine-tuned models improve over the 9.7% baseline')
    p('(range 12-20%), but there is no clear dose-response relationship')
    p('between the proportion of self-correction training data and the')
    p('holistic improvement rate.')
    p()

    # =========================================================================
    # Experiment 5
    # =========================================================================
    p('## Experiment 5: Meta-prompting (all models)')
    p()

    # Load all prompted data
    prompted = {v: defaultdict(list) for v in VARIANTS}
    exp5 = load('whole_response_judge_exp5_results.json')
    for fpath, trials in exp5.items():
        for v in VARIANTS:
            if v in fpath and 'Meta-Llama-3.3-70B' in fpath:
                prompted[v]['Llama 3.3 70B'].extend(trials)
    for v in VARIANTS:
        path = Path(f'whole_response_judge_exp5_{v}_other_models.json')
        if path.exists():
            data = load(str(path))
            for fpath, trials in data.items():
                m = get_model(fpath)
                if m:
                    prompted[v][m].extend(trials)

    for v in VARIANTS:
        p(f'  ### "{VARIANT_NAMES[v]}"')
        p(f'  {"Model":<18} {"Baseline":>9} {"Prompted":>9} {"Explicit":>9} {"Gradual":>9}')
        p('  ' + '-' * 58)
        for m in MODEL_ORDER:
            b = by_model[m]
            pt = prompted[v].get(m, [])
            if pt:
                p(f'  {m:<18} {rate(b):>8.1f}% {rate(pt):>8.1f}% '
                  f'{explicit(pt):>8.1f}% {gradual(pt):>8.1f}%')
        p()

    p('Key finding: Meta-prompting consistently boosts Llama 70B (15.8% ->')
    p('20-24% depending on variant), with "resist manipulation" and')
    p('"self-monitor" being the most effective. Other models show little')
    p('to no effect. The Gemma models already have high baseline improvement')
    p('rates and are not further boosted by meta-prompting. Explicit')
    p('restarts are concentrated in Llama 70B and increase with prompting.')
    p()

    # =========================================================================
    # Overall summary
    # =========================================================================
    p('## Overall summary')
    p()
    p('1. The whole-response judge detects 4-50x more self-correction than')
    p('   the original explicit-phrase judge, primarily via "gradual shift"')
    p('   patterns that the original judge missed entirely.')
    p()
    p('2. The model ranking changes: Gemma models show the highest holistic')
    p('   improvement rates (20-27%), while the original judge found almost')
    p('   zero ESR for Gemma. Llama 70B drops from being the clear leader')
    p('   to the middle of the pack.')
    p()
    p('3. OTD ablation has no effect under the holistic judge, suggesting')
    p('   off-topic detectors drive explicit self-correction specifically.')
    p()
    p('4. Fine-tuning helps (9.7% -> 12-20%) but the effect plateaus;')
    p('   more self-correction training data doesn\'t clearly help more.')
    p()
    p('5. Meta-prompting is Llama-70B-specific: it boosts that model by')
    p('   ~50% relative, but leaves smaller/different-architecture models')
    p('   largely unchanged.')

    out = Path('whole_response_judge_analysis.md')
    out.write_text('\n'.join(lines) + '\n')
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
