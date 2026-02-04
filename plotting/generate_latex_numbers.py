#!/usr/bin/env python3
"""
Generate LaTeX command definitions from paper_numbers_computed.json.

This creates a .tex file with \\newcommand definitions that can be \\input
in the main LaTeX document, allowing automatic number updates.

Usage:
    python plotting/generate_latex_numbers.py [--output-dir plots/]

Output:
    paper_numbers.tex in the specified output directory

In LaTeX, use like:
    \\input{images/paper_numbers}
    The ESR rate for Llama 70B is \\esrRateLlamaSeventy\\%.
"""

import argparse
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def sanitize_command_name(name: str) -> str:
    """Convert a name to a valid LaTeX command name (letters only)."""
    # Replace common patterns
    replacements = {
        "llama_3.3_70b": "LlamaSeventy",
        "llama_3.1_8b": "LlamaEight",
        "llama 3.3 70b": "LlamaSeventy",
        "llama 3.1 8b": "LlamaEight",
        "gemma_2_27b": "GemmaTwentySeven",
        "gemma_2_9b": "GemmaNine",
        "gemma_2_2b": "GemmaTwo",
        "gemma 2 27b": "GemmaTwentySeven",
        "gemma 2 9b": "GemmaNine",
        "gemma 2 2b": "GemmaTwo",
        "70b": "Seventy",
        "27b": "TwentySeven",
        "8b": "Eight",
        "9b": "Nine",
        "2b": "Two",
        "_pct": "Pct",
        "_se": "SE",
        "self_monitor": "SelfMonitor",
        "multi_attempt": "MultiAttempt",
        "esr_rate": "EsrRate",
        "n_trials": "NTrials",
        "n_episodes": "NEpisodes",
        "mean_first_score": "MeanFirstScore",
        "improvement_rate": "ImprovementRate",
        "off_topic": "OffTopic",
        "on_topic": "OnTopic",
        "baseline": "Baseline",
        "ablation": "Ablation",
        "random": "Random",
        "reduction": "Reduction",
    }

    result = name.lower()
    for old, new in replacements.items():
        result = result.replace(old.lower(), new)

    # Remove non-letters and capitalize each word
    words = re.split(r'[^a-zA-Z]+', result)
    result = ''.join(word.capitalize() for word in words if word)

    # Ensure it starts with lowercase for LaTeX convention
    if result:
        result = result[0].lower() + result[1:]

    return result


def format_number(value: float | int, decimals: int = 1) -> str:
    """Format a number for LaTeX output."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", "{,}")
    elif abs(value) < 0.01 and value != 0:
        return f"{value:.4f}"
    elif decimals == 0:
        return f"{int(round(value)):,}".replace(",", "{,}")
    else:
        return f"{value:.{decimals}f}"


def generate_commands(data: dict, prefix: str = "") -> list[tuple[str, str, str]]:
    """
    Recursively generate LaTeX commands from nested dict.

    Returns list of (command_name, value, comment).
    """
    commands = []

    for key, value in data.items():
        cmd_name = sanitize_command_name(prefix + key)

        if isinstance(value, dict):
            # Recurse into nested dicts
            commands.extend(generate_commands(value, prefix + key + "_"))
        elif isinstance(value, list):
            # For lists, just store the count
            commands.append((cmd_name + "Count", str(len(value)), f"Count of {key}"))
        elif isinstance(value, (int, float)):
            # Determine appropriate formatting
            if "pct" in key.lower() or "rate" in key.lower():
                formatted = format_number(value, 1)
            elif "n_" in key.lower() or key.lower().startswith("n"):
                formatted = format_number(value, 0)
            elif "_se" in key.lower():
                formatted = format_number(value, 2)
            else:
                formatted = format_number(value, 1)

            commands.append((cmd_name, formatted, f"{prefix}{key}"))
        elif isinstance(value, str):
            commands.append((cmd_name, value, f"{prefix}{key}"))

    return commands


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX commands from paper numbers JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save paper_numbers.tex (relative paths resolved from experiment base dir). Default: plots/",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Path to paper_numbers_computed.json. Default: <output-dir>/paper_numbers_computed.json",
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input JSON path
    if args.input_json:
        input_json = args.input_json if args.input_json.is_absolute() else (BASE_DIR / args.input_json)
    else:
        input_json = output_dir / "paper_numbers_computed.json"

    if not input_json.exists():
        print(f"Error: Input JSON not found: {input_json}")
        print("Run extract_paper_numbers.py first to generate it.")
        return

    # Load the computed numbers
    with open(input_json) as f:
        data = json.load(f)

    # Generate commands
    all_commands = []

    # Add header comment
    header = f"""%% Auto-generated paper numbers - DO NOT EDIT MANUALLY
%% Generated from: {input_json.name}
%% Regenerate with: python plotting/generate_latex_numbers.py
%%
%% Usage: \\input{{images/paper_numbers}}
%%        The ESR rate is \\esrRateLlamaSeventy\\%.

"""

    # Process each section with clear prefixes
    sections = [
        ("", "experiment_1_esr_across_models", "Experiment 1: ESR Across Models"),
        ("expTwo", "experiment_2_boost_sweep", "Experiment 2: Boost Sweep"),
        ("expThree", "experiment_3_otd_ablation", "Experiment 3: OTD Ablation"),
        ("expFour", "experiment_4_finetuning", "Experiment 4: Fine-tuning"),
        ("expFive", "experiment_5_meta_prompting", "Experiment 5: Meta-Prompting"),
        ("expEight", "experiment_8_no_steering_baseline", "Experiment 8: No-Steering Baseline"),
        ("expNine", "experiment_9_activation_stats", "Experiment 9: Activation Statistics"),
        ("expTen", "experiment_10_random_ablation", "Experiment 10: Random Ablation"),
    ]

    # Key numbers section with manually curated commands
    key_numbers = f"""
%% ====================================================================
%% KEY PAPER NUMBERS (most commonly used)
%% ====================================================================

%% Models
\\newcommand{{\\numModels}}{{5}}
\\newcommand{{\\numPrompts}}{{{data['prompts']['n_prompts']}}}

%% Llama 70B primary results
\\newcommand{{\\llamaSeventyEsrRate}}{{{format_number(data['experiment_1_esr_across_models']['Llama 3.3 70B']['esr_rate'], 1)}}}
\\newcommand{{\\llamaSeventyMultiAttemptPct}}{{{format_number(data['experiment_1_esr_across_models']['Llama 3.3 70B']['multi_attempt_pct'], 1)}}}
\\newcommand{{\\llamaSeventyNTrials}}{{{format_number(data['experiment_1_esr_across_models']['Llama 3.3 70B']['n_trials'], 0)}}}

%% OTD Ablation key results
\\newcommand{{\\numOtdLatents}}{{{data['experiment_3_otd_ablation']['n_otd_latents']}}}
\\newcommand{{\\multiAttemptReductionPct}}{{{format_number(data['experiment_3_otd_ablation']['multi_attempt_reduction_pct'], 0)}}}
\\newcommand{{\\esrReductionPct}}{{{format_number(data['experiment_3_otd_ablation']['esr_reduction_pct'], 0)}}}
\\newcommand{{\\ablationBaselineNTrials}}{{{format_number(data['experiment_3_otd_ablation']['baseline']['n_trials'], 0)}}}
\\newcommand{{\\ablationNTrials}}{{{format_number(data['experiment_3_otd_ablation']['ablation']['n_trials'], 0)}}}
\\newcommand{{\\ablationMultiAttemptBefore}}{{{format_number(data['experiment_3_otd_ablation']['baseline']['multi_attempt_pct'], 1)}}}
\\newcommand{{\\ablationMultiAttemptAfter}}{{{format_number(data['experiment_3_otd_ablation']['ablation']['multi_attempt_pct'], 1)}}}
\\newcommand{{\\ablationEsrBefore}}{{{format_number(data['experiment_3_otd_ablation']['baseline']['esr_rate'], 1)}}}
\\newcommand{{\\ablationEsrAfter}}{{{format_number(data['experiment_3_otd_ablation']['ablation']['esr_rate'], 1)}}}
\\newcommand{{\\ablationFirstScoreBefore}}{{{format_number(data['experiment_3_otd_ablation']['baseline']['mean_first_score'], 1)}}}
\\newcommand{{\\ablationFirstScoreAfter}}{{{format_number(data['experiment_3_otd_ablation']['ablation']['mean_first_score'], 1)}}}

%% Meta-prompting key results
\\newcommand{{\\metaPromptingMultiIncrease}}{{{format_number(data['experiment_5_meta_prompting']['llama_70b_multi_increase_factor'], 1)}}}
\\newcommand{{\\metaPromptingEsrIncrease}}{{{format_number(data['experiment_5_meta_prompting']['llama_70b_esr_increase_factor'], 1)}}}
\\newcommand{{\\metaPromptingMultiBefore}}{{{format_number(data['experiment_5_meta_prompting']['llama_70b_baseline_multi_pct'], 1)}}}
\\newcommand{{\\metaPromptingMultiAfter}}{{{format_number(data['experiment_5_meta_prompting']['llama_70b_self_monitor_multi_pct'], 1)}}}
\\newcommand{{\\metaPromptingEsrBefore}}{{{format_number(data['experiment_5_meta_prompting']['llama_70b_baseline_esr'], 1)}}}
\\newcommand{{\\metaPromptingEsrAfter}}{{{format_number(data['experiment_5_meta_prompting']['llama_70b_self_monitor_esr'], 1)}}}

%% No-steering baseline
\\newcommand{{\\noSteeringNTrials}}{{{format_number(data['experiment_8_no_steering_baseline']['n_trials'], 0)}}}
\\newcommand{{\\noSteeringMultiAttemptPct}}{{{format_number(data['experiment_8_no_steering_baseline']['multi_attempt_pct'], 2)}}}
\\newcommand{{\\noSteeringMeanScore}}{{{format_number(data['experiment_8_no_steering_baseline']['mean_first_score'], 1)}}}

%% Activation statistics
\\newcommand{{\\numSelfCorrectionEpisodes}}{{{data['experiment_9_activation_stats']['n_self_correction_episodes']}}}
\\newcommand{{\\numBaselineEpisodes}}{{{data['experiment_9_activation_stats']['n_baseline_episodes']}}}
\\newcommand{{\\otdRatioOffTopic}}{{{format_number(data['experiment_9_activation_stats']['otd_ratio_off_topic_vs_baseline'], 1)}}}
\\newcommand{{\\otdRatioOnTopic}}{{{format_number(data['experiment_9_activation_stats']['otd_ratio_on_topic_vs_baseline'], 1)}}}
\\newcommand{{\\otdMeanOffTopic}}{{{format_number(data['experiment_9_activation_stats']['otd_off_topic_mean'], 4)}}}
\\newcommand{{\\otdMeanOnTopic}}{{{format_number(data['experiment_9_activation_stats']['otd_on_topic_mean'], 4)}}}
\\newcommand{{\\otdMeanBaseline}}{{{format_number(data['experiment_9_activation_stats']['baseline_otd_mean'], 4)}}}

%% Random ablation control
\\newcommand{{\\randomAblationNTrials}}{{{format_number(data['experiment_10_random_ablation']['random_ablation']['n_trials'], 0)}}}
\\newcommand{{\\randomAblationMultiAttemptPct}}{{{format_number(data['experiment_10_random_ablation']['random_ablation']['multi_attempt_pct'], 1)}}}
\\newcommand{{\\randomAblationEsrRate}}{{{format_number(data['experiment_10_random_ablation']['random_ablation']['esr_rate'], 1)}}}
\\newcommand{{\\randomAblationFirstScore}}{{{format_number(data['experiment_10_random_ablation']['random_ablation']['mean_first_score'], 1)}}}

"""

    # Sample sizes for each model
    model_samples = """
%% ====================================================================
%% SAMPLE SIZES BY MODEL
%% ====================================================================
"""
    for model_name, model_data in data['experiment_1_esr_across_models'].items():
        cmd_prefix = sanitize_command_name(model_name)
        model_samples += f"\\newcommand{{\\{cmd_prefix}NTrials}}{{{format_number(model_data['n_trials'], 0)}}}\n"
        model_samples += f"\\newcommand{{\\{cmd_prefix}EsrRate}}{{{format_number(model_data['esr_rate'], 1)}}}\n"
        model_samples += f"\\newcommand{{\\{cmd_prefix}MultiAttemptPct}}{{{format_number(model_data['multi_attempt_pct'], 1)}}}\n"
        model_samples += f"\\newcommand{{\\{cmd_prefix}NMultiAttempt}}{{{format_number(model_data['n_multi_attempt'], 0)}}}\n"
        model_samples += f"\\newcommand{{\\{cmd_prefix}MeanFirstScore}}{{{format_number(model_data['mean_first_score'], 1)}}}\n"
        model_samples += "\n"

    # Fine-tuning ratios
    finetuning = """
%% ====================================================================
%% FINE-TUNING EXPERIMENT (by mixing ratio)
%% ====================================================================
"""
    for ratio_key, ratio_data in data['experiment_4_finetuning'].items():
        if ratio_key == "base":
            continue
        ratio_num = ratio_key.replace("pct", "")
        finetuning += f"\\newcommand{{\\finetuning{ratio_num}NTrials}}{{{format_number(ratio_data['n_trials'], 0)}}}\n"
        finetuning += f"\\newcommand{{\\finetuning{ratio_num}MultiAttemptPct}}{{{format_number(ratio_data['multi_attempt_pct'], 1)}}}\n"
        finetuning += f"\\newcommand{{\\finetuning{ratio_num}ImprovementRate}}{{{format_number(ratio_data['improvement_rate'], 1)}}}\n"

    # Write output
    output_content = header + key_numbers + model_samples + finetuning

    output_tex = output_dir / "paper_numbers.tex"
    with open(output_tex, 'w') as f:
        f.write(output_content)

    print(f"Generated: {output_tex}")
    print(f"Commands defined: {output_content.count('newcommand')}")


if __name__ == "__main__":
    main()
