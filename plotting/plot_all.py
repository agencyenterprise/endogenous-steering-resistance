#!/usr/bin/env python3
"""
Generate all ESR experiment plots.

This is the Python replacement for the old `generate_all_plots.sh`.
It creates a timestamped output folder under plots/ and runs each plot script.

Usage:
    python plotting/plot_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repo root is importable (these scripts live in plotting/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from result_file_utils import ModelFamily
from plotting.plot_utils import collect_experiment_1_result_files

BASE_DIR = Path(__file__).parent.parent
PLOTTING_DIR = Path(__file__).parent


def _resolve_dir(p: Path) -> Path:
    return p if p.is_absolute() else (BASE_DIR / p)


def _run_plot(
    script_name: str,
    args: list[str],
    *,
    allow_fail: bool = False,
) -> None:
    script_path = PLOTTING_DIR / script_name
    cmd = [sys.executable, str(script_path), *args]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"  Warning: {script_name} failed (exit {e.returncode})")
        else:
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all ESR experiment plots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to save plots. If omitted, uses plots/run_<timestamp>/ "
            "(relative paths are resolved from the experiment base dir)."
        ),
    )
    parser.add_argument(
        "--exp3-model-identifier",
        type=str,
        default="Meta-Llama-3.3-70B-Instruct",
        help="Model identifier argument passed to plot_exp3.py (default: Meta-Llama-3.3-70B-Instruct).",
    )
    parser.add_argument(
        "--resistance-variant-id",
        type=str,
        default="self_monitor",
        help="Variant id passed to plot_exp5.py (default: self_monitor).",
    )
    parser.add_argument(
        "--all-judges",
        action="store_true",
        help="Use results from all judge folders (default: haiku only)",
    )
    parser.add_argument(
        "--exclude-degraded",
        action="store_true",
        help="Filter out degraded (repetitive) outputs instead of including them",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_DIR / "plots" / f"run_{run_ts}"
    else:
        output_dir = _resolve_dir(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print("==========================================")
    print("Generating all ESR experiment plots")
    print("==========================================")
    print(f"Saving plots to: {output_dir}")

    # Experiment 1
    print("\n[1/13] Generating Experiment 1 plots (ESR model comparison)...")
    exp1_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp1_args.append("--haiku-only")
    if args.exclude_degraded:
        exp1_args.append("--exclude-degraded")
    _run_plot("plot_exp1.py", exp1_args)

    # Experiment 2
    print("\n[2/13] Generating Experiment 2 plots (boost level sweep)...")
    exp2_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp2_args.append("--haiku-only")
    if args.exclude_degraded:
        exp2_args.append("--exclude-degraded")
    _run_plot("plot_exp2.py", exp2_args)

    # Experiment 3
    print("\n[3/13] Generating Experiment 3 plot (ablation study - combined 27-latent)...")
    exp3_args = [args.exp3_model_identifier, "--output-dir", str(output_dir)]
    if not args.all_judges:
        exp3_args.append("--haiku-only")
    if args.exclude_degraded:
        exp3_args.append("--exclude-degraded")
    _run_plot("plot_exp3.py", exp3_args, allow_fail=True)

    # Experiment 4
    print("\n[4/13] Generating Experiment 4 plots (fine-tuning ratio sweep)...")
    exp4_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp4_args.append("--haiku-only")
    _run_plot("plot_exp4.py", exp4_args, allow_fail=True)

    # Experiment 5
    print("\n[5/13] Generating Experiment 5 plots (prompt variants)...")
    exp1_baseline_files, _, _ = collect_experiment_1_result_files(
        BASE_DIR,
        excluded_families={ModelFamily.FINETUNED_8B},
        haiku_only=not args.all_judges,
    )
    exp5_args = [
        "--output-dir",
        str(output_dir),
        "--baseline-results",
        *[str(p) for p in exp1_baseline_files],
        "--resistance-variant-id",
        args.resistance_variant_id,
    ]
    if not args.all_judges:
        exp5_args.append("--haiku-only")
    if args.exclude_degraded:
        exp5_args.append("--exclude-degraded")
    _run_plot("plot_exp5.py", exp5_args, allow_fail=True)

    # Experiment 6
    print("\n[6/13] Generating Experiment 6 plots (sequential activations)...")
    exp6_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp6_args.append("--haiku-only")
    _run_plot("plot_exp6.py", exp6_args, allow_fail=True)

    # Experiment 7
    print("\n[7/13] Generating Experiment 7 plots (cross-judge analysis)...")
    exp7_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp7_args.append("--haiku-only")
    _run_plot("plot_exp7.py", exp7_args, allow_fail=True)

    # Experiment 8
    print("\n[8/13] Generating Experiment 8 plots (no-steering baseline)...")
    exp8_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp8_args.append("--haiku-only")
    if args.exclude_degraded:
        exp8_args.append("--exclude-degraded")
    _run_plot("plot_exp8.py", exp8_args, allow_fail=True)

    # Experiment 9
    print("\n[9/13] Generating Experiment 9 plots (activation statistics)...")
    exp9_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp9_args.append("--haiku-only")
    _run_plot("plot_exp9.py", exp9_args, allow_fail=True)

    # Experiment 10
    print("\n[10/13] Generating Experiment 10 plots (random latent ablation control)...")
    exp10_args = ["--output-dir", str(output_dir)]
    if not args.all_judges:
        exp10_args.append("--haiku-only")
    if args.exclude_degraded:
        exp10_args.append("--exclude-degraded")
    _run_plot("plot_exp10.py", exp10_args, allow_fail=True)

    # Combined prompting vs fine-tuning comparison
    print("\n[11/13] Generating combined prompting vs fine-tuning plot...")
    combined_args = [
        "--output-dir", str(output_dir),
        "--best-variant", args.resistance_variant_id,
    ]
    if not args.all_judges:
        combined_args.append("--haiku-only")
    _run_plot("plot_combined_prompting_finetuning.py", combined_args, allow_fail=True)

    # Extract paper numbers (JSON)
    print("\n[12/13] Extracting paper numbers to JSON...")
    extract_args = ["--output-dir", str(output_dir)]
    _run_plot("extract_paper_numbers.py", extract_args, allow_fail=True)

    # Generate LaTeX commands
    print("\n[13/13] Generating LaTeX commands from paper numbers...")
    latex_args = ["--output-dir", str(output_dir)]
    _run_plot("generate_latex_numbers.py", latex_args, allow_fail=True)

    print("\n==========================================")
    print(f"Done! Plots saved to: {output_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()


