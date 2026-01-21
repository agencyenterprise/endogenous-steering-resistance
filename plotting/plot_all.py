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
    print("\n[1/8] Generating Experiment 1 plots (ESR model comparison)...")
    _run_plot("plot_exp1.py", ["--output-dir", str(output_dir)])

    # Experiment 2
    print("\n[2/8] Generating Experiment 2 plots (boost level sweep)...")
    _run_plot("plot_exp2.py", ["--output-dir", str(output_dir)])

    # Experiment 3
    print("\n[3/8] Generating Experiment 3 plot (ablation study - combined 27-latent)...")
    _run_plot(
        "plot_exp3.py",
        [args.exp3_model_identifier, "--output-dir", str(output_dir)],
        allow_fail=True,
    )

    # Experiment 4
    print("\n[4/8] Generating Experiment 4 plots (fine-tuning ratio sweep)...")
    _run_plot("plot_exp4.py", ["--output-dir", str(output_dir)], allow_fail=True)

    # Experiment 5
    print("\n[5/8] Generating Experiment 5 plots (prompt variants)...")
    exp1_baseline_files, _, _ = collect_experiment_1_result_files(
        BASE_DIR,
        excluded_families={ModelFamily.FINETUNED_8B},
    )
    _run_plot(
        "plot_exp5.py",
        [
            "--output-dir",
            str(output_dir),
            "--baseline-results",
            *[str(p) for p in exp1_baseline_files],
            "--resistance-variant-id",
            args.resistance_variant_id,
        ],
        allow_fail=True,
    )

    # Experiment 6
    print("\n[6/8] Generating Experiment 6 plots (sequential activations)...")
    _run_plot("plot_exp6.py", ["--output-dir", str(output_dir)], allow_fail=True)

    # Experiment 7
    print("\n[7/8] Generating Experiment 7 plots (cross-judge analysis)...")
    _run_plot("plot_exp7.py", ["--output-dir", str(output_dir)], allow_fail=True)

    # Experiment 8
    print("\n[8/8] Generating Experiment 8 plots (baseline results)...")
    _run_plot("plot_exp8.py", ["--output-dir", str(output_dir)], allow_fail=True)

    print("\n==========================================")
    print(f"Done! Plots saved to: {output_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()


