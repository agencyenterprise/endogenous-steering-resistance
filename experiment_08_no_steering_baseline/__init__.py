#!/usr/bin/env python3
"""
Experiment 8: No-Steering Baseline

Runs experiment 1 with feature steering disabled to measure baseline ESR rates.
This is a wrapper that calls experiment_01_esr with --no-steering flag.

Results are saved with suffix "_no_steering_baseline.json" and can be plotted
with plotting/plot_exp8.py.

Usage:
    # Run for a specific model using features from an existing results file
    python experiment_8_no_steering_baseline.py 70b --from-results data/experiment_results/claude_haiku_4_5_20251001_judge/experiment_results_Meta-Llama-3.3-70B-Instruct_20260127_184422.json

    # Run with fresh features (not recommended - use --from-results for comparability)
    python experiment_8_no_steering_baseline.py 70b --n-features 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 8: No-steering baseline (wrapper for experiment 1)"
    )
    parser.add_argument(
        "config",
        help="Config name (e.g., '70b', '8b', 'gemma-2-2b'). See experiment_1_esr.py for options."
    )
    parser.add_argument(
        "--from-results",
        type=str,
        required=True,
        help="Path to experiment 1 results file to reuse features/prompts from. "
             "Required for meaningful comparison with steered results."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials per feature"
    )
    parser.add_argument(
        "--fresh-prompts",
        action="store_true",
        help="Sample fresh prompts instead of reusing from --from-results"
    )
    parser.add_argument(
        "--judge",
        "-j",
        type=str,
        default="haiku",
        help="Judge model to use (default: haiku)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 8: No-Steering Baseline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"From results: {args.from_results}")
    print(f"Judge: {args.judge}")
    print("=" * 60)

    # Build command - run experiment_01_esr as a module
    cmd = [
        sys.executable,
        "-m", "experiment_01_esr",
        args.config,
        "--no-steering",
        "--from-results", args.from_results,
        "--output-suffix", "no_steering_baseline",
        "--judge", args.judge,
    ]

    if args.n_trials is not None:
        cmd.extend(["--n-trials", str(args.n_trials)])

    if args.fresh_prompts:
        cmd.append("--fresh-prompts")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run experiment 1 with no-steering flag
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
