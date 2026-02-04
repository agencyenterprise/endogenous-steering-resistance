#!/usr/bin/env python3
"""
Experiment 9: Self-Correction Activation Statistics

Runner script that orchestrates the activation statistics pipeline.

Usage:
    python run_activation_stats.py <command> [options]

Commands:
    find-backtracking     Find backtracking latents via keyword search
    extract-episodes      Extract self-correction episodes from exp1 results
    annotate-boundaries   Annotate episode boundaries with Claude
    collect-activations   Collect per-token SAE activations (requires GPU)
    collect-baseline      Collect baseline activations (requires GPU)
    analyze               Analyze activation statistics
    generate-output       Generate tables and plots
    all                   Run full pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
SCRIPT_DIR = Path(__file__).parent
HAIKU_RESULTS_DIR = BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge"
OUTPUT_DIR = HAIKU_RESULTS_DIR / "activation_stats"


def run_script(script_name: str, extra_args: list[str] = None):
    """Run a script from the local directory."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 9: Self-Correction Activation Statistics"
    )
    parser.add_argument(
        "command",
        choices=[
            "find-backtracking",
            "extract-episodes",
            "annotate-boundaries",
            "collect-activations",
            "collect-baseline",
            "analyze",
            "generate-output",
            "all",
        ],
        help="Command to run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)"
    )
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 9: Self-Correction Activation Statistics")
    print("=" * 60)
    print(f"Scripts directory: {SCRIPT_DIR}")
    print(f"Results directory: {HAIKU_RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    limit_args = ["--limit", str(args.limit)] if args.limit else []

    commands = {
        "find-backtracking": ("find_backtracking_latents.py", []),
        "extract-episodes": ("extract_episodes.py", []),
        "annotate-boundaries": ("annotate_boundaries.py", limit_args),
        "collect-activations": ("collect_activations.py", limit_args),
        "collect-baseline": ("collect_baseline_activations.py", limit_args),
        "analyze": ("analyze_activations.py", []),
        "generate-output": None,  # Special: runs multiple scripts
    }

    if args.command == "all":
        # Run full pipeline
        for cmd_name, (script, script_args) in commands.items():
            if cmd_name == "generate-output":
                run_script("generate_tables.py", [])
                # Plots are generated via plotting/plot_exp9.py (or plot_all.py)
            else:
                run_script(script, script_args)
    elif args.command == "generate-output":
        run_script("generate_tables.py", [])
        # Plots are generated via plotting/plot_exp9.py (or plot_all.py)
    else:
        script, script_args = commands[args.command]
        run_script(script, script_args)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
