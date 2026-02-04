#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "tqdm",
#   "python-dotenv",
# ]
# ///
"""
Experiment 7: Cross-Judge Analysis

Regrades experiment 1 results with multiple judge models to validate ESR findings.
This is a wrapper around regrade_cross_judge.py that:
1. Excludes Haiku 4.5 from judges (since it was the original judge)
2. Saves results to the haiku judge folder

Usage:
    python run_cross_judge.py --n-samples 1000
    python run_cross_judge.py --dry-run --n-samples 500
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports (needed for judge module)
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_07_cross_judge.regrade_cross_judge import main as run_regrade, JUDGE_MODELS

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Haiku results directory
HAIKU_RESULTS_DIR = BASE_DIR / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge"

# Judges to use (excluding Haiku 4.5 since it was the original judge)
CROSS_JUDGE_MODELS = [k for k in JUDGE_MODELS.keys() if "haiku" not in k.lower()]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 7: Cross-judge analysis of ESR results"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to regrade (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show what would be done without making API calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent requests per model",
    )
    parser.add_argument(
        "--min-per-model",
        type=int,
        default=100,
        help="Minimum samples per target model (default: 100)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 7: Cross-Judge Analysis")
    print("=" * 60)
    print(f"Results directory: {HAIKU_RESULTS_DIR}")
    print(f"Output directory: {HAIKU_RESULTS_DIR / 'cross_judge_results'}")
    print(f"Judges to use: {CROSS_JUDGE_MODELS}")
    print(f"N samples: {args.n_samples}")
    print("=" * 60)

    # Build argv for regrade_cross_judge
    sys_argv_backup = sys.argv
    sys.argv = [
        "regrade_cross_judge.py",
        "--n-samples", str(args.n_samples),
        "--models", *CROSS_JUDGE_MODELS,
        "--output-dir", str(HAIKU_RESULTS_DIR / "cross_judge_results"),
        "--experiment-results-dir", str(HAIKU_RESULTS_DIR),
        "--seed", str(args.seed),
        "--max-concurrent", str(args.max_concurrent),
        "--min-per-model", str(args.min_per_model),
    ]
    if args.dry_run:
        sys.argv.append("--dry-run")

    try:
        import asyncio
        asyncio.run(run_regrade())
    finally:
        sys.argv = sys_argv_backup


if __name__ == "__main__":
    main()
