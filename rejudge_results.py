#!/usr/bin/env python3
"""
Re-judge experiment results with a different judge model.

Usage:
    python rejudge_results.py <input_file> --judge gemini-2.5-flash
    python rejudge_results.py <input_file> --judge claude-sonnet-4-5-20250929
    python rejudge_results.py <input_file> --judge openai/gpt-5-mini
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from judge import create_judge, get_judge_folder_name


# Short aliases for common models
MODEL_ALIASES = {
    "gemini-flash": "gemini-2.5-flash-preview-05-20",
    "gemini-pro": "gemini-2.5-pro-preview-05-06",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}


def resolve_model_id(model_spec: str) -> str:
    """Resolve a model spec (alias or full ID) to a full model ID."""
    return MODEL_ALIASES.get(model_spec, model_spec)


async def rejudge_file(
    input_path: Path,
    output_path: Path,
    model_id: str,
    max_concurrent: int = 50,
) -> dict:
    """Re-judge all trials in a results file with a new judge model."""

    with open(input_path) as f:
        data = json.load(f)

    judge = create_judge(model_id, max_concurrent=max_concurrent)

    # Track stats
    total_trials = 0
    errors = 0

    for feature in data.get("results_by_feature", []):
        if feature.get("error"):
            continue

        trials_to_judge = []
        for trial in feature.get("trials", []):
            if trial.get("error") or not trial.get("response"):
                continue
            trials_to_judge.append(trial)

        if not trials_to_judge:
            continue

        # Create tasks for this feature's trials
        tasks = [
            judge.grade_response(
                response=trial["response"],
                prompt=trial["prompt"],
                feature_label=feature.get("feature_label", ""),
            )
            for trial in trials_to_judge
        ]

        # Run with progress bar
        label = feature.get("feature_label", "unknown")[:35]
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"{label}",
        )

        # Store results with a key based on the model ID
        score_key = f"score_{get_judge_folder_name(model_id)}"
        for trial, new_score in zip(trials_to_judge, results):
            trial[score_key] = new_score
            total_trials += 1
            if new_score.get("error"):
                errors += 1

    # Add metadata about rejudging
    if "rejudge_history" not in data:
        data["rejudge_history"] = []
    data["rejudge_history"].append({
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "total_trials": total_trials,
        "errors": errors,
    })

    # Save output
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "total_trials": total_trials,
        "errors": errors,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-judge experiment results with a different model",
        epilog=f"Model aliases: {', '.join(f'{k}={v}' for k, v in MODEL_ALIASES.items())}",
    )
    parser.add_argument("input_file", type=Path, help="Input results JSON file")
    parser.add_argument(
        "--judge", "-j",
        type=str,
        required=True,
        help="Judge model: alias (gemini-flash, sonnet), full name (gemini-2.5-flash), or OpenRouter path (openai/gpt-5-mini)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: adds _rejudged_{judge} suffix)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent API requests (default: 50)",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Resolve model ID
    model_id = resolve_model_id(args.judge)

    # Generate output path if not specified
    if args.output is None:
        stem = args.input_file.stem
        suffix = args.input_file.suffix
        judge_slug = get_judge_folder_name(model_id)
        args.output = args.input_file.parent / f"{stem}_rejudged_{judge_slug}{suffix}"

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output}")
    print(f"Judge:  {model_id}")
    print()

    result = asyncio.run(rejudge_file(
        input_path=args.input_file,
        output_path=args.output,
        model_id=model_id,
        max_concurrent=args.max_concurrent,
    ))

    print()
    print(f"Done! Judged {result['total_trials']} trials ({result['errors']} errors)")
    print(f"Saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
