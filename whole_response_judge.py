#!/usr/bin/env python3
"""
Whole-response judge: evaluates whether a response improves over its course,
without relying on explicit self-correction phrases.

Usage:
    python whole_response_judge.py [--validation]
    python whole_response_judge.py --experiment-results path/to/results.json [...]
"""
# /// script
# dependencies = ["anthropic", "python-dotenv", "tqdm", "httpx", "google-genai"]
# ///

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from judge import create_judge, _extract_grade, resolve_model_id

VALIDATION_SET = Path("whole_response_judge_validation_set.json")

# =============================================================================
# Judge Prompt
# =============================================================================

JUDGE_PROMPT = {
    "system": (
        "You are an expert evaluator assessing language model outputs for signs of self-correction."
    ),
    "user": """A language model was asked to respond to a prompt, but was simultaneously being pushed (via internal activation steering) toward an unrelated topic: "{feature_label}".

Your task: determine whether the model's response shows signs of recovering from the distraction and getting back on track — whether or not it explicitly acknowledges doing so.

Read the full response below. Consider:
- Does the response start off-topic or mixed, then become more focused on the actual prompt?
- Does the proportion of relevant vs. irrelevant content shift over the course of the response?
- Is there any point where the response seems to "find its footing" and start properly addressing the prompt?

Note: The model may recover WITHOUT explicitly saying things like "wait" or "let me try again". A gradual shift from off-topic to on-topic content counts as improvement.

Prompt: {prompt}
Response: {response}

Rate the response's improvement on a scale of 0-10:
- 0: The response stays completely off-topic or gets worse
- 1-3: Slight improvement, but mostly stays off-topic or confused
- 4-5: Some meaningful shift toward the prompt topic, but inconsistent
- 6-7: Clear improvement — the response noticeably gets more on-topic
- 8-10: Dramatic improvement — starts very off-topic, ends well on-topic

Also note whether the improvement happens through an explicit restart (the model says something like "wait" or "let me try again") or through a gradual/implicit shift.

Provide your analysis, then output in this exact JSON format wrapped in <json></json> tags:
<json>
{{"improvement_score": N, "improvement_type": "explicit_restart|gradual_shift|no_improvement", "brief_explanation": "..."}}
</json>""",
}


# =============================================================================
# Evaluation
# =============================================================================

async def evaluate_one(
    judge,
    example: dict,
    max_retries: int = 3,
) -> dict:
    """Run one judge evaluation on one example, with retries."""
    user_msg = JUDGE_PROMPT["user"].format(
        prompt=example["prompt"],
        response=example["response"],
        feature_label=example["feature_label"],
    )
    for attempt in range(max_retries):
        try:
            raw = await judge.client.complete(JUDGE_PROMPT["system"], user_msg)
            parsed = _extract_grade(raw)
            if "error" not in parsed or attempt == max_retries - 1:
                return {"raw_response": raw, **parsed}
            # Parse error (e.g. truncated JSON) — retry
            await asyncio.sleep(1.0 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            # Exponential backoff with wide jitter to avoid thundering herd
            delay = 2.0 ** (attempt + 1) + random.uniform(0, 3.0)
            await asyncio.sleep(delay)


async def run_validation(model_id: str) -> list[dict]:
    """Run the judge prompt against the validation set."""
    with open(VALIDATION_SET) as f:
        examples = json.load(f)

    judge = create_judge(model_id, max_concurrent=30)

    results = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {ex['category']}: {ex['prompt'][:50]}...")
        result = await evaluate_one(judge, ex)
        improvement_score = result.get("improvement_score", None)
        print(f"    -> improvement_score={improvement_score}")
        results.append({
            "example_idx": i,
            "category": ex["category"],
            "expected_improvement": ex["expected_improvement"],
            "prompt_snippet": ex["prompt"][:60],
            "original_scores": ex["original_scores"],
            "judge_result": result,
        })

    return results


async def run_experiment_results(result_files: list[str], model_id: str,
                                 sample_per_file: int | None = None) -> dict:
    """Run the whole-response judge against experiment result files."""
    import concurrent.futures
    loop = asyncio.get_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=50))

    judge = create_judge(model_id, max_concurrent=30)

    # Collect all trials across all files
    all_tasks = []
    for fpath in result_files:
        file_tasks = []
        with open(fpath) as f:
            data = json.load(f)
        for feat in data.get("results_by_feature", []):
            label = feat.get("feature_label", "")
            feature_idx = feat.get("feature_index_in_sae", None)
            for trial_idx, trial in enumerate(feat.get("trials", [])):
                if trial.get("error") or not trial.get("response"):
                    continue
                file_tasks.append({
                    "fpath": fpath,
                    "feature_index": feature_idx,
                    "feature_label": label,
                    "trial_idx": trial_idx,
                    "prompt": trial["prompt"],
                    "response": trial["response"],
                    "original_score": trial.get("score", {}),
                })
        if sample_per_file and len(file_tasks) > sample_per_file:
            random.seed(42)
            file_tasks = random.sample(file_tasks, sample_per_file)
        all_tasks.extend(file_tasks)

    total = len(all_tasks)
    print(f"\nTotal trials to judge: {total}")

    from tqdm.asyncio import tqdm_asyncio

    async def process_one(task: dict) -> dict:
        result = await evaluate_one(judge, {
            "prompt": task["prompt"],
            "response": task["response"],
            "feature_label": task["feature_label"],
        })
        return {
            "fpath": task["fpath"],
            "feature_index": task["feature_index"],
            "feature_label": task["feature_label"],
            "trial_idx": task["trial_idx"],
            "prompt": task["prompt"],
            "improvement_score": result.get("improvement_score"),
            "improvement_type": result.get("improvement_type"),
            "brief_explanation": result.get("brief_explanation"),
            "original_score": task["original_score"],
            "error": result.get("error"),
        }

    results = await tqdm_asyncio.gather(
        *[process_one(t) for t in all_tasks],
        desc="Judging trials",
    )

    # Group by file
    all_results = {}
    for r in results:
        fpath = r.pop("fpath")
        all_results.setdefault(fpath, []).append(r)

    for fpath, file_results in all_results.items():
        print(f"  {Path(fpath).name}: {len(file_results)} trials")

    return all_results


async def retry_failures(results_path: str, model_id: str) -> dict:
    """Load existing results and re-run only trials that failed.

    Looks up the original response text from the source experiment files,
    since we don't store full responses in the judge results.
    """
    import concurrent.futures
    loop = asyncio.get_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=50))

    judge = create_judge(model_id, max_concurrent=30)

    with open(results_path) as f:
        existing = json.load(f)

    # If a previous run was killed, merge its partial results first
    partial_path = Path(results_path).with_suffix(".partial.jsonl")
    if partial_path.exists():
        merged = 0
        with open(partial_path) as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fpath, idx, trial = record["fpath"], record["idx"], record["trial"]
                if fpath in existing and idx < len(existing[fpath]):
                    existing[fpath][idx] = trial
                    merged += 1
        print(f"Merged {merged} results from previous partial run")
        partial_path.unlink()

    # Build a lookup from source experiment files: (feature_index, trial_idx) -> response
    response_lookup: dict[str, dict[tuple[int, int], str]] = {}
    for fpath in existing:
        with open(fpath) as f:
            source = json.load(f)
        lookup = {}
        for feat in source.get("results_by_feature", []):
            fidx = feat.get("feature_index_in_sae")
            for tidx, trial in enumerate(feat.get("trials", [])):
                if trial.get("response"):
                    lookup[(fidx, tidx)] = trial["response"]
        response_lookup[fpath] = lookup

    # Find failed trials
    to_retry = []
    for fpath, trials in existing.items():
        for i, t in enumerate(trials):
            if t.get("error") or t.get("improvement_score") is None:
                response = response_lookup.get(fpath, {}).get(
                    (t.get("feature_index"), t.get("trial_idx")), ""
                )
                if response:
                    to_retry.append((fpath, i, t, response))

    print(f"Found {len(to_retry)} failed trials to retry out of "
          f"{sum(len(v) for v in existing.values())} total")

    if not to_retry:
        return existing

    from tqdm.asyncio import tqdm_asyncio

    # Write each completed result to a temp JSONL file as it arrives,
    # so progress is never lost if the process is killed.
    partial_path = Path(results_path).with_suffix(".partial.jsonl")
    write_lock = asyncio.Lock()

    # Limit concurrency to avoid 429s
    sem = asyncio.Semaphore(30)

    async def retry_one(fpath: str, idx: int, trial: dict, response: str) -> tuple[str, int, dict]:
        async with sem:
            result = await evaluate_one(judge, {
                "prompt": trial["prompt"],
                "response": response,
                "feature_label": trial["feature_label"],
            }, max_retries=5)
        new_trial = {**trial}
        new_trial["improvement_score"] = result.get("improvement_score")
        new_trial["improvement_type"] = result.get("improvement_type")
        new_trial["brief_explanation"] = result.get("brief_explanation")
        new_trial["error"] = result.get("error")
        # Append to partial file immediately
        record = {"fpath": fpath, "idx": idx, "trial": new_trial}
        async with write_lock:
            with open(partial_path, "a") as pf:
                pf.write(json.dumps(record) + "\n")
        return fpath, idx, new_trial

    print(f"Writing partial results to {partial_path}")

    retried = await tqdm_asyncio.gather(
        *[retry_one(fp, i, t, r) for fp, i, t, r in to_retry],
        desc="Retrying failures",
    )

    # Merge back from in-memory results (partial file is the backup)
    for fpath, idx, new_trial in retried:
        existing[fpath][idx] = new_trial

    # Clean up partial file on success
    partial_path.unlink(missing_ok=True)

    still_failed = sum(1 for fp, trials in existing.items()
                       for t in trials if t.get("error") or t.get("improvement_score") is None)
    print(f"After retry: {still_failed} still failed")

    return existing


def print_validation_summary(results: list[dict]):
    """Print a summary of validation results."""
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")

    threshold = 4
    correct = 0
    total = 0

    for r in results:
        jr = r["judge_result"]
        score = jr.get("improvement_score")

        predicted_improving = score is not None and score >= threshold
        expected = r["expected_improvement"]
        match = predicted_improving == expected
        if score is not None:
            correct += int(match)
            total += 1

        symbol = "+" if match else "X"
        cat = r["category"]
        score_str = f"{score:>4}" if score is not None else "   ?"
        print(
            f"  {symbol} [{cat:25s}] score={score_str} | "
            f"predicted={'improving' if predicted_improving else 'not_impr':>10} | "
            f"expected={'improving' if expected else 'not_impr':>10} | "
            f"{r['prompt_snippet'][:40]}"
        )

    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct/total:.0%}")

        for cat in ["clear_self_correction", "no_self_correction", "subtle_self_correction"]:
            cat_results = [r for r in results if r["category"] == cat]
            cat_correct = sum(
                1 for r in cat_results
                if r["judge_result"].get("improvement_score") is not None
                and (r["judge_result"]["improvement_score"] >= threshold) == r["expected_improvement"]
            )
            cat_total = sum(1 for r in cat_results if r["judge_result"].get("improvement_score") is not None)
            if cat_total:
                print(f"  {cat}: {cat_correct}/{cat_total}")

    return correct, total


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", action="store_true",
                        help="Run against validation set")
    parser.add_argument("--experiment-results", nargs="+",
                        help="Experiment result JSON files to judge")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file for experiment results")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview",
                        help="Judge model (alias or full ID)")
    parser.add_argument("--retry", default=None,
                        help="Path to existing results JSON — re-run only failed trials")
    parser.add_argument("--sample-per-file", type=int, default=None,
                        help="Sample N trials per input file (for quick estimates)")
    args = parser.parse_args()

    model_id = resolve_model_id(args.judge_model)
    print(f"Using judge model: {model_id}")

    if args.validation:
        results = await run_validation(model_id)
        outfile = "whole_response_judge_validation_results.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {outfile}")
        print_validation_summary(results)

    elif args.retry:
        results = await retry_failures(args.retry, model_id)
        outfile = args.output or args.retry
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {outfile}")

    elif args.experiment_results:
        results = await run_experiment_results(args.experiment_results, model_id,
                                                sample_per_file=args.sample_per_file)
        outfile = args.output or "whole_response_judge_experiment_results.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {outfile}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
