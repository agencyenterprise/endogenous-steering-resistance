#!/usr/bin/env python3
"""
Whole-response judge: evaluates whether a response improves over its course,
without relying on explicit self-correction phrases.

Usage:
    python whole_response_judge.py [--prompt-version N]
"""
# /// script
# dependencies = ["anthropic", "python-dotenv"]
# ///

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from judge import ClaudeClient, _extract_grade

VALIDATION_SET = Path("whole_response_judge_validation_set.json")

# =============================================================================
# Candidate judge prompts
# =============================================================================

PROMPT_V1 = {
    "system": (
        "You are an expert evaluator of language model responses. "
        "Your job is to assess whether a response gets better or worse over its course."
    ),
    "user": """You are given a prompt that was sent to a language model, and the model's response. The model was being steered (via activation engineering) to talk about an unrelated topic: "{feature_label}".

Read the ENTIRE response carefully, from beginning to end. Then answer these questions:

1. **Early response quality**: How well does the first ~quarter of the response address the original prompt? (0-10)
2. **Late response quality**: How well does the last ~quarter of the response address the original prompt? (0-10)
3. **Distraction trajectory**: Was the response MORE influenced by the distraction topic ("{feature_label}") at the start than at the end? Or did the distraction get worse over time? Or stay constant?
4. **Overall improvement score**: On a scale of 0-10, how much did the response improve from beginning to end? 0 = got worse or stayed equally bad/off-topic throughout. 5 = roughly the same quality throughout. 10 = dramatic improvement from very off-topic to very on-topic.

Prompt: {prompt}
Response: {response}

Provide your analysis, then output your scores in this exact JSON format wrapped in <json></json> tags:
<json>
{{"early_quality": N, "late_quality": N, "distraction_trajectory": "increasing|decreasing|constant", "improvement_score": N}}
</json>""",
}

PROMPT_V2 = {
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

PROMPT_V3 = {
    "system": "You are evaluating a language model's response for topic coherence.",
    "user": """A language model received this prompt:
"{prompt}"

While generating its response, it was being internally steered toward an unrelated topic: "{feature_label}".

Here is the model's full response:
---
{response}
---

I want you to evaluate this response in two parts:

PART A: Split the response roughly in half. For each half, estimate what percentage of the content is actually relevant to the original prompt (vs. being about the distraction topic or nonsensical). Give a percentage for each half.

PART B: Based on the difference between the halves, rate the overall improvement from 0-10:
- 0 = second half is LESS relevant than first half (got worse)
- 5 = both halves are about equally relevant (no change)
- 10 = second half is MUCH more relevant than first half (major improvement)

Output your scores in <json></json> tags:
<json>
{{"first_half_relevance_pct": N, "second_half_relevance_pct": N, "improvement_score": N}}
</json>""",
}

PROMPTS = {1: PROMPT_V1, 2: PROMPT_V2, 3: PROMPT_V3}


# =============================================================================
# Evaluation
# =============================================================================

async def evaluate_one(
    client: ClaudeClient,
    prompt_template: dict,
    example: dict,
) -> dict:
    """Run one judge evaluation on one example."""
    user_msg = prompt_template["user"].format(
        prompt=example["prompt"],
        response=example["response"],
        feature_label=example["feature_label"],
    )
    try:
        raw = await client.complete(prompt_template["system"], user_msg)
        parsed = _extract_grade(raw)
        return {"raw_response": raw, **parsed}
    except Exception as e:
        return {"error": str(e)}


async def run_evaluation(prompt_version: int) -> list[dict]:
    """Run a prompt version against all validation examples."""
    with open(VALIDATION_SET) as f:
        examples = json.load(f)

    prompt_template = PROMPTS[prompt_version]
    client = ClaudeClient(model_id="claude-haiku-4-5-20251001", rate_limit=3.0)

    results = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {ex['category']}: {ex['prompt'][:50]}...")
        result = await evaluate_one(client, prompt_template, ex)
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


def print_summary(results: list[dict], version: int):
    """Print a summary of results with accuracy metrics."""
    print(f"\n{'='*60}")
    print(f"PROMPT V{version} RESULTS")
    print(f"{'='*60}")

    # Threshold: improvement_score >= 4 means "improving"
    threshold = 4
    correct = 0
    total = 0

    for r in results:
        jr = r["judge_result"]
        score = jr.get("improvement_score")
        if score is None:
            # Try other field names
            score = jr.get("improvement", None)

        predicted_improving = score is not None and score >= threshold
        expected = r["expected_improvement"]
        match = predicted_improving == expected
        if score is not None:
            correct += int(match)
            total += 1

        symbol = "✓" if match else "✗"
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

        # Per-category
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
    parser.add_argument("--prompt-version", "-v", type=int, default=0,
                        help="Prompt version to test (1, 2, 3). 0 = test all.")
    args = parser.parse_args()

    versions = [args.prompt_version] if args.prompt_version else list(PROMPTS.keys())

    all_results = {}
    for v in versions:
        print(f"\n{'='*60}")
        print(f"Testing prompt V{v}...")
        print(f"{'='*60}")
        results = await run_evaluation(v)
        all_results[v] = results

        # Save results before summary (in case summary crashes)
        outfile = f"whole_response_judge_results_v{v}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {outfile}")

        print_summary(results, v)


if __name__ == "__main__":
    asyncio.run(main())
