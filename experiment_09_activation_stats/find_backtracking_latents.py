"""
Find backtracking/self-correction latents by searching latent labels.

This script implements a documented, principled process for identifying SAE latents
that are likely to fire during self-correction behavior in LLM responses.

Method:
1. Load latent labels from CSV file
2. Score each label based on weighted keyword matching
3. Exclude labels that are clearly about technical errors (HTTP, encoding, etc.)
4. Output ranked list of candidate backtracking latents
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


# Keywords and their weights for identifying self-correction latents
# Higher weight = more relevant to self-correction behavior
POSITIVE_KEYWORDS = [
    # High relevance (weight 3)
    ("self-correct", 3),
    ("backtrack", 3),
    ("correct a", 3),
    ("correcting", 3),
    ("made a mistake", 3),
    ("acknowledge.*error", 3),
    ("apologize", 3),
    ("apolog", 3),

    # Medium-high relevance (weight 2)
    ("mistake", 2),
    ("incorrect", 2),
    ("revise", 2),
    ("reconsider", 2),
    ("take back", 2),
    ("taking ownership", 2),
    ("responsibility for", 2),

    # Medium relevance (weight 1)
    ("error", 1),
    ("wrong", 1),
    ("acknowledge", 1),
    ("careful and precise", 1),
    ("realiz", 1),  # realize, realizing
]

# Patterns to EXCLUDE - these are about technical errors, not self-correction behavior
EXCLUSION_PATTERNS = [
    r"HTTP",
    r"stack trace",
    r"encoding error",
    r"Python type",
    r"array index",
    r"bounds check",
    r"error handling",
    r"error message",
    r"try-catch",
    r"exception",
    r"Unicode",
    r"replacement character",
    r"syntax error",
    r"compilation",
    r"debug",
    r"line number",
    r"error code",
    r"API error",
    r"network error",
    r"file.*error",
    r"database",
    r"SQL",
    r"validation error",
    r"input validation",
    r"security",
    r"authentication",
    r"password",
    r"permission",
    r"access denied",
    r"timeout",
    r"connection",
    r"404|500|403|401",  # HTTP status codes
    r"await.*async",  # async/await programming (not "wait" in self-correction sense)
    r"loop",  # programming loops (not behavioral)
]


@dataclass
class LatentCandidate:
    """A candidate backtracking latent with scoring information."""
    index: int
    label: str
    score: float
    matched_keywords: list[str]


def score_label(label: str) -> tuple[float, list[str]]:
    """
    Score a label based on keyword matching.

    Returns:
        (score, list of matched keywords)
    """
    label_lower = label.lower()
    score = 0.0
    matched = []

    # Check exclusion patterns first
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, label, re.IGNORECASE):
            return 0.0, []  # Excluded

    # Score based on positive keywords
    for keyword, weight in POSITIVE_KEYWORDS:
        if re.search(keyword, label_lower):
            score += weight
            matched.append(keyword)

    return score, matched


def find_backtracking_latents(labels_file: Path, min_score: float = 1.0) -> list[LatentCandidate]:
    """
    Search through latent labels and find backtracking/self-correction candidates.

    Args:
        labels_file: Path to CSV file with latent labels
        min_score: Minimum score threshold for inclusion

    Returns:
        List of LatentCandidate objects, sorted by score descending
    """
    candidates = []

    with open(labels_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index_in_sae"])
            label = row["label"] if row["label"] else ""

            if not label:
                continue

            score, matched = score_label(label)

            if score >= min_score:
                candidates.append(LatentCandidate(
                    index=idx,
                    label=label,
                    score=score,
                    matched_keywords=matched,
                ))

    # Sort by score descending, then by index
    candidates.sort(key=lambda x: (-x.score, x.index))

    return candidates


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    labels_file = base_dir / "data" / "llama-70b-goodfire-l50.csv"
    output_dir = base_dir / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "activation_stats"
    output_file = output_dir / "backtracking_latent_candidates.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for backtracking latents in: {labels_file}")

    # Find candidates
    candidates = find_backtracking_latents(labels_file, min_score=1.0)

    print(f"\nFound {len(candidates)} candidate backtracking latents\n")

    # Print top candidates
    print("Top 30 candidates by score:")
    print("-" * 100)
    for i, c in enumerate(candidates[:30], 1):
        print(f"{i:2}. [{c.index:5}] (score={c.score:.1f}) {c.label[:80]}")
        print(f"       Matched: {', '.join(c.matched_keywords)}")

    # Save to JSON
    output_data = {
        "method": "keyword_search",
        "positive_keywords": POSITIVE_KEYWORDS,
        "exclusion_patterns": EXCLUSION_PATTERNS,
        "min_score": 1.0,
        "n_candidates": len(candidates),
        "candidates": [
            {
                "index": c.index,
                "label": c.label,
                "score": c.score,
                "matched_keywords": c.matched_keywords,
            }
            for c in candidates
        ]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved candidates to: {output_file}")

    # Print recommended latents (score >= 2)
    recommended = [c for c in candidates if c.score >= 2.0]
    print(f"\n\nRecommended latents (score >= 2.0): {len(recommended)}")
    print("These should be used for activation analysis:")
    print("-" * 80)
    for c in recommended:
        print(f"  {c.index}: {c.label}")


if __name__ == "__main__":
    main()
