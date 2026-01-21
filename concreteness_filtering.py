"""Concreteness filtering utilities for feature steering experiments.

This module provides functions to grade feature labels for concreteness and filter
features to find those above a concreteness threshold.
"""

import asyncio
import json
import os
import time
from typing import List, Dict
from pathlib import Path
import hashlib

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()


class ConcretenessGrader:
    """Grades feature labels for concreteness on-demand with persistent caching."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", cache_dir: Path = None):
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path("data") / "concreteness_cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache file path based on model name
        model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
        self.cache_file = self.cache_dir / f"concreteness_scores_{model_hash}.json"

        # Load existing cache
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, float]:
        """Load concreteness scores from cache file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                print(f"Loaded {len(cache_data)} cached concreteness scores")
                return cache_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load cache file: {e}")
                return {}
        else:
            return {}

    def _save_cache(self):
        """Save concreteness scores to cache file."""
        try:
            # Create a backup if the file exists
            if self.cache_file.exists():
                backup_file = self.cache_file.with_suffix(f'.json.backup.{int(time.time())}')
                self.cache_file.rename(backup_file)

            # Save current cache
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2, sort_keys=True)

            # Clean up old backups (keep only the 3 most recent)
            backup_files = sorted(
                self.cache_dir.glob("concreteness_scores_*.json.backup.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for old_backup in backup_files[3:]:
                old_backup.unlink()

        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")

    async def _grade_single_label_batch(self, labels: List[str]) -> Dict[str, float]:
        """Grade a single batch of labels for concreteness using async API.

        Args:
            labels: List of feature labels to grade

        Returns:
            Dictionary mapping label -> concreteness score (0-100)
        """
        # Format labels for API call
        labels_json = json.dumps(labels)

        prompt = f"""You are assessing feature labels for concreteness and domain-specificity.
Rate each label on a scale of 0-100 where:
0 = Very abstract and general
100 = Very concrete and domain-specific

In particular, if the label concerns conversational styles, e.g. "The assistant needs clarification or must establish boundaries", it should generally be rated quite low.

Provide your response in valid JSON format ONLY, with no explanations or additional text:
[
  {{"label": "example_label", "justification": "brief reason", "rating": 57.0}}
]

Here are the labels to assess:
{labels_json}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=10000,
                system="You are an AI that analyzes feature labels for concreteness and domain specificity. You MUST respond only with valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            content = response.content[0].text

            # Strip any markdown formatting
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()

            # Parse ratings
            batch_results = json.loads(content)

            # Extract scores
            results = {}
            for item in batch_results:
                if "label" in item and "rating" in item:
                    label = item["label"]
                    rating = float(item["rating"])
                    results[label] = rating

            return results

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return {label: 0.0 for label in labels}

        except Exception as e:
            print(f"Error grading labels: {e}")
            return {label: 0.0 for label in labels}

    async def grade_labels_batch(self, labels: List[str], batch_size: int = 50) -> Dict[str, float]:
        """Grade labels in batches using async API calls.

        Args:
            labels: List of feature labels to grade
            batch_size: Number of labels to grade in each API call

        Returns:
            Dictionary mapping label -> concreteness score (0-100)
        """
        # Check cache first
        uncached_labels = [label for label in labels if label not in self._cache]
        results = {label: self._cache[label] for label in labels if label in self._cache}

        if not uncached_labels:
            print(f"Found all {len(labels)} labels in cache")
            return results

        print(f"Found {len(results)} cached labels, grading {len(uncached_labels)} new labels")

        # Split into batches
        batches = [
            uncached_labels[i:i+batch_size]
            for i in range(0, len(uncached_labels), batch_size)
        ]

        # Grade all batches in parallel using asyncio.gather with tqdm
        print(f"Grading {len(batches)} batches in parallel...")
        tasks = [self._grade_single_label_batch(batch) for batch in batches]
        batch_results = await tqdm.gather(*tasks, desc="Grading batches")

        # Merge results and update cache
        new_scores_count = 0
        for batch_result in batch_results:
            for label, score in batch_result.items():
                if label not in self._cache:
                    new_scores_count += 1
                self._cache[label] = score
                results[label] = score

        # Save cache after adding new scores
        if new_scores_count > 0:
            print(f"Graded {new_scores_count} new labels")
            self._save_cache()

        return results

    async def get_concreteness_scores(
        self, feature_labels: Dict[int, str]
    ) -> Dict[int, float]:
        """Get concreteness scores for a dict of feature indices to labels.

        Args:
            feature_labels: Dict mapping feature_index -> label string

        Returns:
            Dict mapping feature_index -> concreteness score (0-100)
        """
        # Extract unique labels
        labels = list(feature_labels.values())

        # Grade all labels
        label_scores = await self.grade_labels_batch(labels, batch_size=50)

        # Map back to feature indices
        feature_scores = {
            idx: label_scores.get(label, 0.0)
            for idx, label in feature_labels.items()
        }

        return feature_scores


async def filter_concrete_features(
    feature_labels: Dict[int, str],
    concreteness_threshold: float,
    grader: ConcretenessGrader = None,
) -> List[int]:
    """Filter features to find those above the concreteness threshold.

    Args:
        feature_labels: Dict mapping feature_index -> label string
        concreteness_threshold: Minimum concreteness score (0-100)
        grader: ConcretenessGrader instance (creates new one if None)

    Returns:
        List of feature indices that meet the concreteness threshold
    """
    if grader is None:
        grader = ConcretenessGrader()

    # Filter out generic labels (starting with "feature_")
    meaningful_features = {
        idx: label for idx, label in feature_labels.items()
        if not label.startswith("feature_")
    }

    if not meaningful_features:
        print("Warning: No meaningful labels found")
        return []

    print(f"Filtering {len(meaningful_features)} features with meaningful labels")

    # Get concreteness scores
    scores = await grader.get_concreteness_scores(meaningful_features)

    # Filter by threshold
    concrete_features = [
        idx for idx, score in scores.items()
        if score >= concreteness_threshold
    ]

    if concrete_features:
        avg_score = sum(scores[idx] for idx in concrete_features) / len(concrete_features)
        print(f"Found {len(concrete_features)}/{len(meaningful_features)} concrete features (avg score: {avg_score:.1f})")
    else:
        print(f"No features met concreteness threshold {concreteness_threshold}")

    return concrete_features
