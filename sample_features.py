"""Feature sampling with concreteness and relevance filtering.

This module provides functionality to sample features that are both concrete
(based on label quality) and irrelevant (zero activation on provided prompts).
"""

import random
from typing import List, Dict, Optional

from concreteness_filtering import ConcretenessGrader, filter_concrete_features
from relevance_filtering import compute_feature_relevance_for_prompts, filter_irrelevant_features


async def sample_filtered_features(
    engine,
    prompts: List[str],
    feature_labels: Dict[int, str],
    n_features: int,
    concreteness_threshold: float,
    num_sae_features: int,
    candidate_multiplier: int = 3,
    grader: Optional[ConcretenessGrader] = None,
    labels_file: Optional[str] = None,
) -> List[int]:
    """Sample n features that are both concrete and irrelevant to all prompts.

    This function:
    1. Randomly samples candidate_multiplier * n_features candidate features
    2. Filters by concreteness (keeps features with labels above threshold)
    3. Filters by relevance (keeps features with zero activation on all prompts)
    4. Returns up to n_features that meet both criteria

    Args:
        engine: Initialized VLLMSteeringEngine instance
        prompts: List of prompt strings to test relevance against
        feature_labels: Dict mapping feature_index -> label string
        n_features: Target number of features to sample
        concreteness_threshold: Minimum concreteness score (0-100)
        num_sae_features: Total number of SAE features available
        candidate_multiplier: Sample this many times n_features as candidates (default: 3)
        grader: ConcretenessGrader instance (creates new one if None)
        labels_file: Path to labels file, or None if no labels available.
            If None, concreteness filtering is skipped (all features treated as concrete).

    Returns:
        List of up to n_features feature indices that are both concrete and irrelevant
    """
    candidate_pool_size = n_features * candidate_multiplier

    print(f"\n{'='*80}")
    print(f"Sampling {n_features} features with concreteness and relevance filtering")
    print(f"{'='*80}")
    print(f"Candidate pool size: {candidate_pool_size}")
    print(f"Total SAE features: {num_sae_features}")
    print(f"Concreteness threshold: {concreteness_threshold}")
    print(f"Number of prompts for relevance testing: {len(prompts)}")
    print(f"{'='*80}\n")

    # Step 1: Randomly sample candidate features
    print(f"Step 1: Randomly sampling {candidate_pool_size} candidate features...")
    candidate_indices = random.sample(range(num_sae_features), candidate_pool_size)
    print(f"✓ Sampled {len(candidate_indices)} candidates\n")

    # Step 2: Filter by concreteness (skip if no labels file available)
    if labels_file is None:
        print(f"Step 2: No labels file - skipping concreteness filtering, treating all features as concrete")
        concrete_features = list(candidate_indices)
        print(f"✓ Using all {len(concrete_features)} candidates as concrete\n")
    else:
        print(f"Step 2: Filtering by concreteness (threshold: {concreteness_threshold})...")
        candidate_labels = {idx: feature_labels.get(idx, f"feature_{idx}") for idx in candidate_indices}

        concrete_features = await filter_concrete_features(
            candidate_labels,
            concreteness_threshold,
            grader=grader,
        )
        print(f"✓ After concreteness filtering: {len(concrete_features)} features\n")

        if not concrete_features:
            print("⚠️ No features passed concreteness filtering")
            return []

    # Step 3: Filter by relevance
    print(f"Step 3: Filtering by relevance (zero activation on all prompts)...")
    relevance_scores = await compute_feature_relevance_for_prompts(
        engine,
        concrete_features,
        prompts,
    )

    irrelevant_features = filter_irrelevant_features(relevance_scores)
    print(f"✓ After relevance filtering: {len(irrelevant_features)} features\n")

    if not irrelevant_features:
        print("⚠️ No features passed relevance filtering")
        return []

    # Step 4: Return up to n_features
    final_features = irrelevant_features[:n_features]

    print(f"{'='*80}")
    print(f"✓ Final result: {len(final_features)}/{n_features} features sampled")
    if len(final_features) < n_features:
        print(f"  Note: Found fewer features than requested")
    print(f"{'='*80}\n")

    return final_features


async def sample_filtered_features_with_retry(
    engine,
    prompts: List[str],
    feature_labels: Dict[int, str],
    n_features: int,
    concreteness_threshold: float,
    num_sae_features: int,
    candidate_multiplier: int = 3,
    max_retries: int = 3,
    grader: Optional[ConcretenessGrader] = None,
    labels_file: Optional[str] = None,
) -> List[int]:
    """Sample filtered features with retry logic if not enough features are found.

    If the initial sampling doesn't yield enough features, this function will retry
    with an increased candidate pool size.

    Args:
        engine: Initialized VLLMSteeringEngine instance
        prompts: List of prompt strings to test relevance against
        feature_labels: Dict mapping feature_index -> label string
        n_features: Target number of features to sample
        concreteness_threshold: Minimum concreteness score (0-100)
        num_sae_features: Total number of SAE features available
        candidate_multiplier: Initial multiplier for candidate pool size
        max_retries: Maximum number of retry attempts with larger pools
        grader: ConcretenessGrader instance (creates new one if None)
        labels_file: Path to labels file, or None if no labels available.
            If None, concreteness filtering is skipped (all features treated as concrete).

    Returns:
        List of feature indices that are both concrete and irrelevant
    """
    sampled_features = set()
    current_multiplier = candidate_multiplier

    for attempt in range(max_retries + 1):
        if len(sampled_features) >= n_features:
            break

        remaining = n_features - len(sampled_features)

        if attempt > 0:
            print(f"\nRetry attempt {attempt}/{max_retries}")
            print(f"Need {remaining} more features, increasing multiplier to {current_multiplier}x")

        # Sample features
        new_features = await sample_filtered_features(
            engine=engine,
            prompts=prompts,
            feature_labels=feature_labels,
            n_features=remaining,
            concreteness_threshold=concreteness_threshold,
            num_sae_features=num_sae_features,
            candidate_multiplier=current_multiplier,
            grader=grader,
            labels_file=labels_file,
        )

        # Add to collected features
        sampled_features.update(new_features)

        # Increase multiplier for next attempt
        current_multiplier *= 2

    final_features = list(sampled_features)[:n_features]

    if len(final_features) < n_features:
        print(f"\n⚠️ Warning: Only found {len(final_features)}/{n_features} features after {max_retries + 1} attempts")

    return final_features
