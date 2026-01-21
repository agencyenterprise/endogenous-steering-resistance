"""Relevance filtering utilities for feature steering experiments.

This module provides functions to compute feature relevance to prompts and filter
features to find those that are irrelevant (zero activation) across all prompts.
"""

import uuid
from typing import List, Dict

import torch
from tqdm import tqdm

from vllm import SamplingParams
from vllm.inputs import TokenInputs


async def get_feature_activations_for_prompt(
    engine,
    prompt: str,
) -> torch.Tensor:
    """Get feature activations for a single prompt without generating tokens.

    Args:
        engine: VLLMSteeringEngine instance (must be initialized)
        prompt: Prompt string

    Returns:
        Feature activation tensor of shape [seq_len, num_sae_features]
    """
    # Prepare conversation
    conversation = [{"role": "user", "content": prompt}]

    # Apply chat template and get token IDs
    # Use add_generation_prompt=False since we're just getting activations
    prompt_token_ids = engine.tokenizer.apply_chat_template(conversation, add_generation_prompt=False)

    # Create token inputs
    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=conversation)

    # Set up sampling parameters - generate 1 token to get activations
    # (vLLM may require at least 1 token generation)
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
    )

    # Get feature activations for the input prompt
    request_id = str(uuid.uuid4())
    results_generator = engine.engine.generate(
        prompt=token_inputs,
        sampling_params=sampling_params,
        request_id=request_id,
        interventions=None,
        is_feature_decode=True,  # This enables feature tensor collection
    )

    # Wait for final output
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if hasattr(final_output, "feature_tensor") and final_output.feature_tensor is not None:
        return final_output.feature_tensor
    else:
        raise RuntimeError(f"No feature tensor found for prompt: {prompt[:50]}...")


def get_feature_relevance_scores(
    feature_tensor: torch.Tensor,
    feature_indices: List[int],
) -> Dict[int, float]:
    """Extract relevance scores for specific features from activation tensor.

    Uses max aggregation across the sequence dimension to get the peak activation.

    Args:
        feature_tensor: Tensor of shape [seq_len, num_sae_features]
        feature_indices: List of feature indices to get scores for

    Returns:
        Dict mapping feature_index -> relevance_score (max activation across sequence)
    """
    if feature_tensor is None or feature_tensor.numel() == 0:
        return {idx: 0.0 for idx in feature_indices}

    # Get max activation across sequence dimension for each feature
    max_activations = feature_tensor.max(dim=0).values

    # Extract scores for requested features
    scores = {}
    for idx in feature_indices:
        if idx < len(max_activations):
            scores[idx] = float(max_activations[idx].item())
        else:
            scores[idx] = 0.0

    return scores


async def compute_feature_relevance_for_prompts(
    engine,
    feature_indices: List[int],
    prompts: List[str],
) -> Dict[int, List[float]]:
    """Compute relevance scores for specific features across multiple prompts.

    Args:
        engine: VLLMSteeringEngine instance (must be initialized)
        feature_indices: List of feature indices to evaluate
        prompts: List of prompt strings

    Returns:
        Dict mapping feature_index -> list of relevance scores (one per prompt)
    """
    # Initialize result structure
    relevance_by_feature = {idx: [] for idx in feature_indices}

    print(f"Computing relevance for {len(feature_indices)} features on {len(prompts)} prompts...")

    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Get feature activations for this prompt (no generation)
        feature_tensor = await get_feature_activations_for_prompt(engine, prompt)

        # Get scores for our specific features
        scores = get_feature_relevance_scores(feature_tensor, feature_indices)

        # Store scores
        for idx in feature_indices:
            relevance_by_feature[idx].append(scores[idx])

    return relevance_by_feature


def filter_irrelevant_features(
    relevance_by_feature: Dict[int, List[float]],
) -> List[int]:
    """Filter features to find those that have zero activation on all prompts.

    A feature is considered irrelevant if it has 0 activation on ALL prompts.

    Args:
        relevance_by_feature: Dict mapping feature_index -> list of relevance scores

    Returns:
        List of feature indices that have zero activation on all prompts
    """
    irrelevant_features = []

    for feature_idx, scores in relevance_by_feature.items():
        # Feature must have zero activation on ALL prompts
        if all(score == 0.0 for score in scores):
            irrelevant_features.append(feature_idx)

    return irrelevant_features
