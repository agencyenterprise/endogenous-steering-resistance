#!/usr/bin/env python
"""Collect SAE latent activations for shuffled and non-shuffled prompt-response pairs.

This script:
1. Generates unsteered responses to all 38 prompts (prompts.txt)
2. Creates shuffled pairs using multiple derangements (each prompt paired with different prompts' responses)
3. Gets SAE activations for ALL latents across all token positions (max-pooled)
4. Saves the full activation tensor for both conditions

The output can then be analyzed to understand how the 27 off-topic detectors were selected.

Usage:
    python collect_activations.py

Configuration:
    NUM_DERANGEMENTS - Number of derangements to generate (default: 2)
                       With 38 prompts and 2 derangements, you get 76 shuffled samples.

Output:
    data/activations_all_latents.npz - Contains:
        - shuffled_activations: [num_prompts * num_derangements, num_latents] tensor
        - normal_activations: [num_prompts, num_latents] tensor
    data/activations_metadata.json - Contains:
        - prompts, normal_responses, shuffled_pairs, etc.
"""

import asyncio
import json
import random
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

# Add AGI-1516 to path for imports
AGI_1516_ROOT = Path(__file__).parent.parent.parent / "AGI-1516-esr-with-vllm"
sys.path.insert(0, str(AGI_1516_ROOT))

from vllm_engine import VLLMSteeringEngine

load_dotenv()

# Configuration (can be overridden by command-line args)
MODEL_NAME = "meta-llama/Meta-Llama-3.3-70B-Instruct"
DEFAULT_PROMPTS_FILE = Path(__file__).parent.parent / "data" / "prompts.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
MAX_COMPLETION_TOKENS = 512
RANDOM_SEED = 42  # For reproducibility
NUM_DERANGEMENTS = 10  # Number of different derangements to generate (each analyzed separately)


def load_prompts(filepath: Path) -> list[str]:
    """Load prompts from file, one per line."""
    with open(filepath) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


async def generate_unsteered_response(
    engine: VLLMSteeringEngine,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.6,
    seed: int = None,
) -> str:
    """Generate an unsteered response to a prompt."""
    convo = [{"role": "user", "content": prompt}]
    response = await engine.generate_with_conversation(
        conversation=convo,
        feature_interventions=None,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    return response


async def get_sae_activations_all_positions(
    engine: VLLMSteeringEngine,
    prompt: str,
    existing_response: str,
) -> dict[str, torch.Tensor]:
    """
    Get SAE latent activations across all token positions for a conversation.

    Feeds the model a conversation with an existing response, generates one more token,
    and captures SAE activations at ALL positions.

    Returns:
        Dict with 'max', 'mean', 'sum' pooled activations (each shape: [num_latents])
    """
    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": existing_response},
    ]

    prompt_token_ids = engine.tokenizer.apply_chat_template(convo)

    from vllm.inputs import TokenInputs
    from vllm import SamplingParams
    import uuid

    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=convo)

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1,  # Only generate 1 token
        repetition_penalty=1.0,
    )

    request_id = str(uuid.uuid4())
    results_generator = engine.engine.generate(
        prompt=token_inputs,
        sampling_params=sampling_params,
        request_id=request_id,
        interventions=None,
        is_feature_decode=True,  # Enable feature extraction
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # Extract feature activations
    # feature_tensor shape: [num_tokens, num_latents]
    features = final_output.feature_tensor.cpu()

    # Max-pool across all token positions
    return features.max(dim=0)[0]  # [num_latents]


def generate_derangement(n: int, seed: int = None) -> list[int]:
    """Generate a derangement: a permutation where no element appears in its original position."""
    if seed is not None:
        random.seed(seed)
    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if all(i != x for i, x in enumerate(perm)):
            return perm


async def main(prompts_file: Path, output_dir: Path):
    """Collect all latent activations for shuffled and non-shuffled pairs."""

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("Collecting SAE latent activations for off-topic detector analysis")
    print("=" * 80)
    print(f"Prompts file: {prompts_file}")
    print(f"Output dir: {output_dir}")

    # Initialize engine
    print("\nInitializing vLLM engine...")
    engine = VLLMSteeringEngine(MODEL_NAME)
    await engine.initialize()
    print("Engine initialized")

    # Load prompts
    prompts = load_prompts(prompts_file)
    print(f"\nLoaded {len(prompts)} prompts")
    
    # Step 1: Generate unsteered responses for all prompts (or load from cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "normal_responses.json"
    
    if responses_file.exists():
        print("\n=== Step 1: Loading cached responses ===")
        with open(responses_file) as f:
            cached = json.load(f)
        normal_responses = cached["normal_responses"]
        print(f"✓ Loaded {len(normal_responses)} cached responses from {responses_file}")
    else:
        print("\n=== Step 1: Generating unsteered responses ===")
        normal_responses = {}
        
        for i, prompt in enumerate(tqdm_asyncio(prompts, desc="Generating responses")):
            # Use deterministic seed for each prompt
            seed = RANDOM_SEED + i
            response = await generate_unsteered_response(
                engine, prompt, max_tokens=MAX_COMPLETION_TOKENS, seed=seed
            )
            normal_responses[prompt] = response
        
        print(f"✓ Generated {len(normal_responses)} responses")
        
        # Save responses immediately (in case we crash later)
        with open(responses_file, "w") as f:
            json.dump({"prompts": prompts, "normal_responses": normal_responses}, f, indent=2)
        print(f"✓ Saved responses to {responses_file}")
    
    # Step 2: Create shuffled pairs using multiple derangements
    print(f"\n=== Step 2: Creating shuffled pairs ({NUM_DERANGEMENTS} derangements) ===")

    shuffled_pairs = []
    all_derangement_indices = []

    for d in range(NUM_DERANGEMENTS):
        # Use different seed for each derangement
        derangement_seed = RANDOM_SEED + d * 100
        shuffled_indices = generate_derangement(len(prompts), seed=derangement_seed)
        all_derangement_indices.append(shuffled_indices)
        print(f"Derangement {d+1}: {shuffled_indices}")

        for i, prompt in enumerate(prompts):
            other_prompt = prompts[shuffled_indices[i]]
            other_response = normal_responses[other_prompt]
            shuffled_pairs.append((prompt, other_response))

    # Shuffle the final list order
    random.seed(RANDOM_SEED + 1000)
    random.shuffle(shuffled_pairs)

    print(f"Created {NUM_DERANGEMENTS} derangements with {len(prompts)} pairs each")

    # Step 3: Get activations for normal pairs (only need to do once)
    print("\n=== Step 3: Getting activations for normal (on-topic) pairs ===")

    normal_activations = []
    for prompt in tqdm_asyncio(prompts, desc="Normal pairs"):
        try:
            response = normal_responses[prompt]
            activations = await get_sae_activations_all_positions(engine, prompt, response)
            normal_activations.append(activations)
        except Exception as e:
            print(f"Error getting activations for normal pair: {e}")
            continue

    if not normal_activations:
        raise ValueError("Failed to get any normal activations")

    normal_tensor = torch.stack(normal_activations)
    print(f"✓ Collected activations for {len(normal_activations)} normal pairs")
    print(f"  Shape: {normal_tensor.shape}")

    # Step 4: Get activations for each derangement separately
    print("\n=== Step 4: Getting activations for shuffled pairs (each derangement) ===")

    shuffled_by_derangement = {}
    for d in range(NUM_DERANGEMENTS):
        derangement_seed = RANDOM_SEED + d * 100
        shuffled_indices = all_derangement_indices[d]

        # Create pairs for this derangement
        pairs = []
        for i, prompt in enumerate(prompts):
            other_prompt = prompts[shuffled_indices[i]]
            other_response = normal_responses[other_prompt]
            pairs.append((prompt, other_response))

        # Collect activations
        activations = []
        for prompt, response in tqdm_asyncio(pairs, desc=f"Derangement {d+1}/{NUM_DERANGEMENTS}"):
            try:
                act = await get_sae_activations_all_positions(engine, prompt, response)
                activations.append(act)
            except Exception as e:
                print(f"Error: {e}")
                continue

        shuffled_by_derangement[d] = torch.stack(activations)
        print(f"  Derangement {d+1}: {shuffled_by_derangement[d].shape}")

    # Step 5: Save everything
    print("\n=== Step 5: Saving data ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save activations - normal once, shuffled per derangement
    output_file = output_dir / "activations_all_latents.npz"
    save_dict = {
        "normal": normal_tensor.float().numpy(),
        "num_derangements": NUM_DERANGEMENTS,
    }
    for d in range(NUM_DERANGEMENTS):
        save_dict[f"shuffled_{d}"] = shuffled_by_derangement[d].float().numpy()

    np.savez_compressed(output_file, **save_dict)
    print(f"✓ Saved activation tensors to {output_file}")
    
    # Save metadata as JSON
    metadata_file = output_dir / "activations_metadata.json"
    metadata = {
        "model_name": MODEL_NAME,
        "prompts_file": str(prompts_file),
        "num_prompts": len(prompts),
        "num_derangements": NUM_DERANGEMENTS,
        "prompts": prompts,
        "normal_responses": normal_responses,
        "derangement_indices": all_derangement_indices,
        "random_seed": RANDOM_SEED,
        "num_latents": normal_tensor.shape[1],
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Number of derangements: {NUM_DERANGEMENTS}")
    print(f"Number of latents: {normal_tensor.shape[1]}")
    print(f"Normal samples: {normal_tensor.shape[0]}")
    print(f"Shuffled samples per derangement: {shuffled_by_derangement[0].shape[0]}")

    print("\n✓ Done! Run analyze_activations.py next to analyze the results.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect SAE latent activations for OTD analysis")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_FILE,
                        help="Path to prompts file (default: prompts.txt)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    asyncio.run(main(args.prompts, args.output_dir))
