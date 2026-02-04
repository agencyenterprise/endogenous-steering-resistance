"""
Collect SAE latent activations for self-correction episodes.

This script runs vLLM forward passes to collect per-token activations
for off-topic detectors and backtracking latents.

Usage:
    python collect_activations.py [--limit N]
"""

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for vllm_engine import
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_engine import VLLMSteeringEngine
from vllm.inputs import TokenInputs
from vllm import SamplingParams


# Off-topic detector latents - loaded from data/off_topic_detectors_old.json
DEFAULT_OTD_FILE = Path(__file__).parent.parent / "data" / "off_topic_detectors_old.json"

def _load_off_topic_detectors(detector_file: Path = DEFAULT_OTD_FILE) -> list[int]:
    with open(detector_file) as f:
        data = json.load(f)
    return data["off_topic_detectors"]

OFF_TOPIC_DETECTORS = _load_off_topic_detectors()

# Top backtracking latents (score >= 5 from find_backtracking_latents.py)
# Selected for high relevance to self-correction behavior
BACKTRACKING_LATENTS = [
    5852,   # "The assistant needs to apologize and correct a mistake" (score=11)
    18311,  # "The assistant needs to apologize and offer further assistance after providing incorrect information" (score=8)
    45478,  # "The assistant needs to apologize for providing incorrect information" (score=8)
    52694,  # "The assistant is apologizing and correcting a mistake in its previous response" (score=8)
    57675,  # "The assistant's formal apology format when correcting mistakes" (score=8)
    63162,  # "The assistant is apologizing and correcting previous mistakes" (score=8)
    3994,   # "The assistant should acknowledge limitations or apologize for inadequate responses" (score=7)
    3473,   # "Assistant taking ownership or responsibility for mistakes" (score=6)
    5215,   # "The assistant is correcting a mistake or acknowledging an error" (score=6)
    7318,   # "The assistant needs to acknowledge confusion or correct a mistake" (score=6)
    53491,  # "The assistant needs clarification or realizes it made a mistake" (score=6)
    890,    # "The assistant is being careful and precise with statements, often when correcting previous errors" (score=5)
    1719,   # "The assistant needs to correct a previous incorrect statement" (score=5)
    28564,  # "The assistant is acknowledging and correcting its mistakes" (score=5)
    34597,  # "The assistant is correcting a previous mistake or misconception" (score=5)
    33044,  # "Sarcastic backtracking after provocative statements" (known from experiment_6)
]


@dataclass
class EpisodeActivations:
    """Activation data for a single episode."""
    episode_id: str
    prompt: str
    response: str
    token_strings: list[str]

    # Boundary positions (token indices)
    response_start_token: int
    off_topic_start_token: int
    correction_start_token: int
    on_topic_start_token: int

    # Character positions (from annotation)
    off_topic_char_start: int
    correction_char_start: int
    on_topic_char_start: int

    # Activations stored separately due to size


def char_to_token_position(
    token_strings: list[str],
    response_start: int,
    char_pos: int,
) -> int:
    """
    Map a character position in the response to a token position.

    Args:
        token_strings: List of decoded token strings for the full input
        response_start: Token index where the response begins
        char_pos: Character position within the response

    Returns:
        Token index
    """
    if char_pos < 0:
        return -1

    # Reconstruct response text token by token
    current_char = 0
    for token_idx in range(response_start, len(token_strings)):
        if current_char >= char_pos:
            return token_idx
        current_char += len(token_strings[token_idx])

    return len(token_strings) - 1


async def get_activations(
    engine: VLLMSteeringEngine,
    conversation: list,
    latent_indices: list[int],
) -> tuple[dict[int, np.ndarray], list[str]]:
    """
    Run forward pass and get activations for specified latents.

    Args:
        engine: Initialized VLLMSteeringEngine
        conversation: Full conversation including assistant response
        latent_indices: List of latent indices to extract activations for

    Returns:
        - Dict mapping latent index to activation array
        - List of token strings
    """
    token_ids = engine.tokenizer.apply_chat_template(conversation)
    token_strings = [engine.tokenizer.decode([tid]) for tid in token_ids]

    token_inputs = TokenInputs(prompt_token_ids=token_ids, prompt=conversation)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1)

    results_generator = engine.engine.generate(
        prompt=token_inputs,
        sampling_params=sampling_params,
        request_id=str(uuid.uuid4()),
        is_feature_decode=True,
    )

    async for request_output in results_generator:
        final_output = request_output

    # Extract activations for each latent
    activations = {}
    feature_tensor = final_output.feature_tensor.cpu().float().numpy()
    for idx in latent_indices:
        activations[idx] = feature_tensor[:, idx]

    return activations, token_strings


def find_response_start(token_strings: list[str]) -> int:
    """Find the token index where the assistant response begins."""
    for i, tok in enumerate(token_strings):
        if i > 10 and "assistant" in tok.lower():
            return i + 1
    # Fallback: look for common chat template markers
    for i, tok in enumerate(token_strings):
        if "<|start_header_id|>" in tok and i + 2 < len(token_strings):
            if "assistant" in token_strings[i + 1].lower():
                return i + 3
    return 0


async def collect_all_activations(
    episodes_file: Path,
    output_file: Path,
    activations_dir: Path,
    limit: Optional[int] = None,
):
    """Collect activations for all episodes."""

    # Load annotated episodes
    with open(episodes_file) as f:
        data = json.load(f)

    episodes = [ep for ep in data["episodes"] if ep.get("boundary_annotations", {}).get("valid", False)]
    if limit:
        episodes = episodes[:limit]

    print(f"Processing {len(episodes)} valid episodes")

    # All latents to track
    all_latents = list(set(OFF_TOPIC_DETECTORS + BACKTRACKING_LATENTS))
    print(f"Tracking {len(all_latents)} latents ({len(OFF_TOPIC_DETECTORS)} OTDs + {len(BACKTRACKING_LATENTS)} backtracking)")

    # Initialize engine
    print("Initializing vLLM engine...")
    engine = VLLMSteeringEngine("meta-llama/Meta-Llama-3.3-70B-Instruct")
    await engine.initialize()
    print("Engine initialized.\n")

    # Process episodes
    results = []
    all_activations = {}

    for i, ep in enumerate(episodes):
        episode_id = ep["episode_id"]
        print(f"[{i+1}/{len(episodes)}] Processing {episode_id}...")

        # Build conversation
        conversation = [
            {"role": "user", "content": ep["prompt"]},
            {"role": "assistant", "content": ep["response"]},
        ]

        # Get activations
        activations, token_strings = await get_activations(engine, conversation, all_latents)

        # Find response start
        response_start = find_response_start(token_strings)

        # Get boundary annotations
        boundaries = ep["boundary_annotations"]

        # Convert character positions to token positions
        off_topic_token = char_to_token_position(
            token_strings, response_start, boundaries["off_topic_char_start"]
        )
        correction_token = char_to_token_position(
            token_strings, response_start, boundaries["correction_char_start"]
        )
        on_topic_token = char_to_token_position(
            token_strings, response_start, boundaries["on_topic_char_start"]
        )

        # Store metadata
        episode_data = EpisodeActivations(
            episode_id=episode_id,
            prompt=ep["prompt"],
            response=ep["response"],
            token_strings=token_strings,
            response_start_token=response_start,
            off_topic_start_token=off_topic_token,
            correction_start_token=correction_token,
            on_topic_start_token=on_topic_token,
            off_topic_char_start=boundaries["off_topic_char_start"],
            correction_char_start=boundaries["correction_char_start"],
            on_topic_char_start=boundaries["on_topic_char_start"],
        )
        results.append(asdict(episode_data))

        # Store activations (convert to lists for JSON serialization)
        all_activations[episode_id] = {
            str(k): v.tolist() for k, v in activations.items()
        }

        print(f"    Tokens: {len(token_strings)}, Response start: {response_start}")
        print(f"    Boundaries (token): off_topic={off_topic_token}, correction={correction_token}, on_topic={on_topic_token}")

        # Save progress periodically
        if (i + 1) % 5 == 0:
            save_results(results, all_activations, output_file, activations_dir)

    # Final save
    save_results(results, all_activations, output_file, activations_dir)

    print(f"\nCollection complete: {len(results)} episodes")


def save_results(results: list, activations: dict, output_file: Path, activations_dir: Path):
    """Save results and activations to files."""

    # Save metadata
    output_data = {
        "n_episodes": len(results),
        "off_topic_detectors": OFF_TOPIC_DETECTORS,
        "backtracking_latents": BACKTRACKING_LATENTS,
        "episodes": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Save activations as compressed numpy
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays for efficient storage
    for episode_id, acts in activations.items():
        ep_file = activations_dir / f"{episode_id}.npz"
        np.savez_compressed(
            ep_file,
            **{k: np.array(v) for k, v in acts.items()}
        )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect SAE activations for self-correction episodes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of episodes to process")
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "activation_stats"
    episodes_file = output_dir / "episodes_annotated.json"
    output_file = output_dir / "episode_metadata.json"
    activations_dir = output_dir / "activations"

    if not episodes_file.exists():
        print(f"Error: Annotated episodes file not found: {episodes_file}")
        print("Please run annotate_boundaries.py first.")
        return

    await collect_all_activations(episodes_file, output_file, activations_dir, limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
