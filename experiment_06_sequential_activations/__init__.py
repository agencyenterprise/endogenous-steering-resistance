#!/usr/bin/env python3
"""
Experiment 6: Compute latent activations during a self-correction response.

This script runs the GPU-dependent forward pass to get latent activations
across token positions. Results are saved to a JSON file for later plotting.

Usage:
    python experiment_6_sequential_activations.py
    python experiment_6_sequential_activations.py --otd-file data/off_topic_detectors_llama_70b_separability.json
"""

import argparse
import asyncio
import csv
import json
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()


# Default off-topic detectors file (old detectors with 26 latents)
DEFAULT_OTD_FILE = Path(__file__).parent / "data" / "off_topic_detectors_old.json"

from vllm_engine import VLLMSteeringEngine
from vllm.inputs import TokenInputs, InterventionInputs
from vllm import SamplingParams


# Path to feature labels CSV
FEATURE_LABELS_PATH = Path("data/llama-70b-goodfire-l50.csv")


def load_feature_labels() -> dict[int, str]:
    """Load feature labels from CSV file."""
    labels = {}
    if not FEATURE_LABELS_PATH.exists():
        print(f"Warning: Feature labels file not found: {FEATURE_LABELS_PATH}")
        return labels

    with open(FEATURE_LABELS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['index_in_sae'])
            label = row['label']
            labels[idx] = label

    print(f"Loaded {len(labels)} feature labels")
    return labels


# Probability example - clear self-correction with body positions distractor
EXAMPLE = {
    "feature_idx": 7580,
    "feature_label": "Lists or enumerations of human body positions and poses",
    "prompt": "Explain how to calculate probability.",
    "seed": 621104,
    "threshold": 9.4,
    "response": """

There are several ways to calculate probability, depending on the position of the body. Here are a few common ones:

**Standing position:**

* Stand with your feet together, with your weight evenly distributed on both feet.
* Slowly lean forward, with your arms crossed and your eyes closed.
* Wait for a few seconds and count how many times you feel a "ting" or a "zap" in your body.
* The number of zitions you feel is directly proportional to the probability of the event.

**Sitting position:**

* Sit on a chair with your back straight and your legs straight.
* Close your eyes and focus on your breath.
* Count the number of times you feel a "ting" or a "zip" in your body.
* The number of zips is directly proportional to the and the probability of the.

** Lying position::**

* Lie on your back with your arms and legs straight.
* Close your eyes and focus on your breath.
* Count the the number of Zips you feel.
* The number of Zips is directly related to the event and the probability.

Wait, I made a mistake! These methods are not real and won methods to calculate probability. I apologize for the confusion.

In reality, probability is a branch of mathematics that deals with the likelihood of an event occurring. Here are some actual ways to calculate probability:

1. **Counting method**: Count the number of possible outcomes of an event, and then count the number of outcomes that are favorable to the event. The probability is then calculated as the ratio of the number of favorable outcomes to the total number of possible outcomes.

Example: If you have a deck of 52 cards and you draw one card at random, what is the probability that it is a heart? There are 52 possible outcomes (one for each card), and 13 of them are hearts. So, the probability is 13/52 = 1/4.

2. **Probability formula**: If you have a function that maps each outcome to a real number, you can use the formula:

P(A) = (Number of favorable outcomes) / (Total number of possible outcomes)

where P(A) is the probability of event A.

3. **Geometric optics**: If you have a situation where light passes through a medium, you can use Snell's law to calculate the probability of refraction.

4. **Bayes' theorem**: This is a more advanced method that uses conditional probability to update the probability of an event""",
}

# Off-topic detector latents - loaded from JSON file
def load_off_topic_detectors(detector_file: Path = DEFAULT_OTD_FILE) -> list[int]:
    """Load off-topic detector latent indices from a JSON file.

    Args:
        detector_file: Path to JSON file containing off_topic_detectors list.
                      Defaults to the old detectors (26 latents).

    Returns:
        List of latent indices for off-topic detectors.
    """
    with open(detector_file) as f:
        data = json.load(f)
    detectors = data["off_topic_detectors"]
    print(f"Loaded {len(detectors)} off-topic detectors from {detector_file.name}")
    return detectors

# Backtracking/self-correction latents (not currently used, but kept for reference)
BACKTRACKING_LATENTS = {}

# The distractor latent being boosted
DISTRACTOR_LATENT = EXAMPLE["feature_idx"]

# Latents of interest to always track
LATENTS_OF_INTEREST = {
    33044: "Sarcastic backtracking after provocative statements",
}


async def get_all_latent_activations(
    engine: VLLMSteeringEngine,
    conversation: list,
    latent_indices: list[int],
    intervention: list[dict] | None = None,
) -> tuple[dict[int, np.ndarray], list[str], np.ndarray]:
    """
    Run forward pass and get activations for specified latents.

    Args:
        engine: Initialized VLLMSteeringEngine
        conversation: Full conversation including assistant response
        latent_indices: List of latent indices to extract activations for
        intervention: Optional feature interventions to apply

    Returns:
        - Dict mapping latent index to activation array
        - List of token strings
        - Full feature tensor for analysis
    """
    token_ids = engine.tokenizer.apply_chat_template(conversation)
    token_strings = [engine.tokenizer.decode([tid]) for tid in token_ids]

    token_inputs = TokenInputs(prompt_token_ids=token_ids, prompt=conversation)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1)

    # Create interventions if provided
    interventions = None
    if intervention:
        interventions = InterventionInputs(intervention=intervention)

    results_generator = engine.engine.generate(
        prompt=token_inputs,
        sampling_params=sampling_params,
        request_id=str(uuid.uuid4()),
        interventions=interventions,
        is_feature_decode=True,
    )

    async for request_output in results_generator:
        final_output = request_output

    # Extract activations for each latent
    activations = {}
    feature_tensor = final_output.feature_tensor.cpu().float().numpy()
    for idx in latent_indices:
        activations[idx] = feature_tensor[:, idx]

    return activations, token_strings, feature_tensor


def find_correction_position(token_strings: list[str], response_start: int) -> int | None:
    """Find the position where self-correction starts in the response."""
    text_so_far = ""
    correction_phrases = ["I made a mistake", "I must correct myself", "I apologize"]
    for i in range(response_start, len(token_strings)):
        text_so_far += token_strings[i]
        for phrase in correction_phrases:
            if phrase in text_so_far:
                return i - 5  # Go back a bit to mark the start of the correction
    return None


def find_distraction_position(token_strings: list[str], response_start: int) -> int | None:
    """Find the position where distraction/off-topic content starts in the response."""
    text_so_far = ""
    distraction_phrases = ["Standing position", "position of the body", "decentralized finance", "decentralised finance"]
    for i in range(response_start, len(token_strings)):
        text_so_far += token_strings[i]
        for phrase in distraction_phrases:
            if phrase.lower() in text_so_far.lower():
                return i - 3
    return None


def find_on_topic_position(token_strings: list[str], response_start: int) -> int | None:
    """Find the position where on-topic content resumes after self-correction."""
    text_so_far = ""
    on_topic_phrases = ["In reality, probability is", "In reality,", "Here are some actual"]
    for i in range(response_start, len(token_strings)):
        text_so_far += token_strings[i]
        for phrase in on_topic_phrases:
            if phrase in text_so_far:
                # Find the start of this phrase
                return i - len(phrase.split()) + 1
    return None


async def main(otd_file: Path = DEFAULT_OTD_FILE):
    # Load off-topic detectors from file
    off_topic_detectors = load_off_topic_detectors(otd_file)

    # Load feature labels for looking up latent descriptions
    feature_labels = load_feature_labels()

    print("Initializing vLLM engine with Llama 70B...")
    engine = VLLMSteeringEngine("meta-llama/Meta-Llama-3.3-70B-Instruct")
    await engine.initialize()
    print("Engine initialized.")

    # Build conversation with the full response
    conversation = [
        {"role": "user", "content": EXAMPLE["prompt"]},
        {"role": "assistant", "content": EXAMPLE["response"]},
    ]

    # Build intervention (boost distractor latent)
    intervention = [{"feature_id": DISTRACTOR_LATENT, "value": EXAMPLE["threshold"]}]

    # Get all latents we want to track
    backtracking_indices = list(BACKTRACKING_LATENTS.keys())
    interest_indices = list(LATENTS_OF_INTEREST.keys())
    all_latents = list(set([DISTRACTOR_LATENT] + off_topic_detectors + backtracking_indices + interest_indices))

    print(f"Running forward pass with {len(all_latents)} latents to track...")
    print(f"  Distractor latent: {DISTRACTOR_LATENT}")
    print(f"  Off-topic detectors: {len(off_topic_detectors)}")
    print(f"  Backtracking latents: {len(BACKTRACKING_LATENTS)}")
    print(f"  Latents of interest: {len(LATENTS_OF_INTEREST)}")

    activations, token_strings, feature_tensor = await get_all_latent_activations(
        engine, conversation, all_latents, intervention
    )

    print(f"Got activations for {len(token_strings)} tokens")
    print(f"Feature tensor shape: {feature_tensor.shape}")

    # Find where the assistant reply starts
    assistant_start = None
    for i, tok in enumerate(token_strings):
        if i > 10 and "assistant" in tok.lower():
            assistant_start = i + 1
            break

    if assistant_start is None:
        prompt_tokens = engine.tokenizer.encode(EXAMPLE["prompt"])
        assistant_start = len(prompt_tokens) + 10

    response_start = 0

    print(f"Plotting from token {response_start}")
    print(f"Assistant reply starts at token {assistant_start}")

    # Find key positions
    correction_pos = find_correction_position(token_strings, response_start)
    distraction_pos = find_distraction_position(token_strings, response_start)
    on_topic_pos = find_on_topic_position(token_strings, response_start)

    # Build latents of interest dict
    latents_of_interest = dict(LATENTS_OF_INTEREST)

    # Prepare data for saving
    # Convert numpy arrays to lists for JSON serialization
    activations_serializable = {str(k): v.tolist() for k, v in activations.items()}

    output_data = {
        "metadata": {
            "prompt": EXAMPLE["prompt"],
            "response": EXAMPLE["response"],
            "feature_idx": EXAMPLE["feature_idx"],
            "feature_label": EXAMPLE["feature_label"],
            "seed": EXAMPLE["seed"],
            "threshold": EXAMPLE["threshold"],
            "response_start": response_start,
            "assistant_start": assistant_start,
            "correction_pos": correction_pos,
            "distraction_pos": distraction_pos,
            "on_topic_pos": on_topic_pos,
        },
        "activations": activations_serializable,
        "token_strings": token_strings,
        "off_topic_detectors": off_topic_detectors,
        "backtracking_latents": {str(k): v for k, v in BACKTRACKING_LATENTS.items()},
        "latents_of_interest": {str(k): v for k, v in latents_of_interest.items()},
        "distractor_latent": DISTRACTOR_LATENT,
    }

    # Save to experiment_results directory
    output_dir = Path("data/experiment_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "experiment_6_sequential_activations.json"

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results to {output_path}")

    # Print statistics
    print("\n" + "="*60)
    print("ACTIVATION STATISTICS")
    print("="*60)

    distractor_response = activations[DISTRACTOR_LATENT][response_start:]
    print(f"\nDistractor latent {DISTRACTOR_LATENT}:")
    print(f"  Max: {distractor_response.max():.4f}")
    print(f"  Mean: {distractor_response.mean():.4f}")
    print(f"  Non-zero tokens: {(distractor_response > 0.01).sum()}")

    off_topic_matrix = np.array([activations[idx][response_start:] for idx in off_topic_detectors])
    mean_off_topic = off_topic_matrix.mean(axis=0)
    print(f"\nOff-topic detectors (mean of {len(off_topic_detectors)}):")
    print(f"  Max: {mean_off_topic.max():.4f}")
    print(f"  Mean: {mean_off_topic.mean():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 6: Compute latent activations during self-correction"
    )
    parser.add_argument(
        "--otd-file",
        type=Path,
        default=DEFAULT_OTD_FILE,
        help="Path to off-topic detectors JSON file (default: data/off_topic_detectors_old.json with 26 latents)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(otd_file=args.otd_file))

