"""
Collect baseline activations from non-self-correction episodes.

These are single-attempt trials where no self-correction occurred,
used for comparison with self-correction episodes.
"""

import asyncio
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AGI-1516-esr-with-vllm"))

from vllm_engine import VLLMSteeringEngine
from vllm.inputs import TokenInputs
from vllm import SamplingParams


# Same latents as collect_activations.py
OFF_TOPIC_DETECTORS = [
    6527, 7517, 9005, 10304, 11390, 17250, 17481, 17516, 23093, 24684,
    26312, 28403, 28540, 37234, 37536, 38956, 39926, 40119, 40792, 44845,
    45078, 56830, 58565, 59483, 61420
]

BACKTRACKING_LATENTS = [
    5852, 18311, 45478, 52694, 57675, 63162, 3994, 3473, 5215, 7318,
    53491, 890, 1719, 28564, 34597, 33044
]


def extract_non_self_correction_episodes(results_dir: Path, n_samples: int = 150) -> list[dict]:
    """Extract non-self-correction episodes (single attempt, good score)."""
    import glob

    episodes = []

    for f in sorted(results_dir.glob("experiment_results_Meta-Llama-3.3-70B-Instruct_*.json")):
        if "no_steering_baseline" in f.name:
            continue

        with open(f) as fp:
            data = json.load(fp)

        for feature in data.get("results_by_feature", []):
            for trial in feature.get("trials", []):
                score = trial.get("score", {})
                attempts = score.get("attempts", [])

                # Single attempt with decent score (not off-topic)
                if len(attempts) == 1 and attempts[0].get("score", 0) >= 50:
                    episodes.append({
                        "prompt": trial.get("prompt", ""),
                        "response": trial.get("response", ""),
                        "feature_label": trial.get("feature_label", ""),
                        "score": attempts[0].get("score", 0),
                    })

    # Random sample
    random.seed(42)
    if len(episodes) > n_samples:
        episodes = random.sample(episodes, n_samples)

    return episodes


async def get_activations(
    engine: VLLMSteeringEngine,
    conversation: list,
    latent_indices: list[int],
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Run forward pass and get activations."""
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

    activations = {}
    feature_tensor = final_output.feature_tensor.cpu().float().numpy()
    for idx in latent_indices:
        activations[idx] = feature_tensor[:, idx]

    return activations, token_strings


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of baseline episodes")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    results_dir = base_dir.parent / "AGI-1516-esr-with-vllm" / "experiment_results"
    output_file = base_dir / "data" / "self-correction" / "baseline_stats.json"

    print(f"Extracting non-self-correction episodes...")
    episodes = extract_non_self_correction_episodes(results_dir, n_samples=args.limit)
    print(f"Found {len(episodes)} baseline episodes")

    all_latents = list(set(OFF_TOPIC_DETECTORS + BACKTRACKING_LATENTS))

    print("Initializing vLLM engine...")
    engine = VLLMSteeringEngine("meta-llama/Meta-Llama-3.3-70B-Instruct")
    await engine.initialize()
    print("Engine initialized.\n")

    # Collect mean activations for each episode
    otd_means = []
    bt_means = []

    for i, ep in enumerate(episodes):
        print(f"[{i+1}/{len(episodes)}] Processing baseline episode...")

        conversation = [
            {"role": "user", "content": ep["prompt"]},
            {"role": "assistant", "content": ep["response"]},
        ]

        activations, token_strings = await get_activations(engine, conversation, all_latents)

        # Find response start
        response_start = 0
        for j, tok in enumerate(token_strings):
            if j > 10 and "assistant" in tok.lower():
                response_start = j + 1
                break

        # Mean OTD activation in response
        otd_acts = [activations[idx][response_start:] for idx in OFF_TOPIC_DETECTORS]
        otd_mean = np.mean([np.mean(a) for a in otd_acts])
        otd_means.append(otd_mean)

        # Mean backtracking activation in response
        bt_acts = [activations[idx][response_start:] for idx in BACKTRACKING_LATENTS]
        bt_mean = np.mean([np.mean(a) for a in bt_acts])
        bt_means.append(bt_mean)

    # Save results
    results = {
        "n_episodes": len(episodes),
        "otd_mean": float(np.mean(otd_means)),
        "otd_std": float(np.std(otd_means)),
        "bt_mean": float(np.mean(bt_means)),
        "bt_std": float(np.std(bt_means)),
        "otd_per_episode": [float(x) for x in otd_means],
        "bt_per_episode": [float(x) for x in bt_means],
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline statistics:")
    print(f"  OTD mean: {results['otd_mean']:.4f} +/- {results['otd_std']:.4f}")
    print(f"  Backtracking mean: {results['bt_mean']:.4f} +/- {results['bt_std']:.4f}")
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
