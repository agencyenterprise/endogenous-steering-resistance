"""
Extract self-correction episodes from experiment results.

This script identifies and extracts episodes where the model successfully
self-corrected within a single response.

Filtering criteria:
1. len(attempts) == 2 (simple self-correction, easier to analyze boundaries)
2. attempts[-1].score > attempts[0].score (successful self-correction - model improved)
3. Omit episodes with >2 attempts (complex multi-correction)
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import glob


@dataclass
class Episode:
    """A self-correction episode extracted from experiment results."""
    episode_id: str  # Unique identifier
    source_file: str  # Original experiment file
    prompt: str  # User prompt
    response: str  # Full model response
    feature_label: str  # Label of the steering feature
    feature_idx: int  # SAE index of the steering feature
    threshold: float  # Steering threshold used
    seed: int  # Random seed for reproducibility
    first_attempt_score: int  # Score of first segment
    final_attempt_score: int  # Score of final segment
    first_attempt_text: str  # Text of first attempt
    final_attempt_text: str  # Text of final attempt


def extract_episodes(results_dir: Path, model_pattern: str = "*70B*") -> list[Episode]:
    """
    Extract self-correction episodes from experiment result files.

    Args:
        results_dir: Directory containing experiment result JSON files
        model_pattern: Glob pattern to filter model-specific files

    Returns:
        List of Episode objects
    """
    episodes = []
    episode_counter = 0

    # Find all matching result files
    pattern = f"experiment_results_{model_pattern}.json"
    result_files = list(results_dir.glob(pattern))

    print(f"Found {len(result_files)} result files matching pattern: {pattern}")

    for result_file in sorted(result_files):
        # Skip baseline files (no steering)
        if "no_steering_baseline" in result_file.name:
            print(f"  Skipping baseline: {result_file.name}")
            continue

        print(f"  Processing: {result_file.name}")

        with open(result_file) as f:
            data = json.load(f)

        file_episodes = 0

        for feature in data.get("results_by_feature", []):
            for trial in feature.get("trials", []):
                score = trial.get("score", {})
                attempts = score.get("attempts", [])

                # Skip if not exactly 2 attempts
                if len(attempts) != 2:
                    continue

                first_score = attempts[0].get("score", 0)
                last_score = attempts[-1].get("score", 0)

                # Skip if not successful self-correction (must improve)
                if last_score <= first_score:
                    continue

                episode_counter += 1
                file_episodes += 1

                episode = Episode(
                    episode_id=f"ep_{episode_counter:04d}",
                    source_file=result_file.name,
                    prompt=trial.get("prompt", ""),
                    response=trial.get("response", ""),
                    feature_label=trial.get("feature_label", ""),
                    feature_idx=trial.get("feature_index_in_sae", 0),
                    threshold=trial.get("threshold", 0.0),
                    seed=trial.get("seed", 0),
                    first_attempt_score=first_score,
                    final_attempt_score=last_score,
                    first_attempt_text=attempts[0].get("attempt_text", ""),
                    final_attempt_text=attempts[-1].get("attempt_text", ""),
                )
                episodes.append(episode)

        print(f"    Found {file_episodes} successful 2-attempt self-corrections")

    return episodes


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge"
    output_dir = base_dir / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "activation_stats"
    output_file = output_dir / "episodes.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting self-correction episodes from: {results_dir}")
    print()

    # Extract episodes
    episodes = extract_episodes(results_dir)

    print()
    print(f"Total episodes extracted: {len(episodes)}")

    # Statistics
    if episodes:
        scores_improved = [e.final_attempt_score - e.first_attempt_score for e in episodes]
        print(f"\nScore improvement statistics:")
        print(f"  Min improvement: {min(scores_improved)}")
        print(f"  Max improvement: {max(scores_improved)}")
        print(f"  Mean improvement: {sum(scores_improved) / len(scores_improved):.1f}")

        # Show a few examples
        print("\nSample episodes:")
        for ep in episodes[:3]:
            print(f"\n  {ep.episode_id}:")
            print(f"    Prompt: {ep.prompt[:60]}...")
            print(f"    Feature: {ep.feature_label[:50]}...")
            print(f"    Scores: {ep.first_attempt_score} -> {ep.final_attempt_score}")
            print(f"    Response preview: {ep.response[:100]}...")

    # Save to JSON
    output_data = {
        "n_episodes": len(episodes),
        "filter_criteria": {
            "n_attempts": 2,
            "requires_improvement": True,
            "model_pattern": "*70B*",
        },
        "episodes": [asdict(ep) for ep in episodes],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved episodes to: {output_file}")


if __name__ == "__main__":
    main()
