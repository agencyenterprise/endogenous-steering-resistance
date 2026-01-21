#!/usr/bin/env python3
"""
Plot latent activations during a self-correction response.

Loads pre-computed activation data from experiment_6_sequential_activations.py
and creates visualization showing how the distractor latent and off-topic
detector latents activate across token positions during a response.

Usage:
    python plot_exp6.py
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Base directory for experiment data
BASE_DIR = Path(__file__).parent.parent


def _resolve_output_dir(output_dir: Path) -> Path:
    """Resolve a user-provided output dir relative to the experiment base dir."""
    return output_dir if output_dir.is_absolute() else (BASE_DIR / output_dir)


def exponential_smoothing(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential smoothing to data."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_activations(
    activations: dict[int, np.ndarray],
    token_strings: list[str],
    response_start: int,
    output_path: Path,
    boost_level: float,
    distractor_latent: int,
    distractor_label: str,
    off_topic_detectors: list[int],
    backtracking_latents: dict[int, str],
    latents_of_interest: dict[int, str],
    assistant_start: int | None = None,
    correction_pos: int | None = None,
    distraction_pos: int | None = None,
    on_topic_pos: int | None = None,
):
    """
    Create 3-row plot showing latent activations over token positions.
    
    Row 1 (top): Labels for temporal regions
    Row 2 (middle): Off-topic detectors (sum)
    Row 3 (bottom): Sarcastic backtracking latent
    """
    # Filter out off-topic detectors that don't fire at all
    active_detectors = []
    for idx in off_topic_detectors:
        if idx in latents_of_interest:
            continue
        detector_acts = activations[idx][response_start:]
        if detector_acts.max() > 0.001:
            active_detectors.append(idx)

    print(f"Active off-topic detectors: {len(active_detectors)} / {len(off_topic_detectors)}")

    # Only plot response portion
    response_length = len(token_strings) - response_start
    x = np.arange(response_length)

    # Calculate relative positions
    assistant_start_relative = assistant_start - response_start if assistant_start is not None else None
    correction_pos_relative = correction_pos - response_start if correction_pos is not None else None
    on_topic_pos_relative = on_topic_pos - response_start if on_topic_pos is not None else None

    # Smoothing coefficient (lower = more smoothing, 1 = no smoothing)
    alpha = 0.5

    # Region colors (colorblind-safe)
    color_prompt = '#E8E8E8'      # Light gray for user prompt area
    color_offtopic = '#9B59B6'    # Purple for off-topic
    color_correction = '#E67E22'  # Orange for self-correction
    color_ontopic = '#2980B9'     # Blue for on-topic
    region_alpha = 0.2

    # Create 3-row figure with shared x-axis
    # Heights: top row (labels) is shorter, middle and bottom rows are equal
    fig, axes = plt.subplots(3, 1, figsize=(14, 5.5), sharex=True,
                              gridspec_kw={'height_ratios': [0.3, 1.2, 1.2]})
    ax_labels, ax_detectors, ax_backtrack = axes

    # Helper function to add region shading to an axis
    def add_regions(ax):
        # User prompt region (before assistant starts)
        if assistant_start_relative is not None and assistant_start_relative > 0:
            ax.axvspan(0, assistant_start_relative, color=color_prompt, alpha=region_alpha)
        # Off-topic region
        if assistant_start_relative is not None and correction_pos_relative is not None:
            ax.axvspan(assistant_start_relative, correction_pos_relative,
                       color=color_offtopic, alpha=region_alpha)
        # Self-correction region
        if correction_pos_relative is not None and on_topic_pos_relative is not None:
            ax.axvspan(correction_pos_relative, on_topic_pos_relative,
                       color=color_correction, alpha=region_alpha)
        # On-topic region
        if on_topic_pos_relative is not None:
            ax.axvspan(on_topic_pos_relative, response_length,
                       color=color_ontopic, alpha=region_alpha)

    # === Row 1: Labels for temporal regions ===
    add_regions(ax_labels)
    ax_labels.set_xlim(0, response_length)
    ax_labels.set_ylim(0, 1)
    ax_labels.axis('off')  # Hide axes for label row

    # Add text labels centered in each region
    label_y = 0.5
    label_fontsize = 13
    
    # User prompt label
    if assistant_start_relative is not None and assistant_start_relative > 0:
        ax_labels.text(assistant_start_relative / 2, label_y, 'User\nprompt',
                       fontsize=label_fontsize, ha='center', va='center',
                       fontweight='bold', color='#555555')
    
    # Off-topic region label
    if assistant_start_relative is not None and correction_pos_relative is not None:
        mid = (assistant_start_relative + correction_pos_relative) / 2
        ax_labels.text(mid, label_y, 'Off-topic generation',
                       fontsize=label_fontsize, ha='center', va='center',
                       fontweight='bold', color='#7D3C98')
    
    # Self-correction label with quote
    if correction_pos_relative is not None and on_topic_pos_relative is not None:
        mid = (correction_pos_relative + on_topic_pos_relative) / 2
        ax_labels.text(mid, label_y, '"Wait, I made\na mistake!..."',
                       fontsize=label_fontsize, ha='center', va='center',
                       fontweight='bold', fontstyle='italic', color='#AF601A')
    
    # On-topic region label with quote
    if on_topic_pos_relative is not None:
        mid = (on_topic_pos_relative + response_length) / 2
        ax_labels.text(mid, label_y, '"In reality,\nprobability is..."',
                       fontsize=label_fontsize, ha='center', va='center',
                       fontweight='bold', fontstyle='italic', color='#1A5276')

    # === Row 2: Off-topic detectors (sum) ===
    add_regions(ax_detectors)
    if active_detectors:
        detector_matrix = np.array([activations[idx][response_start:] for idx in active_detectors])
        sum_detector_acts = detector_matrix.sum(axis=0)
        sum_detector_smoothed = exponential_smoothing(sum_detector_acts, alpha)
        ax_detectors.plot(x, sum_detector_smoothed, color='#17A2B8', linewidth=3.0, alpha=0.9)
    
    ax_detectors.set_ylabel('Off-topic\ndetectors (sum)', fontsize=13, fontweight='bold')
    ax_detectors.tick_params(axis='both', labelsize=11)
    ax_detectors.grid(True, alpha=0.3)
    ax_detectors.set_xlim(0, response_length)

    # === Row 3: Sarcastic backtracking latent ===
    add_regions(ax_backtrack)
    
    # Find the sarcastic backtracking latent (33044)
    backtrack_idx = 33044
    if backtrack_idx in activations:
        backtrack_acts = activations[backtrack_idx][response_start:]
        backtrack_smoothed = exponential_smoothing(backtrack_acts, alpha)
        ax_backtrack.plot(x, backtrack_smoothed, color='#E67E22', linewidth=3.0, alpha=0.9)
        print(f"  Latent {backtrack_idx}: max={backtrack_acts.max():.4f}, mean={backtrack_acts.mean():.4f}")
    
    ax_backtrack.set_ylabel('Sarcastic\nbacktracking', fontsize=13, fontweight='bold')
    ax_backtrack.set_xlabel('Token Position', fontsize=14)
    ax_backtrack.tick_params(axis='both', labelsize=11)
    ax_backtrack.grid(True, alpha=0.3)
    ax_backtrack.set_xlim(0, response_length)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")
    plt.close()

    # Save data JSON alongside the plot
    data_output_path = output_path.with_suffix('.json')

    series = {}

    # Save sum of off-topic detectors
    if active_detectors:
        detector_matrix = np.array([activations[idx][response_start:] for idx in active_detectors])
        sum_detector_acts = detector_matrix.sum(axis=0)
        sum_detector_smoothed = exponential_smoothing(sum_detector_acts, alpha)
        series["Off-topic detectors (sum)"] = sum_detector_smoothed.tolist()

    # Save sarcastic backtracking latent
    backtrack_idx = 33044
    if backtrack_idx in activations:
        backtrack_acts = activations[backtrack_idx][response_start:]
        backtrack_smoothed = exponential_smoothing(backtrack_acts, alpha)
        series["Sarcastic backtracking"] = backtrack_smoothed.tolist()

    regions = {
        "assistant_start": assistant_start_relative,
        "correction_pos": correction_pos_relative,
        "on_topic_pos": on_topic_pos_relative,
    }

    plot_data = {
        "metadata": {
            "distractor_label": distractor_label,
            "boost_level": boost_level,
            "token_count": response_length,
            "smoothing_alpha": alpha,
        },
        "series": series,
        "regions": regions,
    }

    with open(data_output_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Saved data to {data_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 6 sequential activations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Folder to save plots/data (relative paths are resolved from the experiment base dir). Default: plots/",
    )
    args = parser.parse_args()

    # Load pre-computed activation data
    input_path = BASE_DIR / "experiment_results/experiment_6_sequential_activations.json"

    if not input_path.exists():
        print(f"Error: Results file not found: {input_path}")
        print("Run experiment_6_sequential_activations.py first to generate the data.")
        return

    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Extract data
    metadata = data["metadata"]
    token_strings = data["token_strings"]
    distractor_latent = data["distractor_latent"]
    off_topic_detectors = data["off_topic_detectors"]

    # Convert string keys back to ints and lists back to numpy arrays
    activations = {int(k): np.array(v) for k, v in data["activations"].items()}
    backtracking_latents = {int(k): v for k, v in data["backtracking_latents"].items()}
    latents_of_interest = {int(k): v for k, v in data["latents_of_interest"].items()}
    
    # Filter out "Hesitation and uncertainty markers" latent (40119)
    latents_of_interest.pop(40119, None)

    print(f"Loaded activations for {len(token_strings)} tokens")
    print(f"  Distractor latent: {distractor_latent}")
    print(f"  Off-topic detectors: {len(off_topic_detectors)}")
    print(f"  Backtracking latents: {len(backtracking_latents)}")
    print(f"  Latents of interest: {len(latents_of_interest)}")

    # Print latents of interest
    print("\n" + "="*60)
    print("LATENTS OF INTEREST (full labels)")
    print("="*60)
    for idx, label in latents_of_interest.items():
        print(f"\n  Latent {idx}:")
        print(f"    {label}")

    # Create output directory and plot
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_activations(
        activations=activations,
        token_strings=token_strings,
        response_start=metadata["response_start"],
        output_path=output_dir / "experiment_6_sequential_activations.png",
        boost_level=metadata["threshold"],
        distractor_latent=distractor_latent,
        distractor_label=metadata["feature_label"],
        off_topic_detectors=off_topic_detectors,
        backtracking_latents=backtracking_latents,
        latents_of_interest=latents_of_interest,
        # Hard-coded token positions for key events in the response:
        #   assistant_start: 40  (where assistant response begins)
        #   correction_pos: 268  (where "Wait, I made a mistake" appears)
        #   on_topic_pos: 293    (where "In reality, probability is" begins)
        assistant_start=40,
        correction_pos=268,
        distraction_pos=metadata.get("distraction_pos"),
        on_topic_pos=293,
    )

    # Print statistics
    response_start = metadata["response_start"]

    print("\n" + "="*60)
    print("ACTIVATION STATISTICS")
    print("="*60)

    distractor_response = activations[distractor_latent][response_start:]
    print(f"\nDistractor latent {distractor_latent}:")
    print(f"  Max: {distractor_response.max():.4f}")
    print(f"  Mean: {distractor_response.mean():.4f}")
    print(f"  Non-zero tokens: {(distractor_response > 0.01).sum()}")

    off_topic_matrix = np.array([activations[idx][response_start:] for idx in off_topic_detectors])
    mean_off_topic = off_topic_matrix.mean(axis=0)
    print(f"\nOff-topic detectors (mean of {len(off_topic_detectors)}):")
    print(f"  Max: {mean_off_topic.max():.4f}")
    print(f"  Mean: {mean_off_topic.mean():.4f}")


if __name__ == "__main__":
    main()

