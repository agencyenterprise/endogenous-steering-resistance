"""
Analyze activation statistics for self-correction episodes.

Computes:
1. Mean activation per latent in each region (off-topic, correction, on-topic)
2. Statistical tests comparing regions (Wilcoxon, t-test)
3. Effect sizes (Cohen's d, Cliff's delta)

Usage:
    python analyze_activations.py
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


# Off-topic detector latents - loaded from data/off_topic_detectors_old.json
DEFAULT_OTD_FILE = Path(__file__).parent.parent / "data" / "off_topic_detectors_old.json"

def _load_off_topic_detectors(detector_file: Path = DEFAULT_OTD_FILE) -> list[int]:
    with open(detector_file) as f:
        data = json.load(f)
    return data["off_topic_detectors"]

OFF_TOPIC_DETECTORS = _load_off_topic_detectors()

BACKTRACKING_LATENTS = [
    5852, 18311, 45478, 52694, 57675, 63162, 3994, 3473, 5215, 7318,
    53491, 890, 1719, 28564, 34597, 33044
]


@dataclass
class RegionStats:
    """Statistics for a single region."""
    mean: float
    std: float
    median: float
    n_tokens: int
    n_nonzero: int
    pct_nonzero: float


@dataclass
class LatentStats:
    """Statistics for a single latent across all episodes."""
    latent_idx: int
    latent_label: str

    # Per-region aggregated statistics
    off_topic: RegionStats
    correction: RegionStats
    on_topic: RegionStats

    # Statistical tests: off_topic vs on_topic
    wilcoxon_stat: float
    wilcoxon_p: float
    ttest_stat: float
    ttest_p: float

    # Effect sizes
    cohens_d: float
    cliffs_delta: float


@dataclass
class GroupStats:
    """Statistics for a group of latents."""
    group_name: str
    n_latents: int

    # Aggregated across latents
    mean_off_topic: float
    mean_correction: float
    mean_on_topic: float
    std_off_topic: float
    std_correction: float
    std_on_topic: float

    # Statistical tests on aggregated means
    wilcoxon_stat: float
    wilcoxon_p: float
    ttest_stat: float
    ttest_p: float
    cohens_d: float


def load_episode_activations(
    metadata_file: Path,
    activations_dir: Path,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    """Load episode metadata and activations."""

    with open(metadata_file) as f:
        data = json.load(f)

    episodes = data["episodes"]
    activations = {}

    for ep in episodes:
        episode_id = ep["episode_id"]
        act_file = activations_dir / f"{episode_id}.npz"
        if act_file.exists():
            loaded = np.load(act_file)
            activations[episode_id] = {k: loaded[k] for k in loaded.files}

    return episodes, activations


def compute_region_activations(
    activations: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """Extract activations for a specific region."""
    if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
        return np.array([])
    return activations[start_idx:end_idx]


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta effect size (non-parametric)."""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0

    n1, n2 = len(group1), len(group2)
    more = 0
    less = 0

    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                more += 1
            elif x1 < x2:
                less += 1

    return (more - less) / (n1 * n2)


def analyze_latent(
    latent_idx: int,
    latent_label: str,
    episodes: list[dict],
    activations: dict[str, dict[str, np.ndarray]],
) -> Optional[LatentStats]:
    """Analyze a single latent across all episodes."""

    off_topic_vals = []
    correction_vals = []
    on_topic_vals = []

    for ep in episodes:
        episode_id = ep["episode_id"]
        if episode_id not in activations:
            continue

        ep_acts = activations[episode_id]
        latent_key = str(latent_idx)
        if latent_key not in ep_acts:
            continue

        acts = ep_acts[latent_key]

        # Get region boundaries
        response_start = ep["response_start_token"]
        off_topic_start = ep["off_topic_start_token"]
        correction_start = ep["correction_start_token"]
        on_topic_start = ep["on_topic_start_token"]

        # Extract per-region activations
        # Off-topic region: from off_topic_start to correction_start
        off_topic_acts = compute_region_activations(acts, off_topic_start, correction_start)
        # Correction region: from correction_start to on_topic_start
        correction_acts = compute_region_activations(acts, correction_start, on_topic_start)
        # On-topic region: from on_topic_start to end
        on_topic_acts = compute_region_activations(acts, on_topic_start, len(acts))

        # Use mean activation per episode for statistical tests
        if len(off_topic_acts) > 0:
            off_topic_vals.append(np.mean(off_topic_acts))
        if len(correction_acts) > 0:
            correction_vals.append(np.mean(correction_acts))
        if len(on_topic_acts) > 0:
            on_topic_vals.append(np.mean(on_topic_acts))

    if len(off_topic_vals) < 2 or len(on_topic_vals) < 2:
        return None

    off_topic_arr = np.array(off_topic_vals)
    correction_arr = np.array(correction_vals) if correction_vals else np.array([0.0])
    on_topic_arr = np.array(on_topic_vals)

    # Compute per-region statistics
    def make_region_stats(arr: np.ndarray) -> RegionStats:
        return RegionStats(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            n_tokens=len(arr),
            n_nonzero=int(np.sum(arr > 0.01)),
            pct_nonzero=float(np.mean(arr > 0.01) * 100),
        )

    # Statistical tests: off_topic vs on_topic
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(off_topic_arr, on_topic_arr, alternative="greater")
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    try:
        ttest_stat, ttest_p = stats.ttest_rel(off_topic_arr, on_topic_arr)
    except ValueError:
        ttest_stat, ttest_p = 0.0, 1.0

    return LatentStats(
        latent_idx=latent_idx,
        latent_label=latent_label,
        off_topic=make_region_stats(off_topic_arr),
        correction=make_region_stats(correction_arr),
        on_topic=make_region_stats(on_topic_arr),
        wilcoxon_stat=float(wilcoxon_stat),
        wilcoxon_p=float(wilcoxon_p),
        ttest_stat=float(ttest_stat),
        ttest_p=float(ttest_p / 2),  # One-tailed
        cohens_d=cohens_d(off_topic_arr, on_topic_arr),
        cliffs_delta=cliffs_delta(off_topic_arr, on_topic_arr),
    )


def analyze_group(
    group_name: str,
    latent_indices: list[int],
    latent_stats: list[LatentStats],
) -> GroupStats:
    """Analyze aggregated statistics for a group of latents."""

    off_topic_means = [s.off_topic.mean for s in latent_stats if s is not None]
    correction_means = [s.correction.mean for s in latent_stats if s is not None]
    on_topic_means = [s.on_topic.mean for s in latent_stats if s is not None]

    off_topic_arr = np.array(off_topic_means)
    correction_arr = np.array(correction_means)
    on_topic_arr = np.array(on_topic_means)

    # Statistical tests on latent-level means
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(off_topic_arr, on_topic_arr, alternative="greater")
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    try:
        ttest_stat, ttest_p = stats.ttest_rel(off_topic_arr, on_topic_arr)
    except ValueError:
        ttest_stat, ttest_p = 0.0, 1.0

    return GroupStats(
        group_name=group_name,
        n_latents=len(latent_indices),
        mean_off_topic=float(np.mean(off_topic_arr)),
        mean_correction=float(np.mean(correction_arr)),
        mean_on_topic=float(np.mean(on_topic_arr)),
        std_off_topic=float(np.std(off_topic_arr)),
        std_correction=float(np.std(correction_arr)),
        std_on_topic=float(np.std(on_topic_arr)),
        wilcoxon_stat=float(wilcoxon_stat),
        wilcoxon_p=float(wilcoxon_p),
        ttest_stat=float(ttest_stat),
        ttest_p=float(ttest_p / 2),  # One-tailed
        cohens_d=cohens_d(off_topic_arr, on_topic_arr),
    )


def load_latent_labels(labels_file: Path) -> dict[int, str]:
    """Load latent labels from CSV."""
    import csv
    labels = {}
    with open(labels_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index_in_sae"])
            labels[idx] = row["label"] if row["label"] else f"feature_{idx}"
    return labels


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "activation_stats"
    metadata_file = output_dir / "episode_metadata.json"
    activations_dir = output_dir / "activations"
    labels_file = base_dir / "data" / "llama-70b-goodfire-l50.csv"
    output_file = output_dir / "analysis_results.json"

    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        print("Please run collect_activations.py first.")
        return

    print("Loading data...")
    episodes, activations = load_episode_activations(metadata_file, activations_dir)
    labels = load_latent_labels(labels_file)

    print(f"Loaded {len(episodes)} episodes with activations")

    # Analyze OTD latents
    print("\nAnalyzing Off-Topic Detector latents...")
    otd_stats = []
    for latent_idx in OFF_TOPIC_DETECTORS:
        label = labels.get(latent_idx, f"feature_{latent_idx}")
        stat = analyze_latent(latent_idx, label, episodes, activations)
        if stat:
            otd_stats.append(stat)
            print(f"  {latent_idx}: off={stat.off_topic.mean:.4f}, corr={stat.correction.mean:.4f}, on={stat.on_topic.mean:.4f}, p={stat.wilcoxon_p:.4f}")

    # Analyze backtracking latents
    print("\nAnalyzing Backtracking latents...")
    bt_stats = []
    for latent_idx in BACKTRACKING_LATENTS:
        label = labels.get(latent_idx, f"feature_{latent_idx}")
        stat = analyze_latent(latent_idx, label, episodes, activations)
        if stat:
            bt_stats.append(stat)
            print(f"  {latent_idx}: off={stat.off_topic.mean:.4f}, corr={stat.correction.mean:.4f}, on={stat.on_topic.mean:.4f}, p={stat.wilcoxon_p:.4f}")

    # Group-level analysis
    print("\nGroup-level statistics:")
    otd_group = analyze_group("Off-Topic Detectors", OFF_TOPIC_DETECTORS, otd_stats)
    bt_group = analyze_group("Backtracking Latents", BACKTRACKING_LATENTS, bt_stats)

    print(f"\nOff-Topic Detectors ({otd_group.n_latents} latents):")
    print(f"  Off-topic region: {otd_group.mean_off_topic:.4f} +/- {otd_group.std_off_topic:.4f}")
    print(f"  Correction region: {otd_group.mean_correction:.4f} +/- {otd_group.std_correction:.4f}")
    print(f"  On-topic region: {otd_group.mean_on_topic:.4f} +/- {otd_group.std_on_topic:.4f}")
    print(f"  Wilcoxon p-value (off > on): {otd_group.wilcoxon_p:.6f}")
    print(f"  Cohen's d: {otd_group.cohens_d:.4f}")

    print(f"\nBacktracking Latents ({bt_group.n_latents} latents):")
    print(f"  Off-topic region: {bt_group.mean_off_topic:.4f} +/- {bt_group.std_off_topic:.4f}")
    print(f"  Correction region: {bt_group.mean_correction:.4f} +/- {bt_group.std_correction:.4f}")
    print(f"  On-topic region: {bt_group.mean_on_topic:.4f} +/- {bt_group.std_on_topic:.4f}")
    print(f"  Wilcoxon p-value (off > on): {bt_group.wilcoxon_p:.6f}")
    print(f"  Cohen's d: {bt_group.cohens_d:.4f}")

    # Save results
    output_data = {
        "n_episodes": len(episodes),
        "otd_group": asdict(otd_group),
        "backtracking_group": asdict(bt_group),
        "otd_latent_stats": [asdict(s) for s in otd_stats],
        "backtracking_latent_stats": [asdict(s) for s in bt_stats],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
