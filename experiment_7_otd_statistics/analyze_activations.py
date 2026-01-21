#!/usr/bin/env python
"""Analyze collected SAE latent activations to understand off-topic detector selection.

This script:
1. Loads the collected activation data
2. Reports statistics for the 25 off-topic detectors from the original experiment
3. Explores what criteria might have been used to select them

Usage:
    python analyze_activations.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings from constant/near-constant arrays
warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation')

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# The 25 off-topic detectors from the original experiment
# (excluding 33044, which is a self-correction trigger, not an off-topic detector)
OFF_TOPIC_DETECTORS = [
    40119, 34765, 59483, 28540, 61420, 17516, 37536, 24684, 58565,
    34002, 27331, 15375, 7517, 17481, 10304, 11977, 61116, 45078,
    41038, 40792, 49897, 54311, 46037, 9168, 3675
]
OFF_TOPIC_DETECTORS_UNIQUE = sorted(set(OFF_TOPIC_DETECTORS))


def load_labels(labels_file: Path) -> dict[int, str]:
    """Load latent labels from CSV file."""
    df = pd.read_csv(labels_file)
    return dict(zip(df['index_in_sae'], df['label']))


def compute_auc(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute AUC-ROC (Separability) treating this as binary classification (group1=positive)."""
    from sklearn.metrics import roc_auc_score
    # group1 = shuffled (off-topic, positive class)
    # group2 = normal (on-topic, negative class)
    y_true = np.concatenate([np.ones(len(group1)), np.zeros(len(group2))])
    y_score = np.concatenate([group1, group2])
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def analyze_latent(
    latent_idx: int,
    shuffled_activations: np.ndarray,
    normal_activations: np.ndarray,
    zero_threshold: float = 1e-8,
) -> dict:
    """Compute basic statistics for a single latent."""
    shuffled_vals = shuffled_activations[:, latent_idx]
    normal_vals = normal_activations[:, latent_idx]

    # Frequency of non-zero activation
    shuffled_nonzero_freq = (shuffled_vals >= zero_threshold).mean()
    normal_nonzero_freq = (normal_vals >= zero_threshold).mean()

    # All normal samples zero?
    all_normal_zero = (normal_vals < zero_threshold).all()

    return {
        "latent_idx": latent_idx,
        "shuffled_nonzero_pct": shuffled_nonzero_freq * 100,
        "normal_nonzero_pct": normal_nonzero_freq * 100,
        "all_normal_zero": bool(all_normal_zero),
    }


def main(data_dir: Path = DEFAULT_DATA_DIR):
    print("=" * 80)
    print("Analyzing SAE latent activations for off-topic detector analysis")
    print("=" * 80)
    print(f"Data dir: {data_dir}")

    # Load activation data
    activations_file = data_dir / "activations_all_latents.npz"
    if not activations_file.exists():
        print(f"ERROR: {activations_file} not found!")
        print("Please run collect_activations.py first.")
        return
    
    print(f"\nLoading activations from {activations_file}...")
    data = np.load(activations_file)

    # New format: multiple derangements stored separately
    if "normal" in data and "shuffled_0" in data:
        normal_activations = data["normal"]
        num_derangements = int(data["num_derangements"])
        shuffled_by_derangement = {
            d: data[f"shuffled_{d}"] for d in range(num_derangements)
        }
        print(f"Loaded {num_derangements} derangements")
    # Legacy format
    elif "shuffled_max" in data:
        normal_activations = data["normal_max"]
        shuffled_by_derangement = {0: data["shuffled_max"]}
        num_derangements = 1
        print("Loaded legacy format (1 derangement)")
    else:
        normal_activations = data["normal_activations"] if "normal_activations" in data else data["normal"]
        shuffled_by_derangement = {0: data["shuffled_activations"] if "shuffled_activations" in data else data["shuffled_0"]}
        num_derangements = 1
        print("Loaded legacy format (1 derangement)")

    num_normal_samples, num_latents = normal_activations.shape
    num_shuffled_samples = shuffled_by_derangement[0].shape[0]

    print(f"Normal activations shape: {normal_activations.shape}")
    print(f"Shuffled activations shape per derangement: ({num_shuffled_samples}, {num_latents})")

    # Filter to latents with any activation (speeds up analysis significantly)
    all_activations = [normal_activations] + [shuffled_by_derangement[d] for d in range(num_derangements)]
    combined = np.vstack(all_activations)
    active_mask = combined.max(axis=0) > 1e-8
    active_latent_indices = np.where(active_mask)[0]
    print(f"Active latents (non-zero in at least one sample): {len(active_latent_indices)}/{num_latents}")

    # For backward compatibility with rest of script
    shuffled_activations = shuffled_by_derangement[0]

    # Load labels
    labels_file = data_dir / "llama-70b-goodfire-l50.csv"
    labels = load_labels(labels_file) if labels_file.exists() else {}
    
    # =========================================================================
    # Section 1: Analyze the 25 off-topic detectors
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: The 25 off-topic detectors")
    print("=" * 80)
    
    results = []
    for latent_idx in OFF_TOPIC_DETECTORS_UNIQUE:
        stats = analyze_latent(latent_idx, shuffled_activations, normal_activations)
        stats["label"] = labels.get(latent_idx, "UNKNOWN")
        results.append(stats)
    
    # Sort by shuffled frequency (descending)
    results.sort(key=lambda x: x["shuffled_nonzero_pct"], reverse=True)
    
    print(f"\n{'Latent':<8} | {'Shuffled %':>10} | {'Normal %':>10} | {'All Norm=0':>10} | Label")
    print("-" * 100)
    
    for r in results:
        all_zero_str = "✓" if r["all_normal_zero"] else "✗"
        label = r["label"][:50] if r["label"] else "UNKNOWN"
        print(f"{r['latent_idx']:<8} | {r['shuffled_nonzero_pct']:>9.1f}% | {r['normal_nonzero_pct']:>9.1f}% | {all_zero_str:>10} | {label}")
    
    # Summary stats
    all_normal_zero_count = sum(1 for r in results if r["all_normal_zero"])
    avg_shuffled_freq = np.mean([r["shuffled_nonzero_pct"] for r in results])
    min_shuffled_freq = min(r["shuffled_nonzero_pct"] for r in results)
    max_shuffled_freq = max(r["shuffled_nonzero_pct"] for r in results)
    
    print(f"\nSummary for the 25 detectors:")
    print(f"  All normal=0: {all_normal_zero_count}/{len(results)}")
    print(f"  Shuffled activation frequency: min={min_shuffled_freq:.1f}%, max={max_shuffled_freq:.1f}%, avg={avg_shuffled_freq:.1f}%")
    
    # =========================================================================
    # Section 2: Find all latents matching different criteria
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Exploring selection criteria")
    print("=" * 80)
    
    # Analyze active latents to find patterns
    all_stats = []
    for latent_idx in tqdm(active_latent_indices, desc="Analyzing latents"):
        stats = analyze_latent(latent_idx, shuffled_activations, normal_activations)
        all_stats.append(stats)
    
    # Criteria 1: All normal samples are zero
    zero_normal_latents = [s for s in all_stats if s["all_normal_zero"]]
    print(f"\nLatents with ALL normal samples = 0: {len(zero_normal_latents)}")
    
    # Criteria 2: Zero normal + various shuffled thresholds
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        matching = [s for s in zero_normal_latents if s["shuffled_nonzero_pct"] >= threshold * 100]
        our_detectors_in = [s for s in matching if s["latent_idx"] in OFF_TOPIC_DETECTORS_UNIQUE]
        print(f"  + shuffled >= {threshold*100:.0f}%: {len(matching)} latents ({len(our_detectors_in)} of our 25 match)")
    
    our_set = set(OFF_TOPIC_DETECTORS_UNIQUE)

    # =========================================================================
    # Section 3: Separability (AUC-ROC) analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Latents by Separability (AUC-ROC)")
    print("=" * 80)

    # Use first derangement
    shuffled_d = shuffled_by_derangement[0]

    # Compute Separability for each active latent
    sep_stats = []
    for latent_idx in tqdm(active_latent_indices, desc="Computing separability"):
        shuf_vals = shuffled_d[:, latent_idx]
        norm_vals = normal_activations[:, latent_idx]
        auc = compute_auc(shuf_vals, norm_vals)
        shuf_pct = (shuf_vals > 1e-8).mean() * 100
        norm_pct = (norm_vals > 1e-8).mean() * 100
        sep_stats.append({
            "latent_idx": latent_idx,
            "auc_roc": auc,
            "shuffled_pct": shuf_pct,
            "normal_pct": norm_pct,
        })

    # Sort by Separability
    sep_stats.sort(key=lambda x: x["auc_roc"], reverse=True)

    # Find latents with perfect separability (AUC >= 1.0)
    SEPARABILITY_THRESHOLD = 1.0
    perfect_sep = [s for s in sep_stats if s["auc_roc"] >= SEPARABILITY_THRESHOLD]

    print(f"\nLatents with Separability >= {SEPARABILITY_THRESHOLD}: {len(perfect_sep)}")
    print(f"\n{'Rank':<6} | {'Latent':<8} | {'AUC-ROC':>8} | {'Shuf %':>8} | {'Norm %':>8} | {'In 25?':>6} | Label")
    print("-" * 110)

    for i, s in enumerate(perfect_sep):
        in_25 = "✓" if s["latent_idx"] in our_set else ""
        label = labels.get(s["latent_idx"], "")[:35]
        print(f"{i+1:<6} | {s['latent_idx']:<8} | {s['auc_roc']:>8.3f} | {s['shuffled_pct']:>7.1f}% | {s['normal_pct']:>7.1f}% | {in_25:>6} | {label}")

    # Show next 25 below threshold for context
    print(f"\nNext 25 latents (Separability < {SEPARABILITY_THRESHOLD}):")
    print("-" * 110)
    below_threshold = [s for s in sep_stats if s["auc_roc"] < SEPARABILITY_THRESHOLD][:25]
    for i, s in enumerate(below_threshold):
        in_25 = "✓" if s["latent_idx"] in our_set else ""
        label = labels.get(s["latent_idx"], "")[:35]
        print(f"{len(perfect_sep)+i+1:<6} | {s['latent_idx']:<8} | {s['auc_roc']:>8.3f} | {s['shuffled_pct']:>7.1f}% | {s['normal_pct']:>7.1f}% | {in_25:>6} | {label}")

    # Summary stats
    perfect_sep_set = set(s["latent_idx"] for s in perfect_sep)

    print(f"\nRecovery of 25 original detectors:")
    print(f"  With Separability >= {SEPARABILITY_THRESHOLD}: {len(our_set & perfect_sep_set)}/25")

    # Show which original detectors have perfect separability vs not
    recovered = our_set & perfect_sep_set
    missing = our_set - perfect_sep_set

    print(f"\nOriginal detectors with Separability >= {SEPARABILITY_THRESHOLD}: {sorted(recovered)}")
    print(f"Original detectors below threshold: {sorted(missing)}")

    # Show details for missing ones
    if missing:
        print(f"\nMissing detectors details:")
        latent_to_auc = {s["latent_idx"]: s["auc_roc"] for s in sep_stats}
        for idx in sorted(missing):
            auc = latent_to_auc.get(idx, 0)
            label = labels.get(idx, "")[:40]
            print(f"  {idx:<6} (AUC={auc:.3f}) {label}")

    print("\n✓ Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze SAE latent activations for OTD selection")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Data directory with activations (default: data/)")
    args = parser.parse_args()

    main(args.data_dir)
