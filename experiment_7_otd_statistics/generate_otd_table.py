#!/usr/bin/env python
"""Generate LaTeX table of activation statistics for off-topic detectors.

This script produces a LaTeX table showing, for each of the 25 off-topic detectors:
1. Mean activation strength on shuffled (off-topic) samples
2. Percentage of shuffled samples with non-zero activation
3. Mean activation strength on normal (on-topic) samples
4. Percentage of normal samples with non-zero activation

Usage:
    python generate_otd_table.py
    python generate_otd_table.py --output table.tex
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# The 25 separability-based off-topic detectors
OFF_TOPIC_DETECTORS = [
    6527, 7517, 9005, 10304, 11390, 17250, 17481, 17516, 23093, 24684,
    26312, 28403, 28540, 37234, 37536, 38956, 39926, 40119, 40792, 44845,
    45078, 56830, 58565, 59483, 61420
]


def load_labels(labels_file: Path) -> dict[int, str]:
    """Load latent labels from CSV file."""
    df = pd.read_csv(labels_file)
    return dict(zip(df['index_in_sae'], df['label']))


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def main(data_dir: Path = DEFAULT_DATA_DIR, output_file: Path = None):
    print("Loading activation data...")

    # Load activations
    activations_file = data_dir / "activations_all_latents.npz"
    if not activations_file.exists():
        print(f"ERROR: {activations_file} not found!")
        print("Please run collect_activations.py first.")
        return

    data = np.load(activations_file)
    normal_activations = data["normal"]
    shuffled_activations = data["shuffled_0"]  # First derangement

    print(f"Normal samples: {normal_activations.shape[0]}")
    print(f"Shuffled samples: {shuffled_activations.shape[0]}")
    print(f"Number of latents: {normal_activations.shape[1]}")

    # Load labels
    labels_file = data_dir / "llama-70b-goodfire-l50.csv"
    labels = load_labels(labels_file) if labels_file.exists() else {}

    # Compute statistics for each OTD
    ZERO_THRESHOLD = 1e-8

    rows = []
    for latent_idx in OFF_TOPIC_DETECTORS:
        shuffled_vals = shuffled_activations[:, latent_idx]
        normal_vals = normal_activations[:, latent_idx]

        # Shuffled (off-topic) stats
        shuffled_mean = shuffled_vals.mean()
        shuffled_nonzero_pct = (shuffled_vals > ZERO_THRESHOLD).mean() * 100

        # Normal (on-topic) stats
        normal_mean = normal_vals.mean()
        normal_nonzero_pct = (normal_vals > ZERO_THRESHOLD).mean() * 100

        label = labels.get(latent_idx, "")

        rows.append({
            "latent_idx": latent_idx,
            "label": label,
            "shuffled_mean": shuffled_mean,
            "shuffled_nonzero_pct": shuffled_nonzero_pct,
            "normal_mean": normal_mean,
            "normal_nonzero_pct": normal_nonzero_pct,
        })

    # Sort by latent index
    rows.sort(key=lambda x: x["latent_idx"])

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Activation statistics for the 25 separability-based off-topic detectors. Shuffled pairs are off-topic (prompt paired with different prompt's response); normal pairs are on-topic. All values are max-pooled across token positions.}")
    latex_lines.append(r"\label{tab:otd-activations}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{r l r r r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Latent & Label & \multicolumn{2}{c}{Shuffled (Off-topic)} & \multicolumn{2}{c}{Normal (On-topic)} \\")
    latex_lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    latex_lines.append(r" & & Mean & \% Active & Mean & \% Active \\")
    latex_lines.append(r"\midrule")

    for row in rows:
        # Truncate label to fit
        label = row["label"][:40]
        if len(row["label"]) > 40:
            label = label[:37] + "..."
        label = escape_latex(label)

        latex_lines.append(
            f"{row['latent_idx']} & {label} & "
            f"{row['shuffled_mean']:.3f} & {row['shuffled_nonzero_pct']:.1f}\\% & "
            f"{row['normal_mean']:.3f} & {row['normal_nonzero_pct']:.1f}\\% \\\\"
        )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_output = "\n".join(latex_lines)

    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(latex_output)
        print(f"\nLaTeX table written to {output_file}")
    else:
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80)
        print(latex_output)

    # Also print a summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    avg_shuffled_nonzero = np.mean([r["shuffled_nonzero_pct"] for r in rows])
    avg_normal_nonzero = np.mean([r["normal_nonzero_pct"] for r in rows])
    avg_shuffled_mean = np.mean([r["shuffled_mean"] for r in rows])
    avg_normal_mean = np.mean([r["normal_mean"] for r in rows])

    print(f"Average across 25 OTDs:")
    print(f"  Shuffled: mean activation = {avg_shuffled_mean:.4f}, % active = {avg_shuffled_nonzero:.1f}%")
    print(f"  Normal:   mean activation = {avg_normal_mean:.4f}, % active = {avg_normal_nonzero:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX table of OTD activation statistics")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Data directory with activations (default: data/)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output file for LaTeX (default: print to stdout)")
    args = parser.parse_args()

    main(args.data_dir, args.output)
