"""Load feature labels from CSV files."""

import csv
from typing import Dict


def load_labels(csv_path: str) -> Dict[int, str]:
    """
    Load feature labels from a CSV file.

    Args:
        csv_path: Path to CSV file with columns: index_in_sae, label, status, notes, uuid

    Returns:
        Dictionary mapping feature index to label
    """
    labels = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index_in_sae"])
            label = row["label"]

            # Use the label if it exists, otherwise use a placeholder
            if label:
                labels[idx] = label
            else:
                labels[idx] = f"feature_{idx}"

    return labels
