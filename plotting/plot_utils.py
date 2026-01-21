"""
Shared utilities for plotting scripts.

This module centralizes:
- Which result files should be ignored (known-bad runs, merged artifacts, etc.)
- Degradation detection for filtering repetitive / broken generations
- Common experiment-results file discovery logic used across plot scripts
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from result_file_utils import (
    CanonicalModelInfo,
    ModelFamily,
    canonicalize_model_name,
    parse_results_filename,
)

#
# Ignore list + helpers
#

# Files to ignore (problematic runs that should not be included)
IGNORED_FILES: set[str] = {
    # gemma-9b early runs: various configuration issues
    "experiment_results_gemma-2-9b-it-res-16k-layer-20_20251124_134523.json",
    "experiment_results_gemma-2-9b-it-res-16k-layer-20_20251124_132251.json",
    "experiment_results_gemma-2-9b-it-res-16k-layer-20_20251124_130027.json",
    "experiment_results_gemma-2-9b-it-res-16k-layer-20_20251124_120931.json",
    "experiment_results_gemma-2-9b-res-16k-layer-26_20251124_112341.json",
    "experiment_results_gemma-2-9b-res-16k-layer-26_20251124_111954.json",
    "experiment_results_gemma-2-9b-res-16k-layer-26_20251124_111141.json",
    "experiment_results_gemma-2-9b-res-16k-layer-26_20251124_105500.json",
    # llama-70b early run: configuration issue
    "experiment_results_Meta-Llama-3.3-70B-Instruct_20251030_093427.json",
    # gemma-27b: threshold upper_bound was too low (80 or 1500), all thresholds hit ceiling,
    # steering had no effect, first attempt scores ~95%
    "experiment_results_gemma-2-27b-it-res-131k-layer-22_20251125_123304.json",
    "experiment_results_gemma-2-27b-it-res-131k-layer-22_20251127_164235.json",
    "experiment_results_gemma-2-27b-it-res-131k-layer-22_20251211_104732.json",
    "experiment_results_gemma-2-27b-it-res-131k-layer-22_20251211_133028.json",
    "experiment_results_gemma-2-27b-it-res-131k-layer-22_20251125_165543.json",
    # gemma-2b: first attempt score ~88%, borderline problematic
    "experiment_results_gemma-2-2b-it-res-16k-layer-16_20251124_203814.json",
}


def should_ignore_file(filename: str) -> bool:
    """
    Check if a file should be ignored based on filename.

    Args:
        filename: The filename (not full path) to check

    Returns:
        True if the file should be ignored
    """
    # Check explicit ignore list
    if filename in IGNORED_FILES:
        return True

    # Also ignore files with certain patterns
    filename_lower = filename.lower()
    ignore_patterns = [
        "-merged",
        "masked-",
        "pct",
        "no_steering_baseline",
        "ablation",
        "multi_boost",
    ]

    return any(pattern in filename_lower for pattern in ignore_patterns)


def is_degraded_output(response: str, min_repeats: int = 5) -> bool:
    """
    Check if a response is degraded (contains repetitive patterns).

    Args:
        response: The generated response text
        min_repeats: Minimum consecutive repetitions to count as degraded

    Returns:
        True if the response appears degraded
    """
    words = response.split()
    if len(words) < min_repeats:
        return False

    max_repeat = 1
    current_repeat = 1

    for i in range(1, len(words)):
        if words[i] == words[i - 1] and len(words[i]) > 1:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1

    return max_repeat >= min_repeats


#
# Experiment-results discovery helpers
#

def iter_experiment_results_jsons(results_dir: Path) -> Iterable[Path]:
    """Yield all JSON files in a results directory, sorted for stability."""
    yield from sorted(results_dir.glob("*.json"))


def collect_experiment_1_result_files(
    base_dir: Path,
    *,
    excluded_families: set[ModelFamily] | None = None,
) -> tuple[list[Path], dict[Path, CanonicalModelInfo], dict[str, list[Path]]]:
    """
    Collect Experiment 1-style ESR result files (non-ablation, non-multi-boost) and group them by model.

    This matches the selection logic previously duplicated in `plotting/plot_exp1.py` and `generate_all_plots.sh`.

    Returns:
        - selected_files: flat list of result file Paths
        - model_info_map: Path -> CanonicalModelInfo
        - model_files: display_name -> list[Path] (sorted by mtime desc)
    """
    excluded_families = excluded_families or {ModelFamily.FINETUNED_8B}

    result_dir = base_dir / "experiment_results"
    all_json_files = list(iter_experiment_results_jsons(result_dir))

    model_info_map: dict[Path, CanonicalModelInfo] = {}
    model_files: dict[str, list[Path]] = defaultdict(list)

    for result_file in all_json_files:
        if should_ignore_file(result_file.name):
            continue

        parsed = parse_results_filename(result_file)
        if parsed is None:
            continue

        if parsed.is_ablation:
            continue

        if parsed.experiment_type == "multi_boost":
            continue

        model_info = canonicalize_model_name(parsed.model_name)
        if model_info.family in excluded_families:
            continue

        model_info_map[result_file] = model_info
        model_files[model_info.display_name].append(result_file)

    # Sort files by modification time within each model (most recent first)
    for model_name in list(model_files.keys()):
        model_files[model_name] = sorted(
            model_files[model_name],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

    selected_files = [f for files in model_files.values() for f in files]
    return selected_files, model_info_map, dict(model_files)


