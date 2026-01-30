"""
Utilities for parsing experiment result filenames and canonicalizing model names.

Filename format:
    experiment_{experiment_type}_{model_name}_{YYYYMMDD}_{HHMMSS}[_with_ablation].json

Where:
    - experiment_type: "multi_boost" or "results"
    - model_name: The model identifier (may contain underscores)
    - YYYYMMDD: Date (8 digits)
    - HHMMSS: Time (6 digits)
    - _with_ablation: Optional suffix for ablation experiments
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ParsedResultFile:
    """Parsed components of a result filename."""
    experiment_type: str  # "multi_boost" or "results"
    model_name: str       # Raw model name from filename
    date: str             # YYYYMMDD
    time: str             # HHMMSS
    is_ablation: bool     # Whether this is an ablation experiment
    filepath: Path        # Original filepath


def parse_results_filename(filepath: Path | str) -> Optional[ParsedResultFile]:
    """
    Parse a result filename into its components.

    Args:
        filepath: Path to the result file (can be full path or just filename)

    Returns:
        ParsedResultFile with extracted components, or None if parsing fails

    Examples:
        >>> parse_results_filename("experiment_multi_boost_Meta-Llama-3.1-8B-Instruct_20251127_124258.json")
        ParsedResultFile(experiment_type='multi_boost', model_name='Meta-Llama-3.1-8B-Instruct',
                        date='20251127', time='124258', is_ablation=False, ...)

        >>> parse_results_filename("experiment_results_gemma-2-9b-it-res-16k-layer-20_20251124_171352.json")
        ParsedResultFile(experiment_type='results', model_name='gemma-2-9b-it-res-16k-layer-20',
                        date='20251124', time='171352', is_ablation=False, ...)

        >>> parse_results_filename("experiment_results_Meta-Llama-3.3-70B-Instruct_20260127_165832_layer33_repro.json")
        ParsedResultFile(experiment_type='results', model_name='Meta-Llama-3.3-70B-Instruct',
                        date='20260127', time='165832', is_ablation=False, ...)
    """
    filepath = Path(filepath)
    filename = filepath.name

    # Remove .json extension
    if not filename.endswith('.json'):
        return None
    base = filename[:-5]  # Remove '.json'

    # Check for and remove _with_ablation suffix
    is_ablation = base.endswith('_with_ablation')
    if is_ablation:
        base = base[:-14]  # Remove '_with_ablation'

    # Pattern: experiment_{type}_{model}_{date}_{time}[_{suffix}]
    # Parse from the right since model names can contain underscores
    # The timestamp is always DATE (8 digits) followed by TIME (6 digits)
    parts = base.split('_')

    if len(parts) < 4:
        return None

    # First part should be "experiment"
    if parts[0] != 'experiment':
        return None

    # Find the date_time pattern: look for 8-digit date followed by 6-digit time
    # Search from the end backwards for the timestamp
    date_idx = None
    time_str = None
    for i in range(len(parts) - 1, 1, -1):  # Start from end, stop before experiment type
        if len(parts[i]) == 6 and parts[i].isdigit():
            # Potential time, check if previous part is date
            if i > 1 and len(parts[i-1]) == 8 and parts[i-1].isdigit():
                date_idx = i - 1
                time_str = parts[i]
                break

    # Also try to find date-only pattern (no time, just date followed by suffix)
    if date_idx is None:
        for i in range(len(parts) - 1, 1, -1):
            if len(parts[i]) == 8 and parts[i].isdigit():
                # Check if this looks like a date (starts with 20xx)
                if parts[i].startswith('20'):
                    date_idx = i
                    time_str = "000000"  # Default time for date-only files
                    break

    if date_idx is None:
        return None

    date_str = parts[date_idx]

    # Determine experiment type and model name
    if parts[1] == 'multi' and len(parts) > 2 and parts[2] == 'boost':
        experiment_type = 'multi_boost'
        model_name = '_'.join(parts[3:date_idx])
    elif parts[1] == 'results':
        experiment_type = 'results'
        model_name = '_'.join(parts[2:date_idx])
    else:
        return None

    if not model_name:
        return None

    return ParsedResultFile(
        experiment_type=experiment_type,
        model_name=model_name,
        date=date_str,
        time=time_str,
        is_ablation=is_ablation,
        filepath=filepath,
    )


# Model family classification
class ModelFamily:
    LLAMA_8B = "llama-8b"
    LLAMA_70B = "llama-70b"
    GEMMA_2B = "gemma-2b"
    GEMMA_9B = "gemma-9b"
    GEMMA_27B = "gemma-27b"
    FINETUNED_8B = "finetuned-8b"
    UNKNOWN = "unknown"


@dataclass
class CanonicalModelInfo:
    """Canonical model information."""
    display_name: str       # Human-readable name for plots
    family: str             # Model family (from ModelFamily)
    is_finetuned: bool      # Whether this is a finetuned model
    finetuning_config: Optional[str] = None  # e.g., "ratio-80-20", "run-1"
    param_count_b: float = 0  # Parameter count in billions (for sorting)


def canonicalize_model_name(raw_model_name: str) -> CanonicalModelInfo:
    """
    Convert a raw model name to canonical model information.

    This function handles model names from both:
    1. Filenames (e.g., "Meta-Llama-3.1-8B-Instruct", "gemma-2-9b-it-res-16k-layer-20")
    2. JSON config (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct", paths with "run-1-merged")

    Args:
        raw_model_name: The model name from filename or JSON config

    Returns:
        CanonicalModelInfo with display name, family, and finetuning info
    """
    name_lower = raw_model_name.lower()

    # Check for finetuned models first (they have specific patterns)
    # Patterns: run-1-merged, run-2-mixed-80-20-merged, ratio-XX-YY-merged
    finetuned_patterns = [
        (r'run-(\d+)-merged', 'run-{}'),
        (r'run-(\d+)-mixed-(\d+)-(\d+)-merged', 'run-{}-mixed-{}-{}'),
        (r'ratio-(\d+)-(\d+)-merged', 'ratio-{}-{}'),
    ]

    for pattern, config_fmt in finetuned_patterns:
        match = re.search(pattern, name_lower)
        if match:
            groups = match.groups()
            config = config_fmt.format(*groups)
            return CanonicalModelInfo(
                display_name=f"Llama 8B FT ({config})",
                family=ModelFamily.FINETUNED_8B,
                is_finetuned=True,
                finetuning_config=config,
                param_count_b=8.0,
            )

    # Check for Llama models
    if 'llama' in name_lower:
        if '70b' in name_lower:
            return CanonicalModelInfo(
                display_name="Llama 3.3 70B",
                family=ModelFamily.LLAMA_70B,
                is_finetuned=False,
                param_count_b=70.0,
            )
        elif '8b' in name_lower:
            return CanonicalModelInfo(
                display_name="Llama 3.1 8B",
                family=ModelFamily.LLAMA_8B,
                is_finetuned=False,
                param_count_b=8.0,
            )

    # Check for Gemma models
    if 'gemma' in name_lower:
        if '27b' in name_lower:
            return CanonicalModelInfo(
                display_name="Gemma 2 27B",
                family=ModelFamily.GEMMA_27B,
                is_finetuned=False,
                param_count_b=27.0,
            )
        elif '9b' in name_lower:
            return CanonicalModelInfo(
                display_name="Gemma 2 9B",
                family=ModelFamily.GEMMA_9B,
                is_finetuned=False,
                param_count_b=9.0,
            )
        elif '2b' in name_lower:
            return CanonicalModelInfo(
                display_name="Gemma 2 2B",
                family=ModelFamily.GEMMA_2B,
                is_finetuned=False,
                param_count_b=2.0,
            )

    # Fallback: use the raw name
    return CanonicalModelInfo(
        display_name=raw_model_name,
        family=ModelFamily.UNKNOWN,
        is_finetuned=False,
        param_count_b=0,
    )


def get_model_color(model_info: CanonicalModelInfo, scheme: str = "target_models") -> str:
    """
    Get a consistent color for a model based on its family.

    Args:
        model_info: The canonical model info
        scheme: Color scheme to use:
            - "target_models": Blue/green scheme for experiment 1 (target model analysis)
            - "boost_analysis": Purple/teal scheme for experiments 2, 3 (boost/steering analysis)
    """
    if scheme == "target_models":
        # Blue/green scheme for experiment 1
        family_colors = {
            ModelFamily.LLAMA_70B: "#1A5276",    # Dark blue
            ModelFamily.LLAMA_8B: "#5DADE2",     # Light blue
            ModelFamily.GEMMA_27B: "#1E8449",    # Dark green
            ModelFamily.GEMMA_9B: "#27AE60",     # Green
            ModelFamily.GEMMA_2B: "#82E0AA",     # Light green
            ModelFamily.FINETUNED_8B: "#E67E22", # Orange
            ModelFamily.UNKNOWN: "#888888",      # Gray
        }
    elif scheme == "boost_analysis":
        # Purple/teal scheme for experiments 2, 3 (colorblind-friendly)
        family_colors = {
            ModelFamily.LLAMA_70B: "#6A0DAD",    # Deep purple
            ModelFamily.LLAMA_8B: "#9B59B6",     # Medium purple
            ModelFamily.GEMMA_27B: "#0E6655",    # Deep teal
            ModelFamily.GEMMA_9B: "#17A2B8",     # Teal
            ModelFamily.GEMMA_2B: "#48C9B0",     # Light teal
            ModelFamily.FINETUNED_8B: "#D4AC0D", # Gold
            ModelFamily.UNKNOWN: "#888888",      # Gray
        }
    else:
        # Default fallback
        family_colors = {
            ModelFamily.UNKNOWN: "#888888",
        }

    return family_colors.get(model_info.family, "#888888")


# Convenience function combining parsing and canonicalization
def get_model_info_from_file(filepath: Path | str) -> Optional[tuple[ParsedResultFile, CanonicalModelInfo]]:
    """
    Parse a result file and get canonical model info.

    Args:
        filepath: Path to the result file

    Returns:
        Tuple of (ParsedResultFile, CanonicalModelInfo) or None if parsing fails
    """
    parsed = parse_results_filename(filepath)
    if parsed is None:
        return None

    model_info = canonicalize_model_name(parsed.model_name)
    return parsed, model_info


if __name__ == "__main__":
    # Test the parsing with actual filenames
    test_files = [
        "experiment_multi_boost_Meta-Llama-3.1-8B-Instruct_20251127_124258.json",
        "experiment_multi_boost_gemma-2-9b-it-res-16k-layer-20_20251127_161615.json",
        "experiment_results_Meta-Llama-3.3-70B-Instruct_20251023_145148_with_ablation.json",
        "experiment_results_gemma-2-27b-it-res-131k-layer-22_20251125_123304.json",
        "experiment_results_ratio-80-20-merged_20251125_161937.json",
        "experiment_results_run-1-merged_20251030_130016.json",
        "experiment_results_run-2-mixed-80-20-merged_20251125_141711.json",
    ]

    print("Testing parse_results_filename and canonicalize_model_name:")
    print("=" * 80)

    for filename in test_files:
        parsed = parse_results_filename(filename)
        if parsed:
            model_info = canonicalize_model_name(parsed.model_name)
            print(f"\nFile: {filename}")
            print(f"  Experiment type: {parsed.experiment_type}")
            print(f"  Raw model name: {parsed.model_name}")
            print(f"  Date/Time: {parsed.date} {parsed.time}")
            print(f"  Is ablation: {parsed.is_ablation}")
            print(f"  → Display name: {model_info.display_name}")
            print(f"  → Family: {model_info.family}")
            print(f"  → Is finetuned: {model_info.is_finetuned}")
            if model_info.finetuning_config:
                print(f"  → FT config: {model_info.finetuning_config}")
        else:
            print(f"\nFailed to parse: {filename}")
