"""Shared dataclasses for experiment results."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureInfo:
    """Simple feature info holder."""

    index_in_sae: int
    label: str


@dataclass
class TrialResult:
    """Results from a single trial."""

    prompt: str
    feature_index_in_sae: int
    feature_label: str
    threshold: float  # For multi-boost, this is the boost_value
    seed: int
    response: str
    score: dict
    error: Optional[str] = None


@dataclass
class FeatureResult:
    """Results for a single feature."""

    feature_index_in_sae: int
    feature_label: str
    threshold: Optional[float]  # For multi-boost, this is baseline_threshold
    trials: List[TrialResult]
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    """Full experiment results."""

    experiment_config: dict  # Using dict to avoid circular import
    results_by_feature: List[FeatureResult]
    # Optional fields for multi-boost experiments
    n_boost_levels: Optional[int] = None
    boost_range_std: Optional[float] = None
