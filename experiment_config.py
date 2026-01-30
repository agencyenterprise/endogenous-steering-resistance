"""Configuration for endogenous steering resistance experiment."""

import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from pathlib import Path

EXP_ROOT = Path(__file__).parent


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Prompts for the model
    prompts_file: str

    # Model configuration
    model_name: str  # e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Feature labels
    labels_file: Optional[str] = None
    _labels: Optional[Dict[int, str]] = None

    # Judge configuration
    judge_model_name: str = "claude-sonnet-4-5-20250929"

    # Steering configuration
    # If True, do not apply any feature interventions (i.e., "zero steering" baseline).
    disable_steering: bool = False

    # Threshold finding parameters
    target_score_normalized: float = 0.5
    threshold_n_trials: int = 20
    threshold_samples_per_trial: int = 1  # Number of samples to average per trial
    per_prompt_calibration: bool = False  # If True, calibrate threshold for each prompt-feature pair
    threshold_lower_bound: float = 0.0
    threshold_upper_bound: float = 5.0
    threshold_prior_mean: float = 1.0
    threshold_prior_std: float = 0.34

    # Experiment parameters
    n_possible_seeds: int = 1000000
    seed_start: int = 0
    max_completion_tokens: int = 512
    n_trials_per_feature: int = 10
    n_features: int = 80
    n_simultaneous_features: int = 10

    # Feature filtering
    min_feature_concreteness: float = 65.0  # Minimum concreteness score for features

    # Provenance tracking
    source_results_file: Optional[str] = None  # Path to --from-results file if used

    def get_prompts(self) -> List[str]:
        """Load prompts from file."""
        if not hasattr(self, "prompts"):
            with open(EXP_ROOT / self.prompts_file, "r") as f:
                self.prompts = [line.strip("\n") for line in f.readlines()]
        return self.prompts

    def get_labels(self, num_features: Optional[int] = None) -> Dict[int, str]:
        """
        Load feature labels from CSV file, or generate dummy labels if no file provided.

        Args:
            num_features: If labels_file is None, generate this many dummy labels

        Returns dictionary mapping feature index to label.
        """
        if self._labels is not None:
            return self._labels

        if self.labels_file is None:
            # Generate dummy labels if no labels file provided
            if num_features is None:
                # Default to a reasonable number for modern SAEs
                num_features = 16384
            self._labels = {i: f"feature_{i}" for i in range(num_features)}
            return self._labels

        self._labels = {}
        with open(EXP_ROOT / self.labels_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index_in_sae"])
                label = row["label"]
                # Use the label if it exists, otherwise use a placeholder
                if label:
                    self._labels[idx] = label
                else:
                    self._labels[idx] = f"feature_{idx}"

        return self._labels

    def get_threshold_cache_file(self) -> str:
        """Get the path to the threshold cache file based on model name."""
        short_model_name = self.model_name.split("/")[-1]
        return str(EXP_ROOT / "data" / f"threshold_cache_{short_model_name}.json")

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = {
            "prompts_file": self.prompts_file,
            "model_name": self.model_name,
            "labels_file": self.labels_file,
            "judge_model_name": self.judge_model_name,
            "disable_steering": self.disable_steering,
            "target_score_normalized": self.target_score_normalized,
            "threshold_n_trials": self.threshold_n_trials,
            "threshold_samples_per_trial": self.threshold_samples_per_trial,
            "per_prompt_calibration": self.per_prompt_calibration,
            "threshold_lower_bound": self.threshold_lower_bound,
            "threshold_upper_bound": self.threshold_upper_bound,
            "threshold_prior_mean": self.threshold_prior_mean,
            "threshold_prior_std": self.threshold_prior_std,
            "n_possible_seeds": self.n_possible_seeds,
            "seed_start": self.seed_start,
            "max_completion_tokens": self.max_completion_tokens,
            "n_trials_per_feature": self.n_trials_per_feature,
            "n_features": self.n_features,
            "n_simultaneous_features": self.n_simultaneous_features,
            "min_feature_concreteness": self.min_feature_concreteness,
            "source_results_file": self.source_results_file,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create from dict."""
        return cls(**data)
