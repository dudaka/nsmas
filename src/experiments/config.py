"""
Experiment Configuration

Defines configuration for Phase 6 experiment runs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List


class RunType(str, Enum):
    """Type of experiment run."""
    BASELINE_GPT4O = "baseline_gpt4o"
    BASELINE_GPT4O_MINI = "baseline_gpt4o_mini"
    BASELINE_SC = "baseline_sc"  # CoT + Self-Consistency
    NSMAS_FIXED_SLOW = "nsmas_fixed_slow"
    NSMAS_BANDIT = "nsmas_bandit"
    NSMAS_RANDOM = "nsmas_random"


class DatasetVariant(str, Enum):
    """Dataset variant."""
    BASE = "base"
    P1 = "p1"
    P2 = "p2"
    NOOP = "noop"


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    run_type: RunType
    name: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 30.0

    # For bandit runs
    bandit_cold_start: bool = True
    bandit_model_path: Optional[Path] = None

    # For random router
    random_fast_probability: float = 0.5


@dataclass
class ExperimentConfig:
    """Configuration for the experiment pipeline."""

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("output"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    # Dataset selection
    variants: List[DatasetVariant] = field(
        default_factory=lambda: [
            DatasetVariant.BASE,
            DatasetVariant.P1,
            DatasetVariant.P2,
            DatasetVariant.NOOP,
        ]
    )

    # Execution settings
    max_concurrent: int = 5  # Async workers (conservative for rate limits)
    checkpoint_interval: int = 50  # Save every N problems
    retry_on_error: int = 3  # Retry failed API calls

    # Rate limiting
    requests_per_minute: int = 400  # Conservative for 500 RPM limit
    backoff_base: float = 2.0  # Exponential backoff base
    backoff_max: float = 60.0  # Max backoff seconds

    # Reproducibility
    seed: int = 42

    # Pilot settings
    pilot_size: int = 500
    pilot_templates: int = 10  # Use first N templates for pilot

    def get_dataset_path(self, variant: DatasetVariant) -> Path:
        """Get path to dataset file for a variant."""
        return self.data_dir / f"gsm_{variant.value}.jsonl"

    def get_results_path(self, run_type: RunType, variant: DatasetVariant) -> Path:
        """Get path to results file for a run."""
        return self.results_dir / run_type.value / f"{variant.value}.jsonl"

    def get_checkpoint_path(self, run_type: RunType, variant: DatasetVariant) -> Path:
        """Get path to checkpoint file for a run."""
        return self.results_dir / "checkpoints" / f"{run_type.value}_{variant.value}.json"


# Predefined run configurations
BASELINE_GPT4O_CONFIG = RunConfig(
    run_type=RunType.BASELINE_GPT4O,
    name="GPT-4o Zero-Shot CoT",
    model="gpt-4o",
    temperature=0.0,
)

BASELINE_GPT4O_MINI_CONFIG = RunConfig(
    run_type=RunType.BASELINE_GPT4O_MINI,
    name="GPT-4o-mini Zero-Shot CoT",
    model="gpt-4o-mini",
    temperature=0.0,
)

NSMAS_FIXED_SLOW_CONFIG = RunConfig(
    run_type=RunType.NSMAS_FIXED_SLOW,
    name="NS-MAS Fixed Slow (GVR Always)",
    model="gpt-4o",
    temperature=0.0,
)

NSMAS_BANDIT_CONFIG = RunConfig(
    run_type=RunType.NSMAS_BANDIT,
    name="NS-MAS Bandit (Cold Start)",
    model="gpt-4o",
    temperature=0.0,
    bandit_cold_start=True,
)

NSMAS_RANDOM_CONFIG = RunConfig(
    run_type=RunType.NSMAS_RANDOM,
    name="NS-MAS Random Router (50/50)",
    model="gpt-4o",
    temperature=0.0,
    random_fast_probability=0.5,
)
