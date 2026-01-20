"""
Metrics Collection and Result Types

Defines data structures for experiment results and metrics calculation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class PathTaken(str, Enum):
    """Which path was taken for solving."""
    FAST = "fast"
    SLOW = "slow"
    BASELINE = "baseline"  # For pure LLM baselines


@dataclass
class ExperimentResult:
    """Result for a single problem."""

    # Problem identification
    problem_id: str
    template_id: str
    variant: str

    # Input
    question: str
    ground_truth: int

    # System info
    system: str  # e.g., "baseline_gpt4o", "nsmas_bandit"
    path_taken: PathTaken

    # Output
    final_answer: Optional[int]
    correct: bool

    # Performance metrics
    tokens_input: int = 0
    tokens_output: int = 0
    execution_time_ms: float = 0.0

    # For NS-MAS runs
    iterations: int = 0
    solver_feedback: Optional[str] = None
    error_type: Optional[str] = None
    self_corrected: bool = False

    # For bandit runs
    bandit_action: Optional[str] = None  # "fast" or "slow"
    bandit_probability: Optional[float] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["path_taken"] = self.path_taken.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        d = d.copy()
        d["path_taken"] = PathTaken(d["path_taken"])
        return cls(**d)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.tokens_input + self.tokens_output


@dataclass
class MetricsSummary:
    """Aggregated metrics for a run."""

    # Identification
    run_type: str
    variant: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Counts
    total_problems: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    error_count: int = 0

    # Accuracy
    accuracy: float = 0.0

    # Token usage
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    avg_tokens_per_problem: float = 0.0
    tokens_per_correct: float = 0.0

    # Timing
    total_time_ms: float = 0.0
    avg_time_per_problem_ms: float = 0.0

    # NS-MAS specific
    avg_iterations: float = 0.0
    self_correction_rate: float = 0.0

    # Path distribution (for bandit/random)
    fast_path_count: int = 0
    slow_path_count: int = 0
    fast_path_accuracy: float = 0.0
    slow_path_accuracy: float = 0.0


class MetricsCollector:
    """Collects and aggregates experiment results."""

    def __init__(self):
        self.results: List[ExperimentResult] = []

    def add(self, result: ExperimentResult):
        """Add a result."""
        self.results.append(result)

    def add_batch(self, results: List[ExperimentResult]):
        """Add multiple results."""
        self.results.extend(results)

    def clear(self):
        """Clear all results."""
        self.results = []

    def compute_summary(self, run_type: str, variant: str) -> MetricsSummary:
        """Compute aggregated metrics."""
        if not self.results:
            return MetricsSummary(run_type=run_type, variant=variant)

        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        errors = sum(1 for r in self.results if r.error_type is not None)

        total_tokens_in = sum(r.tokens_input for r in self.results)
        total_tokens_out = sum(r.tokens_output for r in self.results)
        total_tokens = total_tokens_in + total_tokens_out

        total_time = sum(r.execution_time_ms for r in self.results)

        # Path distribution
        fast_results = [r for r in self.results if r.path_taken == PathTaken.FAST]
        slow_results = [r for r in self.results if r.path_taken == PathTaken.SLOW]

        fast_correct = sum(1 for r in fast_results if r.correct) if fast_results else 0
        slow_correct = sum(1 for r in slow_results if r.correct) if slow_results else 0

        # Self-correction (NS-MAS)
        self_corrected = sum(1 for r in self.results if r.self_corrected)
        avg_iterations = (
            sum(r.iterations for r in self.results) / total if total > 0 else 0
        )

        return MetricsSummary(
            run_type=run_type,
            variant=variant,
            total_problems=total,
            correct_count=correct,
            incorrect_count=total - correct - errors,
            error_count=errors,
            accuracy=correct / total if total > 0 else 0.0,
            total_tokens_input=total_tokens_in,
            total_tokens_output=total_tokens_out,
            avg_tokens_per_problem=total_tokens / total if total > 0 else 0.0,
            tokens_per_correct=total_tokens / correct if correct > 0 else 0.0,
            total_time_ms=total_time,
            avg_time_per_problem_ms=total_time / total if total > 0 else 0.0,
            avg_iterations=avg_iterations,
            self_correction_rate=self_corrected / total if total > 0 else 0.0,
            fast_path_count=len(fast_results),
            slow_path_count=len(slow_results),
            fast_path_accuracy=(
                fast_correct / len(fast_results) if fast_results else 0.0
            ),
            slow_path_accuracy=(
                slow_correct / len(slow_results) if slow_results else 0.0
            ),
        )

    def save_results(self, path: Path):
        """Save results to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict()) + "\n")

    def load_results(self, path: Path) -> int:
        """Load results from JSONL file. Returns count loaded."""
        if not path.exists():
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.results.append(ExperimentResult.from_dict(json.loads(line)))
                    count += 1
        return count

    def get_processed_ids(self) -> set:
        """Get set of already processed problem IDs."""
        return {r.problem_id for r in self.results}
