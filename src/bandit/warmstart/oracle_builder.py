"""
Oracle Label Builder for Warm-Start Training.

Constructs oracle labels by merging Phase 6b (baseline) and Phase 6d
(NS-MAS Fixed Slow) results to determine the optimal routing action
for each problem.

Oracle Decision Logic:
- Both correct -> Fast (cheaper)
- Only Slow correct -> Slow (need verification)
- Only Fast correct -> Fast
- Both wrong -> Fast (fail cheap)

Usage:
    python -m src.bandit.warmstart.oracle_builder \\
        --results-dir results \\
        --output data/oracle_training_set.json

    # Programmatic usage
    builder = OracleBuilder()
    labels = builder.build_oracle(
        fast_results_dir="results/baseline_gpt4o_mini",
        slow_results_dir="results/nsmas_fixed_slow"
    )
    builder.save(labels, "data/oracle_training_set.json")
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OracleLabel:
    """Oracle label for a single problem."""

    problem_id: str
    question: str
    ground_truth: int
    variant: str

    # Fast path results (GPT-4o-mini baseline)
    fast_answer: Optional[int]
    fast_correct: bool

    # Slow path results (NS-MAS Fixed Slow / GVR)
    slow_answer: Optional[int]
    slow_correct: bool

    # Oracle decision
    oracle_action: str  # "fast" or "slow"
    oracle_reason: str  # Explanation for decision

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OracleLabel":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OracleStats:
    """Statistics about oracle label distribution."""

    total: int = 0
    both_correct: int = 0
    only_slow_correct: int = 0
    only_fast_correct: int = 0
    both_wrong: int = 0

    @property
    def fast_rate(self) -> float:
        """Fraction of problems where fast is optimal."""
        if self.total == 0:
            return 0.0
        return (self.both_correct + self.only_fast_correct + self.both_wrong) / self.total

    @property
    def oracle_accuracy(self) -> float:
        """Theoretical accuracy with perfect routing."""
        if self.total == 0:
            return 0.0
        return (self.both_correct + self.only_slow_correct + self.only_fast_correct) / self.total

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "fast_rate": self.fast_rate,
            "oracle_accuracy": self.oracle_accuracy
        }


class OracleBuilder:
    """
    Builds oracle labels from experiment results.

    Merges fast path (baseline) and slow path (NS-MAS) results to
    determine optimal routing decisions for warm-start training.
    """

    def __init__(self, variants: list[str] = None):
        """
        Initialize oracle builder.

        Args:
            variants: Dataset variants to process (default: all)
        """
        self.variants = variants or ["base", "p1", "p2", "noop"]

    def load_results(self, results_dir: Path, variant: str) -> dict[str, dict]:
        """
        Load results from JSONL file.

        Args:
            results_dir: Directory containing result files
            variant: Dataset variant (base, p1, p2, noop)

        Returns:
            Dict mapping problem_id to result record
        """
        result_file = results_dir / f"{variant}.jsonl"
        if not result_file.exists():
            logger.warning(f"Results file not found: {result_file}")
            return {}

        results = {}
        with open(result_file, "r") as f:
            for line in f:
                record = json.loads(line)
                results[record["problem_id"]] = record

        return results

    def determine_oracle_action(
        self,
        fast_correct: bool,
        slow_correct: bool
    ) -> tuple[str, str]:
        """
        Determine optimal routing action.

        Args:
            fast_correct: Whether fast path got correct answer
            slow_correct: Whether slow path got correct answer

        Returns:
            Tuple of (action, reason)
        """
        if fast_correct and slow_correct:
            return "fast", "both_correct"
        elif slow_correct and not fast_correct:
            return "slow", "only_slow_correct"
        elif fast_correct and not slow_correct:
            return "fast", "only_fast_correct"
        else:
            return "fast", "both_wrong"

    def build_oracle_for_variant(
        self,
        fast_results: dict[str, dict],
        slow_results: dict[str, dict],
        variant: str
    ) -> tuple[list[OracleLabel], OracleStats]:
        """
        Build oracle labels for a single variant.

        Args:
            fast_results: Fast path results by problem_id
            slow_results: Slow path results by problem_id
            variant: Dataset variant name

        Returns:
            Tuple of (labels list, stats)
        """
        labels = []
        stats = OracleStats()

        # Find common problem IDs
        common_ids = set(fast_results.keys()) & set(slow_results.keys())
        logger.info(f"Variant {variant}: {len(common_ids)} common problems")

        for problem_id in sorted(common_ids):
            fast = fast_results[problem_id]
            slow = slow_results[problem_id]

            fast_correct = fast.get("correct", False)
            slow_correct = slow.get("correct", False)

            action, reason = self.determine_oracle_action(fast_correct, slow_correct)

            # Update stats
            stats.total += 1
            if reason == "both_correct":
                stats.both_correct += 1
            elif reason == "only_slow_correct":
                stats.only_slow_correct += 1
            elif reason == "only_fast_correct":
                stats.only_fast_correct += 1
            else:
                stats.both_wrong += 1

            label = OracleLabel(
                problem_id=problem_id,
                question=fast.get("question", ""),
                ground_truth=fast.get("ground_truth", 0),
                variant=variant,
                fast_answer=fast.get("final_answer"),
                fast_correct=fast_correct,
                slow_answer=slow.get("final_answer"),
                slow_correct=slow_correct,
                oracle_action=action,
                oracle_reason=reason
            )
            labels.append(label)

        return labels, stats

    def build_oracle(
        self,
        fast_results_dir: str | Path,
        slow_results_dir: str | Path
    ) -> tuple[list[OracleLabel], dict[str, OracleStats]]:
        """
        Build oracle labels from experiment results.

        Args:
            fast_results_dir: Directory with fast path (baseline) results
            slow_results_dir: Directory with slow path (NS-MAS) results

        Returns:
            Tuple of (all labels, stats by variant)
        """
        fast_results_dir = Path(fast_results_dir)
        slow_results_dir = Path(slow_results_dir)

        all_labels = []
        all_stats = {}

        for variant in self.variants:
            logger.info(f"Processing variant: {variant}")

            fast_results = self.load_results(fast_results_dir, variant)
            slow_results = self.load_results(slow_results_dir, variant)

            if not fast_results or not slow_results:
                logger.warning(f"Skipping variant {variant}: missing results")
                continue

            labels, stats = self.build_oracle_for_variant(
                fast_results, slow_results, variant
            )

            all_labels.extend(labels)
            all_stats[variant] = stats

            logger.info(
                f"  {variant}: {stats.total} labels, "
                f"fast_rate={stats.fast_rate:.2%}, "
                f"oracle_acc={stats.oracle_accuracy:.2%}"
            )

        # Compute overall stats
        overall = OracleStats()
        for stats in all_stats.values():
            overall.total += stats.total
            overall.both_correct += stats.both_correct
            overall.only_slow_correct += stats.only_slow_correct
            overall.only_fast_correct += stats.only_fast_correct
            overall.both_wrong += stats.both_wrong

        all_stats["overall"] = overall

        logger.info(
            f"Overall: {overall.total} labels, "
            f"fast_rate={overall.fast_rate:.2%}, "
            f"oracle_acc={overall.oracle_accuracy:.2%}"
        )

        return all_labels, all_stats

    def save(
        self,
        labels: list[OracleLabel],
        output_path: str | Path,
        stats: dict[str, OracleStats] = None
    ) -> None:
        """
        Save oracle labels to JSON file.

        Args:
            labels: List of oracle labels
            output_path: Output file path
            stats: Optional stats to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "labels": [label.to_dict() for label in labels],
            "stats": {k: v.to_dict() for k, v in stats.items()} if stats else None,
            "total": len(labels)
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Oracle labels saved to {output_path}")

    @staticmethod
    def load(path: str | Path) -> tuple[list[OracleLabel], dict]:
        """
        Load oracle labels from JSON file.

        Args:
            path: Path to oracle JSON file

        Returns:
            Tuple of (labels, stats dict)
        """
        with open(path, "r") as f:
            data = json.load(f)

        labels = [OracleLabel.from_dict(d) for d in data["labels"]]
        stats = data.get("stats", {})

        return labels, stats


def main():
    """CLI for oracle builder."""
    parser = argparse.ArgumentParser(
        description="Build oracle labels from experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory"
    )
    parser.add_argument(
        "--fast-dir",
        type=str,
        default=None,
        help="Fast path results directory (default: baseline_gpt4o_mini)"
    )
    parser.add_argument(
        "--slow-dir",
        type=str,
        default=None,
        help="Slow path results directory (default: nsmas_fixed_slow)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/oracle_training_set.json",
        help="Output path for oracle labels"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "p1", "p2", "noop"],
        help="Dataset variants to process"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Determine paths
    results_dir = Path(args.results_dir)
    fast_dir = Path(args.fast_dir) if args.fast_dir else results_dir / "baseline_gpt4o_mini"
    slow_dir = Path(args.slow_dir) if args.slow_dir else results_dir / "nsmas_fixed_slow"

    logger.info(f"Fast results: {fast_dir}")
    logger.info(f"Slow results: {slow_dir}")

    # Build oracle
    builder = OracleBuilder(variants=args.variants)
    labels, stats = builder.build_oracle(fast_dir, slow_dir)

    # Save
    builder.save(labels, args.output, stats)

    # Report
    overall = stats.get("overall", OracleStats())
    print(f"\nOracle Labels Built:")
    print(f"  Total labels: {len(labels)}")
    print(f"  Both correct: {overall.both_correct} ({overall.both_correct/overall.total*100:.1f}%)")
    print(f"  Only slow correct: {overall.only_slow_correct} ({overall.only_slow_correct/overall.total*100:.1f}%)")
    print(f"  Only fast correct: {overall.only_fast_correct} ({overall.only_fast_correct/overall.total*100:.1f}%)")
    print(f"  Both wrong: {overall.both_wrong} ({overall.both_wrong/overall.total*100:.1f}%)")
    print(f"  Oracle fast rate: {overall.fast_rate*100:.1f}%")
    print(f"  Oracle accuracy: {overall.oracle_accuracy*100:.1f}%")
    print(f"  Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
