"""
Offline Policy Evaluation (OPE) Tuner for Phase 6c.

Uses baseline logs from Phase 6b to tune bandit hyperparameters without
additional API calls. Constructs oracle labels by matching GPT-4o (slow path)
and GPT-4o-mini (fast path) results.

Key hyperparameters to tune:
- gamma_scale: SquareCB greediness (higher = more greedy)
- alpha_incorrect: Penalty for incorrect answers
- beta_cost: Weight for action costs
"""

import json
import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.bandit.config import BanditConfig
from src.bandit.reward import (
    calculate_loss,
    compute_oracle_action,
    compute_regret,
    RewardTracker,
)

logger = logging.getLogger(__name__)


@dataclass
class OracleRecord:
    """A problem with oracle labels from both paths."""
    problem_id: str
    question: str
    ground_truth: int
    variant: str
    template_id: str

    # Fast path (GPT-4o-mini) outcome
    fast_answer: Optional[int]
    fast_correct: bool

    # Slow path (GPT-4o) outcome
    slow_answer: Optional[int]
    slow_correct: bool

    # Computed oracle
    oracle_action: int  # 0=fast, 1=slow

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "variant": self.variant,
            "template_id": self.template_id,
            "fast_answer": self.fast_answer,
            "fast_correct": self.fast_correct,
            "slow_answer": self.slow_answer,
            "slow_correct": self.slow_correct,
            "oracle_action": self.oracle_action,
        }


@dataclass
class OPEResult:
    """Result from offline policy evaluation."""
    config: Dict
    accuracy: float
    avg_loss: float
    avg_regret: float
    fast_rate: float
    cost_savings: float  # vs always-slow baseline

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "accuracy": self.accuracy,
            "avg_loss": self.avg_loss,
            "avg_regret": self.avg_regret,
            "fast_rate": self.fast_rate,
            "cost_savings": self.cost_savings,
        }


@dataclass
class OPETuningResult:
    """Result from hyperparameter tuning."""
    best_config: Dict
    best_result: OPEResult
    all_results: List[OPEResult]
    oracle_stats: Dict

    def to_dict(self) -> dict:
        return {
            "best_config": self.best_config,
            "best_result": self.best_result.to_dict(),
            "all_results": [r.to_dict() for r in self.all_results],
            "oracle_stats": self.oracle_stats,
        }


class OPETuner:
    """
    Offline Policy Evaluation tuner for bandit hyperparameters.

    Uses baseline logs to construct oracle labels and evaluate
    different hyperparameter configurations without API calls.
    """

    def __init__(
        self,
        results_dir: Path,
        fast_run_name: str = "baseline_gpt4o_mini",
        slow_run_name: str = "baseline_gpt4o",
    ):
        """
        Initialize OPE tuner.

        Args:
            results_dir: Directory containing baseline results
            fast_run_name: Name of fast path baseline run
            slow_run_name: Name of slow path baseline run
        """
        self.results_dir = Path(results_dir)
        self.fast_run_name = fast_run_name
        self.slow_run_name = slow_run_name
        self.oracle_records: List[OracleRecord] = []

    def load_baseline_results(self, run_name: str) -> Dict[str, dict]:
        """
        Load baseline results from JSONL files.

        Args:
            run_name: Name of the run (directory name)

        Returns:
            Dict mapping problem_id to result record
        """
        run_dir = self.results_dir / run_name
        results = {}

        for jsonl_file in run_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        results[record["problem_id"]] = record

        logger.info(f"Loaded {len(results)} results from {run_name}")
        return results

    def construct_oracle(self) -> List[OracleRecord]:
        """
        Construct oracle labels by matching fast and slow path results.

        Returns:
            List of OracleRecord with both path outcomes
        """
        # Load both baselines
        fast_results = self.load_baseline_results(self.fast_run_name)
        slow_results = self.load_baseline_results(self.slow_run_name)

        # Match by problem_id
        matched = 0
        unmatched = 0

        for problem_id, slow_record in slow_results.items():
            if problem_id in fast_results:
                fast_record = fast_results[problem_id]

                # Compute oracle action
                oracle_action = compute_oracle_action(
                    fast_correct=fast_record["correct"],
                    slow_correct=slow_record["correct"],
                )

                oracle_record = OracleRecord(
                    problem_id=problem_id,
                    question=slow_record["question"],
                    ground_truth=slow_record["ground_truth"],
                    variant=slow_record["variant"],
                    template_id=slow_record["template_id"],
                    fast_answer=fast_record.get("final_answer"),
                    fast_correct=fast_record["correct"],
                    slow_answer=slow_record.get("final_answer"),
                    slow_correct=slow_record["correct"],
                    oracle_action=oracle_action,
                )
                self.oracle_records.append(oracle_record)
                matched += 1
            else:
                unmatched += 1

        logger.info(f"Constructed oracle: {matched} matched, {unmatched} unmatched")
        return self.oracle_records

    def compute_oracle_stats(self) -> Dict:
        """
        Compute statistics about the oracle decisions.

        Returns:
            Dict with oracle statistics
        """
        if not self.oracle_records:
            self.construct_oracle()

        total = len(self.oracle_records)
        fast_correct = sum(1 for r in self.oracle_records if r.fast_correct)
        slow_correct = sum(1 for r in self.oracle_records if r.slow_correct)
        both_correct = sum(1 for r in self.oracle_records if r.fast_correct and r.slow_correct)
        both_wrong = sum(1 for r in self.oracle_records if not r.fast_correct and not r.slow_correct)
        only_slow_correct = sum(1 for r in self.oracle_records if not r.fast_correct and r.slow_correct)
        oracle_fast = sum(1 for r in self.oracle_records if r.oracle_action == 0)

        # By variant
        by_variant = {}
        for r in self.oracle_records:
            if r.variant not in by_variant:
                by_variant[r.variant] = {"total": 0, "fast_correct": 0, "slow_correct": 0, "oracle_fast": 0}
            by_variant[r.variant]["total"] += 1
            by_variant[r.variant]["fast_correct"] += int(r.fast_correct)
            by_variant[r.variant]["slow_correct"] += int(r.slow_correct)
            by_variant[r.variant]["oracle_fast"] += int(r.oracle_action == 0)

        return {
            "total": total,
            "fast_accuracy": fast_correct / total if total else 0,
            "slow_accuracy": slow_correct / total if total else 0,
            "both_correct_rate": both_correct / total if total else 0,
            "both_wrong_rate": both_wrong / total if total else 0,
            "only_slow_correct_rate": only_slow_correct / total if total else 0,
            "oracle_fast_rate": oracle_fast / total if total else 0,
            "oracle_slow_rate": (total - oracle_fast) / total if total else 0,
            "by_variant": by_variant,
        }

    def evaluate_policy(
        self,
        policy_fast_rate: float,
        config: BanditConfig,
    ) -> OPEResult:
        """
        Evaluate a simulated policy with given fast rate.

        For cold-start OPE, we simulate a policy that chooses fast path
        with probability `policy_fast_rate` (exploration) and evaluate
        its expected performance.

        Args:
            policy_fast_rate: Probability of choosing fast path
            config: Bandit config with loss parameters

        Returns:
            OPEResult with evaluation metrics
        """
        if not self.oracle_records:
            self.construct_oracle()

        tracker = RewardTracker(config)

        # For each problem, compute expected loss under the policy
        total_loss = 0.0
        total_regret = 0.0
        correct_count = 0

        for record in self.oracle_records:
            # Expected loss = P(fast) * Loss(fast) + P(slow) * Loss(slow)
            loss_fast = calculate_loss(record.fast_correct, 0, config)
            loss_slow = calculate_loss(record.slow_correct, 1, config)

            expected_loss = policy_fast_rate * loss_fast + (1 - policy_fast_rate) * loss_slow
            total_loss += expected_loss

            # Expected accuracy
            expected_correct = (
                policy_fast_rate * int(record.fast_correct) +
                (1 - policy_fast_rate) * int(record.slow_correct)
            )
            correct_count += expected_correct

            # Regret vs oracle
            oracle_loss = calculate_loss(
                record.fast_correct if record.oracle_action == 0 else record.slow_correct,
                record.oracle_action,
                config,
            )
            total_regret += expected_loss - oracle_loss

        total = len(self.oracle_records)

        # Cost savings vs always-slow baseline
        always_slow_loss = sum(
            calculate_loss(r.slow_correct, 1, config)
            for r in self.oracle_records
        )
        cost_savings = (always_slow_loss - total_loss) / always_slow_loss if always_slow_loss else 0

        return OPEResult(
            config={
                "policy_fast_rate": policy_fast_rate,
                "alpha_incorrect": config.alpha_incorrect,
                "beta_cost": config.beta_cost,
            },
            accuracy=correct_count / total if total else 0,
            avg_loss=total_loss / total if total else 0,
            avg_regret=total_regret / total if total else 0,
            fast_rate=policy_fast_rate,
            cost_savings=cost_savings,
        )

    def evaluate_oracle_policy(self, config: BanditConfig) -> OPEResult:
        """
        Evaluate the oracle (optimal) policy.

        This gives us the theoretical best performance.

        Args:
            config: Bandit config with loss parameters

        Returns:
            OPEResult for oracle policy
        """
        if not self.oracle_records:
            self.construct_oracle()

        total_loss = 0.0
        correct_count = 0
        fast_count = 0

        for record in self.oracle_records:
            action = record.oracle_action
            correct = record.fast_correct if action == 0 else record.slow_correct
            loss = calculate_loss(correct, action, config)

            total_loss += loss
            correct_count += int(correct)
            fast_count += int(action == 0)

        total = len(self.oracle_records)

        # Cost savings vs always-slow
        always_slow_loss = sum(
            calculate_loss(r.slow_correct, 1, config)
            for r in self.oracle_records
        )
        cost_savings = (always_slow_loss - total_loss) / always_slow_loss if always_slow_loss else 0

        return OPEResult(
            config={"policy": "oracle"},
            accuracy=correct_count / total if total else 0,
            avg_loss=total_loss / total if total else 0,
            avg_regret=0.0,  # Oracle has zero regret by definition
            fast_rate=fast_count / total if total else 0,
            cost_savings=cost_savings,
        )

    def tune_hyperparameters(
        self,
        alpha_grid: List[float] = [5.0, 10.0, 15.0, 20.0],
        beta_grid: List[float] = [0.5, 1.0, 2.0],
        fast_rate_grid: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    ) -> OPETuningResult:
        """
        Grid search over hyperparameters.

        Args:
            alpha_grid: Values for alpha_incorrect
            beta_grid: Values for beta_cost
            fast_rate_grid: Values for policy fast rate (exploration)

        Returns:
            OPETuningResult with best config and all results
        """
        if not self.oracle_records:
            self.construct_oracle()

        oracle_stats = self.compute_oracle_stats()
        all_results = []

        logger.info(f"Starting grid search: {len(alpha_grid)}x{len(beta_grid)}x{len(fast_rate_grid)} = "
                   f"{len(alpha_grid) * len(beta_grid) * len(fast_rate_grid)} configs")

        for alpha, beta, fast_rate in product(alpha_grid, beta_grid, fast_rate_grid):
            config = BanditConfig(
                alpha_incorrect=alpha,
                beta_cost=beta,
            )
            result = self.evaluate_policy(fast_rate, config)
            result.config["alpha_incorrect"] = alpha
            result.config["beta_cost"] = beta
            all_results.append(result)

        # Find best config by minimizing avg_loss while maintaining accuracy
        # Use Pareto-optimal selection: minimize loss subject to accuracy >= oracle_fast_accuracy
        oracle_result = self.evaluate_oracle_policy(BanditConfig())

        # Sort by loss (lower is better)
        all_results.sort(key=lambda r: r.avg_loss)

        # Find best that maintains reasonable accuracy
        best_result = all_results[0]
        for result in all_results:
            # Prefer lower loss with acceptable accuracy (within 5% of slow path)
            if result.accuracy >= oracle_stats["slow_accuracy"] * 0.95:
                if result.avg_loss < best_result.avg_loss:
                    best_result = result

        logger.info(f"Best config: {best_result.config}")
        logger.info(f"  Accuracy: {best_result.accuracy:.2%}")
        logger.info(f"  Avg Loss: {best_result.avg_loss:.3f}")
        logger.info(f"  Cost Savings: {best_result.cost_savings:.2%}")

        return OPETuningResult(
            best_config=best_result.config,
            best_result=best_result,
            all_results=all_results,
            oracle_stats=oracle_stats,
        )

    def save_results(self, output_path: Path, result: OPETuningResult):
        """Save tuning results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved OPE results to {output_path}")

    def save_oracle(self, output_path: Path):
        """Save oracle records to JSONL for inspection."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in self.oracle_records:
                f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(f"Saved {len(self.oracle_records)} oracle records to {output_path}")


def run_ope_tuning(
    results_dir: str = "results",
    output_dir: str = "results/ope_tuning",
) -> OPETuningResult:
    """
    Run OPE tuning using Phase 6b baseline logs.

    Args:
        results_dir: Directory containing baseline results
        output_dir: Directory to save tuning results

    Returns:
        OPETuningResult with best hyperparameters
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_path = Path(results_dir)
    output_path = Path(output_dir)

    # Initialize tuner
    tuner = OPETuner(results_path)

    # Construct oracle
    tuner.construct_oracle()

    # Print oracle stats
    stats = tuner.compute_oracle_stats()
    print("\n" + "=" * 60)
    print("ORACLE STATISTICS")
    print("=" * 60)
    print(f"Total problems: {stats['total']}")
    print(f"Fast path (GPT-4o-mini) accuracy: {stats['fast_accuracy']:.2%}")
    print(f"Slow path (GPT-4o) accuracy: {stats['slow_accuracy']:.2%}")
    print(f"Both correct: {stats['both_correct_rate']:.2%}")
    print(f"Only slow correct: {stats['only_slow_correct_rate']:.2%}")
    print(f"Both wrong: {stats['both_wrong_rate']:.2%}")
    print(f"Oracle chooses fast: {stats['oracle_fast_rate']:.2%}")
    print(f"Oracle chooses slow: {stats['oracle_slow_rate']:.2%}")

    print("\nBy variant:")
    for variant, vstats in stats["by_variant"].items():
        print(f"  {variant}: fast={vstats['fast_correct']/vstats['total']:.2%}, "
              f"slow={vstats['slow_correct']/vstats['total']:.2%}, "
              f"oracle_fast={vstats['oracle_fast']/vstats['total']:.2%}")

    # Evaluate oracle policy
    oracle_result = tuner.evaluate_oracle_policy(BanditConfig())
    print("\n" + "=" * 60)
    print("ORACLE POLICY PERFORMANCE (Theoretical Best)")
    print("=" * 60)
    print(f"Accuracy: {oracle_result.accuracy:.2%}")
    print(f"Avg Loss: {oracle_result.avg_loss:.3f}")
    print(f"Fast Rate: {oracle_result.fast_rate:.2%}")
    print(f"Cost Savings: {oracle_result.cost_savings:.2%}")

    # Run grid search
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    result = tuner.tune_hyperparameters(
        alpha_grid=[5.0, 10.0, 15.0, 20.0],
        beta_grid=[0.5, 1.0, 2.0],
        fast_rate_grid=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    )

    # Save results
    tuner.save_results(output_path / "tuning_results.json", result)
    tuner.save_oracle(output_path / "oracle_records.jsonl")

    # Print best config
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Alpha (incorrect penalty): {result.best_config.get('alpha_incorrect', 10.0)}")
    print(f"Beta (cost weight): {result.best_config.get('beta_cost', 1.0)}")
    print(f"Policy fast rate: {result.best_config.get('policy_fast_rate', 0.5):.2%}")
    print(f"\nExpected performance:")
    print(f"  Accuracy: {result.best_result.accuracy:.2%}")
    print(f"  Avg Loss: {result.best_result.avg_loss:.3f}")
    print(f"  Cost Savings: {result.best_result.cost_savings:.2%}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OPE hyperparameter tuning")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing baseline results",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ope_tuning",
        help="Directory to save tuning results",
    )

    args = parser.parse_args()
    run_ope_tuning(args.results_dir, args.output_dir)
