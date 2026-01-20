"""
Training Harness for Bandit Router.

Implements:
1. Oracle Construction - Run both paths on a subset to create ground truth labels
2. Policy Training - Train VW on oracle-labeled data
3. Evaluation - Measure accuracy, cost savings, and regret

Training Protocol (from Phase 5 research):
1. Select representative subset (2000 problems) from GSM-Symbolic
2. Run BOTH Fast and Slow paths on every problem
3. Label each problem with oracle action (optimal choice)
4. Train VW policy using offline policy evaluation
5. Evaluate on held-out test set
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from .config import BanditConfig
from .fast_solver import FastSolver
from .reward import (
    RewardTracker,
    calculate_loss,
    compute_oracle_action,
    compute_regret,
)
from .router import BanditRouter

logger = logging.getLogger(__name__)


@dataclass
class OracleRecord:
    """Record from oracle construction."""

    question: str
    answer: int
    fast_answer: Optional[int]
    slow_answer: Optional[int]
    fast_correct: bool
    slow_correct: bool
    oracle_action: int  # 0=fast, 1=slow

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "fast_answer": self.fast_answer,
            "slow_answer": self.slow_answer,
            "fast_correct": self.fast_correct,
            "slow_correct": self.slow_correct,
            "oracle_action": self.oracle_action,
        }


@dataclass
class TrainingMetrics:
    """Metrics from training run."""

    total_examples: int
    accuracy: float
    avg_loss: float
    avg_regret: float
    fast_rate: float
    cost_saved: float  # Relative to always-slow baseline
    training_time_seconds: float

    def to_dict(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "accuracy": self.accuracy,
            "avg_loss": self.avg_loss,
            "avg_regret": self.avg_regret,
            "fast_rate": self.fast_rate,
            "cost_saved": self.cost_saved,
            "training_time_seconds": self.training_time_seconds,
        }


class BanditTrainer:
    """
    Training harness for Bandit Router.

    Handles oracle construction and policy training.

    Usage:
        trainer = BanditTrainer(bandit_config, agent_config)

        # Construct oracle (run both paths on dataset)
        oracle_data = trainer.construct_oracle(dataset)

        # Train policy
        metrics = trainer.train(oracle_data)

        # Evaluate on test set
        eval_metrics = trainer.evaluate(test_data)
    """

    def __init__(
        self,
        bandit_config: Optional[BanditConfig] = None,
        agent_config=None,  # Optional[AgentConfig]
    ):
        """
        Initialize trainer.

        Args:
            bandit_config: Bandit configuration
            agent_config: Agent configuration for slow path (optional)
        """
        self.bandit_config = bandit_config or BanditConfig()
        self.agent_config = agent_config

        self._router: Optional[BanditRouter] = None
        self._fast_solver: Optional[FastSolver] = None
        self._slow_agent = None

    @property
    def router(self) -> BanditRouter:
        """Lazy init router."""
        if self._router is None:
            self._router = BanditRouter(self.bandit_config)
        return self._router

    @property
    def fast_solver(self) -> FastSolver:
        """Lazy init fast solver."""
        if self._fast_solver is None:
            self._fast_solver = FastSolver(self.bandit_config)
        return self._fast_solver

    @property
    def slow_agent(self):
        """Lazy init slow agent."""
        if self._slow_agent is None:
            if self.agent_config is None:
                from src.agent import Agent, AgentConfig

                self.agent_config = AgentConfig()

            from src.agent import Agent

            self._slow_agent = Agent(self.agent_config)
        return self._slow_agent

    def construct_oracle(
        self,
        dataset: list[dict],
        output_path: Optional[Path] = None,
        skip_slow: bool = False,
    ) -> list[OracleRecord]:
        """
        Run both paths on dataset to create oracle labels.

        This is the most expensive part of training - runs both
        fast and slow paths on every problem.

        Args:
            dataset: List of {"question": str, "answer": int}
            output_path: Optional path to save oracle data
            skip_slow: If True, only run fast path (for quick testing)

        Returns:
            List of OracleRecord with oracle labels
        """
        logger.info(f"Constructing oracle for {len(dataset)} examples")
        results = []

        for item in tqdm(dataset, desc="Constructing oracle"):
            question = item["question"]
            answer = item["answer"]

            # Run fast path
            fast_answer = self.fast_solver.solve(question)
            fast_correct = fast_answer == answer

            # Run slow path (expensive)
            if skip_slow:
                slow_answer = None
                slow_correct = True  # Assume slow always works (for testing)
            else:
                slow_result = self.slow_agent.solve(question, expected_answer=answer)
                slow_answer = slow_result.get("final_answer")
                slow_correct = slow_result.get("status") == "success"

            # Compute oracle action
            oracle_action = compute_oracle_action(
                fast_correct, slow_correct, self.bandit_config
            )

            record = OracleRecord(
                question=question,
                answer=answer,
                fast_answer=fast_answer,
                slow_answer=slow_answer,
                fast_correct=fast_correct,
                slow_correct=slow_correct,
                oracle_action=oracle_action,
            )
            results.append(record)

        # Log summary
        fast_correct_count = sum(1 for r in results if r.fast_correct)
        slow_correct_count = sum(1 for r in results if r.slow_correct)
        logger.info(
            f"Oracle construction complete: "
            f"fast_correct={fast_correct_count}/{len(results)}, "
            f"slow_correct={slow_correct_count}/{len(results)}"
        )

        # Save if path provided
        if output_path:
            self._save_oracle(results, output_path)

        return results

    def _save_oracle(self, records: list[OracleRecord], path: Path) -> None:
        """Save oracle data to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(f"Saved oracle data to {path}")

    def load_oracle(self, path: Path) -> list[OracleRecord]:
        """Load oracle data from JSONL file."""
        records = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                records.append(
                    OracleRecord(
                        question=data["question"],
                        answer=data["answer"],
                        fast_answer=data["fast_answer"],
                        slow_answer=data["slow_answer"],
                        fast_correct=data["fast_correct"],
                        slow_correct=data["slow_correct"],
                        oracle_action=data["oracle_action"],
                    )
                )
        logger.info(f"Loaded {len(records)} oracle records from {path}")
        return records

    def train(
        self,
        oracle_data: list[OracleRecord],
        epochs: int = 1,
        shuffle: bool = True,
    ) -> TrainingMetrics:
        """
        Train policy on oracle-labeled data.

        Args:
            oracle_data: List of oracle records
            epochs: Number of passes through data
            shuffle: Whether to shuffle data each epoch

        Returns:
            Training metrics
        """
        import random
        import time

        logger.info(f"Training on {len(oracle_data)} examples for {epochs} epochs")
        start_time = time.time()

        # Enable learning mode temporarily
        original_setting = self.bandit_config.enable_online_learning
        self.bandit_config.enable_online_learning = True

        tracker = RewardTracker(self.bandit_config)

        for epoch in range(epochs):
            data = list(oracle_data)
            if shuffle:
                random.shuffle(data)

            for record in tqdm(data, desc=f"Epoch {epoch + 1}/{epochs}"):
                # Get bandit prediction
                action, prob = self.router.predict(record.question)

                # Compute correctness based on chosen action
                correct = record.fast_correct if action == 0 else record.slow_correct

                # Compute loss
                loss = calculate_loss(correct, action, self.bandit_config)

                # Update policy
                self.router.learn(record.question, action, loss, prob)

                # Track metrics
                tracker.record(
                    correct=correct,
                    action=action,
                    fast_correct=record.fast_correct,
                    slow_correct=record.slow_correct,
                )

        # Restore setting
        self.bandit_config.enable_online_learning = original_setting

        # Save trained model
        self.router.save()

        # Compute final metrics
        metrics_dict = tracker.get_metrics()
        training_time = time.time() - start_time

        # Compute cost saved vs always-slow baseline
        # If we always used slow, cost would be 1.0 per example
        # With routing, cost is fast_rate * 0.1 + slow_rate * 1.0
        avg_cost = metrics_dict["fast_rate"] * 0.1 + metrics_dict["slow_rate"] * 1.0
        cost_saved = 1.0 - avg_cost  # Relative savings

        return TrainingMetrics(
            total_examples=metrics_dict["total"],
            accuracy=metrics_dict["accuracy"],
            avg_loss=metrics_dict["avg_loss"],
            avg_regret=metrics_dict["avg_regret"],
            fast_rate=metrics_dict["fast_rate"],
            cost_saved=cost_saved,
            training_time_seconds=training_time,
        )

    def evaluate(
        self,
        test_data: list[OracleRecord],
        deterministic: bool = True,
    ) -> TrainingMetrics:
        """
        Evaluate trained policy on test data.

        Args:
            test_data: List of oracle records
            deterministic: If True, use argmax instead of sampling

        Returns:
            Evaluation metrics
        """
        import time

        logger.info(f"Evaluating on {len(test_data)} examples")
        start_time = time.time()

        tracker = RewardTracker(self.bandit_config)

        for record in tqdm(test_data, desc="Evaluating"):
            # Get prediction
            if deterministic:
                action, prob = self.router.predict_deterministic(record.question)
            else:
                action, prob = self.router.predict(record.question)

            # Compute correctness
            correct = record.fast_correct if action == 0 else record.slow_correct

            # Track metrics
            tracker.record(
                correct=correct,
                action=action,
                fast_correct=record.fast_correct,
                slow_correct=record.slow_correct,
            )

        metrics_dict = tracker.get_metrics()
        eval_time = time.time() - start_time

        avg_cost = metrics_dict["fast_rate"] * 0.1 + metrics_dict["slow_rate"] * 1.0
        cost_saved = 1.0 - avg_cost

        return TrainingMetrics(
            total_examples=metrics_dict["total"],
            accuracy=metrics_dict["accuracy"],
            avg_loss=metrics_dict["avg_loss"],
            avg_regret=metrics_dict["avg_regret"],
            fast_rate=metrics_dict["fast_rate"],
            cost_saved=cost_saved,
            training_time_seconds=eval_time,
        )


def load_gsm_dataset(path: Path, limit: Optional[int] = None) -> list[dict]:
    """
    Load GSM-Symbolic dataset from JSONL.

    Args:
        path: Path to JSONL file
        limit: Optional limit on number of examples

    Returns:
        List of {"question": str, "answer": int}
    """
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            data.append(
                {
                    "question": item["question"],
                    "answer": int(item["answer"]),
                }
            )
    return data


def train_from_gsm(
    dataset_path: Path,
    output_dir: Path,
    training_size: int = 2000,
    test_size: int = 500,
    epochs: int = 1,
    skip_slow: bool = False,
) -> dict:
    """
    End-to-end training from GSM-Symbolic dataset.

    Args:
        dataset_path: Path to GSM-Symbolic JSONL
        output_dir: Directory for outputs (model, oracle data, logs)
        training_size: Number of examples for training
        test_size: Number of examples for testing
        epochs: Training epochs
        skip_slow: Skip slow path (for quick testing)

    Returns:
        Dictionary with training and evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    full_data = load_gsm_dataset(dataset_path)

    # Split into train/test
    train_data = full_data[:training_size]
    test_data = full_data[training_size : training_size + test_size]

    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Initialize trainer
    config = BanditConfig(model_dir=str(output_dir / "models"))
    trainer = BanditTrainer(config)

    # Check for existing oracle data
    oracle_path = output_dir / "oracle_train.jsonl"
    if oracle_path.exists():
        logger.info("Loading existing oracle data")
        oracle_train = trainer.load_oracle(oracle_path)
    else:
        logger.info("Constructing oracle (this may take a while)")
        oracle_train = trainer.construct_oracle(
            train_data, output_path=oracle_path, skip_slow=skip_slow
        )

    # Train
    logger.info("Training policy")
    train_metrics = trainer.train(oracle_train, epochs=epochs)
    logger.info(f"Training metrics: {train_metrics.to_dict()}")

    # Evaluate on test set
    if test_data:
        oracle_test_path = output_dir / "oracle_test.jsonl"
        if oracle_test_path.exists():
            oracle_test = trainer.load_oracle(oracle_test_path)
        else:
            oracle_test = trainer.construct_oracle(
                test_data, output_path=oracle_test_path, skip_slow=skip_slow
            )

        eval_metrics = trainer.evaluate(oracle_test)
        logger.info(f"Evaluation metrics: {eval_metrics.to_dict()}")
    else:
        eval_metrics = None

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "training_size": len(train_data),
        "test_size": len(test_data),
        "epochs": epochs,
        "train_metrics": train_metrics.to_dict(),
        "eval_metrics": eval_metrics.to_dict() if eval_metrics else None,
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    return results
