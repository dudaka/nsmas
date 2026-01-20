"""
Reward and Loss Calculation for Bandit Router.

Implements the loss function for VW training:
    Loss = α × I_incorrect + β × cost(action)

Where:
- α = 10.0 (high penalty for incorrect answers)
- β = 1.0 (normalized cost weight)
- cost(fast) = 0.1, cost(slow) = 1.0

This ensures the router never prefers a "cheap failure" over an "expensive success".

Loss Matrix:
              | Fast (a=0) | Slow (a=1)
--------------+------------+-----------
Correct       |    0.1     |    1.0
Incorrect     |   10.1     |   11.0
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .config import BanditConfig

logger = logging.getLogger(__name__)


@dataclass
class LossResult:
    """Result from loss calculation."""

    loss: float
    correct: bool
    action: int
    action_cost: float
    incorrect_penalty: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "correct": self.correct,
            "action": self.action,
            "action_cost": self.action_cost,
            "incorrect_penalty": self.incorrect_penalty,
        }


def calculate_loss(
    correct: bool,
    action: int,
    config: Optional[BanditConfig] = None,
) -> float:
    """
    Calculate loss for VW (which minimizes cost).

    Loss = α × I_incorrect + β × cost(action)

    Args:
        correct: Whether the answer was correct
        action: Action taken (0=fast, 1=slow)
        config: Bandit config (uses defaults if not provided)

    Returns:
        Loss value (lower is better)
    """
    if config is None:
        config = BanditConfig()

    # Incorrect penalty
    incorrect_penalty = 0.0 if correct else config.alpha_incorrect

    # Action cost (normalized)
    action_cost = config.fast_cost if action == 0 else config.slow_cost
    cost_penalty = config.beta_cost * action_cost

    return incorrect_penalty + cost_penalty


def calculate_loss_detailed(
    correct: bool,
    action: int,
    config: Optional[BanditConfig] = None,
) -> LossResult:
    """
    Calculate loss with detailed breakdown.

    Args:
        correct: Whether the answer was correct
        action: Action taken (0=fast, 1=slow)
        config: Bandit config

    Returns:
        LossResult with detailed breakdown
    """
    if config is None:
        config = BanditConfig()

    incorrect_penalty = 0.0 if correct else config.alpha_incorrect
    action_cost = config.fast_cost if action == 0 else config.slow_cost

    loss = incorrect_penalty + (config.beta_cost * action_cost)

    return LossResult(
        loss=loss,
        correct=correct,
        action=action,
        action_cost=action_cost,
        incorrect_penalty=incorrect_penalty,
    )


def calculate_reward(
    correct: bool,
    action: int,
    config: Optional[BanditConfig] = None,
) -> float:
    """
    Calculate reward (negative loss).

    Useful when thinking in terms of maximizing reward
    rather than minimizing loss.

    Args:
        correct: Whether the answer was correct
        action: Action taken
        config: Bandit config

    Returns:
        Reward value (higher is better)
    """
    return -calculate_loss(correct, action, config)


def compute_oracle_action(
    fast_correct: bool,
    slow_correct: bool,
    config: Optional[BanditConfig] = None,
) -> int:
    """
    Compute the oracle (optimal) action given outcomes.

    Oracle Logic:
    - If fast is correct: choose fast (cheaper)
    - If only slow is correct: choose slow
    - If both fail: choose fast (minimize loss)

    Args:
        fast_correct: Whether fast path would be correct
        slow_correct: Whether slow path would be correct
        config: Bandit config

    Returns:
        Optimal action (0=fast, 1=slow)
    """
    if fast_correct:
        return 0  # Fast works and is cheaper
    elif slow_correct:
        return 1  # Only slow works
    else:
        return 0  # Both fail, minimize cost


def compute_regret(
    chosen_action: int,
    fast_correct: bool,
    slow_correct: bool,
    config: Optional[BanditConfig] = None,
) -> float:
    """
    Compute regret for a decision.

    Regret = Loss(chosen) - Loss(oracle)

    Args:
        chosen_action: Action that was taken
        fast_correct: Whether fast would have been correct
        slow_correct: Whether slow would have been correct
        config: Bandit config

    Returns:
        Regret value (0 = optimal choice)
    """
    if config is None:
        config = BanditConfig()

    # Compute loss of chosen action
    chosen_correct = fast_correct if chosen_action == 0 else slow_correct
    chosen_loss = calculate_loss(chosen_correct, chosen_action, config)

    # Compute oracle loss
    oracle_action = compute_oracle_action(fast_correct, slow_correct, config)
    oracle_correct = fast_correct if oracle_action == 0 else slow_correct
    oracle_loss = calculate_loss(oracle_correct, oracle_action, config)

    return chosen_loss - oracle_loss


class RewardTracker:
    """
    Track rewards and compute aggregate metrics.

    Useful for monitoring training progress.
    """

    def __init__(self, config: Optional[BanditConfig] = None):
        self.config = config or BanditConfig()
        self.records: list[dict] = []

    def record(
        self,
        correct: bool,
        action: int,
        fast_correct: Optional[bool] = None,
        slow_correct: Optional[bool] = None,
    ) -> LossResult:
        """
        Record an outcome and compute loss.

        Args:
            correct: Whether chosen action was correct
            action: Action taken
            fast_correct: Whether fast would have been correct (for regret)
            slow_correct: Whether slow would have been correct (for regret)

        Returns:
            Loss result
        """
        loss_result = calculate_loss_detailed(correct, action, self.config)

        record = loss_result.to_dict()
        record["fast_correct"] = fast_correct
        record["slow_correct"] = slow_correct

        if fast_correct is not None and slow_correct is not None:
            record["regret"] = compute_regret(
                action, fast_correct, slow_correct, self.config
            )
            record["oracle_action"] = compute_oracle_action(
                fast_correct, slow_correct, self.config
            )
        else:
            record["regret"] = None
            record["oracle_action"] = None

        self.records.append(record)
        return loss_result

    def get_metrics(self) -> dict:
        """Compute aggregate metrics."""
        if not self.records:
            return {
                "total": 0,
                "accuracy": 0.0,
                "avg_loss": 0.0,
                "avg_regret": 0.0,
                "fast_rate": 0.0,
                "slow_rate": 0.0,
            }

        total = len(self.records)
        correct = sum(1 for r in self.records if r["correct"])
        fast_count = sum(1 for r in self.records if r["action"] == 0)
        total_loss = sum(r["loss"] for r in self.records)

        regrets = [r["regret"] for r in self.records if r["regret"] is not None]
        avg_regret = sum(regrets) / len(regrets) if regrets else 0.0

        return {
            "total": total,
            "accuracy": correct / total,
            "avg_loss": total_loss / total,
            "avg_regret": avg_regret,
            "fast_rate": fast_count / total,
            "slow_rate": (total - fast_count) / total,
        }

    def clear(self):
        """Clear all records."""
        self.records = []
