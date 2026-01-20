"""
Tests for reward/loss calculation utilities.
"""

import pytest

from src.bandit.config import BanditConfig
from src.bandit.reward import (
    calculate_loss,
    calculate_loss_detailed,
    calculate_reward,
    compute_oracle_action,
    compute_regret,
    LossResult,
    RewardTracker,
)


class TestCalculateLoss:
    """Test loss calculation functions."""

    @pytest.fixture
    def config(self):
        """Default config for testing."""
        return BanditConfig(
            alpha_incorrect=10.0,
            beta_cost=1.0,
            fast_cost=0.1,
            slow_cost=1.0,
        )

    def test_loss_correct_fast(self, config):
        """Test loss for correct answer via fast path."""
        loss = calculate_loss(correct=True, action=0, config=config)

        # Loss = 0 (correct) + 1.0 * 0.1 (fast cost) = 0.1
        assert loss == pytest.approx(0.1)

    def test_loss_correct_slow(self, config):
        """Test loss for correct answer via slow path."""
        loss = calculate_loss(correct=True, action=1, config=config)

        # Loss = 0 (correct) + 1.0 * 1.0 (slow cost) = 1.0
        assert loss == pytest.approx(1.0)

    def test_loss_incorrect_fast(self, config):
        """Test loss for incorrect answer via fast path."""
        loss = calculate_loss(correct=False, action=0, config=config)

        # Loss = 10.0 (incorrect) + 1.0 * 0.1 (fast cost) = 10.1
        assert loss == pytest.approx(10.1)

    def test_loss_incorrect_slow(self, config):
        """Test loss for incorrect answer via slow path."""
        loss = calculate_loss(correct=False, action=1, config=config)

        # Loss = 10.0 (incorrect) + 1.0 * 1.0 (slow cost) = 11.0
        assert loss == pytest.approx(11.0)

    def test_loss_detailed_returns_result(self, config):
        """Test detailed loss returns LossResult object."""
        result = calculate_loss_detailed(correct=True, action=0, config=config)

        assert isinstance(result, LossResult)
        assert result.loss == pytest.approx(0.1)
        assert result.correct is True
        assert result.action == 0
        assert result.action_cost == 0.1
        assert result.incorrect_penalty == 0.0

    def test_reward_is_negative_loss(self, config):
        """Test reward is negative of loss."""
        loss = calculate_loss(correct=True, action=0, config=config)
        reward = calculate_reward(correct=True, action=0, config=config)

        assert reward == -loss


class TestComputeOracleAction:
    """Test oracle action computation."""

    def test_oracle_both_correct_chooses_fast(self):
        """When both paths work, oracle chooses fast (cheaper)."""
        action = compute_oracle_action(fast_correct=True, slow_correct=True)

        assert action == 0  # Fast

    def test_oracle_only_slow_correct(self):
        """When only slow works, oracle chooses slow."""
        action = compute_oracle_action(fast_correct=False, slow_correct=True)

        assert action == 1  # Slow

    def test_oracle_only_fast_correct(self):
        """When only fast works, oracle chooses fast."""
        action = compute_oracle_action(fast_correct=True, slow_correct=False)

        assert action == 0  # Fast

    def test_oracle_both_fail_chooses_fast(self):
        """When both fail, oracle chooses fast (minimize cost)."""
        action = compute_oracle_action(fast_correct=False, slow_correct=False)

        assert action == 0  # Fast (minimize loss)


class TestComputeRegret:
    """Test regret computation."""

    @pytest.fixture
    def config(self):
        return BanditConfig()

    def test_regret_optimal_choice(self, config):
        """Test regret is zero when choosing optimally."""
        # Both correct, fast is optimal
        regret = compute_regret(
            chosen_action=0,
            fast_correct=True,
            slow_correct=True,
            config=config,
        )

        assert regret == pytest.approx(0.0)

    def test_regret_suboptimal_choice(self, config):
        """Test regret is positive when choosing suboptimally."""
        # Both correct, but chose slow (suboptimal)
        regret = compute_regret(
            chosen_action=1,
            fast_correct=True,
            slow_correct=True,
            config=config,
        )

        # Regret = slow_cost - fast_cost = 1.0 - 0.1 = 0.9
        assert regret == pytest.approx(0.9)

    def test_regret_necessary_slow(self, config):
        """Test no regret when slow is necessary."""
        # Only slow works, chose slow
        regret = compute_regret(
            chosen_action=1,
            fast_correct=False,
            slow_correct=True,
            config=config,
        )

        assert regret == pytest.approx(0.0)

    def test_regret_wrong_path(self, config):
        """Test high regret when choosing wrong path."""
        # Only slow works, but chose fast
        regret = compute_regret(
            chosen_action=0,
            fast_correct=False,
            slow_correct=True,
            config=config,
        )

        # Regret = loss(fast, incorrect) - loss(slow, correct)
        # = 10.1 - 1.0 = 9.1
        assert regret == pytest.approx(9.1)


class TestRewardTracker:
    """Test RewardTracker class."""

    @pytest.fixture
    def tracker(self):
        return RewardTracker(BanditConfig())

    def test_record_returns_loss_result(self, tracker):
        """Test record returns LossResult."""
        result = tracker.record(correct=True, action=0)

        assert isinstance(result, LossResult)

    def test_get_metrics_empty(self, tracker):
        """Test metrics on empty tracker."""
        metrics = tracker.get_metrics()

        assert metrics["total"] == 0
        assert metrics["accuracy"] == 0.0

    def test_get_metrics_with_data(self, tracker):
        """Test metrics calculation."""
        # Add some records
        tracker.record(correct=True, action=0, fast_correct=True, slow_correct=True)
        tracker.record(correct=True, action=1, fast_correct=False, slow_correct=True)
        tracker.record(correct=False, action=0, fast_correct=False, slow_correct=True)
        tracker.record(correct=True, action=0, fast_correct=True, slow_correct=True)

        metrics = tracker.get_metrics()

        assert metrics["total"] == 4
        assert metrics["accuracy"] == 0.75  # 3/4 correct
        assert metrics["fast_rate"] == 0.75  # 3/4 chose fast
        assert metrics["slow_rate"] == 0.25  # 1/4 chose slow

    def test_clear_tracker(self, tracker):
        """Test clearing tracker."""
        tracker.record(correct=True, action=0)
        tracker.record(correct=True, action=1)

        tracker.clear()

        assert len(tracker.records) == 0
        assert tracker.get_metrics()["total"] == 0
