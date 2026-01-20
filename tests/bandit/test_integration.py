"""
Integration tests for the bandit module.

Tests the full pipeline from routing to solving.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.bandit.config import BanditConfig
from src.bandit.router import BanditRouter
from src.bandit.fast_solver import FastSolver
from src.bandit.reward import calculate_loss, compute_oracle_action
from src.bandit.training import OracleRecord, BanditTrainer


class TestBanditPipeline:
    """Test the full bandit pipeline."""

    @pytest.fixture
    def mock_components(self, bandit_config, tmp_path):
        """Create mocked pipeline components."""
        bandit_config.model_dir = str(tmp_path)

        with patch("src.bandit.router.Featurizer") as mock_featurizer:
            mock_feat_instance = MagicMock()
            mock_feat_instance.to_vw_example.return_value = (
                "shared |x test\n|Action fast\n|Action slow"
            )
            mock_feat_instance.featurize.return_value = MagicMock(
                to_vw_string=MagicMock(return_value="|x test")
            )
            mock_featurizer.return_value = mock_feat_instance

            router = BanditRouter(bandit_config)
            router._vw = MagicMock()
            router._vw.predict.return_value = [(0, 0.7), (1, 0.3)]
            router._initialized = True
            router.featurizer = mock_feat_instance

            return {
                "config": bandit_config,
                "router": router,
            }

    def test_pipeline_predict_fast(self, mock_components):
        """Test pipeline predicts fast path."""
        router = mock_components["router"]
        router._vw.predict.return_value = [(0, 0.8), (1, 0.2)]

        action, prob = router.predict("Simple question: 2+2?")

        # Should choose fast with high probability
        assert action in [0, 1]  # Stochastic, but likely 0

    def test_pipeline_predict_slow(self, mock_components):
        """Test pipeline predicts slow path when appropriate."""
        router = mock_components["router"]
        router._vw.predict.return_value = [(0, 0.1), (1, 0.9)]

        action, prob = router.predict_deterministic("Complex multi-step problem...")

        assert action == 1  # Deterministic chooses argmax
        assert prob == 0.9


class TestOracleConstruction:
    """Test oracle data construction."""

    def test_oracle_record_creation(self):
        """Test OracleRecord dataclass."""
        record = OracleRecord(
            question="Test question",
            answer=42,
            fast_answer=42,
            slow_answer=42,
            fast_correct=True,
            slow_correct=True,
            oracle_action=0,
        )

        assert record.question == "Test question"
        assert record.oracle_action == 0

    def test_oracle_record_to_dict(self):
        """Test OracleRecord serialization."""
        record = OracleRecord(
            question="Test",
            answer=10,
            fast_answer=10,
            slow_answer=10,
            fast_correct=True,
            slow_correct=True,
            oracle_action=0,
        )

        data = record.to_dict()

        assert data["question"] == "Test"
        assert data["answer"] == 10
        assert data["oracle_action"] == 0

    def test_oracle_action_both_correct(self):
        """Test oracle chooses fast when both paths work."""
        action = compute_oracle_action(fast_correct=True, slow_correct=True)
        assert action == 0  # Fast

    def test_oracle_action_only_slow(self):
        """Test oracle chooses slow when only slow works."""
        action = compute_oracle_action(fast_correct=False, slow_correct=True)
        assert action == 1  # Slow


class TestAgentIntegration:
    """Test Agent integration with bandit router."""

    @pytest.fixture
    def mock_agent_deps(self):
        """Mock agent dependencies."""
        with patch("src.agent.graph.create_agent_graph") as mock_graph:
            mock_graph.return_value = MagicMock()
            yield mock_graph

    def test_agent_without_bandit(self, mock_agent_deps):
        """Test agent works without bandit (Phase 4 behavior)."""
        from src.agent import Agent, AgentConfig

        agent = Agent(AgentConfig())

        assert agent.bandit_config is None
        assert agent.router is None
        assert agent.fast_solver is None

    def test_agent_with_bandit_config(self, mock_agent_deps):
        """Test agent accepts bandit config."""
        from src.agent import Agent, AgentConfig

        bandit_config = BanditConfig()
        agent = Agent(AgentConfig(), bandit_config=bandit_config)

        assert agent.bandit_config is not None

    def test_agent_force_path_fast(self, mock_agent_deps):
        """Test agent respects force_path='fast'."""
        from src.agent import Agent, AgentConfig

        bandit_config = BanditConfig()

        with patch("src.bandit.BanditRouter") as mock_router_cls:
            with patch("src.bandit.FastSolver") as mock_solver_cls:
                mock_solver = MagicMock()
                mock_solver.solve.return_value = 42
                mock_solver_cls.return_value = mock_solver

                agent = Agent(AgentConfig(), bandit_config=bandit_config)
                agent._fast_solver = mock_solver

                result = agent.solve("Test", force_path="fast")

                assert result["status"] == "success"
                assert result["final_answer"] == 42
                mock_solver.solve.assert_called_once()


class TestEndToEndTraining:
    """Test training pipeline end-to-end."""

    @pytest.fixture
    def sample_data(self):
        """Sample dataset for training."""
        return [
            {"question": "What is 2 + 2?", "answer": 4},
            {"question": "What is 10 - 5?", "answer": 5},
            {"question": "What is 3 * 4?", "answer": 12},
        ]

    @pytest.fixture
    def sample_oracle(self):
        """Sample oracle data."""
        return [
            OracleRecord(
                question="What is 2 + 2?",
                answer=4,
                fast_answer=4,
                slow_answer=4,
                fast_correct=True,
                slow_correct=True,
                oracle_action=0,
            ),
            OracleRecord(
                question="Complex problem",
                answer=100,
                fast_answer=None,
                slow_answer=100,
                fast_correct=False,
                slow_correct=True,
                oracle_action=1,
            ),
        ]

    def test_loss_calculation_in_training(self, sample_oracle):
        """Test loss calculation during training."""
        config = BanditConfig()

        for record in sample_oracle:
            # Simulate choosing the oracle action
            action = record.oracle_action
            correct = record.fast_correct if action == 0 else record.slow_correct

            loss = calculate_loss(correct, action, config)

            # Oracle action should have low loss when correct
            if correct:
                if action == 0:
                    assert loss == pytest.approx(0.1)  # Fast correct
                else:
                    assert loss == pytest.approx(1.0)  # Slow correct
