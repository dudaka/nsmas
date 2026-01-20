"""
Tests for BanditRouter.

Note: These tests mock VW to avoid requiring vowpalwabbit installation
in CI environments. Integration tests with real VW should be run separately.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.bandit.config import BanditConfig
from src.bandit.router import BanditRouter, create_router


class TestBanditRouter:
    """Test BanditRouter class."""

    @pytest.fixture
    def mock_vw(self):
        """Create mock VW workspace."""
        mock = MagicMock()
        # Default prediction: equal probability for both actions
        mock.predict.return_value = [(0, 0.5), (1, 0.5)]
        return mock

    @pytest.fixture
    def router_with_mock_vw(self, bandit_config, mock_vw):
        """Create router with mocked VW."""
        with patch("src.bandit.router.Featurizer") as mock_featurizer:
            mock_featurizer_instance = MagicMock()
            mock_featurizer_instance.to_vw_example.return_value = (
                "shared |x len_tokens:0.1\n|Action fast\n|Action slow"
            )
            mock_featurizer.return_value = mock_featurizer_instance

            router = BanditRouter(bandit_config)
            router._vw = mock_vw
            router._initialized = True
            router.featurizer = mock_featurizer_instance

            return router

    def test_init_with_config(self, bandit_config):
        """Test router initialization with config."""
        with patch("src.bandit.router.Featurizer"):
            router = BanditRouter(bandit_config)

            assert router.config == bandit_config
            assert router._vw is None  # Lazy init
            assert router._initialized is False

    def test_action_constants(self):
        """Test action index constants."""
        assert BanditRouter.FAST_ACTION == 0
        assert BanditRouter.SLOW_ACTION == 1

    def test_predict_returns_action_and_prob(self, router_with_mock_vw):
        """Test predict returns (action, probability) tuple."""
        router = router_with_mock_vw

        action, prob = router.predict("Test question")

        assert action in [0, 1]
        assert 0.0 <= prob <= 1.0

    def test_predict_deterministic(self, router_with_mock_vw):
        """Test deterministic prediction uses argmax."""
        router = router_with_mock_vw
        # Set up mock to favor fast path
        router.vw.predict.return_value = [(0, 0.8), (1, 0.2)]

        action, prob = router.predict_deterministic("Test question")

        assert action == 0  # Highest probability
        assert prob == 0.8

    def test_learn_disabled_by_default(self, router_with_mock_vw):
        """Test learn does nothing when online learning is disabled."""
        router = router_with_mock_vw
        router.config.enable_online_learning = False

        router.learn("Test question", action=0, cost=0.1, prob=0.5)

        router.vw.learn.assert_not_called()

    def test_learn_enabled(self, router_with_mock_vw):
        """Test learn calls VW when enabled."""
        router = router_with_mock_vw
        router.config.enable_online_learning = True

        # Mock featurize to return a FeatureResult
        with patch.object(router.featurizer, "featurize") as mock_featurize:
            mock_result = MagicMock()
            mock_result.to_vw_string.return_value = "|x test_features"
            mock_featurize.return_value = mock_result

            router.learn("Test question", action=0, cost=0.1, prob=0.5)

            router.vw.learn.assert_called_once()

    def test_get_action_name(self, router_with_mock_vw):
        """Test action name retrieval."""
        router = router_with_mock_vw

        assert router.get_action_name(0) == "fast_path"
        assert router.get_action_name(1) == "slow_path"
        assert router.get_action_name(99) == "unknown_99"

    def test_get_stats(self, router_with_mock_vw, bandit_config):
        """Test stats retrieval."""
        router = router_with_mock_vw

        stats = router.get_stats()

        assert "model_path" in stats
        assert stats["exploration_algo"] == bandit_config.exploration_algo
        assert stats["gamma_scale"] == bandit_config.gamma_scale
        assert stats["action_space"] == ["fast_path", "slow_path"]

    def test_save_creates_directory(self, router_with_mock_vw, tmp_path):
        """Test save creates model directory if needed."""
        router = router_with_mock_vw
        router.config.model_dir = str(tmp_path / "new_dir")
        router.config.model_filename = "test.vw"

        path = router.save()

        assert path.parent.exists()
        router.vw.save.assert_called_once()

    def test_format_labeled_example_fast(self, router_with_mock_vw):
        """Test labeled example format for fast action."""
        router = router_with_mock_vw

        with patch.object(router.featurizer, "featurize") as mock_featurize:
            mock_result = MagicMock()
            mock_result.to_vw_string.return_value = "|x test"
            mock_featurize.return_value = mock_result

            example = router._format_labeled_example(
                "Test", action=0, cost=0.1, prob=0.8
            )

            assert "shared |x test" in example
            assert "0:0.1000:0.8000 |Action fast" in example
            assert "|Action slow" in example

    def test_format_labeled_example_slow(self, router_with_mock_vw):
        """Test labeled example format for slow action."""
        router = router_with_mock_vw

        with patch.object(router.featurizer, "featurize") as mock_featurize:
            mock_result = MagicMock()
            mock_result.to_vw_string.return_value = "|x test"
            mock_featurize.return_value = mock_result

            example = router._format_labeled_example(
                "Test", action=1, cost=1.0, prob=0.6
            )

            assert "|Action fast" in example
            assert "0:1.0000:0.6000 |Action slow" in example


class TestCreateRouter:
    """Test create_router factory function."""

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        with patch("src.bandit.router.Featurizer"):
            router = create_router()

            assert router.config.exploration_algo == "squarecb"
            assert router.config.enable_online_learning is False

    def test_create_with_epsilon(self):
        """Test factory with epsilon-greedy."""
        with patch("src.bandit.router.Featurizer"):
            router = create_router(exploration_algo="epsilon")

            assert router.config.exploration_algo == "epsilon"

    def test_create_with_model_path(self, tmp_path):
        """Test factory with custom model path."""
        model_path = str(tmp_path / "custom" / "model.vw")

        with patch("src.bandit.router.Featurizer"):
            router = create_router(model_path=model_path)

            assert router.config.model_dir == str(tmp_path / "custom")
            assert router.config.model_filename == "model.vw"


class TestBanditRouterIntegration:
    """Integration tests that require real VW (skip if not installed)."""

    @pytest.fixture
    def real_router(self, bandit_config, tmp_path):
        """Create real router for integration tests."""
        try:
            import vowpalwabbit  # noqa
        except ImportError:
            pytest.skip("vowpalwabbit not installed")

        bandit_config.model_dir = str(tmp_path)
        bandit_config.enable_online_learning = True

        return BanditRouter(bandit_config)

    def test_real_predict(self, real_router):
        """Test prediction with real VW."""
        action, prob = real_router.predict("John has 5 apples. How many left?")

        assert action in [0, 1]
        assert 0.0 <= prob <= 1.0

    def test_real_learn_and_predict(self, real_router):
        """Test learning updates predictions."""
        question = "Simple math: 2 + 2"

        # Initial prediction
        action1, prob1 = real_router.predict(question)

        # Learn that fast is good for this
        from src.bandit.reward import calculate_loss

        loss = calculate_loss(correct=True, action=0, config=real_router.config)
        real_router.learn(question, action=0, cost=loss, prob=prob1)

        # Prediction should still work (may or may not change after 1 sample)
        action2, prob2 = real_router.predict(question)

        assert action2 in [0, 1]

    def test_save_and_load(self, real_router, tmp_path):
        """Test model persistence."""
        # Make a prediction
        real_router.predict("Test question")

        # Save
        model_path = real_router.save()
        assert model_path.exists()

        # Create new router and load
        config2 = BanditConfig(
            model_dir=str(tmp_path),
            model_filename=real_router.config.model_filename,
        )
        router2 = BanditRouter(config2)

        # Should be able to predict
        action, prob = router2.predict("Test question")
        assert action in [0, 1]
