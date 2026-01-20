"""
Tests for BanditConfig.
"""

import pytest
from pathlib import Path

from src.bandit.config import BanditConfig
from src.agent.config import LLMProvider


class TestBanditConfig:
    """Test BanditConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BanditConfig()

        assert config.exploration_algo == "squarecb"
        assert config.gamma_scale == 10.0
        assert config.epsilon == 0.2
        assert config.alpha_incorrect == 10.0
        assert config.beta_cost == 1.0
        assert config.fast_cost == 0.1
        assert config.slow_cost == 1.0
        assert config.fast_model == "gpt-4o-mini"
        assert config.enable_online_learning is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BanditConfig(
            exploration_algo="epsilon",
            gamma_scale=5.0,
            alpha_incorrect=20.0,
            fast_model="gpt-3.5-turbo",
        )

        assert config.exploration_algo == "epsilon"
        assert config.gamma_scale == 5.0
        assert config.alpha_incorrect == 20.0
        assert config.fast_model == "gpt-3.5-turbo"

    def test_model_path_property(self):
        """Test model_path property returns correct path."""
        config = BanditConfig(
            model_dir="/custom/dir",
            model_filename="custom.vw",
        )

        assert config.model_path == Path("/custom/dir/custom.vw")

    def test_get_vw_args_squarecb(self):
        """Test VW args generation for SquareCB."""
        config = BanditConfig(
            exploration_algo="squarecb",
            gamma_scale=10.0,
        )

        args = config.get_vw_args(load_existing=False)

        assert "--cb_explore_adf" in args
        assert "--squarecb" in args
        assert "--gamma_scale 10.0" in args
        assert "--interactions ::" in args
        assert "--quiet" in args

    def test_get_vw_args_epsilon(self):
        """Test VW args generation for epsilon-greedy."""
        config = BanditConfig(
            exploration_algo="epsilon",
            epsilon=0.3,
        )

        args = config.get_vw_args(load_existing=False)

        assert "--epsilon" in args
        assert "--epsilon 0.3" in args

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = BanditConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_exploration_algo(self):
        """Test validation fails for invalid exploration algorithm."""
        config = BanditConfig(exploration_algo="invalid")

        with pytest.raises(ValueError, match="Invalid exploration_algo"):
            config.validate()

    def test_validate_invalid_gamma_scale(self):
        """Test validation fails for non-positive gamma_scale."""
        config = BanditConfig(gamma_scale=-1.0)

        with pytest.raises(ValueError, match="gamma_scale must be positive"):
            config.validate()

    def test_validate_invalid_epsilon(self):
        """Test validation fails for epsilon out of range."""
        config = BanditConfig(epsilon=1.5)

        with pytest.raises(ValueError, match="epsilon must be in"):
            config.validate()

    def test_action_names_default(self):
        """Test default action names."""
        config = BanditConfig()

        assert config.action_names == ("fast_path", "slow_path")
        assert len(config.action_names) == 2
