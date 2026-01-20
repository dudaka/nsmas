"""
Tests for WarmTrainer.
"""

import subprocess
from unittest.mock import Mock, patch

import pytest

from src.bandit.warmstart.warm_trainer import (
    WarmTrainer,
    WarmTrainerConfig,
    TrainingResult,
)


class TestWarmTrainerConfig:
    """Test WarmTrainerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WarmTrainerConfig()

        assert config.learning_rate == 0.01
        assert config.passes == 20
        assert config.loss_function == "squared"
        assert config.epsilon == 0.0
        assert config.l2_regularization == 0.0001
        assert config.cb_type == "mtr"
        assert config.holdout_off is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WarmTrainerConfig(
            learning_rate=0.1,
            passes=50,
            epsilon=0.05,
        )

        assert config.learning_rate == 0.1
        assert config.passes == 50
        assert config.epsilon == 0.05


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_success_result(self):
        """Test creating successful result."""
        result = TrainingResult(
            success=True,
            model_path="/path/to/model.vw",
            avg_loss=0.123,
            n_examples=1000,
            vw_output="training output",
        )

        assert result.success is True
        assert result.model_path == "/path/to/model.vw"
        assert result.avg_loss == 0.123
        assert result.n_examples == 1000

    def test_failure_result(self):
        """Test creating failure result."""
        result = TrainingResult(
            success=False,
            error_message="Training failed",
        )

        assert result.success is False
        assert result.error_message == "Training failed"
        assert result.model_path is None


class TestWarmTrainer:
    """Test WarmTrainer class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        trainer = WarmTrainer()

        assert trainer.config.learning_rate == 0.01
        assert trainer.config.passes == 20

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = WarmTrainerConfig(learning_rate=0.05)
        trainer = WarmTrainer(config)

        assert trainer.config.learning_rate == 0.05

    def test_build_vw_command_basic(self, temp_dir):
        """Test building basic VW command."""
        trainer = WarmTrainer()

        cmd = trainer.build_vw_command(
            data_path=temp_dir / "train.dat",
            output_path=temp_dir / "model.vw",
        )

        assert "vw" in cmd
        assert "--cb_explore_adf" in cmd
        assert "--data" in cmd
        assert "--final_regressor" in cmd
        assert "--learning_rate" in cmd
        assert "--passes" in cmd

    def test_build_vw_command_with_cache(self, temp_dir):
        """Test VW command includes cache file."""
        trainer = WarmTrainer()
        cache_path = temp_dir / "cache.vw"

        cmd = trainer.build_vw_command(
            data_path=temp_dir / "train.dat",
            output_path=temp_dir / "model.vw",
            cache_path=cache_path,
        )

        assert "--cache_file" in cmd
        assert str(cache_path) in cmd

    def test_build_vw_command_auto_cache(self, temp_dir):
        """Test VW command with auto cache."""
        trainer = WarmTrainer()

        cmd = trainer.build_vw_command(
            data_path=temp_dir / "train.dat",
            output_path=temp_dir / "model.vw",
            cache_path=None,
        )

        assert "--cache" in cmd
        assert "--cache_file" not in cmd

    def test_build_vw_command_with_epsilon(self, temp_dir):
        """Test VW command with exploration epsilon."""
        config = WarmTrainerConfig(epsilon=0.1)
        trainer = WarmTrainer(config)

        cmd = trainer.build_vw_command(
            data_path=temp_dir / "train.dat",
            output_path=temp_dir / "model.vw",
        )

        assert "--epsilon" in cmd
        assert "0.1" in cmd

    def test_build_vw_command_no_epsilon_when_zero(self, temp_dir):
        """Test VW command omits epsilon when 0."""
        config = WarmTrainerConfig(epsilon=0.0)
        trainer = WarmTrainer(config)

        cmd = trainer.build_vw_command(
            data_path=temp_dir / "train.dat",
            output_path=temp_dir / "model.vw",
        )

        assert "--epsilon" not in cmd

    def test_build_vw_command_holdout_off(self, temp_dir):
        """Test VW command includes holdout_off."""
        trainer = WarmTrainer()

        cmd = trainer.build_vw_command(
            data_path=temp_dir / "train.dat",
            output_path=temp_dir / "model.vw",
        )

        assert "--holdout_off" in cmd

    def test_parse_avg_loss(self):
        """Test parsing average loss from VW output."""
        trainer = WarmTrainer()
        vw_output = """
        number of examples = 1000
        average loss = 0.123456
        best constant loss = 0.5
        """

        avg_loss = trainer._parse_avg_loss(vw_output)

        assert avg_loss == pytest.approx(0.123456)

    def test_parse_avg_loss_not_found(self):
        """Test parsing when avg loss not in output."""
        trainer = WarmTrainer()
        vw_output = "some other output"

        avg_loss = trainer._parse_avg_loss(vw_output)

        assert avg_loss is None

    def test_parse_n_examples(self):
        """Test parsing number of examples from VW output."""
        trainer = WarmTrainer()
        vw_output = """
        number of examples = 5000
        average loss = 0.1
        """

        n_examples = trainer._parse_n_examples(vw_output)

        assert n_examples == 5000

    def test_parse_n_examples_not_found(self):
        """Test parsing when n_examples not in output."""
        trainer = WarmTrainer()
        vw_output = "some other output"

        n_examples = trainer._parse_n_examples(vw_output)

        assert n_examples == 0

    def test_train_data_not_found(self, temp_dir):
        """Test training with non-existent data file."""
        trainer = WarmTrainer()

        result = trainer.train(
            data_path=temp_dir / "nonexistent.dat",
            output_path=temp_dir / "model.vw",
        )

        assert result.success is False
        assert "not found" in result.error_message

    @patch("subprocess.run")
    def test_train_success(self, mock_run, temp_dir):
        """Test successful training."""
        # Create dummy data file
        data_path = temp_dir / "train.dat"
        data_path.write_text("shared |x f1:1.0\n0:0:1 |a fast\n1:1:0 |a slow\n\n")

        # Mock VW output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr="number of examples = 100\naverage loss = 0.05",
        )

        trainer = WarmTrainer()
        result = trainer.train(
            data_path=data_path,
            output_path=temp_dir / "model.vw",
        )

        assert result.success is True
        assert result.avg_loss == 0.05
        assert result.n_examples == 100

    @patch("subprocess.run")
    def test_train_vw_failure(self, mock_run, temp_dir):
        """Test training when VW fails."""
        data_path = temp_dir / "train.dat"
        data_path.write_text("invalid data")

        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="VW error: invalid format",
        )

        trainer = WarmTrainer()
        result = trainer.train(
            data_path=data_path,
            output_path=temp_dir / "model.vw",
        )

        assert result.success is False
        assert "VW error" in result.error_message

    @patch("subprocess.run")
    def test_train_creates_output_directory(self, mock_run, temp_dir):
        """Test that training creates output directory."""
        data_path = temp_dir / "train.dat"
        data_path.write_text("shared |x f1:1.0\n0:0:1 |a fast\n1:1:0 |a slow\n\n")

        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr="number of examples = 10\naverage loss = 0.1",
        )

        trainer = WarmTrainer()
        output_path = temp_dir / "nested" / "dir" / "model.vw"

        result = trainer.train(
            data_path=data_path,
            output_path=output_path,
        )

        assert output_path.parent.exists()

    @patch("subprocess.run")
    def test_train_override_params(self, mock_run, temp_dir):
        """Test overriding config params in train call."""
        data_path = temp_dir / "train.dat"
        data_path.write_text("shared |x f1:1.0\n0:0:1 |a fast\n1:1:0 |a slow\n\n")

        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr="number of examples = 10\naverage loss = 0.1",
        )

        trainer = WarmTrainer()
        trainer.train(
            data_path=data_path,
            output_path=temp_dir / "model.vw",
            passes=50,
            learning_rate=0.1,
        )

        # Config should be updated
        assert trainer.config.passes == 50
        assert trainer.config.learning_rate == 0.1

    @patch("subprocess.run")
    def test_evaluate_success(self, mock_run, temp_dir):
        """Test successful evaluation."""
        model_path = temp_dir / "model.vw"
        model_path.write_text("")
        test_path = temp_dir / "test.dat"
        test_path.write_text("shared |x f1:1.0\n|a fast\n|a slow\n\n")

        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr="number of examples = 50\naverage loss = 0.08",
        )

        trainer = WarmTrainer()
        result = trainer.evaluate(model_path, test_path)

        assert "error" not in result
        assert result["avg_loss"] == 0.08
        assert result["n_examples"] == 50

    def test_evaluate_model_not_found(self, temp_dir):
        """Test evaluation with non-existent model."""
        trainer = WarmTrainer()
        test_path = temp_dir / "test.dat"
        test_path.write_text("")

        with pytest.raises(FileNotFoundError, match="Model not found"):
            trainer.evaluate(
                model_path=temp_dir / "nonexistent.vw",
                test_data_path=test_path,
            )

    def test_evaluate_test_data_not_found(self, temp_dir):
        """Test evaluation with non-existent test data."""
        trainer = WarmTrainer()
        model_path = temp_dir / "model.vw"
        model_path.write_text("")

        with pytest.raises(FileNotFoundError, match="Test data not found"):
            trainer.evaluate(
                model_path=model_path,
                test_data_path=temp_dir / "nonexistent.dat",
            )


class TestWarmTrainerIntegration:
    """Integration tests requiring actual VW installation."""

    @pytest.mark.skipif(
        subprocess.run(["which", "vw"], capture_output=True).returncode != 0,
        reason="VW not installed"
    )
    def test_real_vw_training(self, temp_dir):
        """Test actual VW training with real binary."""
        # Create valid VW ADF data
        data_path = temp_dir / "train.dat"
        data_content = """shared |x len:0.5 chars:0.3
0:0.0:1.0 |a fast
1:1.0:0.0 |a slow

shared |x len:0.8 chars:0.6
0:1.0:0.0 |a fast
1:0.0:1.0 |a slow

shared |x len:0.3 chars:0.2
0:0.0:1.0 |a fast
1:1.0:0.0 |a slow

"""
        data_path.write_text(data_content)

        trainer = WarmTrainer(WarmTrainerConfig(passes=5))
        result = trainer.train(
            data_path=data_path,
            output_path=temp_dir / "model.vw",
        )

        assert result.success is True
        assert result.n_examples > 0
        assert (temp_dir / "model.vw").exists()
