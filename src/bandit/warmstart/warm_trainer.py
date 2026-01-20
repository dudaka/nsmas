"""
Warm-Start Trainer for Offline VW Pre-Training with Doubly Robust Estimation.

Trains a VW contextual bandit policy on oracle labels using Doubly Robust (DR)
estimation for counterfactual risk minimization. DR combines IPS with a direct
method to reduce variance while remaining unbiased.

VW Training Command (DR Estimator):
    vw --cb_explore_adf \\
       --cb_type dr \\
       --data train_adf.dat \\
       --passes 20 \\
       --l2 1e-6 \\
       --cache_file cache.vw \\
       -f warm_policy.vw

Key Design:
- Uses --cb_type dr (Doubly Robust) instead of IPS/MTR
- DR requires propensity < 1.0 in training data (provided by VWConverter)
- Combines regression model with IPS for lower variance

Usage:
    python -m src.bandit.warmstart.warm_trainer \\
        --data data/train_adf.dat \\
        --output models/warm_policy.vw \\
        --passes 20

    # Programmatic usage
    trainer = WarmTrainer()
    trainer.train(
        data_path="data/train_adf.dat",
        output_path="models/warm_policy.vw",
        passes=20
    )
"""

import argparse
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WarmTrainerConfig:
    """Configuration for warm-start training with Doubly Robust estimation."""

    # VW algorithm settings
    learning_rate: float = 0.01
    passes: int = 20

    # Exploration settings (for warm-start, we want low exploration)
    epsilon: float = 0.0  # No exploration during offline training

    # Regularization
    l2_regularization: float = 1e-6  # Light regularization as per research spec

    # VW options - KEY CHANGE: use DR estimator
    cb_type: str = "dr"  # Doubly Robust estimator (was "mtr")
    holdout_off: bool = True  # Use all data for training


@dataclass
class TrainingResult:
    """Result from training run."""

    success: bool
    model_path: Optional[str] = None
    avg_loss: Optional[float] = None
    n_examples: int = 0
    vw_output: str = ""
    error_message: Optional[str] = None


class WarmTrainer:
    """
    Offline VW trainer for warm-start pre-training.

    Uses VW's contextual bandit mode with behavior cloning to
    learn a policy from oracle labels.
    """

    def __init__(self, config: WarmTrainerConfig = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or WarmTrainerConfig()

    def build_vw_command(
        self,
        data_path: Path,
        output_path: Path,
        cache_path: Optional[Path] = None
    ) -> list[str]:
        """
        Build VW command line arguments for DR estimation.

        Args:
            data_path: Path to training data
            output_path: Path for output model
            cache_path: Optional cache file path

        Returns:
            List of command arguments
        """
        cmd = [
            "vw",
            "--cb_explore_adf",
            "--cb_type", self.config.cb_type,  # "dr" for Doubly Robust
            "--data", str(data_path),
            "--final_regressor", str(output_path),
            "--learning_rate", str(self.config.learning_rate),
            "--passes", str(self.config.passes),
            "--l2", str(self.config.l2_regularization),
        ]

        if self.config.epsilon > 0:
            cmd.extend(["--epsilon", str(self.config.epsilon)])

        if self.config.holdout_off:
            cmd.append("--holdout_off")

        if cache_path:
            cmd.extend(["--cache_file", str(cache_path)])
        else:
            cmd.append("--cache")  # Use auto cache

        return cmd

    def train(
        self,
        data_path: str | Path,
        output_path: str | Path,
        passes: int = None,
        learning_rate: float = None
    ) -> TrainingResult:
        """
        Train VW model on oracle data.

        Args:
            data_path: Path to VW training data
            output_path: Path for output model
            passes: Override number of passes
            learning_rate: Override learning rate

        Returns:
            TrainingResult with success status and metrics
        """
        data_path = Path(data_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not data_path.exists():
            return TrainingResult(
                success=False,
                error_message=f"Training data not found: {data_path}"
            )

        # Override config if specified
        if passes is not None:
            self.config.passes = passes
        if learning_rate is not None:
            self.config.learning_rate = learning_rate

        # Create temp cache file
        with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as cache_file:
            cache_path = Path(cache_file.name)

        try:
            # Build and run VW command
            cmd = self.build_vw_command(data_path, output_path, cache_path)
            logger.info(f"Running VW: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"VW training failed: {result.stderr}")
                return TrainingResult(
                    success=False,
                    error_message=result.stderr,
                    vw_output=result.stdout + result.stderr
                )

            # Parse VW output for metrics
            avg_loss = self._parse_avg_loss(result.stderr)
            n_examples = self._parse_n_examples(result.stderr)

            logger.info(f"Training complete. Avg loss: {avg_loss}, Examples: {n_examples}")

            return TrainingResult(
                success=True,
                model_path=str(output_path),
                avg_loss=avg_loss,
                n_examples=n_examples,
                vw_output=result.stderr
            )

        finally:
            # Cleanup cache file
            if cache_path.exists():
                cache_path.unlink()

    def _parse_avg_loss(self, vw_output: str) -> Optional[float]:
        """Parse average loss from VW output."""
        import re
        # VW outputs "average loss = X.XXX"
        match = re.search(r"average loss\s*=\s*([\d.]+)", vw_output)
        if match:
            return float(match.group(1))
        return None

    def _parse_n_examples(self, vw_output: str) -> int:
        """Parse number of examples from VW output."""
        import re
        # VW outputs "number of examples = X"
        match = re.search(r"number of examples\s*=\s*(\d+)", vw_output)
        if match:
            return int(match.group(1))
        return 0

    def evaluate(
        self,
        model_path: str | Path,
        test_data_path: str | Path
    ) -> dict:
        """
        Evaluate trained model on test data.

        Args:
            model_path: Path to trained VW model
            test_data_path: Path to test data

        Returns:
            Dictionary with evaluation metrics
        """
        model_path = Path(model_path)
        test_data_path = Path(test_data_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")

        cmd = [
            "vw",
            "--initial_regressor", str(model_path),
            "--testonly",
            "--data", str(test_data_path),
            "--cb_explore_adf",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            return {"error": result.stderr}

        avg_loss = self._parse_avg_loss(result.stderr)
        n_examples = self._parse_n_examples(result.stderr)

        return {
            "avg_loss": avg_loss,
            "n_examples": n_examples,
            "vw_output": result.stderr
        }


def main():
    """CLI for warm-start trainer."""
    parser = argparse.ArgumentParser(
        description="Train VW warm-start policy on oracle data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to VW training data (ADF format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/warm_policy.vw",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=20,
        help="Number of training passes (default: 20)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0001,
        help="L2 regularization (default: 0.0001)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Optional test data for evaluation"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create trainer
    config = WarmTrainerConfig(
        learning_rate=args.learning_rate,
        passes=args.passes,
        l2_regularization=args.l2
    )
    trainer = WarmTrainer(config)

    # Train
    logger.info(f"Training on {args.data}")
    result = trainer.train(
        data_path=args.data,
        output_path=args.output
    )

    if not result.success:
        print(f"\nTraining FAILED: {result.error_message}")
        return 1

    print(f"\nTraining Complete:")
    print(f"  Model: {result.model_path}")
    print(f"  Examples: {result.n_examples}")
    print(f"  Avg Loss: {result.avg_loss:.6f}" if result.avg_loss else "  Avg Loss: N/A")
    print(f"  Passes: {args.passes}")
    print(f"  Learning Rate: {args.learning_rate}")

    # Evaluate if test data provided
    if args.test_data:
        logger.info(f"Evaluating on {args.test_data}")
        eval_result = trainer.evaluate(args.output, args.test_data)

        if "error" not in eval_result:
            print(f"\nTest Evaluation:")
            print(f"  Examples: {eval_result['n_examples']}")
            print(f"  Avg Loss: {eval_result['avg_loss']:.6f}" if eval_result['avg_loss'] else "  Avg Loss: N/A")

    return 0


if __name__ == "__main__":
    exit(main())
