"""
Contextual Bandit Router using Vowpal Wabbit.

Implements the SquareCB algorithm for adaptive exploration
between Fast and Slow paths.

Design Decisions:
- Binary action space (Fast=0, Slow=1) for clear cost gradient
- ADF format for forward-compatibility with more actions
- Offline learning by default (no online updates during inference)
- SquareCB for adaptive exploration based on confidence gap
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config import BanditConfig
from .featurizer import Featurizer

logger = logging.getLogger(__name__)


class BanditRouter:
    """
    Contextual Bandit Router using Vowpal Wabbit.

    Routes problems between Fast (zero-shot) and Slow (GVR) paths
    based on learned problem complexity patterns.

    Usage:
        config = BanditConfig()
        router = BanditRouter(config)

        # Predict action
        action, prob = router.predict("John has 5 apples...")
        # action: 0 (fast) or 1 (slow)
        # prob: probability of chosen action

        # Learn from outcome (training only)
        router.learn(question, action=0, cost=0.1, prob=0.8)

        # Save model
        router.save()
    """

    # Action indices
    FAST_ACTION = 0
    SLOW_ACTION = 1

    def __init__(self, config: Optional[BanditConfig] = None):
        """
        Initialize the Bandit Router.

        Args:
            config: Bandit configuration (uses defaults if not provided)
        """
        self.config = config or BanditConfig()
        self.config.validate()

        self.featurizer = Featurizer(self.config)
        self._vw = None
        self._initialized = False

    @property
    def vw(self):
        """Lazy initialization of VW workspace."""
        if self._vw is None:
            self._vw = self._init_vw()
            self._initialized = True
        return self._vw

    def _init_vw(self):
        """Initialize Vowpal Wabbit workspace."""
        try:
            import vowpalwabbit

            vw_args = self.config.get_vw_args(load_existing=True)
            logger.info(f"Initializing VW with args: {vw_args}")

            workspace = vowpalwabbit.Workspace(vw_args)

            if self.config.model_path.exists():
                logger.info(f"Loaded existing model from {self.config.model_path}")
            else:
                logger.info("Initialized new VW model (no existing model found)")

            return workspace

        except ImportError:
            raise ImportError(
                "vowpalwabbit is required for BanditRouter. "
                "Install with: pip install vowpalwabbit"
            )

    def predict(self, question: str) -> tuple[int, float]:
        """
        Select action for a question.

        Args:
            question: The math problem text

        Returns:
            (action_index, probability) where:
            - action_index: 0 (fast) or 1 (slow)
            - probability: probability of the chosen action
        """
        # Get VW-formatted example
        vw_example = self.featurizer.to_vw_example(question)

        # Get probability mass function from VW
        try:
            pmf = self.vw.predict(vw_example)

            # Handle different VW return formats
            if isinstance(pmf, list) and len(pmf) >= 2:
                # Check if it's a list of tuples or list of floats
                if isinstance(pmf[0], tuple):
                    # List of (action_id, probability) tuples
                    probs = [p for _, p in pmf]
                else:
                    # List of probabilities directly
                    probs = list(pmf)
            elif isinstance(pmf, (int, float)):
                # Single value - treat as action index or use uniform
                logger.warning(f"VW returned single value: {pmf}, using uniform")
                probs = [0.5, 0.5]
            else:
                # Fallback if unexpected format
                logger.warning(f"Unexpected PMF format: {type(pmf)} {pmf}, using uniform")
                probs = [0.5, 0.5]

            # Sample action from PMF
            action = np.random.choice(len(probs), p=probs)
            prob = probs[action]

            logger.debug(
                f"Predicted action={action} ({self.config.action_names[action]}) "
                f"with prob={prob:.3f}"
            )

            return action, prob

        except Exception as e:
            logger.error(f"VW prediction failed: {e}. Defaulting to slow path.")
            return self.SLOW_ACTION, 1.0

    def predict_deterministic(self, question: str) -> tuple[int, float]:
        """
        Select action deterministically (argmax).

        Useful for evaluation where we want consistent behavior.

        Args:
            question: The math problem text

        Returns:
            (action_index, probability) - highest probability action
        """
        vw_example = self.featurizer.to_vw_example(question)

        try:
            pmf = self.vw.predict(vw_example)

            # Handle different VW return formats
            if isinstance(pmf, list) and len(pmf) >= 2:
                if isinstance(pmf[0], tuple):
                    probs = [p for _, p in pmf]
                else:
                    probs = list(pmf)
                action = int(np.argmax(probs))
                prob = probs[action]
            else:
                action = self.SLOW_ACTION
                prob = 1.0

            return action, prob

        except Exception as e:
            logger.error(f"VW prediction failed: {e}. Defaulting to slow path.")
            return self.SLOW_ACTION, 1.0

    def learn(
        self,
        question: str,
        action: int,
        cost: float,
        prob: float,
    ) -> None:
        """
        Update policy with observed outcome.

        Only updates if enable_online_learning is True in config.

        Args:
            question: The problem text
            action: Action that was taken (0=fast, 1=slow)
            cost: Observed cost/loss
            prob: Probability with which action was taken
        """
        if not self.config.enable_online_learning:
            logger.debug("Online learning disabled, skipping update")
            return

        # Build labeled example in ADF format
        labeled_example = self._format_labeled_example(question, action, cost, prob)

        try:
            self.vw.learn(labeled_example)
            logger.debug(f"Learned from example: action={action}, cost={cost:.3f}")
        except Exception as e:
            logger.error(f"VW learn failed: {e}")

    def _format_labeled_example(
        self,
        question: str,
        action: int,
        cost: float,
        prob: float,
    ) -> str:
        """
        Format a labeled example for VW learning.

        In CB ADF format, the label goes on the chosen action line:
            shared |features...
            0:cost:prob |Action fast   <- label on action 0
            |Action slow

        Args:
            question: Problem text
            action: Chosen action (0 or 1)
            cost: Observed cost
            prob: Selection probability

        Returns:
            VW-formatted labeled example
        """
        features = self.featurizer.featurize(question)
        feature_str = features.to_vw_string(include_embedding=True)

        # Build action lines with label on chosen action
        if action == 0:
            action_lines = [
                f"0:{cost:.4f}:{prob:.4f} |Action fast",
                "|Action slow",
            ]
        else:
            action_lines = [
                "|Action fast",
                f"0:{cost:.4f}:{prob:.4f} |Action slow",
            ]

        return f"shared {feature_str}\n" + "\n".join(action_lines)

    def save(self) -> Path:
        """
        Save model to disk.

        Returns:
            Path to saved model file
        """
        model_path = self.config.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self.vw.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

        return model_path

    def get_action_name(self, action: int) -> str:
        """Get human-readable name for action."""
        if 0 <= action < len(self.config.action_names):
            return self.config.action_names[action]
        return f"unknown_{action}"

    def get_stats(self) -> dict:
        """
        Get router statistics.

        Returns:
            Dictionary with model info
        """
        return {
            "model_path": str(self.config.model_path),
            "model_exists": self.config.model_path.exists(),
            "exploration_algo": self.config.exploration_algo,
            "gamma_scale": self.config.gamma_scale,
            "online_learning_enabled": self.config.enable_online_learning,
            "action_space": list(self.config.action_names),
        }

    def __del__(self):
        """Cleanup VW workspace."""
        if self._vw is not None:
            try:
                self._vw.finish()
            except Exception:
                pass


def create_router(
    model_path: Optional[str] = None,
    exploration_algo: str = "squarecb",
    enable_online_learning: bool = False,
) -> BanditRouter:
    """
    Factory function to create a BanditRouter.

    Args:
        model_path: Path to existing model file (optional)
        exploration_algo: "squarecb" or "epsilon"
        enable_online_learning: Whether to update during inference

    Returns:
        Configured BanditRouter
    """
    config = BanditConfig(
        exploration_algo=exploration_algo,
        enable_online_learning=enable_online_learning,
    )

    if model_path:
        config.model_dir = str(Path(model_path).parent)
        config.model_filename = Path(model_path).name

    return BanditRouter(config)
