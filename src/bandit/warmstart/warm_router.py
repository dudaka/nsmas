"""
Warm-Start Bandit Router with PCA Features and Semantic Indicators.

Uses a pre-trained VW model (trained with DR estimator) with 50 PCA
components + 4 semantic features for inference.

Feature format matches VWConverter for consistency:
- |x namespace: 50 PCA features
- |e namespace: 4 semantic features (token_cnt, has_neg, has_logic, has_quant)

Usage:
    router = WarmStartRouter(
        model_path="models/warm_policy.vw",
        pca_path="models/pca_50.pkl"
    )
    action, prob = router.predict("John has 5 apples...")
"""

import logging
import re
from pathlib import Path
from typing import Tuple

import numpy as np

from .pca_transformer import PCATransformer

logger = logging.getLogger(__name__)


class WarmStartRouter:
    """
    Warm-start Bandit Router using PCA + semantic features.

    Uses the same feature format as VWConverter for consistency:
    - 50 PCA components in |x namespace
    - 4 semantic features in |e namespace
    """

    # Action indices
    FAST_ACTION = 0
    SLOW_ACTION = 1

    # Semantic detection patterns (must match VWConverter)
    NEGATION_PATTERN = re.compile(r"\b(no|not|never|neither|none|nobody|nothing|nowhere|without)\b", re.IGNORECASE)
    LOGIC_PATTERN = re.compile(r"\b(if|then|implies|therefore|thus|hence|because|since|when|unless)\b", re.IGNORECASE)
    QUANTIFIER_PATTERN = re.compile(r"\b(all|every|each|some|any|most|few|many|several|none)\b", re.IGNORECASE)

    def __init__(
        self,
        model_path: str | Path,
        pca_path: str | Path,
    ):
        """
        Initialize the warm-start router.

        Args:
            model_path: Path to pre-trained VW model (trained with DR)
            pca_path: Path to fitted PCA transformer (50 components)
        """
        self.model_path = Path(model_path)
        self.pca_path = Path(pca_path)

        # Load PCA transformer
        logger.info(f"Loading PCA transformer from {pca_path}")
        self.pca = PCATransformer.load(pca_path)

        # Initialize VW
        self._vw = self._init_vw()

    def _init_vw(self):
        """Initialize VW workspace with pre-trained model."""
        try:
            import vowpalwabbit

            # Load model for prediction only
            vw_args = [
                "--cb_explore_adf",
                f"--initial_regressor={self.model_path}",
                "--quiet",
            ]
            vw = vowpalwabbit.Workspace(" ".join(vw_args))
            logger.info(f"Loaded VW model from {self.model_path}")
            return vw

        except ImportError:
            raise ImportError("vowpalwabbit required. Install with: pip install vowpalwabbit")

    def _extract_explicit_features(self, question: str) -> dict:
        """
        Extract 4 semantic features from question text.

        Must match VWConverter exactly for model compatibility.
        """
        # Normalized token count (0-1 range)
        len_tokens = len(question.split())
        token_cnt = min(len_tokens / 200, 1.0)

        # Boolean semantic features (0 or 1)
        has_neg = 1.0 if self.NEGATION_PATTERN.search(question) else 0.0
        has_logic = 1.0 if self.LOGIC_PATTERN.search(question) else 0.0
        has_quant = 1.0 if self.QUANTIFIER_PATTERN.search(question) else 0.0

        return {
            "token_cnt": token_cnt,
            "has_neg": has_neg,
            "has_logic": has_logic,
            "has_quant": has_quant,
        }

    def _format_vw_example(
        self,
        pca_features: np.ndarray,
        explicit_features: dict
    ) -> str:
        """
        Format features as VW ADF example.

        Uses same format as VWConverter for compatibility:
        - |x namespace for PCA features
        - |e namespace for explicit semantic features
        """
        # Format PCA features (|x namespace)
        pca_parts = [
            f"pca_{i}:{v:.4f}"
            for i, v in enumerate(pca_features)
            if not np.isnan(v)
        ]
        pca_str = " ".join(pca_parts)

        # Format explicit features (|e namespace)
        explicit_parts = []
        for k, v in explicit_features.items():
            if k.startswith("has_"):
                explicit_parts.append(f"{k}:{int(v)}")
            else:
                explicit_parts.append(f"{k}:{v:.4f}")
        explicit_str = " ".join(explicit_parts)

        # Shared line with both namespaces
        shared_line = f"shared |x {pca_str} |e {explicit_str}"

        # Action lines (no labels for prediction)
        fast_line = "|a fast"
        slow_line = "|a slow"

        return f"{shared_line}\n{fast_line}\n{slow_line}"

    def predict(self, question: str) -> Tuple[int, float]:
        """
        Predict which path to take.

        Args:
            question: The math problem text

        Returns:
            (action_index, probability) where:
            - action_index: 0 (fast) or 1 (slow)
            - probability: probability of the chosen action
        """
        # Extract embeddings and apply PCA
        embedding = self.pca.extract_embeddings_batch([question])[0]
        pca_features = self.pca.transform(embedding.reshape(1, -1))[0]

        # Extract explicit features
        explicit = self._extract_explicit_features(question)

        # Format VW example
        vw_example = self._format_vw_example(pca_features, explicit)

        # Get prediction from VW
        try:
            pmf = self._vw.predict(vw_example)

            # Parse PMF (action probabilities)
            if isinstance(pmf, list) and len(pmf) >= 2:
                if isinstance(pmf[0], tuple):
                    probs = [p for _, p in pmf]
                else:
                    probs = list(pmf)
            else:
                logger.warning(f"Unexpected PMF format: {pmf}, using uniform")
                probs = [0.5, 0.5]

            # Sample action from PMF
            action = np.random.choice(len(probs), p=probs)
            prob = probs[action]

            logger.debug(
                f"Predicted action={action} "
                f"({'fast' if action == 0 else 'slow'}) "
                f"with prob={prob:.3f}"
            )

            return action, prob

        except Exception as e:
            logger.error(f"VW prediction failed: {e}. Defaulting to slow path.")
            return self.SLOW_ACTION, 1.0

    def predict_deterministic(self, question: str) -> Tuple[int, float]:
        """
        Predict deterministically (argmax).

        Args:
            question: The math problem text

        Returns:
            (action_index, probability) - highest probability action
        """
        # Extract embeddings and apply PCA
        embedding = self.pca.extract_embeddings_batch([question])[0]
        pca_features = self.pca.transform(embedding.reshape(1, -1))[0]

        # Extract explicit features
        explicit = self._extract_explicit_features(question)

        # Format VW example
        vw_example = self._format_vw_example(pca_features, explicit)

        try:
            pmf = self._vw.predict(vw_example)

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
