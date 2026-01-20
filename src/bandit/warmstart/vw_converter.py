"""
VW ADF Format Converter for Warm-Start Training.

Converts oracle labels with PCA-reduced features to Vowpal Wabbit
Action-Dependent Features (ADF) format for Doubly Robust (DR) estimation.

VW ADF Format (with simulated exploration):
    shared |x pca_0:0.12 ... |e token_cnt:0.45 has_neg:1 has_logic:0 has_quant:1
    0:-1.0:0.9 |a fast
    |a slow

Where:
- 0:-1.0:0.9 means action 0, cost -1.0 (reward), probability 0.9 (simulated logging)
- Simulated propensity (0.9/0.1) enables DR estimator to learn counterfactuals

Key Design Decisions:
- 50 PCA components + 4 semantic features = 54-dim context
- Simulated logging policy: 90% logged action, 10% exploration
- DR estimator requires propensity < 1.0 to estimate counterfactuals

Usage:
    python -m src.bandit.warmstart.vw_converter \\
        --oracle data/oracle_training_set.json \\
        --pca models/pca_50.pkl \\
        --output data/train_adf.dat

    # Programmatic usage
    converter = VWConverter(pca_path="models/pca_50.pkl")
    converter.convert(
        oracle_path="data/oracle_training_set.json",
        output_path="data/train_adf.dat"
    )
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .pca_transformer import PCATransformer, PCATransformerConfig
from .oracle_builder import OracleLabel

logger = logging.getLogger(__name__)


@dataclass
class VWConverterConfig:
    """Configuration for VW converter."""

    # Costs for loss function
    # Use negative cost for correct action so VW learns to prefer it
    correct_cost: float = -1.0  # Negative cost = reward for correct action
    incorrect_cost: float = 1.0  # Cost for incorrect action

    # Simulated logging policy propensity
    # Key insight: DR estimator needs p < 1.0 to estimate counterfactuals
    # We simulate a stochastic logging policy: 90% logged action, 10% exploration
    logged_action_prob: float = 0.9  # Probability of the logged (oracle) action
    exploration_prob: float = 0.1  # Probability of the other action

    # Feature settings
    include_explicit_features: bool = True
    normalize_pca: bool = True  # Normalize PCA features to [-1, 1]


class VWConverter:
    """
    Converts oracle labels to VW ADF format with DR-compatible propensities.

    Combines PCA-reduced embeddings with 4 semantic features to create
    VW-compatible training data for Doubly Robust estimation.
    """

    # Regex patterns for explicit feature extraction
    NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

    # Semantic detection patterns (from research specification)
    NEGATION_PATTERN = re.compile(r"\b(no|not|never|neither|none|nobody|nothing|nowhere|without)\b", re.IGNORECASE)
    LOGIC_PATTERN = re.compile(r"\b(if|then|implies|therefore|thus|hence|because|since|when|unless)\b", re.IGNORECASE)
    QUANTIFIER_PATTERN = re.compile(r"\b(all|every|each|some|any|most|few|many|several|none)\b", re.IGNORECASE)

    def __init__(
        self,
        pca_path: str | Path,
        config: VWConverterConfig = None
    ):
        """
        Initialize VW converter.

        Args:
            pca_path: Path to fitted PCA transformer
            config: Converter configuration
        """
        self.pca = PCATransformer.load(pca_path)
        self.config = config or VWConverterConfig()

    def extract_explicit_features(self, text: str) -> dict[str, float]:
        """
        Extract 4 semantic features from question text.

        These complement PCA features with high-signal reasoning indicators:
        - token_cnt: Normalized token count (proxy for complexity)
        - has_neg: Presence of negation words
        - has_logic: Presence of logical connectives
        - has_quant: Presence of quantifiers

        Args:
            text: Question text

        Returns:
            Dictionary of feature name -> value
        """
        # Normalized token count (0-1 range)
        len_tokens = len(text.split())
        token_cnt = min(len_tokens / 200, 1.0)  # Cap at 200 tokens

        # Boolean semantic features (0 or 1)
        has_neg = 1.0 if self.NEGATION_PATTERN.search(text) else 0.0
        has_logic = 1.0 if self.LOGIC_PATTERN.search(text) else 0.0
        has_quant = 1.0 if self.QUANTIFIER_PATTERN.search(text) else 0.0

        return {
            "token_cnt": token_cnt,
            "has_neg": has_neg,
            "has_logic": has_logic,
            "has_quant": has_quant,
        }

    def format_pca_features(self, embedding: np.ndarray) -> str:
        """
        Format PCA features as VW namespace string.

        Args:
            embedding: PCA-reduced embedding (n_components,)

        Returns:
            VW feature string like "pca_0:0.12 pca_1:-0.04 ..."
        """
        parts = []
        for i, v in enumerate(embedding):
            if self.config.normalize_pca:
                # Clip to reasonable range
                v = np.clip(v, -5, 5) / 5
            if abs(v) > 1e-6:  # Sparse format
                parts.append(f"pca_{i}:{v:.4f}")
        return " ".join(parts)

    def format_explicit_features(self, features: dict[str, float]) -> str:
        """
        Format explicit semantic features as VW string.

        Args:
            features: Dictionary of feature values

        Returns:
            VW feature string like "token_cnt:0.45 has_neg:1"
        """
        # Format with appropriate precision (integers for boolean features)
        parts = []
        for k, v in features.items():
            if k.startswith("has_"):
                # Boolean features: use integer format
                parts.append(f"{k}:{int(v)}")
            else:
                # Continuous features: use decimal format
                parts.append(f"{k}:{v:.4f}")
        return " ".join(parts)

    def format_example(
        self,
        label: OracleLabel,
        pca_embedding: np.ndarray,
        rng: np.random.Generator = None
    ) -> str:
        """
        Format a single oracle label as VW ADF example with simulated stochastic exploration.

        Key insight: We simulate a stochastic logging policy that explores with 10% probability.
        This means 10% of examples will log the NON-oracle action, which may have cost=1.0.
        This provides the counterfactual signal DR needs to learn.

        Args:
            label: Oracle label with routing decision
            pca_embedding: PCA-reduced embedding
            rng: Random number generator for exploration simulation

        Returns:
            VW ADF formatted example string
        """
        if rng is None:
            rng = np.random.default_rng()

        # Build feature string
        pca_str = self.format_pca_features(pca_embedding)

        feature_parts = [f"|x {pca_str}"]

        if self.config.include_explicit_features:
            explicit = self.extract_explicit_features(label.question)
            # Use separate namespace |e for explicit features
            explicit_str = self.format_explicit_features(explicit)
            feature_parts.append(f"|e {explicit_str}")

        shared_line = "shared " + " ".join(feature_parts)

        # Compute actual costs based on correctness
        fast_cost = 0.0 if label.fast_correct else 1.0
        slow_cost = 0.0 if label.slow_correct else 1.0

        # Simulate stochastic logging: 90% oracle action, 10% exploration
        # This provides counterfactual signal for DR estimator
        explore = rng.random() < self.config.exploration_prob  # 10% exploration

        if explore:
            # Log the non-oracle action (exploration)
            if label.oracle_action == "fast":
                # Explore: log slow action
                logged_action = "slow"
                logged_cost = slow_cost
                logged_prob = self.config.exploration_prob  # 0.1
            else:
                # Explore: log fast action
                logged_action = "fast"
                logged_cost = fast_cost
                logged_prob = self.config.exploration_prob  # 0.1
        else:
            # Log the oracle action (exploit)
            logged_action = label.oracle_action
            if logged_action == "fast":
                logged_cost = fast_cost
            else:
                logged_cost = slow_cost
            logged_prob = self.config.logged_action_prob  # 0.9

        # CB_ADF format: only the logged action has cost:probability
        if logged_action == "fast":
            fast_line = f"0:{logged_cost}:{logged_prob} |a fast"
            slow_line = "|a slow"
        else:
            fast_line = "|a fast"
            slow_line = f"1:{logged_cost}:{logged_prob} |a slow"

        return f"{shared_line}\n{fast_line}\n{slow_line}"

    def convert(
        self,
        oracle_path: str | Path,
        output_path: str | Path,
        batch_size: int = 100,
        random_seed: int = 42
    ) -> int:
        """
        Convert oracle labels to VW ADF format with simulated stochastic exploration.

        Args:
            oracle_path: Path to oracle JSON file
            output_path: Output path for VW data
            batch_size: Batch size for embedding extraction
            random_seed: Seed for reproducible exploration simulation

        Returns:
            Number of examples converted
        """
        oracle_path = Path(oracle_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create RNG for reproducible exploration simulation
        rng = np.random.default_rng(random_seed)

        # Load oracle labels
        logger.info(f"Loading oracle labels from {oracle_path}")
        with open(oracle_path, "r") as f:
            data = json.load(f)

        labels = [OracleLabel.from_dict(d) for d in data["labels"]]
        logger.info(f"Loaded {len(labels)} oracle labels")

        # Extract questions for batch embedding
        questions = [label.question for label in labels]

        # Extract embeddings in batches
        logger.info("Extracting embeddings...")
        embeddings = self.pca.extract_embeddings_batch(questions, batch_size=batch_size)

        # Apply PCA
        logger.info("Applying PCA transformation...")
        pca_embeddings = self.pca.transform(embeddings)

        # Convert to VW format with simulated exploration
        logger.info(f"Converting to VW format with {self.config.exploration_prob*100:.0f}% exploration...")
        explore_count = 0
        with open(output_path, "w") as f:
            for i, (label, pca_emb) in enumerate(zip(labels, pca_embeddings)):
                # Check if this example will explore (for logging)
                will_explore = rng.random() < self.config.exploration_prob
                if will_explore:
                    explore_count += 1

                # Reset and regenerate same random state for format_example
                example = self.format_example(label, pca_emb, rng)
                f.write(example + "\n\n")  # Blank line between examples

                if (i + 1) % 1000 == 0:
                    logger.info(f"  Converted {i + 1}/{len(labels)} examples")

        logger.info(f"VW data saved to {output_path}")
        logger.info(f"Exploration rate: {explore_count}/{len(labels)} ({100*explore_count/len(labels):.1f}%)")
        return len(labels)

    def convert_single(self, question: str, oracle_action: str) -> str:
        """
        Convert a single question to VW format.

        Useful for online prediction format.

        Args:
            question: Question text
            oracle_action: "fast" or "slow"

        Returns:
            VW formatted example
        """
        # Create temporary label
        label = OracleLabel(
            problem_id="",
            question=question,
            ground_truth=0,
            variant="",
            fast_answer=None,
            fast_correct=False,
            slow_answer=None,
            slow_correct=False,
            oracle_action=oracle_action,
            oracle_reason=""
        )

        # Get PCA embedding
        pca_emb = self.pca.transform_text(question)

        return self.format_example(label, pca_emb)

    def format_prediction_example(self, question: str) -> str:
        """
        Format question for VW prediction (no labels).

        Args:
            question: Question text

        Returns:
            VW formatted example for prediction
        """
        # Get PCA embedding
        pca_emb = self.pca.transform_text(question)
        pca_str = self.format_pca_features(pca_emb)

        feature_parts = [f"|x {pca_str}"]

        if self.config.include_explicit_features:
            explicit = self.extract_explicit_features(question)
            explicit_str = self.format_explicit_features(explicit)
            feature_parts.append(f"|e {explicit_str}")

        shared_line = "shared " + " ".join(feature_parts)

        # Action lines without costs (prediction mode)
        fast_line = "|a fast"
        slow_line = "|a slow"

        return f"{shared_line}\n{fast_line}\n{slow_line}"


def main():
    """CLI for VW converter."""
    parser = argparse.ArgumentParser(
        description="Convert oracle labels to VW ADF format"
    )
    parser.add_argument(
        "--oracle",
        type=str,
        required=True,
        help="Path to oracle JSON file"
    )
    parser.add_argument(
        "--pca",
        type=str,
        required=True,
        help="Path to fitted PCA model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_adf.dat",
        help="Output path for VW data"
    )
    parser.add_argument(
        "--correct-cost",
        type=float,
        default=0.0,
        help="Cost for correct action (default: 0.0)"
    )
    parser.add_argument(
        "--incorrect-cost",
        type=float,
        default=1.0,
        help="Cost for incorrect action (default: 1.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding extraction"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create converter
    config = VWConverterConfig(
        correct_cost=args.correct_cost,
        incorrect_cost=args.incorrect_cost
    )
    converter = VWConverter(pca_path=args.pca, config=config)

    # Convert
    n_examples = converter.convert(
        oracle_path=args.oracle,
        output_path=args.output,
        batch_size=args.batch_size
    )

    print(f"\nVW Conversion Complete:")
    print(f"  Examples: {n_examples}")
    print(f"  PCA components: {converter.pca.n_components}")
    print(f"  Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
