"""
PCA Transformer for Dimensionality Reduction.

Reduces MiniLM embeddings from 384 to 50 dimensions to address the
regret bound O(d√T) limitation that caused cold-start failure in Phase 7.

Note: all-MiniLM-L6-v2 produces 384-dim embeddings, not 768.
With 50 PCA components + 4 explicit features = 54-dim context.

Usage:
    # Fit PCA on dataset
    python -m src.bandit.warmstart.pca_transformer \\
        --data-dir output \\
        --output models/pca_50.pkl \\
        --n-components 50

    # Programmatic usage
    transformer = PCATransformer(n_components=50)
    transformer.fit_from_jsonl("output/gsm_base.jsonl")
    transformer.save("models/pca_50.pkl")

    # Transform new embeddings
    reduced = transformer.transform(embedding_384d)
"""

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class PCATransformerConfig:
    """Configuration for PCA transformer."""

    n_components: int = 50  # Reduced from 64 for better sample complexity
    whiten: bool = True  # Whitening helps SGD convergence
    random_state: int = 42
    embedding_model: str = "all-MiniLM-L6-v2"  # Produces 384-dim embeddings


@dataclass
class PCATransformer:
    """
    PCA transformer for embedding dimensionality reduction.

    Reduces 384-dimensional MiniLM embeddings to a configurable
    number of components (default 50) while preserving ~85% variance.
    """

    config: PCATransformerConfig = field(default_factory=PCATransformerConfig)
    pca: Optional[PCA] = field(default=None, repr=False)
    _embedding_model: Optional[object] = field(default=None, repr=False)

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = self._load_embedding_model()
        return self._embedding_model

    def _load_embedding_model(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            model = SentenceTransformer(self.config.embedding_model)
            logger.info("Embedding model loaded successfully")
            return model
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )

    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract 384-dim embedding from text."""
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def extract_embeddings_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for multiple texts."""
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True
        )

    def fit(self, embeddings: np.ndarray) -> "PCATransformer":
        """
        Fit PCA on embedding matrix.

        Args:
            embeddings: (N, 384) matrix of embeddings

        Returns:
            self for chaining
        """
        logger.info(f"Fitting PCA: {embeddings.shape} -> {self.config.n_components} components")

        self.pca = PCA(
            n_components=self.config.n_components,
            whiten=self.config.whiten,
            random_state=self.config.random_state
        )
        self.pca.fit(embeddings)

        variance_retained = sum(self.pca.explained_variance_ratio_) * 100
        logger.info(f"PCA fitted. Variance retained: {variance_retained:.2f}%")

        return self

    def fit_from_jsonl(self, *jsonl_paths: str | Path, question_key: str = "question") -> "PCATransformer":
        """
        Fit PCA from JSONL dataset files.

        Args:
            jsonl_paths: Paths to JSONL files containing problems
            question_key: Key for question text in JSON records

        Returns:
            self for chaining
        """
        questions = []

        for path in jsonl_paths:
            path = Path(path)
            logger.info(f"Loading questions from {path}")

            with open(path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    questions.append(record[question_key])

        logger.info(f"Loaded {len(questions)} questions")

        # Extract embeddings
        logger.info("Extracting embeddings...")
        embeddings = self.extract_embeddings_batch(questions)

        return self.fit(embeddings)

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        """
        Transform embedding(s) to reduced dimensions.

        Args:
            embedding: Single (384,) or batch (N, 384) embeddings

        Returns:
            Reduced (n_components,) or (N, n_components) embeddings
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            return self.pca.transform(embedding)[0]

        return self.pca.transform(embedding)

    def transform_text(self, text: str) -> np.ndarray:
        """
        Extract embedding and transform to reduced dimensions.

        Args:
            text: Input text

        Returns:
            Reduced (n_components,) embedding
        """
        embedding = self.extract_embedding(text)
        return self.transform(embedding)

    def save(self, path: str | Path) -> None:
        """Save fitted PCA transformer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save config as dict to avoid pickle class issues
        config_dict = {
            "n_components": self.config.n_components,
            "whiten": self.config.whiten,
            "random_state": self.config.random_state,
            "embedding_model": self.config.embedding_model,
        }

        with open(path, "wb") as f:
            pickle.dump({
                "config_dict": config_dict,
                "pca": self.pca
            }, f)

        logger.info(f"PCA transformer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PCATransformer":
        """Load fitted PCA transformer from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Handle both old format (config object) and new format (config_dict)
        if "config_dict" in data:
            config = PCATransformerConfig(**data["config_dict"])
        else:
            # Legacy format with config object
            config = data["config"]

        transformer = cls(config=config)
        transformer.pca = data["pca"]

        logger.info(f"PCA transformer loaded from {path}")
        return transformer

    @property
    def n_components(self) -> int:
        """Number of PCA components."""
        return self.config.n_components

    @property
    def variance_retained(self) -> float:
        """Fraction of variance retained by PCA."""
        if self.pca is None:
            return 0.0
        return sum(self.pca.explained_variance_ratio_)


def main():
    """CLI for PCA transformer training."""
    parser = argparse.ArgumentParser(
        description="Train PCA transformer on GSM dataset embeddings"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output",
        help="Directory containing GSM JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pca_50.pkl",
        help="Output path for fitted PCA model"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of PCA components (default: 50)"
    )
    parser.add_argument(
        "--no-whiten",
        action="store_true",
        help="Disable PCA whitening"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "p1", "p2", "noop"],
        help="Dataset variants to include"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Build paths
    data_dir = Path(args.data_dir)
    jsonl_paths = [data_dir / f"gsm_{variant}.jsonl" for variant in args.variants]

    # Filter to existing files
    existing_paths = [p for p in jsonl_paths if p.exists()]
    if not existing_paths:
        logger.error(f"No dataset files found in {data_dir}")
        return 1

    logger.info(f"Found {len(existing_paths)} dataset files")

    # Create and fit transformer
    config = PCATransformerConfig(
        n_components=args.n_components,
        whiten=not args.no_whiten
    )
    transformer = PCATransformer(config=config)
    transformer.fit_from_jsonl(*existing_paths)

    # Save
    transformer.save(args.output)

    # Report
    print(f"\nPCA Transformer trained:")
    print(f"  Components: {transformer.n_components}")
    print(f"  Variance retained: {transformer.variance_retained * 100:.2f}%")
    print(f"  Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
