"""
Hybrid Feature Extraction for Bandit Router.

Combines fast regex-based features with semantic embeddings
for problem complexity estimation.

Feature Namespaces:
- |x : Explicit features (regex, string operations) - <1ms
- |e : Embedding features (MiniLM) - ~20-50ms

VW Input Format (ADF):
    shared |x len_tokens:0.4 n_numbers:5 |e 0:0.04 1:-0.2 ...
    |Action fast
    |Action slow
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import BanditConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """Result from feature extraction."""

    # Explicit features
    len_chars: int
    len_tokens: int  # Approximate token count
    n_sentences: int
    n_numbers: int
    has_fraction: bool
    has_percentage: bool
    has_money: bool
    has_comparison: bool
    has_time: bool
    question_type: str  # "how_many", "how_much", "what_is", "other"

    # Embedding features (optional)
    embedding: Optional[np.ndarray] = None

    def to_vw_string(self, include_embedding: bool = True) -> str:
        """
        Format features for VW input.

        Returns:
            VW-formatted feature string (without action lines)
        """
        # Explicit features namespace |x
        explicit = [
            f"len_chars:{self.len_chars / 1000:.3f}",  # Normalize
            f"len_tokens:{self.len_tokens / 500:.3f}",  # Normalize
            f"n_sentences:{self.n_sentences}",
            f"n_numbers:{self.n_numbers}",
            f"has_fraction:{int(self.has_fraction)}",
            f"has_percentage:{int(self.has_percentage)}",
            f"has_money:{int(self.has_money)}",
            f"has_comparison:{int(self.has_comparison)}",
            f"has_time:{int(self.has_time)}",
            f"qtype_{self.question_type}:1",  # One-hot encoding
        ]
        x_namespace = "|x " + " ".join(explicit)

        if include_embedding and self.embedding is not None:
            # Embedding features namespace |e
            # Format: |e 0:val0 1:val1 ... (sparse format)
            embedding_parts = [
                f"{i}:{v:.4f}" for i, v in enumerate(self.embedding) if abs(v) > 1e-6
            ]
            e_namespace = "|e " + " ".join(embedding_parts)
            return f"{x_namespace} {e_namespace}"

        return x_namespace


class Featurizer:
    """
    Hybrid feature extractor for bandit routing.

    Combines:
    1. Fast regex-based explicit features (<1ms)
    2. Semantic embeddings via MiniLM (~20-50ms)
    """

    # Regex patterns for feature extraction
    NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
    SENTENCE_PATTERN = re.compile(r"[.!?]+")
    FRACTION_KEYWORDS = re.compile(
        r"\b(half|quarter|third|fourth|fifth|fraction|divided by)\b", re.IGNORECASE
    )
    PERCENTAGE_PATTERN = re.compile(r"\b\d+\s*%|\bpercent\b", re.IGNORECASE)
    MONEY_PATTERN = re.compile(r"\$\d+|\b(dollar|cent|price|cost|pay|paid)\b", re.IGNORECASE)
    COMPARISON_KEYWORDS = re.compile(
        r"\b(more|less|fewer|greater|smaller|times|twice|triple|double)\b", re.IGNORECASE
    )
    TIME_KEYWORDS = re.compile(
        r"\b(hours?|minutes?|seconds?|days?|weeks?|months?|years?|morning|afternoon|evening)\b",
        re.IGNORECASE,
    )
    QUESTION_HOW_MANY = re.compile(r"\bhow many\b", re.IGNORECASE)
    QUESTION_HOW_MUCH = re.compile(r"\bhow much\b", re.IGNORECASE)
    QUESTION_WHAT_IS = re.compile(r"\bwhat is\b", re.IGNORECASE)

    def __init__(self, config: BanditConfig):
        """
        Initialize featurizer.

        Args:
            config: Bandit configuration
        """
        self.config = config
        self._embedding_model = None
        self._embedding_cache: dict[str, np.ndarray] = {}

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
            logger.warning(
                "sentence-transformers not installed. "
                "Embeddings will be disabled. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}. Embeddings disabled.")
            return None

    def extract_explicit_features(self, text: str) -> dict:
        """
        Extract regex-based explicit features.

        Fast extraction (<1ms) for basic problem characteristics.

        Args:
            text: Input problem text

        Returns:
            Dictionary of explicit features
        """
        # Basic counts
        len_chars = len(text)
        len_tokens = len(text.split())  # Approximate
        n_sentences = len(self.SENTENCE_PATTERN.split(text))
        n_numbers = len(self.NUMBER_PATTERN.findall(text))

        # Boolean features
        has_fraction = bool(self.FRACTION_KEYWORDS.search(text))
        has_percentage = bool(self.PERCENTAGE_PATTERN.search(text))
        has_money = bool(self.MONEY_PATTERN.search(text))
        has_comparison = bool(self.COMPARISON_KEYWORDS.search(text))
        has_time = bool(self.TIME_KEYWORDS.search(text))

        # Question type classification
        if self.QUESTION_HOW_MANY.search(text):
            question_type = "how_many"
        elif self.QUESTION_HOW_MUCH.search(text):
            question_type = "how_much"
        elif self.QUESTION_WHAT_IS.search(text):
            question_type = "what_is"
        else:
            question_type = "other"

        return {
            "len_chars": len_chars,
            "len_tokens": len_tokens,
            "n_sentences": n_sentences,
            "n_numbers": n_numbers,
            "has_fraction": has_fraction,
            "has_percentage": has_percentage,
            "has_money": has_money,
            "has_comparison": has_comparison,
            "has_time": has_time,
            "question_type": question_type,
        }

    def extract_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract semantic embedding using MiniLM.

        Args:
            text: Input problem text

        Returns:
            Embedding vector or None if unavailable
        """
        if self.embedding_model is None:
            return None

        # Check cache
        if self.config.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)

            # Cache if enabled
            if self.config.cache_embeddings:
                self._embedding_cache[text] = embedding

            return embedding
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return None

    def featurize(self, text: str, include_embedding: bool = True) -> FeatureResult:
        """
        Extract all features from problem text.

        Args:
            text: Input problem text
            include_embedding: Whether to include semantic embeddings

        Returns:
            FeatureResult with all extracted features
        """
        # Extract explicit features (fast)
        explicit = self.extract_explicit_features(text)

        # Extract embedding (slower, optional)
        embedding = None
        if include_embedding:
            embedding = self.extract_embedding(text)

        return FeatureResult(
            len_chars=explicit["len_chars"],
            len_tokens=explicit["len_tokens"],
            n_sentences=explicit["n_sentences"],
            n_numbers=explicit["n_numbers"],
            has_fraction=explicit["has_fraction"],
            has_percentage=explicit["has_percentage"],
            has_money=explicit["has_money"],
            has_comparison=explicit["has_comparison"],
            has_time=explicit["has_time"],
            question_type=explicit["question_type"],
            embedding=embedding,
        )

    def to_vw_example(
        self, text: str, include_embedding: bool = True
    ) -> str:
        """
        Convert problem text to VW ADF example format.

        Args:
            text: Input problem text
            include_embedding: Whether to include embeddings

        Returns:
            VW-formatted example string (ready for predict/learn)
        """
        features = self.featurize(text, include_embedding)
        feature_str = features.to_vw_string(include_embedding)

        # ADF format: shared features + action lines
        return f"shared {feature_str}\n|Action fast\n|Action slow"

    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
