"""
Tests for Featurizer.
"""

import pytest
import numpy as np

from src.bandit.config import BanditConfig
from src.bandit.featurizer import Featurizer, FeatureResult


class TestFeaturizer:
    """Test Featurizer class."""

    @pytest.fixture
    def featurizer(self, bandit_config):
        """Create featurizer for testing."""
        return Featurizer(bandit_config)

    def test_extract_explicit_features_simple(self, featurizer):
        """Test explicit feature extraction on simple text."""
        text = "John has 5 apples."

        features = featurizer.extract_explicit_features(text)

        assert features["len_chars"] == len(text)
        assert features["len_tokens"] == 4  # "John", "has", "5", "apples."
        assert features["n_numbers"] == 1  # "5"
        assert features["has_fraction"] is False
        assert features["has_percentage"] is False
        assert features["has_money"] is False

    def test_extract_explicit_features_with_numbers(self, featurizer):
        """Test extraction with multiple numbers."""
        text = "Tom has 10 apples and 5 oranges. He gave away 3."

        features = featurizer.extract_explicit_features(text)

        assert features["n_numbers"] == 3  # 10, 5, 3

    def test_extract_explicit_features_with_fraction(self, featurizer):
        """Test detection of fraction keywords."""
        text = "Half of the apples were eaten."

        features = featurizer.extract_explicit_features(text)

        assert features["has_fraction"] is True

    def test_extract_explicit_features_with_percentage(self, featurizer):
        """Test detection of percentage."""
        text = "Sales increased by 20%."

        features = featurizer.extract_explicit_features(text)

        assert features["has_percentage"] is True

    def test_extract_explicit_features_with_money(self, featurizer):
        """Test detection of money-related keywords."""
        text = "The shirt costs $25."

        features = featurizer.extract_explicit_features(text)

        assert features["has_money"] is True

    def test_extract_explicit_features_with_comparison(self, featurizer):
        """Test detection of comparison keywords."""
        text = "John has twice as many apples as Mary."

        features = featurizer.extract_explicit_features(text)

        assert features["has_comparison"] is True

    def test_extract_explicit_features_with_time(self, featurizer):
        """Test detection of time keywords."""
        text = "She worked for 8 hours."

        features = featurizer.extract_explicit_features(text)

        assert features["has_time"] is True

    def test_question_type_how_many(self, featurizer):
        """Test question type detection for 'how many'."""
        text = "How many apples does John have?"

        features = featurizer.extract_explicit_features(text)

        assert features["question_type"] == "how_many"

    def test_question_type_how_much(self, featurizer):
        """Test question type detection for 'how much'."""
        text = "How much money did she spend?"

        features = featurizer.extract_explicit_features(text)

        assert features["question_type"] == "how_much"

    def test_question_type_what_is(self, featurizer):
        """Test question type detection for 'what is'."""
        text = "What is the total?"

        features = featurizer.extract_explicit_features(text)

        assert features["question_type"] == "what_is"

    def test_question_type_other(self, featurizer):
        """Test question type detection for other questions."""
        text = "Calculate the sum."

        features = featurizer.extract_explicit_features(text)

        assert features["question_type"] == "other"

    def test_featurize_returns_feature_result(self, featurizer):
        """Test featurize returns FeatureResult object."""
        text = "John has 5 apples."

        result = featurizer.featurize(text, include_embedding=False)

        assert isinstance(result, FeatureResult)
        assert result.len_chars == len(text)
        assert result.n_numbers == 1
        assert result.embedding is None  # Embeddings disabled

    def test_feature_result_to_vw_string_without_embedding(self):
        """Test VW string generation without embedding."""
        result = FeatureResult(
            len_chars=100,
            len_tokens=20,
            n_sentences=2,
            n_numbers=3,
            has_fraction=False,
            has_percentage=True,
            has_money=False,
            has_comparison=False,
            has_time=False,
            question_type="how_many",
            embedding=None,
        )

        vw_str = result.to_vw_string(include_embedding=False)

        assert "|x" in vw_str
        assert "len_chars:" in vw_str
        assert "len_tokens:" in vw_str
        assert "n_numbers:3" in vw_str
        assert "has_percentage:1" in vw_str
        assert "has_fraction:0" in vw_str
        assert "qtype_how_many:1" in vw_str
        assert "|e" not in vw_str  # No embedding namespace

    def test_feature_result_to_vw_string_with_embedding(self, mock_embedding):
        """Test VW string generation with embedding."""
        result = FeatureResult(
            len_chars=100,
            len_tokens=20,
            n_sentences=2,
            n_numbers=3,
            has_fraction=False,
            has_percentage=False,
            has_money=False,
            has_comparison=False,
            has_time=False,
            question_type="other",
            embedding=mock_embedding,
        )

        vw_str = result.to_vw_string(include_embedding=True)

        assert "|x" in vw_str
        assert "|e" in vw_str
        # Check embedding values are present
        assert "0:" in vw_str  # First embedding dimension

    def test_to_vw_example_format(self, featurizer):
        """Test VW ADF example format."""
        text = "John has 5 apples."

        vw_example = featurizer.to_vw_example(text, include_embedding=False)

        assert vw_example.startswith("shared |x")
        assert "|Action fast" in vw_example
        assert "|Action slow" in vw_example
        # Should have 3 lines: shared features, fast action, slow action
        lines = vw_example.strip().split("\n")
        assert len(lines) == 3

    def test_embedding_cache(self, featurizer, sample_questions):
        """Test embedding caching behavior."""
        text = sample_questions[0]

        # First call should cache
        featurizer.featurize(text, include_embedding=False)

        # Check cache is being used when enabled
        if featurizer.config.cache_embeddings:
            featurizer.featurize(text, include_embedding=False)
            # Cache should be populated (embeddings disabled in test)

    def test_clear_cache(self, featurizer):
        """Test cache clearing."""
        featurizer._embedding_cache["test"] = np.zeros(384)

        featurizer.clear_cache()

        assert len(featurizer._embedding_cache) == 0
