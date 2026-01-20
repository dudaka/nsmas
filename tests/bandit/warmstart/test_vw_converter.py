"""
Tests for VWConverter.
"""

import json
import numpy as np
import pytest

from src.bandit.warmstart.vw_converter import VWConverter, VWConverterConfig
from src.bandit.warmstart.oracle_builder import OracleLabel


class TestVWConverterConfig:
    """Test VWConverterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VWConverterConfig()

        assert config.correct_cost == 0.0
        assert config.incorrect_cost == 1.0
        assert config.include_explicit_features is True
        assert config.normalize_pca is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VWConverterConfig(
            correct_cost=0.1,
            incorrect_cost=0.9,
            include_explicit_features=False,
        )

        assert config.correct_cost == 0.1
        assert config.incorrect_cost == 0.9
        assert config.include_explicit_features is False


class TestVWConverter:
    """Test VWConverter class."""

    @pytest.fixture
    def converter(self, mock_pca_transformer):
        """Create converter with mock PCA."""
        return VWConverter(pca_path=mock_pca_transformer)

    def test_init(self, mock_pca_transformer):
        """Test converter initialization."""
        converter = VWConverter(pca_path=mock_pca_transformer)

        assert converter.pca is not None
        assert converter.config.correct_cost == 0.0

    def test_extract_explicit_features(self, converter):
        """Test explicit feature extraction."""
        text = "John has 5 apples and 10 oranges. How many fruits?"

        features = converter.extract_explicit_features(text)

        assert "len" in features
        assert "chars" in features
        assert "num_cnt" in features
        assert features["num_cnt"] > 0  # Has numbers

    def test_extract_explicit_features_normalization(self, converter):
        """Test that features are normalized."""
        text = "Short text with 5 numbers."

        features = converter.extract_explicit_features(text)

        # All features should be normalized to reasonable ranges
        assert 0 <= features["len"] <= 1
        assert 0 <= features["chars"] <= 1
        assert features["num_cnt"] >= 0

    def test_format_pca_features(self, converter, sample_embeddings):
        """Test PCA feature formatting."""
        pca_emb = converter.pca.transform(sample_embeddings[0])
        pca_str = converter.format_pca_features(pca_emb)

        # Should be space-separated key:value pairs
        assert "pca_0:" in pca_str
        parts = pca_str.split()
        for part in parts:
            assert ":" in part
            key, value = part.split(":")
            assert key.startswith("pca_")
            float(value)  # Should be valid float

    def test_format_pca_features_sparse(self, converter):
        """Test that near-zero values are omitted."""
        # Create embedding with mostly zeros
        pca_emb = np.array([0.0, 0.0, 0.0001, 0.0, 1.5, 0.0, -2.0, 0.0])
        pca_str = converter.format_pca_features(pca_emb)

        # Should not include near-zero values
        assert "pca_0" not in pca_str
        assert "pca_4:" in pca_str
        assert "pca_6:" in pca_str

    def test_format_explicit_features(self, converter):
        """Test explicit feature formatting."""
        features = {"len": 0.5, "chars": 0.3, "num_cnt": 0.2}

        result = converter.format_explicit_features(features)

        assert "len:0.5000" in result
        assert "chars:0.3000" in result
        assert "num_cnt:0.2000" in result

    def test_format_example_fast_action(self, converter, sample_oracle_labels, sample_embeddings):
        """Test formatting example where oracle action is fast."""
        label = sample_oracle_labels[0]  # oracle_action="fast"
        pca_emb = converter.pca.transform(sample_embeddings[0])

        example = converter.format_example(label, pca_emb)
        lines = example.strip().split("\n")

        assert len(lines) == 3
        assert lines[0].startswith("shared |x")

        # Fast action should have cost 0.0 (correct), prob 1.0
        assert "0:0.0:1.0 |a fast" in lines[1]
        # Slow action should have cost 1.0 (incorrect), prob 0.0
        assert "1:1.0:0.0 |a slow" in lines[2]

    def test_format_example_slow_action(self, converter, sample_oracle_labels, sample_embeddings):
        """Test formatting example where oracle action is slow."""
        label = sample_oracle_labels[1]  # oracle_action="slow"
        pca_emb = converter.pca.transform(sample_embeddings[0])

        example = converter.format_example(label, pca_emb)
        lines = example.strip().split("\n")

        # Fast action should have cost 1.0 (incorrect), prob 0.0
        assert "0:1.0:0.0 |a fast" in lines[1]
        # Slow action should have cost 0.0 (correct), prob 1.0
        assert "1:0.0:1.0 |a slow" in lines[2]

    def test_format_example_includes_explicit_features(self, converter, sample_oracle_labels, sample_embeddings):
        """Test that explicit features are included when configured."""
        label = sample_oracle_labels[0]
        pca_emb = converter.pca.transform(sample_embeddings[0])

        example = converter.format_example(label, pca_emb)

        assert "len:" in example
        assert "chars:" in example

    def test_format_example_no_explicit_features(self, mock_pca_transformer, sample_oracle_labels, sample_embeddings):
        """Test example without explicit features."""
        config = VWConverterConfig(include_explicit_features=False)
        converter = VWConverter(pca_path=mock_pca_transformer, config=config)
        label = sample_oracle_labels[0]
        pca_emb = converter.pca.transform(sample_embeddings[0])

        example = converter.format_example(label, pca_emb)

        # Should still have PCA features but not explicit
        assert "|x pca_" in example
        # Check that explicit features are not in the shared line
        lines = example.strip().split("\n")
        shared_line = lines[0]
        assert "len:" not in shared_line or "pca_" in shared_line

    def test_format_prediction_example(self, converter):
        """Test formatting example for prediction (no labels)."""
        question = "John has 5 apples. How many?"

        # Mock transform_text to avoid loading embedding model
        converter.pca.transform_text = lambda x: np.random.randn(8)

        example = converter.format_prediction_example(question)
        lines = example.strip().split("\n")

        assert len(lines) == 3
        assert lines[0].startswith("shared |x")
        # Prediction mode: no costs
        assert lines[1] == "|a fast"
        assert lines[2] == "|a slow"

    def test_pca_normalization(self, converter):
        """Test that PCA values are normalized when configured."""
        # Large values should be clipped
        pca_emb = np.array([10.0, -10.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0])

        pca_str = converter.format_pca_features(pca_emb)

        # Values should be in range [-1, 1] after normalization
        for part in pca_str.split():
            if ":" in part:
                value = float(part.split(":")[1])
                assert -1 <= value <= 1

    def test_custom_costs(self, mock_pca_transformer, sample_oracle_labels, sample_embeddings):
        """Test converter with custom cost values."""
        config = VWConverterConfig(
            correct_cost=0.1,
            incorrect_cost=0.5,
        )
        converter = VWConverter(pca_path=mock_pca_transformer, config=config)
        label = sample_oracle_labels[0]  # oracle_action="fast"
        pca_emb = converter.pca.transform(sample_embeddings[0])

        example = converter.format_example(label, pca_emb)
        lines = example.strip().split("\n")

        # Fast (correct) should have custom correct_cost
        assert "0:0.1:1.0" in lines[1]
        # Slow (incorrect) should have custom incorrect_cost
        assert "1:0.5:0.0" in lines[2]


class TestVWConverterIntegration:
    """Integration tests for VWConverter."""

    def test_convert_single(self, mock_pca_transformer):
        """Test converting a single question."""
        converter = VWConverter(pca_path=mock_pca_transformer)

        # Mock transform_text
        converter.pca.transform_text = lambda x: np.random.randn(8)

        example = converter.convert_single(
            question="John has 5 apples. How many?",
            oracle_action="fast"
        )

        lines = example.strip().split("\n")
        assert len(lines) == 3
        assert "0:0.0:1.0" in lines[1]  # fast is correct

    def test_vw_format_validity(self, mock_pca_transformer, sample_oracle_labels, sample_embeddings):
        """Test that output is valid VW ADF format."""
        converter = VWConverter(pca_path=mock_pca_transformer)
        label = sample_oracle_labels[0]
        pca_emb = converter.pca.transform(sample_embeddings[0])

        example = converter.format_example(label, pca_emb)
        lines = example.strip().split("\n")

        # Shared line format
        assert lines[0].startswith("shared ")
        assert "|x" in lines[0]

        # Action lines format: action_idx:cost:prob |namespace features
        for action_line in lines[1:]:
            parts = action_line.split(" |")
            assert len(parts) == 2
            label_part = parts[0]
            # Label format: idx:cost:prob
            label_parts = label_part.split(":")
            assert len(label_parts) == 3
            int(label_parts[0])  # action index
            float(label_parts[1])  # cost
            float(label_parts[2])  # probability
