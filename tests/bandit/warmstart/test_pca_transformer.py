"""
Tests for PCATransformer.
"""

import numpy as np
import pytest

from src.bandit.warmstart.pca_transformer import PCATransformer, PCATransformerConfig


class TestPCATransformerConfig:
    """Test PCATransformerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PCATransformerConfig()

        assert config.n_components == 64
        assert config.whiten is True
        assert config.random_state == 42
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PCATransformerConfig(
            n_components=32,
            whiten=False,
            random_state=123,
        )

        assert config.n_components == 32
        assert config.whiten is False
        assert config.random_state == 123


class TestPCATransformer:
    """Test PCATransformer class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        transformer = PCATransformer()

        assert transformer.config.n_components == 64
        assert transformer.pca is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = PCATransformerConfig(n_components=32)
        transformer = PCATransformer(config=config)

        assert transformer.config.n_components == 32

    def test_fit(self, sample_embeddings):
        """Test fitting PCA on embeddings."""
        config = PCATransformerConfig(n_components=8)
        transformer = PCATransformer(config=config)

        result = transformer.fit(sample_embeddings)

        assert result is transformer  # Returns self for chaining
        assert transformer.pca is not None
        assert transformer.n_components == 8
        assert 0 < transformer.variance_retained <= 1

    def test_transform_single(self, sample_embeddings):
        """Test transforming a single embedding."""
        config = PCATransformerConfig(n_components=8)
        transformer = PCATransformer(config=config)
        transformer.fit(sample_embeddings)

        single_embedding = sample_embeddings[0]
        reduced = transformer.transform(single_embedding)

        assert reduced.shape == (8,)
        assert np.issubdtype(reduced.dtype, np.floating)  # Accept any float type

    def test_transform_batch(self, sample_embeddings):
        """Test transforming a batch of embeddings."""
        config = PCATransformerConfig(n_components=8)
        transformer = PCATransformer(config=config)
        transformer.fit(sample_embeddings)

        reduced = transformer.transform(sample_embeddings)

        assert reduced.shape == (20, 8)  # 20 samples with 8 components

    def test_transform_without_fit_raises(self):
        """Test that transform raises error if not fitted."""
        transformer = PCATransformer()
        dummy_embedding = np.random.randn(768)

        with pytest.raises(ValueError, match="PCA not fitted"):
            transformer.transform(dummy_embedding)

    def test_save_and_load(self, sample_embeddings, temp_dir):
        """Test saving and loading transformer."""
        config = PCATransformerConfig(n_components=8)
        transformer = PCATransformer(config=config)
        transformer.fit(sample_embeddings)

        # Save
        save_path = temp_dir / "pca_model.pkl"
        transformer.save(save_path)

        assert save_path.exists()

        # Load
        loaded = PCATransformer.load(save_path)

        assert loaded.config.n_components == 8
        assert loaded.pca is not None

        # Verify transform produces same results
        original_reduced = transformer.transform(sample_embeddings[0])
        loaded_reduced = loaded.transform(sample_embeddings[0])

        np.testing.assert_array_almost_equal(original_reduced, loaded_reduced)

    def test_n_components_property(self, sample_embeddings):
        """Test n_components property."""
        config = PCATransformerConfig(n_components=8)  # Must be <= n_samples
        transformer = PCATransformer(config=config)
        transformer.fit(sample_embeddings)

        assert transformer.n_components == 8

    def test_variance_retained_property(self, sample_embeddings):
        """Test variance_retained property."""
        config = PCATransformerConfig(n_components=4)
        transformer = PCATransformer(config=config)
        transformer.fit(sample_embeddings)

        variance = transformer.variance_retained

        assert 0 < variance <= 1
        # With random 768-dim data and few samples, variance may be low
        # Just verify it's a positive fraction

    def test_variance_retained_before_fit(self):
        """Test variance_retained returns 0 before fit."""
        transformer = PCATransformer()

        assert transformer.variance_retained == 0.0

    def test_whiten_effect(self, sample_embeddings):
        """Test that whitening produces unit variance components."""
        config_white = PCATransformerConfig(n_components=4, whiten=True)
        config_no_white = PCATransformerConfig(n_components=4, whiten=False)

        transformer_white = PCATransformer(config=config_white)
        transformer_no_white = PCATransformer(config=config_no_white)

        transformer_white.fit(sample_embeddings)
        transformer_no_white.fit(sample_embeddings)

        reduced_white = transformer_white.transform(sample_embeddings)
        reduced_no_white = transformer_no_white.transform(sample_embeddings)

        # Whitened components should have approximately unit variance
        white_vars = np.var(reduced_white, axis=0)
        no_white_vars = np.var(reduced_no_white, axis=0)

        # Whitened variances should be close to 1
        np.testing.assert_array_almost_equal(white_vars, np.ones(4), decimal=1)
        # Non-whitened variances should vary
        assert not np.allclose(no_white_vars, np.ones(4), atol=0.5)

    def test_deterministic_results(self, sample_embeddings):
        """Test that results are deterministic with fixed random state."""
        config = PCATransformerConfig(n_components=4, random_state=42)

        transformer1 = PCATransformer(config=config)
        transformer2 = PCATransformer(config=config)

        transformer1.fit(sample_embeddings)
        transformer2.fit(sample_embeddings)

        reduced1 = transformer1.transform(sample_embeddings)
        reduced2 = transformer2.transform(sample_embeddings)

        np.testing.assert_array_almost_equal(reduced1, reduced2)

    def test_create_output_directory(self, temp_dir):
        """Test that save creates parent directories."""
        config = PCATransformerConfig(n_components=4)
        transformer = PCATransformer(config=config)

        # Create dummy PCA
        from sklearn.decomposition import PCA
        transformer.pca = PCA(n_components=4)
        np.random.seed(42)
        transformer.pca.fit(np.random.randn(10, 768))

        # Save to nested path
        nested_path = temp_dir / "nested" / "path" / "pca.pkl"
        transformer.save(nested_path)

        assert nested_path.exists()
