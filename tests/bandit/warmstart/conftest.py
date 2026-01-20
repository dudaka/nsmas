"""
Test fixtures for warmstart module tests.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.bandit.warmstart.oracle_builder import OracleLabel


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        "John has 5 apples. Mary has 3 apples. How many apples do they have in total?",
        "A store has 24 shirts. If 1/3 are sold, how many shirts remain?",
        "Tom earns $15 per hour. He works 8 hours. How much does he earn?",
        "Sarah has twice as many marbles as Tom. Tom has 7 marbles. How many does Sarah have?",
        "A factory produces 100 widgets per day. After a 20% increase, how many are produced?",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample 768-dim embeddings for testing."""
    np.random.seed(42)
    # Need at least n_components samples for PCA to fit
    return np.random.randn(20, 768).astype(np.float32)


@pytest.fixture
def sample_oracle_labels():
    """Sample oracle labels for testing."""
    return [
        OracleLabel(
            problem_id="gsm_001",
            question="John has 5 apples. Mary has 3. How many total?",
            ground_truth=8,
            variant="base",
            fast_answer=8,
            fast_correct=True,
            slow_answer=8,
            slow_correct=True,
            oracle_action="fast",
            oracle_reason="both_correct",
        ),
        OracleLabel(
            problem_id="gsm_002",
            question="A store has 24 shirts. 1/3 sold. How many remain?",
            ground_truth=16,
            variant="base",
            fast_answer=8,
            fast_correct=False,
            slow_answer=16,
            slow_correct=True,
            oracle_action="slow",
            oracle_reason="only_slow_correct",
        ),
        OracleLabel(
            problem_id="gsm_003",
            question="What is 10 + 5?",
            ground_truth=15,
            variant="base",
            fast_answer=15,
            fast_correct=True,
            slow_answer=10,
            slow_correct=False,
            oracle_action="fast",
            oracle_reason="only_fast_correct",
        ),
        OracleLabel(
            problem_id="gsm_004",
            question="Complex problem with many steps.",
            ground_truth=42,
            variant="base",
            fast_answer=10,
            fast_correct=False,
            slow_answer=20,
            slow_correct=False,
            oracle_action="fast",
            oracle_reason="both_wrong",
        ),
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_jsonl_data(temp_dir):
    """Create sample JSONL dataset file."""
    data = [
        {"problem_id": "gsm_001", "question": "John has 5 apples. How many?", "answer": 5},
        {"problem_id": "gsm_002", "question": "Mary has 10 oranges. How many?", "answer": 10},
        {"problem_id": "gsm_003", "question": "Tom has 3 bananas. How many?", "answer": 3},
    ]

    jsonl_path = temp_dir / "test_data.jsonl"
    with open(jsonl_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    return jsonl_path


@pytest.fixture
def sample_results_dirs(temp_dir):
    """Create sample experiment result directories."""
    fast_dir = temp_dir / "baseline_fast"
    slow_dir = temp_dir / "nsmas_slow"
    fast_dir.mkdir()
    slow_dir.mkdir()

    # Create fast results (baseline)
    fast_results = [
        {"problem_id": "gsm_001", "question": "Q1", "ground_truth": 8, "final_answer": 8, "correct": True},
        {"problem_id": "gsm_002", "question": "Q2", "ground_truth": 16, "final_answer": 8, "correct": False},
        {"problem_id": "gsm_003", "question": "Q3", "ground_truth": 15, "final_answer": 15, "correct": True},
        {"problem_id": "gsm_004", "question": "Q4", "ground_truth": 42, "final_answer": 10, "correct": False},
    ]

    # Create slow results (NS-MAS)
    slow_results = [
        {"problem_id": "gsm_001", "question": "Q1", "ground_truth": 8, "final_answer": 8, "correct": True},
        {"problem_id": "gsm_002", "question": "Q2", "ground_truth": 16, "final_answer": 16, "correct": True},
        {"problem_id": "gsm_003", "question": "Q3", "ground_truth": 15, "final_answer": 10, "correct": False},
        {"problem_id": "gsm_004", "question": "Q4", "ground_truth": 42, "final_answer": 20, "correct": False},
    ]

    with open(fast_dir / "base.jsonl", "w") as f:
        for r in fast_results:
            f.write(json.dumps(r) + "\n")

    with open(slow_dir / "base.jsonl", "w") as f:
        for r in slow_results:
            f.write(json.dumps(r) + "\n")

    return fast_dir, slow_dir


@pytest.fixture
def mock_pca_transformer(temp_dir, sample_embeddings):
    """Create a mock fitted PCA transformer."""
    from sklearn.decomposition import PCA
    from src.bandit.warmstart.pca_transformer import PCATransformer, PCATransformerConfig

    # Fit PCA on sample data
    config = PCATransformerConfig(n_components=8, whiten=True)
    transformer = PCATransformer(config=config)

    # Manual fit without loading embedding model
    transformer.pca = PCA(n_components=8, whiten=True, random_state=42)
    transformer.pca.fit(sample_embeddings)

    # Save and return path
    pca_path = temp_dir / "test_pca.pkl"
    transformer.save(pca_path)

    return pca_path
