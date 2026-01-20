"""
Test fixtures for bandit module tests.
"""

import pytest
import numpy as np

from src.bandit.config import BanditConfig
from src.agent.config import LLMProvider


@pytest.fixture
def bandit_config():
    """Default bandit configuration for testing."""
    return BanditConfig(
        exploration_algo="squarecb",
        gamma_scale=10.0,
        alpha_incorrect=10.0,
        beta_cost=1.0,
        fast_cost=0.1,
        slow_cost=1.0,
        model_dir=".test_bandit",
        model_filename="test_policy.vw",
        enable_online_learning=False,
    )


@pytest.fixture
def bandit_config_with_learning(bandit_config):
    """Bandit config with online learning enabled."""
    bandit_config.enable_online_learning = True
    return bandit_config


@pytest.fixture
def sample_questions():
    """Sample math questions for testing."""
    return [
        # Simple questions (likely fast path)
        "John has 5 apples. He eats 2. How many apples does he have left?",
        "Mary has 10 dollars. She spends 3 dollars. How much money does she have?",
        # Medium questions
        "A store has 24 shirts. If 1/3 are sold, how many shirts remain?",
        "Tom earns $15 per hour. He works 8 hours. How much does he earn?",
        # Complex questions (likely slow path)
        "A factory produces 500 widgets per day. If production increases by 20% each week "
        "for 3 weeks, and then decreases by 10%, how many widgets are produced on the final day?",
        "Sarah has twice as many marbles as Tom. Tom has 3 more marbles than Jane. "
        "If Jane has 7 marbles, how many marbles does Sarah have?",
    ]


@pytest.fixture
def sample_oracle_data():
    """Sample oracle-labeled data for testing."""
    return [
        {
            "question": "John has 5 apples. He eats 2. How many left?",
            "answer": 3,
            "fast_correct": True,
            "slow_correct": True,
            "oracle_action": 0,  # Fast (cheaper)
        },
        {
            "question": "A store has 24 shirts. 1/3 are sold. How many remain?",
            "answer": 16,
            "fast_correct": False,
            "slow_correct": True,
            "oracle_action": 1,  # Slow (fast fails)
        },
        {
            "question": "What is 10 + 5?",
            "answer": 15,
            "fast_correct": True,
            "slow_correct": True,
            "oracle_action": 0,  # Fast
        },
        {
            "question": "Complex multi-step problem that both paths fail on.",
            "answer": 42,
            "fast_correct": False,
            "slow_correct": False,
            "oracle_action": 0,  # Fast (minimize cost when both fail)
        },
    ]


@pytest.fixture
def mock_embedding():
    """Mock embedding vector for testing."""
    np.random.seed(42)
    return np.random.randn(384).astype(np.float32)
