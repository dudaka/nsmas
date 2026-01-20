"""
Pytest fixtures for agent tests.
"""

import pytest
from unittest.mock import patch

from src.agent.config import AgentConfig, LLMConfig, LLMProvider
from src.agent.state import create_initial_state
from tests.agent.mocks.mock_llm import MockLLMProvider, create_mock_for_simple_problem


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return create_mock_for_simple_problem()


@pytest.fixture
def agent_config():
    """Create a default agent configuration."""
    return AgentConfig(
        generator_llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            temperature=0.0,
        ),
        max_retries=3,
        enable_entity_extraction=True,
        enable_cycle_detection=True,
        solver_timeout_ms=5000,
        trace_mode="OFF",
    )


@pytest.fixture
def simple_question():
    """A simple addition problem for testing."""
    return "John has 10 apples. Mary has 5 apples. How many apples do they have in total?"


@pytest.fixture
def simple_state(simple_question):
    """Create initial state for simple problem."""
    return create_initial_state(
        question=simple_question,
        expected_answer=15,
        max_retries=3,
    )


@pytest.fixture
def mock_openai(mock_llm):
    """Patch OpenAI provider to use mock LLM."""
    with patch("src.agent.llm_provider.OpenAIProvider") as MockProvider:
        instance = MockProvider.return_value
        instance.generate = mock_llm.generate
        yield mock_llm


@pytest.fixture
def mock_anthropic(mock_llm):
    """Patch Anthropic provider to use mock LLM."""
    with patch("src.agent.llm_provider.AnthropicProvider") as MockProvider:
        instance = MockProvider.return_value
        instance.generate = mock_llm.generate
        yield mock_llm
