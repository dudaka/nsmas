"""
Tests for FastSolver.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.bandit.config import BanditConfig
from src.bandit.fast_solver import FastSolver, OpenAIClient, FAST_SOLVER_PROMPT


class TestFastSolver:
    """Test FastSolver class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def solver_with_mock(self, bandit_config, mock_client):
        """Create solver with mocked client."""
        solver = FastSolver(bandit_config)
        solver._client = mock_client
        return solver

    def test_extract_answer_standard_format(self, solver_with_mock):
        """Test answer extraction with standard ANSWER: format."""
        response = "Let me solve this step by step...\n\nANSWER: 42"

        answer = solver_with_mock._extract_answer(response)

        assert answer == 42

    def test_extract_answer_case_insensitive(self, solver_with_mock):
        """Test answer extraction is case insensitive."""
        response = "The calculation gives us...\n\nanswer: 15"

        answer = solver_with_mock._extract_answer(response)

        assert answer == 15

    def test_extract_answer_with_spaces(self, solver_with_mock):
        """Test answer extraction handles spaces."""
        response = "ANSWER:   123"

        answer = solver_with_mock._extract_answer(response)

        assert answer == 123

    def test_extract_answer_fallback_final_answer(self, solver_with_mock):
        """Test fallback pattern 'final answer is'."""
        response = "After calculation, the final answer is 99."

        answer = solver_with_mock._extract_answer(response)

        assert answer == 99

    def test_extract_answer_fallback_equals(self, solver_with_mock):
        """Test fallback pattern with equals sign."""
        response = "Total = 55"

        answer = solver_with_mock._extract_answer(response)

        assert answer == 55

    def test_extract_answer_fallback_last_number(self, solver_with_mock):
        """Test fallback to last number in text."""
        response = "We have 10 apples, minus 3, which gives us 7"

        answer = solver_with_mock._extract_answer(response)

        assert answer == 7

    def test_extract_answer_with_decimal(self, solver_with_mock):
        """Test answer extraction rounds decimals."""
        response = "ANSWER: 12.7"

        answer = solver_with_mock._extract_answer(response)

        assert answer == 13  # Rounded

    def test_extract_answer_negative(self, solver_with_mock):
        """Test extraction of negative numbers."""
        response = "ANSWER: -5"

        answer = solver_with_mock._extract_answer(response)

        assert answer == -5

    def test_extract_answer_no_number(self, solver_with_mock):
        """Test extraction returns None when no number found."""
        response = "I cannot solve this problem."

        answer = solver_with_mock._extract_answer(response)

        assert answer is None

    def test_solve_success(self, solver_with_mock, mock_client):
        """Test successful solve."""
        mock_client.generate.return_value = "Step 1: 5 + 3\nANSWER: 8"

        answer = solver_with_mock.solve("What is 5 + 3?")

        assert answer == 8
        mock_client.generate.assert_called_once()

    def test_solve_extraction_failure(self, solver_with_mock, mock_client):
        """Test solve when extraction fails."""
        mock_client.generate.return_value = "I don't know"

        answer = solver_with_mock.solve("Unsolvable question")

        assert answer is None

    def test_solve_client_error(self, solver_with_mock, mock_client):
        """Test solve handles client errors gracefully."""
        mock_client.generate.side_effect = Exception("API Error")

        answer = solver_with_mock.solve("Test question")

        assert answer is None

    def test_solve_with_metadata(self, solver_with_mock, mock_client):
        """Test solve_with_metadata returns full response."""
        mock_client.generate.return_value = "Calculation:\n5 + 5 = 10\nANSWER: 10"

        result = solver_with_mock.solve_with_metadata("What is 5 + 5?")

        assert result["answer"] == 10
        assert result["success"] is True
        assert "Calculation" in result["raw_response"]
        assert result["error"] is None

    def test_solve_with_metadata_failure(self, solver_with_mock, mock_client):
        """Test solve_with_metadata on failure."""
        mock_client.generate.side_effect = Exception("Network error")

        result = solver_with_mock.solve_with_metadata("Test")

        assert result["answer"] is None
        assert result["success"] is False
        assert "Network error" in result["error"]


class TestOpenAIClient:
    """Test OpenAI client wrapper."""

    def test_init(self):
        """Test client initialization."""
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=256,
        )

        assert client.api_key == "test-key"
        assert client.model == "gpt-4o-mini"
        assert client._client is None  # Lazy init

    def test_generate_calls_openai(self):
        """Test generate calls OpenAI API."""
        with patch("openai.OpenAI") as mock_openai:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "ANSWER: 42"

            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_instance

            client = OpenAIClient(api_key="test-key")
            result = client.generate("Test prompt")

            assert result == "ANSWER: 42"
            mock_instance.chat.completions.create.assert_called_once()


class TestFastSolverPrompt:
    """Test system prompt configuration."""

    def test_prompt_contains_format_instruction(self):
        """Test prompt includes answer format instruction."""
        assert "ANSWER:" in FAST_SOLVER_PROMPT
        assert "numeric answer" in FAST_SOLVER_PROMPT.lower()

    def test_prompt_contains_example(self):
        """Test prompt includes example."""
        assert "42" in FAST_SOLVER_PROMPT or "example" in FAST_SOLVER_PROMPT.lower()
