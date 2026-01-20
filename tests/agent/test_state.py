"""
Tests for agent state management.
"""

import pytest

from src.agent.state import AgentState, VerificationResult, create_initial_state


class TestVerificationResult:
    """Tests for VerificationResult TypedDict."""

    def test_create_success_result(self):
        """Test creating a successful verification result."""
        result = VerificationResult(
            status="SAT",
            answer=42,
            error_type=None,
            error_message="",
            solve_time_ms=100.5,
            models=[["final_answer(42)"]],
        )
        assert result["status"] == "SAT"
        assert result["answer"] == 42
        assert result["error_type"] is None

    def test_create_error_result(self):
        """Test creating an error verification result."""
        result = VerificationResult(
            status="ERROR",
            answer=None,
            error_type="SYNTAX",
            error_message="unexpected token",
            solve_time_ms=10.0,
            models=[],
        )
        assert result["status"] == "ERROR"
        assert result["answer"] is None
        assert result["error_type"] == "SYNTAX"


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_create_initial_state(self):
        """Test creating initial agent state."""
        state = create_initial_state(
            question="What is 2 + 2?",
            expected_answer=4,
            max_retries=5,
        )

        assert state["question"] == "What is 2 + 2?"
        assert state["expected_answer"] == 4
        assert state["max_retries"] == 5
        assert state["iteration_count"] == 0
        assert state["status"] == "running"
        assert state["asp_code"] == ""
        assert state["parsed_entities"] == []
        assert state["asp_code_history"] == []
        assert state["critique_history"] == []

    def test_initial_state_defaults(self):
        """Test initial state with default values."""
        state = create_initial_state(question="Test question")

        assert state["expected_answer"] is None
        assert state["max_retries"] == 5
        assert state["final_answer"] is None

    def test_verification_result_in_state(self):
        """Test that verification result is properly initialized."""
        state = create_initial_state(question="Test")

        result = state["verification_result"]
        assert result["status"] == ""
        assert result["answer"] is None
        assert result["error_type"] is None
        assert result["models"] == []
