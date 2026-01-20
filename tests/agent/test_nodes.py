"""
Tests for agent node implementations.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.config import AgentConfig, LLMConfig, LLMProvider
from src.agent.state import create_initial_state
from src.agent.nodes.verify_asp import create_verify_asp_node
from src.agent.nodes.generate_asp import extract_asp_code


class TestVerifyASPNode:
    """Tests for the verify_asp node."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AgentConfig(
            solver_timeout_ms=5000,
            load_asp_libraries=True,
            trace_mode="OFF",
        )

    def test_verify_valid_asp(self, config):
        """Test verification of valid ASP code."""
        node = create_verify_asp_node(config)

        state = create_initial_state(question="Test")
        state["asp_code"] = """
            quantity(john, apples, 10).
            quantity(mary, apples, 5).
            total(T) :- quantity(john, apples, A), quantity(mary, apples, B), T = @add(A, B).
            final_answer(T) :- total(T).
        """

        result = node(state)

        assert result["verification_result"]["status"] == "SAT"
        assert result["verification_result"]["answer"] == 15
        assert result["status"] == "success"
        assert result["final_answer"] == 15

    def test_verify_syntax_error(self, config):
        """Test verification of ASP code with syntax error."""
        node = create_verify_asp_node(config)

        state = create_initial_state(question="Test")
        state["asp_code"] = "invalid asp code without period"

        result = node(state)

        assert result["verification_result"]["status"] == "ERROR"
        assert result["verification_result"]["error_type"] == "SYNTAX"
        assert "SYNTAX" in result["solver_feedback"]

    def test_verify_empty_code(self, config):
        """Test verification of empty ASP code."""
        node = create_verify_asp_node(config)

        state = create_initial_state(question="Test")
        state["asp_code"] = ""

        result = node(state)

        assert result["verification_result"]["status"] == "ERROR"
        assert "Empty" in result["solver_feedback"] or "No ASP" in result["solver_feedback"]

    def test_verify_unsat(self, config):
        """Test verification of unsatisfiable ASP code."""
        node = create_verify_asp_node(config)

        state = create_initial_state(question="Test")
        state["asp_code"] = """
            value(x, 1).
            value(x, 2).
            :- value(x, A), value(x, B), A != B.
        """

        result = node(state)

        assert result["verification_result"]["status"] == "UNSAT"
        assert result["verification_result"]["error_type"] == "UNSAT"


class TestExtractASPCode:
    """Tests for ASP code extraction from LLM responses."""

    def test_extract_from_asp_block(self):
        """Test extraction from ```asp code block."""
        response = """Here is the code:
```asp
final_answer(42).
```
That should work!"""

        code = extract_asp_code(response)
        assert code == "final_answer(42)."

    def test_extract_from_plain_block(self):
        """Test extraction from ``` code block."""
        response = """```
final_answer(42).
```"""

        code = extract_asp_code(response)
        assert code == "final_answer(42)."

    def test_extract_from_prolog_block(self):
        """Test extraction from ```prolog code block."""
        response = """```prolog
final_answer(42).
```"""

        code = extract_asp_code(response)
        assert code == "final_answer(42)."

    def test_extract_multiline(self):
        """Test extraction of multiline ASP code."""
        response = """```asp
quantity(john, apples, 10).
quantity(mary, apples, 5).
final_answer(15).
```"""

        code = extract_asp_code(response)
        assert "quantity(john, apples, 10)." in code
        assert "quantity(mary, apples, 5)." in code
        assert "final_answer(15)." in code

    def test_extract_fallback(self):
        """Test fallback when no code block found."""
        response = "final_answer(42)."

        code = extract_asp_code(response)
        assert "final_answer(42)." in code
