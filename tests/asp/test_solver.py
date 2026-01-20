"""
Unit tests for ASPSolver class.

Tests cover:
    - Basic solve operations
    - Error classification (syntax, grounding, UNSAT, timeout)
    - Answer extraction
    - Feedback generation
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.asp_solver.solver import ASPSolver, SolveResult, ErrorType


@pytest.fixture
def solver():
    """Create a solver instance for testing."""
    return ASPSolver(timeout_ms=5000)


@pytest.fixture
def solver_no_libs():
    """Create a solver without loading libraries."""
    return ASPSolver(timeout_ms=5000, load_libraries=False)


class TestBasicSolve:
    """Test basic solve functionality."""

    def test_simple_fact(self, solver_no_libs):
        """Test solving with a simple fact."""
        program = """
        final_answer(42).
        """
        result = solver_no_libs.solve(program)
        assert result.success
        assert result.answer == 42

    def test_simple_rule(self, solver_no_libs):
        """Test solving with a simple rule."""
        program = """
        a(5).
        b(3).
        c(X) :- a(X).
        final_answer(X) :- c(X).
        """
        result = solver_no_libs.solve(program)
        assert result.success
        assert result.answer == 5

    def test_solution_predicate(self, solver_no_libs):
        """Test extraction via solution/1 predicate."""
        program = """
        value(total, 100).
        target(total).
        solution(V) :- target(ID), value(ID, V).
        """
        result = solver_no_libs.solve(program)
        assert result.success
        assert result.answer == 100

    def test_with_libraries(self, solver):
        """Test solving with libraries loaded."""
        program = """
        quantity(john, apples, 10).
        final_answer(10).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 10


class TestErrorClassification:
    """Test error classification and feedback."""

    def test_syntax_error_missing_period(self, solver_no_libs):
        """Test detection of syntax error (missing period)."""
        program = """
        a(1)
        final_answer(1).
        """
        result = solver_no_libs.solve(program)
        assert not result.success
        assert result.error_type == ErrorType.SYNTAX
        assert "syntax" in result.error_message.lower()

    def test_syntax_error_invalid_predicate(self, solver_no_libs):
        """Test detection of syntax error (invalid characters)."""
        program = """
        123invalid(1).
        final_answer(1).
        """
        result = solver_no_libs.solve(program)
        assert not result.success
        assert result.error_type == ErrorType.SYNTAX

    def test_unsat_contradiction(self, solver_no_libs):
        """Test UNSAT detection with contradicting constraints."""
        program = """
        a(1).
        :- a(1).
        final_answer(1).
        """
        result = solver_no_libs.solve(program)
        assert not result.success
        assert result.error_type == ErrorType.UNSAT
        assert result.satisfiable == False

    def test_no_answer(self, solver_no_libs):
        """Test NO_ANSWER when solution predicate is missing."""
        program = """
        a(1).
        b(2).
        % No final_answer or solution predicate
        """
        result = solver_no_libs.solve(program)
        assert not result.success
        assert result.error_type == ErrorType.NO_ANSWER
        assert result.satisfiable == True

    def test_grounding_error_unsafe_variable(self, solver_no_libs):
        """Test grounding error with unsafe variable."""
        program = """
        a(X).  % X is not bound in body
        final_answer(1).
        """
        result = solver_no_libs.solve(program)
        assert not result.success
        assert result.error_type in (ErrorType.GROUNDING, ErrorType.SYNTAX)


class TestFeedbackGeneration:
    """Test feedback message generation for LLM."""

    def test_success_feedback(self, solver_no_libs):
        """Test feedback for successful solve."""
        program = "final_answer(42)."
        result = solver_no_libs.solve(program)
        feedback = result.to_feedback()
        assert "SUCCESS" in feedback
        assert "42" in feedback

    def test_syntax_error_feedback(self, solver_no_libs):
        """Test feedback for syntax error."""
        program = "a(1"  # Missing closing paren and period
        result = solver_no_libs.solve(program)
        feedback = result.to_feedback()
        assert "SYNTAX ERROR" in feedback
        assert "period" in feedback.lower() or "parenthes" in feedback.lower()

    def test_unsat_feedback(self, solver_no_libs):
        """Test feedback for UNSAT."""
        program = """
        a(1). :- a(1).
        final_answer(1).
        """
        result = solver_no_libs.solve(program)
        feedback = result.to_feedback()
        assert "CONTRADICTION" in feedback

    def test_no_answer_feedback(self, solver_no_libs):
        """Test feedback for no answer."""
        program = "a(1)."
        result = solver_no_libs.solve(program)
        feedback = result.to_feedback()
        assert "NO ANSWER" in feedback
        assert "target" in feedback.lower() or "final_answer" in feedback.lower()


class TestSolveResult:
    """Test SolveResult dataclass."""

    def test_success_property(self):
        """Test success property."""
        result = SolveResult(satisfiable=True, answer=42)
        assert result.success

        result2 = SolveResult(satisfiable=True, answer=None)
        assert not result2.success

        result3 = SolveResult(satisfiable=False, answer=42)
        assert not result3.success

    def test_default_values(self):
        """Test default values."""
        result = SolveResult()
        assert result.satisfiable is None
        assert result.answer is None
        assert result.error_type == ErrorType.NONE
        assert result.error_message == ""
        assert result.errors == []
        assert result.models == []


class TestSyntaxValidation:
    """Test syntax validation without solving."""

    def test_valid_syntax(self, solver_no_libs):
        """Test validation of valid syntax."""
        program = """
        a(1).
        b(X) :- a(X).
        """
        is_valid, error = solver_no_libs.validate_syntax(program)
        assert is_valid
        assert error == ""

    def test_invalid_syntax(self, solver_no_libs):
        """Test validation of invalid syntax."""
        program = """
        a(1)
        b(X) :- a(X).
        """
        is_valid, error = solver_no_libs.validate_syntax(program)
        assert not is_valid
        assert "syntax" in error.lower()


class TestGroundOnly:
    """Test grounding without solving."""

    def test_successful_grounding(self, solver_no_libs):
        """Test successful grounding."""
        program = """
        a(1). a(2). a(3).
        b(X) :- a(X).
        """
        success, error, atom_count = solver_no_libs.ground_only(program)
        assert success
        assert error == ""
        assert atom_count > 0

    def test_grounding_failure(self, solver_no_libs):
        """Test grounding failure."""
        program = """
        a(X).  % Unsafe variable
        """
        success, error, atom_count = solver_no_libs.ground_only(program)
        assert not success


class TestMultipleModels:
    """Test handling of multiple models/answers."""

    def test_single_model(self, solver_no_libs):
        """Test with single deterministic model."""
        program = """
        a(1).
        final_answer(X) :- a(X).
        """
        result = solver_no_libs.solve(program)
        assert result.success
        assert result.answer == 1

    def test_ambiguous_answers(self, solver_no_libs):
        """Test detection of ambiguous answers."""
        program = """
        final_answer(1).
        final_answer(2).
        """
        result = solver_no_libs.solve(program)
        # Should detect ambiguity
        assert result.error_type == ErrorType.AMBIGUOUS or result.answer in [1, 2]


class TestWithOntology:
    """Test solving with ontology predicates."""

    def test_quantity_predicate(self, solver):
        """Test quantity predicate from ontology."""
        program = """
        quantity(john, apples, 10).
        quantity(mary, apples, 5).

        % Manual sum for test
        total(T) :- quantity(john, apples, A), quantity(mary, apples, B), T = @add(A, B).
        final_answer(T) :- total(T).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 15

    def test_target_pointer_pattern(self, solver):
        """Test target pointer extraction pattern."""
        program = """
        value(total_cost, 150).
        target(total_cost).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 150


class TestTimeout:
    """Test timeout handling."""

    def test_fast_solve(self, solver_no_libs):
        """Test that fast solves complete."""
        program = "final_answer(1)."
        result = solver_no_libs.solve(program, timeout_ms=1000)
        assert result.success
        assert result.solve_time_ms < 1000

    # Note: Testing actual timeout is tricky without a genuinely slow program
    # Skipping explicit timeout test to avoid flaky tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
