"""
Unit tests for lib_math.lp - Arithmetic operations and @calc hooks.

Tests cover:
    - Basic arithmetic: add, sub, mul, div, mod
    - Comparison operations: gt, lt, gte, lte, eq, neq
    - Advanced operations: abs, neg, max, min, pow
    - Percentage and ratio calculations
    - Unit conversion
    - Decimal/fixed-point arithmetic
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.asp_solver.solver import ASPSolver, ErrorType


@pytest.fixture
def solver():
    """Create a solver instance for testing."""
    return ASPSolver(timeout_ms=5000)


class TestBasicArithmetic:
    """Test basic arithmetic operations via @calc hooks."""

    def test_addition(self, solver):
        """Test @add hook."""
        program = """
        val(a, 10). val(b, 25).
        result(R) :- val(a, A), val(b, B), R = @add(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 35

    def test_subtraction(self, solver):
        """Test @sub hook."""
        program = """
        val(a, 100). val(b, 37).
        result(R) :- val(a, A), val(b, B), R = @sub(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 63

    def test_subtraction_negative_result(self, solver):
        """Test @sub with negative result."""
        program = """
        val(a, 10). val(b, 25).
        result(R) :- val(a, A), val(b, B), R = @sub(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == -15

    def test_multiplication(self, solver):
        """Test @mul hook."""
        program = """
        val(a, 12). val(b, 8).
        result(R) :- val(a, A), val(b, B), R = @mul(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 96

    def test_multiplication_large_numbers(self, solver):
        """Test @mul with large numbers (would cause grounding explosion without hooks)."""
        program = """
        val(a, 50000). val(b, 200).
        result(R) :- val(a, A), val(b, B), R = @mul(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 10000000

    def test_division(self, solver):
        """Test @div hook (integer division)."""
        program = """
        val(a, 100). val(b, 4).
        result(R) :- val(a, A), val(b, B), R = @div(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 25

    def test_division_floor(self, solver):
        """Test @div with non-exact division (floor behavior)."""
        program = """
        val(a, 10). val(b, 3).
        result(R) :- val(a, A), val(b, B), R = @div(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 3  # Floor division

    def test_modulo(self, solver):
        """Test @mod hook."""
        program = """
        val(a, 17). val(b, 5).
        result(R) :- val(a, A), val(b, B), R = @mod(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success, f"Failed: {result.to_feedback()}"
        assert result.answer == 2


class TestComparisonOperations:
    """Test comparison operations."""

    def test_greater_than_true(self, solver):
        """Test @gt returns 1 when true."""
        program = """
        val(a, 10). val(b, 5).
        result(R) :- val(a, A), val(b, B), R = @gt(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1

    def test_greater_than_false(self, solver):
        """Test @gt returns 0 when false."""
        program = """
        val(a, 5). val(b, 10).
        result(R) :- val(a, A), val(b, B), R = @gt(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 0

    def test_less_than(self, solver):
        """Test @lt hook."""
        program = """
        val(a, 3). val(b, 7).
        result(R) :- val(a, A), val(b, B), R = @lt(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1

    def test_equal(self, solver):
        """Test @eq hook."""
        program = """
        val(a, 42). val(b, 42).
        result(R) :- val(a, A), val(b, B), R = @eq(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1

    def test_not_equal(self, solver):
        """Test @neq hook."""
        program = """
        val(a, 10). val(b, 20).
        result(R) :- val(a, A), val(b, B), R = @neq(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1


class TestAdvancedOperations:
    """Test advanced arithmetic operations."""

    def test_absolute_value_positive(self, solver):
        """Test @abs_val with positive number."""
        program = """
        val(a, 25).
        result(R) :- val(a, A), R = @abs_val(A).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 25

    def test_absolute_value_negative(self, solver):
        """Test @abs_val with negative number."""
        program = """
        val(a, -25).
        result(R) :- val(a, A), R = @abs_val(A).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 25

    def test_negation(self, solver):
        """Test @neg hook."""
        program = """
        val(a, 15).
        result(R) :- val(a, A), R = @neg(A).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == -15

    def test_maximum(self, solver):
        """Test @max_val hook."""
        program = """
        val(a, 30). val(b, 45).
        result(R) :- val(a, A), val(b, B), R = @max_val(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 45

    def test_minimum(self, solver):
        """Test @min_val hook."""
        program = """
        val(a, 30). val(b, 45).
        result(R) :- val(a, A), val(b, B), R = @min_val(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 30

    def test_power(self, solver):
        """Test @pow_val hook."""
        program = """
        val(base, 2). val(exp, 10).
        result(R) :- val(base, B), val(exp, E), R = @pow_val(B, E).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1024


class TestPercentageOperations:
    """Test percentage and ratio operations."""

    def test_percent_of(self, solver):
        """Test @percent_of: 25% of 200 = 50."""
        program = """
        val(percent, 25). val(whole, 200).
        result(R) :- val(percent, P), val(whole, W), R = @percent_of(P, W).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 50

    def test_as_percent(self, solver):
        """Test @as_percent: 30 is what % of 120? = 25%."""
        program = """
        val(part, 30). val(whole, 120).
        result(R) :- val(part, P), val(whole, W), R = @as_percent(P, W).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 25

    def test_ratio_scale(self, solver):
        """Test @ratio_scale: 100 scaled by 3/4 = 75."""
        program = """
        val(v, 100). val(num, 3). val(denom, 4).
        result(R) :- val(v, V), val(num, N), val(denom, D), R = @ratio_scale(V, N, D).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 75


class TestUnitConversion:
    """Test unit conversion helpers."""

    def test_convert_multiply(self, solver):
        """Test conversion by multiplication factor."""
        program = """
        val(feet, 5). val(factor, 12).
        inches(R) :- val(feet, F), val(factor, C), R = @convert(F, C).
        final_answer(R) :- inches(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 60  # 5 feet = 60 inches

    def test_convert_divide(self, solver):
        """Test conversion by division factor."""
        program = """
        val(minutes, 180). val(factor, 60).
        hours(R) :- val(minutes, M), val(factor, C), R = @convert_div(M, C).
        final_answer(R) :- hours(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 3  # 180 minutes = 3 hours


class TestDecimalArithmetic:
    """Test fixed-point decimal arithmetic."""

    def test_decimal_add(self, solver):
        """Test decimal addition: 12.50 + 7.25 = 19.75 (stored as 1250 + 725 = 1975)."""
        program = """
        val(a, 1250). val(b, 725).
        result(R) :- val(a, A), val(b, B), R = @decimal_add(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1975

    def test_decimal_sub(self, solver):
        """Test decimal subtraction: 20.00 - 5.50 = 14.50."""
        program = """
        val(a, 2000). val(b, 550).
        result(R) :- val(a, A), val(b, B), R = @decimal_sub(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1450

    def test_decimal_mul(self, solver):
        """Test decimal multiplication: 10.00 * 2.50 = 25.00."""
        program = """
        val(a, 1000). val(b, 250).
        result(R) :- val(a, A), val(b, B), R = @decimal_mul(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 2500

    def test_to_decimal(self, solver):
        """Test conversion to decimal: 12 and 50 -> 1250."""
        program = """
        val(whole, 12). val(frac, 50).
        result(R) :- val(whole, W), val(frac, F), R = @to_decimal(W, F).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 1250

    def test_from_decimal_whole(self, solver):
        """Test extracting whole part: 1275 -> 12."""
        program = """
        val(scaled, 1275).
        result(R) :- val(scaled, S), R = @from_decimal_whole(S).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 12

    def test_from_decimal_frac(self, solver):
        """Test extracting fractional part: 1275 -> 75."""
        program = """
        val(scaled, 1275).
        result(R) :- val(scaled, S), R = @from_decimal_frac(S).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 75


class TestChainedOperations:
    """Test chained arithmetic operations (multi-step calculations)."""

    def test_chained_add_mul(self, solver):
        """Test (5 + 3) * 4 = 32."""
        program = """
        val(a, 5). val(b, 3). val(c, 4).
        step1(R) :- val(a, A), val(b, B), R = @add(A, B).
        step2(R) :- step1(S1), val(c, C), R = @mul(S1, C).
        final_answer(R) :- step2(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 32

    def test_complex_calculation(self, solver):
        """Test complex multi-step: ((100 - 20) * 3) / 4 = 60."""
        program = """
        val(a, 100). val(b, 20). val(c, 3). val(d, 4).
        step1(R) :- val(a, A), val(b, B), R = @sub(A, B).
        step2(R) :- step1(S1), val(c, C), R = @mul(S1, C).
        step3(R) :- step2(S2), val(d, D), R = @div(S2, D).
        final_answer(R) :- step3(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 60


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_addition(self, solver):
        """Test adding zero."""
        program = """
        val(a, 42). val(b, 0).
        result(R) :- val(a, A), val(b, B), R = @add(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 42

    def test_multiply_by_zero(self, solver):
        """Test multiplying by zero."""
        program = """
        val(a, 1000000). val(b, 0).
        result(R) :- val(a, A), val(b, B), R = @mul(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 0

    def test_negative_numbers(self, solver):
        """Test operations with negative numbers."""
        program = """
        val(a, -10). val(b, -5).
        result(R) :- val(a, A), val(b, B), R = @mul(A, B).
        final_answer(R) :- result(R).
        """
        result = solver.solve(program)
        assert result.success
        assert result.answer == 50  # -10 * -5 = 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
