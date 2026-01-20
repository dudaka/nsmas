"""
ASP Context - Python hooks for Clingo arithmetic operations.

This module defines the Context class that provides Python functions
callable from ASP programs via the @function_name syntax.

Usage:
    The Context instance is passed to clingo.Control.ground(), making
    all methods available as @method_name in ASP rules.
"""

from clingo import Number, String, Function


class MathContext:
    """
    Context class providing Python arithmetic hooks for ASP programs.

    All public methods are callable from ASP via @method_name syntax.
    Methods receive clingo.Symbol objects and must return clingo.Symbol.
    """

    # =========================================================================
    # Core Arithmetic Operations
    # =========================================================================

    def add(self, a, b):
        """Add two numbers: a + b"""
        return Number(a.number + b.number)

    def sub(self, a, b):
        """Subtract two numbers: a - b"""
        return Number(a.number - b.number)

    def mul(self, a, b):
        """Multiply two numbers: a * b"""
        return Number(a.number * b.number)

    def div(self, a, b):
        """Integer division: a // b (floor division)"""
        if b.number == 0:
            return Function("error_div_zero")
        return Number(a.number // b.number)

    def mod(self, a, b):
        """Modulo operation: a % b"""
        if b.number == 0:
            return Function("error_mod_zero")
        return Number(a.number % b.number)

    # =========================================================================
    # Comparison Operations (return 1 for true, 0 for false)
    # =========================================================================

    def gt(self, a, b):
        """Greater than: a > b"""
        return Number(1 if a.number > b.number else 0)

    def lt(self, a, b):
        """Less than: a < b"""
        return Number(1 if a.number < b.number else 0)

    def gte(self, a, b):
        """Greater than or equal: a >= b"""
        return Number(1 if a.number >= b.number else 0)

    def lte(self, a, b):
        """Less than or equal: a <= b"""
        return Number(1 if a.number <= b.number else 0)

    def eq(self, a, b):
        """Equal: a == b"""
        return Number(1 if a.number == b.number else 0)

    def neq(self, a, b):
        """Not equal: a != b"""
        return Number(1 if a.number != b.number else 0)

    # =========================================================================
    # Advanced Arithmetic Operations
    # =========================================================================

    def abs_val(self, a):
        """Absolute value: |a|"""
        return Number(abs(a.number))

    def neg(self, a):
        """Negation: -a"""
        return Number(-a.number)

    def max_val(self, a, b):
        """Maximum of two values"""
        return Number(max(a.number, b.number))

    def min_val(self, a, b):
        """Minimum of two values"""
        return Number(min(a.number, b.number))

    def pow_val(self, base, exp):
        """Power: base^exp (for non-negative integer exponents)"""
        if exp.number < 0:
            return Function("error_neg_exp")
        return Number(base.number ** exp.number)

    # =========================================================================
    # Decimal/Fixed-Point Arithmetic (scale factor of 100)
    # =========================================================================

    def decimal_add(self, a, b):
        """Add two fixed-point decimals (already scaled by 100)"""
        return Number(a.number + b.number)

    def decimal_sub(self, a, b):
        """Subtract two fixed-point decimals"""
        return Number(a.number - b.number)

    def decimal_mul(self, a, b):
        """Multiply two fixed-point decimals, adjust for scale"""
        return Number((a.number * b.number) // 100)

    def decimal_div(self, a, b):
        """Divide two fixed-point decimals"""
        if b.number == 0:
            return Function("error_div_zero")
        return Number((a.number * 100) // b.number)

    def to_decimal(self, whole, frac):
        """Convert whole.frac to fixed-point (e.g., 12, 50 -> 1250)"""
        return Number(whole.number * 100 + frac.number)

    def from_decimal_whole(self, scaled):
        """Extract whole part from fixed-point value"""
        return Number(scaled.number // 100)

    def from_decimal_frac(self, scaled):
        """Extract fractional part from fixed-point value"""
        return Number(abs(scaled.number) % 100)

    # =========================================================================
    # Percentage and Ratio Operations
    # =========================================================================

    def percent_of(self, percent, whole):
        """Calculate percent% of whole (returns integer result)"""
        return Number((percent.number * whole.number) // 100)

    def as_percent(self, part, whole):
        """Calculate what percent 'part' is of 'whole'"""
        if whole.number == 0:
            return Function("error_div_zero")
        return Number((part.number * 100) // whole.number)

    def ratio_scale(self, value, num, denom):
        """Scale value by ratio num/denom"""
        if denom.number == 0:
            return Function("error_div_zero")
        return Number((value.number * num.number) // denom.number)

    # =========================================================================
    # Unit Conversion Helper
    # =========================================================================

    def convert(self, value, factor):
        """Convert value by multiplication factor"""
        return Number(value.number * factor.number)

    def convert_div(self, value, factor):
        """Convert value by division factor"""
        if factor.number == 0:
            return Function("error_div_zero")
        return Number(value.number // factor.number)
