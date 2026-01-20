"""
ASP Solver - Core solver interface for NS-MAS.

This module provides the ASPSolver class that wraps Clingo operations,
handles error classification, and extracts answers from ASP models.

Architecture:
    - Clorm for input validation (type safety at LLM boundary)
    - Raw Clingo API for solver core (flexibility, custom callbacks)
    - Custom logger for granular error classification
"""

import re
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import clingo
from clingo import Control, Symbol, MessageCode

from .context import MathContext


class ErrorType(Enum):
    """Classification of ASP solver errors for LLM feedback."""

    NONE = auto()           # No error - successful solve
    SYNTAX = auto()         # Invalid ASP syntax (parsing failed)
    GROUNDING = auto()      # Unsafe variables, undefined predicates
    TIMEOUT = auto()        # Solve exceeded time limit
    UNSAT = auto()          # Logical contradiction - no model exists
    NO_ANSWER = auto()      # SAT but no solution/final_answer found
    AMBIGUOUS = auto()      # Multiple conflicting answers found
    RUNTIME = auto()        # Python hook error or other runtime issue


@dataclass
class SolveResult:
    """Result of an ASP solve operation."""

    satisfiable: Optional[bool] = None  # True=SAT, False=UNSAT, None=Unknown
    answer: Optional[int] = None        # Extracted numeric answer
    error_type: ErrorType = ErrorType.NONE
    error_message: str = ""             # Human-readable error description
    errors: list[dict] = field(default_factory=list)  # Error predicates from model
    models: list[list[str]] = field(default_factory=list)  # All atoms in model(s)
    solve_time_ms: float = 0.0          # Time spent solving

    @property
    def success(self) -> bool:
        """True if solve succeeded with a valid answer."""
        return self.satisfiable is True and self.answer is not None

    def to_feedback(self) -> str:
        """Generate actionable feedback for LLM self-correction."""
        if self.success:
            return f"SUCCESS: The answer is {self.answer}."

        feedback_map = {
            ErrorType.SYNTAX: (
                f"SYNTAX ERROR: {self.error_message}\n"
                "Check for missing periods, mismatched parentheses, or invalid predicate names."
            ),
            ErrorType.GROUNDING: (
                f"GROUNDING ERROR: {self.error_message}\n"
                "Ensure all variables in rule heads appear in rule bodies. "
                "Check for undefined predicates."
            ),
            ErrorType.TIMEOUT: (
                "TIMEOUT: The reasoning took too long.\n"
                "You may have created a combinatorial explosion. "
                "Avoid large number ranges and ensure all arithmetic uses @calc hooks."
            ),
            ErrorType.UNSAT: (
                "CONTRADICTION: The constraints are logically inconsistent.\n"
                "No valid solution exists. Check if you set conflicting values "
                "for the same variable or violated a constraint."
            ),
            ErrorType.NO_ANSWER: (
                "NO ANSWER: The program ran successfully but no solution was derived.\n"
                "Ensure you have a target(ID) fact and corresponding value(ID, V) "
                "or final_answer(V) predicate."
            ),
            ErrorType.AMBIGUOUS: (
                f"AMBIGUOUS: Multiple answers found: {self.models}\n"
                "The logic produced multiple solutions. Add constraints to ensure "
                "a unique answer."
            ),
            ErrorType.RUNTIME: (
                f"RUNTIME ERROR: {self.error_message}\n"
                "A Python hook failed. Check arithmetic operations for division by zero "
                "or invalid inputs."
            ),
        }

        base_feedback = feedback_map.get(
            self.error_type,
            f"UNKNOWN ERROR: {self.error_message}"
        )

        # Add error predicates if present
        if self.errors:
            error_details = "\n".join(
                f"  - {e['type']}: {e.get('details', 'No details')}"
                for e in self.errors
            )
            base_feedback += f"\n\nDetected issues:\n{error_details}"

        return base_feedback


class ASPSolver:
    """
    Answer Set Programming solver for mathematical word problems.

    This class provides a high-level interface to the Clingo ASP solver,
    with automatic loading of the NS-MAS ontology and math libraries.

    Usage:
        solver = ASPSolver()
        result = solver.solve('''
            quantity(john, apples, 10).
            quantity(mary, apples, 5).
            combine(total, john, mary, apples).
            target(total).
        ''')
        print(result.answer)  # 15
    """

    # Path to ASP library files (relative to this module)
    LIB_DIR = Path(__file__).parent

    def __init__(
        self,
        timeout_ms: int = 30000,
        load_libraries: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize the ASP solver.

        Args:
            timeout_ms: Maximum time for solve operation (default 30 seconds)
            load_libraries: Whether to auto-load lib_math.lp, lib_time.lp, ontology.lp
            strict_validation: If True, use Clorm for input validation
        """
        self.timeout_ms = timeout_ms
        self.load_libraries = load_libraries
        self.strict_validation = strict_validation

        # Error collection during solve
        self._errors: list[str] = []
        self._warnings: list[str] = []

        # Math context for Python hooks
        self._context = MathContext()

        # Library file paths
        self._lib_math = self.LIB_DIR / "lib_math.lp"
        self._lib_time = self.LIB_DIR / "lib_time.lp"
        self._ontology = self.LIB_DIR / "ontology.lp"

    def _logger(self, code: MessageCode, message: str) -> None:
        """
        Custom logger callback for Clingo messages.

        Captures errors and warnings for later classification.
        """
        # Capture errors - RuntimeError, AtomUndefined, GlobalVariable, OperationUndefined
        error_codes = (
            MessageCode.RuntimeError,
            MessageCode.AtomUndefined,
            MessageCode.GlobalVariable,
            MessageCode.OperationUndefined,
            MessageCode.VariableUnbounded,
        )
        if code in error_codes:
            self._errors.append(message)
        elif code == MessageCode.Other:
            # Could be warning or info
            self._warnings.append(message)

    def _create_control(self) -> Control:
        """Create a new Clingo Control object with our logger."""
        return Control(
            arguments=["--warn=none"],  # Suppress default warnings
            logger=self._logger,
        )

    def _load_library_files(self, ctl: Control) -> None:
        """Load the standard library files into the control object."""
        if self._lib_math.exists():
            ctl.load(str(self._lib_math))
        if self._lib_time.exists():
            ctl.load(str(self._lib_time))
        if self._ontology.exists():
            ctl.load(str(self._ontology))

    def _classify_error(self, message: str) -> tuple[ErrorType, str]:
        """
        Classify an error message into an ErrorType.

        Returns:
            Tuple of (ErrorType, cleaned message for feedback)
        """
        message_lower = message.lower()

        # Syntax errors (parsing failures)
        syntax_indicators = ["syntax error", "unexpected", "parsing failed", "parse error"]
        if any(indicator in message_lower for indicator in syntax_indicators):
            # Extract line number if present
            line_match = re.search(r"<string>:(\d+)", message)
            if line_match:
                return ErrorType.SYNTAX, f"Syntax error on line {line_match.group(1)}: {message}"
            return ErrorType.SYNTAX, f"Syntax error: {message}"

        # Grounding errors (unsafe variables, grounding stopped)
        grounding_indicators = ["unsafe", "global variable", "grounding stopped"]
        if any(indicator in message_lower for indicator in grounding_indicators):
            return ErrorType.GROUNDING, f"Grounding error: {message}"

        # Undefined predicate
        if "undefined" in message_lower:
            return ErrorType.GROUNDING, f"Undefined predicate: {message}"

        # Runtime errors (Python hooks)
        if "error" in message_lower and ("python" in message_lower or "script" in message_lower):
            return ErrorType.RUNTIME, f"Python hook error: {message}"

        # Division by zero
        if "division" in message_lower or "zero" in message_lower:
            return ErrorType.RUNTIME, f"Division by zero: {message}"

        return ErrorType.RUNTIME, message

    def _extract_answer(self, model: clingo.Model) -> tuple[Optional[int], list[dict]]:
        """
        Extract the answer and any error predicates from a model.

        Returns:
            Tuple of (answer or None, list of error dicts)
        """
        answer = None
        answers = []
        errors = []

        for atom in model.symbols(shown=True):
            # Look for solution/1 or final_answer/1
            if atom.name == "solution" and len(atom.arguments) == 1:
                arg = atom.arguments[0]
                if arg.type == clingo.SymbolType.Number:
                    answers.append(arg.number)

            elif atom.name == "final_answer" and len(atom.arguments) == 1:
                arg = atom.arguments[0]
                if arg.type == clingo.SymbolType.Number:
                    answers.append(arg.number)

            # Collect error predicates: error(type, ...)
            elif atom.name == "error":
                error_info = {"type": str(atom.arguments[0]) if atom.arguments else "unknown"}
                if len(atom.arguments) > 1:
                    error_info["details"] = ", ".join(str(a) for a in atom.arguments[1:])
                errors.append(error_info)

        # Handle multiple/no answers
        if len(answers) == 1:
            answer = answers[0]
        elif len(answers) > 1:
            # Multiple answers - keep them all for ambiguity detection
            answer = answers[0]  # Return first, but flag as ambiguous

        return answer, errors

    def solve(
        self,
        program: str,
        additional_facts: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> SolveResult:
        """
        Solve an ASP program and extract the answer.

        Args:
            program: The ASP program (facts and rules) to solve
            additional_facts: Optional additional facts to add
            timeout_ms: Override default timeout for this solve

        Returns:
            SolveResult with answer, errors, and feedback
        """
        import time

        # Reset error collection
        self._errors = []
        self._warnings = []

        result = SolveResult()
        timeout = timeout_ms or self.timeout_ms

        start_time = time.time()

        try:
            # Create control object
            ctl = self._create_control()

            # Load library files
            if self.load_libraries:
                self._load_library_files(ctl)

            # Add the user program
            try:
                ctl.add("base", [], program)
                if additional_facts:
                    ctl.add("base", [], additional_facts)
            except RuntimeError as e:
                error_type, error_msg = self._classify_error(str(e))
                result.error_type = error_type
                result.error_message = error_msg
                return result

            # Ground the program with MathContext for Python hooks
            try:
                ctl.ground([("base", [])], context=self._context)
            except RuntimeError as e:
                error_type, error_msg = self._classify_error(str(e))
                result.error_type = error_type
                result.error_message = error_msg
                return result

            # Check for grounding errors collected by logger
            if self._errors:
                error_type, error_msg = self._classify_error(self._errors[0])
                result.error_type = error_type
                result.error_message = error_msg
                return result

            # Solve with timeout
            answers = []
            all_errors = []
            models = []

            solve_future = None
            timed_out = False

            def on_model(model: clingo.Model) -> bool:
                """Callback for each model found."""
                answer, errors = self._extract_answer(model)
                if answer is not None:
                    answers.append(answer)
                all_errors.extend(errors)
                models.append([str(atom) for atom in model.symbols(shown=True)])
                return True  # Continue searching for more models

            # Run solve with timeout using threading
            solve_result = [None]
            solve_error = [None]

            def solve_thread():
                try:
                    solve_result[0] = ctl.solve(on_model=on_model)
                except Exception as e:
                    solve_error[0] = e

            thread = threading.Thread(target=solve_thread)
            thread.start()
            thread.join(timeout=timeout / 1000.0)

            if thread.is_alive():
                # Timeout - interrupt the solve
                ctl.interrupt()
                thread.join(timeout=1.0)
                timed_out = True

            result.solve_time_ms = (time.time() - start_time) * 1000

            if timed_out:
                result.error_type = ErrorType.TIMEOUT
                result.error_message = f"Solve timed out after {timeout}ms"
                return result

            if solve_error[0]:
                error_type, error_msg = self._classify_error(str(solve_error[0]))
                result.error_type = error_type
                result.error_message = error_msg
                return result

            # Process solve result
            clingo_result = solve_result[0]
            result.models = models
            result.errors = all_errors

            if clingo_result is None:
                result.error_type = ErrorType.RUNTIME
                result.error_message = "Solve returned no result"
                return result

            result.satisfiable = clingo_result.satisfiable

            if not clingo_result.satisfiable:
                result.error_type = ErrorType.UNSAT
                result.error_message = "No satisfying model exists (contradiction)"
                return result

            # Check answers
            if not answers:
                result.error_type = ErrorType.NO_ANSWER
                result.error_message = "No solution/final_answer predicate found in model"
                return result

            # Check for ambiguity (multiple different answers)
            unique_answers = list(set(answers))
            if len(unique_answers) > 1:
                result.error_type = ErrorType.AMBIGUOUS
                result.error_message = f"Multiple answers found: {unique_answers}"
                result.answer = unique_answers[0]  # Return first
                return result

            # Success!
            result.answer = unique_answers[0]
            result.error_type = ErrorType.NONE

            # Check for error predicates (soft constraint violations)
            if all_errors:
                # We have an answer but also errors - report them
                result.error_message = "Answer found but with warnings"

            return result

        except Exception as e:
            result.error_type = ErrorType.RUNTIME
            result.error_message = f"Unexpected error: {str(e)}"
            return result

    def validate_syntax(self, program: str) -> tuple[bool, str]:
        """
        Check if an ASP program has valid syntax without solving.

        Returns:
            Tuple of (is_valid, error_message)
        """
        self._errors = []

        try:
            ctl = self._create_control()
            ctl.add("base", [], program)
            return True, ""
        except RuntimeError as e:
            _, error_msg = self._classify_error(str(e))
            return False, error_msg

    def ground_only(self, program: str) -> tuple[bool, str, int]:
        """
        Ground a program without solving, to check for grounding errors.

        Returns:
            Tuple of (success, error_message, atom_count)
        """
        self._errors = []

        try:
            ctl = self._create_control()

            if self.load_libraries:
                self._load_library_files(ctl)

            ctl.add("base", [], program)
            ctl.ground([("base", [])], context=self._context)

            if self._errors:
                _, error_msg = self._classify_error(self._errors[0])
                return False, error_msg, 0

            # Count atoms
            atom_count = sum(1 for _ in ctl.symbolic_atoms)

            return True, "", atom_count

        except RuntimeError as e:
            _, error_msg = self._classify_error(str(e))
            return False, error_msg, 0


# =============================================================================
# Clorm Schema Definitions (for strict validation)
# =============================================================================

try:
    from clorm import Predicate, IntegerField, StringField, ConstantField

    class QuantityPredicate(Predicate):
        """quantity(Entity, Item, Value)"""
        entity = ConstantField()
        item = ConstantField()
        value = IntegerField()

        class Meta:
            name = "quantity"

    class TransferPredicate(Predicate):
        """transfer(From, To, Item, Amount)"""
        from_entity = ConstantField()
        to_entity = ConstantField()
        item = ConstantField()
        amount = IntegerField()

        class Meta:
            name = "transfer"

    class TargetPredicate(Predicate):
        """target(QueryID)"""
        query_id = ConstantField()

        class Meta:
            name = "target"

    class FinalAnswerPredicate(Predicate):
        """final_answer(Value)"""
        value = IntegerField()

        class Meta:
            name = "final_answer"

    CLORM_AVAILABLE = True

except ImportError:
    CLORM_AVAILABLE = False
