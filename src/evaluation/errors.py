"""
Error Categorization Module

Analyzes and categorizes failure modes across experiment results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Counter as TypingCounter
from collections import Counter
from enum import Enum

from .data_loader import SystemResults


class ErrorCategory(str, Enum):
    """High-level error categories for analysis."""

    # ASP Solver errors (from src.asp_solver.solver.ErrorType)
    SYNTAX = "SYNTAX"  # Invalid ASP syntax
    GROUNDING = "GROUNDING"  # Unsafe variables, undefined predicates
    TIMEOUT = "TIMEOUT"  # Solver exceeded time limit
    UNSAT = "UNSAT"  # Logical contradiction
    NO_ANSWER = "NO_ANSWER"  # SAT but no final_answer found
    AMBIGUOUS = "AMBIGUOUS"  # Multiple conflicting answers
    RUNTIME = "RUNTIME"  # Python hook error

    # Agent-level errors
    CYCLE_DETECTED = "cycle_detected"  # GVR loop detected repeated ASP code
    MAX_ITERATIONS = "max_iterations"  # Exceeded max retries

    # Result-level (no explicit error but wrong)
    INCORRECT_ANSWER = "INCORRECT_ANSWER"  # Correct format but wrong value

    # Baseline (no error tracking)
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorBreakdown:
    """Breakdown of errors for a single system/variant."""

    system_name: str
    variant: Optional[str] = None  # None = all variants

    # Counts by category
    error_counts: Dict[str, int] = field(default_factory=dict)
    total_errors: int = 0
    total_problems: int = 0
    n_correct: int = 0

    # Error rate metrics
    error_rate: float = 0.0  # Fraction of problems with errors
    accuracy: float = 0.0

    def __str__(self) -> str:
        variant_str = f" ({self.variant})" if self.variant else ""
        lines = [
            f"Error Breakdown: {self.system_name}{variant_str}",
            f"  Total Problems: {self.total_problems}",
            f"  Correct: {self.n_correct} ({self.accuracy:.2%})",
            f"  Errors: {self.total_errors} ({self.error_rate:.2%})",
            "  By Category:",
        ]

        for cat, count in sorted(
            self.error_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / self.total_problems * 100 if self.total_problems > 0 else 0
            lines.append(f"    {cat}: {count} ({pct:.1f}%)")

        return "\n".join(lines)

    def get_top_errors(self, n: int = 5) -> List[tuple]:
        """Get top N error categories by count."""
        return Counter(self.error_counts).most_common(n)

    @property
    def error_distribution(self) -> Dict[str, float]:
        """Get error distribution as percentages of total problems."""
        if self.total_problems == 0:
            return {}
        return {
            cat: count / self.total_problems
            for cat, count in self.error_counts.items()
        }


@dataclass
class ErrorComparison:
    """Comparison of error patterns across systems."""

    systems: Dict[str, ErrorBreakdown] = field(default_factory=dict)
    all_categories: List[str] = field(default_factory=list)

    def summary_table(self) -> str:
        """Generate markdown table of error distributions."""
        if not self.all_categories:
            self._compute_all_categories()

        # Header
        lines = ["| System | " + " | ".join(self.all_categories) + " |"]
        lines.append("|--------|" + "|".join(["-------"] * len(self.all_categories)) + "|")

        # Data rows
        for sys_name, breakdown in self.systems.items():
            row = [sys_name]
            for cat in self.all_categories:
                count = breakdown.error_counts.get(cat, 0)
                pct = count / breakdown.total_problems * 100 if breakdown.total_problems > 0 else 0
                row.append(f"{pct:.1f}%")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _compute_all_categories(self):
        """Compute union of all error categories."""
        all_cats = set()
        for breakdown in self.systems.values():
            all_cats.update(breakdown.error_counts.keys())
        self.all_categories = sorted(all_cats)


class ErrorAnalyzer:
    """Analyzes error patterns in experiment results."""

    def __init__(self):
        # Map raw error_type strings to categories
        self.error_mapping = {
            "SYNTAX": ErrorCategory.SYNTAX,
            "GROUNDING": ErrorCategory.GROUNDING,
            "TIMEOUT": ErrorCategory.TIMEOUT,
            "UNSAT": ErrorCategory.UNSAT,
            "NO_ANSWER": ErrorCategory.NO_ANSWER,
            "AMBIGUOUS": ErrorCategory.AMBIGUOUS,
            "RUNTIME": ErrorCategory.RUNTIME,
            "cycle_detected": ErrorCategory.CYCLE_DETECTED,
            "max_iterations": ErrorCategory.MAX_ITERATIONS,
        }

    def analyze_system(
        self,
        system_results: SystemResults,
        variant: Optional[str] = None,
    ) -> ErrorBreakdown:
        """Analyze error distribution for a single system.

        Args:
            system_results: Results for the system
            variant: Specific variant or None for all

        Returns:
            ErrorBreakdown with error counts by category
        """
        if variant:
            results = system_results.results_by_variant.get(variant, [])
        else:
            results = system_results.all_results

        if not results:
            return ErrorBreakdown(
                system_name=system_results.system_name,
                variant=variant,
            )

        error_counts: TypingCounter[str] = Counter()
        n_correct = 0

        for r in results:
            if r.correct:
                n_correct += 1
            else:
                # Categorize the error
                if r.error_type:
                    # Map to our category
                    category = self.error_mapping.get(
                        r.error_type, ErrorCategory.UNKNOWN
                    )
                    error_counts[category.value if hasattr(category, 'value') else str(category)] += 1
                else:
                    # No explicit error but incorrect -> wrong answer
                    error_counts[ErrorCategory.INCORRECT_ANSWER.value] += 1

        total_errors = sum(error_counts.values())

        return ErrorBreakdown(
            system_name=system_results.system_name,
            variant=variant,
            error_counts=dict(error_counts),
            total_errors=total_errors,
            total_problems=len(results),
            n_correct=n_correct,
            error_rate=total_errors / len(results) if results else 0.0,
            accuracy=n_correct / len(results) if results else 0.0,
        )

    def analyze_all_systems(
        self,
        systems: Dict[str, SystemResults],
        variant: Optional[str] = None,
    ) -> ErrorComparison:
        """Analyze error patterns across all systems.

        Args:
            systems: Dictionary of system key -> SystemResults
            variant: Specific variant or None for all

        Returns:
            ErrorComparison with all system breakdowns
        """
        comparison = ErrorComparison()

        for sys_key, sys_results in systems.items():
            breakdown = self.analyze_system(sys_results, variant=variant)
            comparison.systems[sys_key] = breakdown

        comparison._compute_all_categories()
        return comparison

    def analyze_by_variant(
        self,
        system_results: SystemResults,
    ) -> Dict[str, ErrorBreakdown]:
        """Analyze error patterns for each variant separately.

        Args:
            system_results: Results for the system

        Returns:
            Dictionary of variant -> ErrorBreakdown
        """
        breakdowns = {}
        for variant in system_results.variants:
            breakdowns[variant] = self.analyze_system(system_results, variant=variant)
        return breakdowns

    def correlation_analysis(
        self,
        breakdown1: ErrorBreakdown,
        breakdown2: ErrorBreakdown,
    ) -> Dict[str, tuple]:
        """Analyze correlation of error patterns between two systems.

        Returns dict of category -> (count1, count2, diff)
        """
        all_categories = set(breakdown1.error_counts.keys()) | set(
            breakdown2.error_counts.keys()
        )

        correlations = {}
        for cat in all_categories:
            c1 = breakdown1.error_counts.get(cat, 0)
            c2 = breakdown2.error_counts.get(cat, 0)
            correlations[cat] = (c1, c2, c1 - c2)

        return correlations

    def generate_narrative(self, comparison: ErrorComparison) -> str:
        """Generate narrative text for paper discussing error patterns."""
        # Find dominant error categories
        all_errors: TypingCounter[str] = Counter()
        for breakdown in comparison.systems.values():
            all_errors.update(breakdown.error_counts)

        top_errors = all_errors.most_common(3)

        narrative = f"""
**Error Analysis**

The distribution of failure modes provides critical insight into the architectural
strengths and weaknesses of each system. Understanding *why* systems fail is as
important as knowing *how often* they fail.

**Dominant Error Categories:**
"""
        for i, (category, count) in enumerate(top_errors, 1):
            narrative += f"\n{i}. **{category}**: {count} occurrences total"

        narrative += f"""

**Error Distribution by System:**
{comparison.summary_table()}

**Key Observations:**

1. **Cycle Detection**: The NS-MAS systems exhibit cycle_detected errors when the
   reflection loop fails to converge, indicating the LLM generates semantically
   equivalent but syntactically different ASP code. This reveals a limitation in
   the self-correction mechanism.

2. **INCORRECT_ANSWER**: Problems where the ASP program executes successfully but
   produces a wrong result suggest translation errors—the LLM correctly formalizes
   the problem structure but misinterprets a numerical relationship.

3. **UNSAT/NO_ANSWER**: These indicate logical modeling errors where the constraints
   are either contradictory or incomplete, preventing the solver from finding a
   valid model.

**Implications for Future Work:**

The error distribution suggests that improvements should focus on:
- Better cycle detection and breaking strategies
- Enhanced prompt engineering for arithmetic translation
- Additional domain rules to catch common logical errors early
"""
        return narrative.strip()
