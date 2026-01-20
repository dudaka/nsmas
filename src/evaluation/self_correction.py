"""
Self-Correction Analysis Module

Analyzes the effectiveness of the reflection loop (GVR) in NS-MAS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

from .data_loader import SystemResults
from .statistical import StatisticalAnalyzer, BootstrapResult


@dataclass
class SelfCorrectionResult:
    """Result of self-correction analysis for a single system."""

    system_name: str
    variant: Optional[str] = None

    # Iteration statistics
    total_problems: int = 0
    single_iteration: int = 0  # Solved on first try
    multi_iteration: int = 0  # Required reflection
    self_corrected: int = 0  # Marked as self_corrected

    # Iteration distribution
    iteration_distribution: Dict[int, int] = field(default_factory=dict)
    avg_iterations: float = 0.0
    max_iterations: int = 0

    # Self-correction effectiveness
    multi_iter_correct: int = 0  # Multi-iteration that ended correct
    multi_iter_incorrect: int = 0  # Multi-iteration that still failed
    correction_success_rate: float = 0.0  # multi_iter_correct / multi_iteration

    # Overall accuracy
    accuracy: float = 0.0
    first_try_accuracy: float = 0.0  # Accuracy on single-iteration problems

    # Bootstrap CI
    correction_rate_ci: Optional[BootstrapResult] = None

    @property
    def reflection_usage_rate(self) -> float:
        """Fraction of problems requiring >1 iteration."""
        if self.total_problems == 0:
            return 0.0
        return self.multi_iteration / self.total_problems

    def __str__(self) -> str:
        variant_str = f" ({self.variant})" if self.variant else ""
        lines = [
            f"Self-Correction Analysis: {self.system_name}{variant_str}",
            f"  Total Problems: {self.total_problems}",
            f"  Single Iteration: {self.single_iteration} ({self.single_iteration/max(1,self.total_problems):.1%})",
            f"  Multi-Iteration: {self.multi_iteration} ({self.reflection_usage_rate:.1%})",
            f"  Avg Iterations: {self.avg_iterations:.2f}",
            f"  Max Iterations: {self.max_iterations}",
            "",
            f"  Self-Correction Success Rate: {self.correction_success_rate:.1%}",
            f"    - Multi-iter correct: {self.multi_iter_correct}",
            f"    - Multi-iter incorrect: {self.multi_iter_incorrect}",
            "",
            f"  Overall Accuracy: {self.accuracy:.2%}",
            f"  First-Try Accuracy: {self.first_try_accuracy:.2%}",
        ]
        return "\n".join(lines)


@dataclass
class SelfCorrectionComparison:
    """Comparison of self-correction across systems/variants."""

    systems: Dict[str, SelfCorrectionResult] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Generate markdown summary table."""
        lines = [
            "| System | Reflection Rate | Success Rate | Avg Iters | Accuracy |",
            "|--------|-----------------|--------------|-----------|----------|",
        ]
        for result in sorted(
            self.systems.values(), key=lambda x: x.correction_success_rate, reverse=True
        ):
            lines.append(
                f"| {result.system_name} | {result.reflection_usage_rate:.1%} | "
                f"{result.correction_success_rate:.1%} | {result.avg_iterations:.2f} | "
                f"{result.accuracy:.2%} |"
            )
        return "\n".join(lines)


class SelfCorrectionAnalyzer:
    """Analyzes self-correction effectiveness in NS-MAS."""

    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        self.stats = statistical_analyzer or StatisticalAnalyzer()

    def analyze_system(
        self,
        system_results: SystemResults,
        variant: Optional[str] = None,
        compute_ci: bool = True,
        n_bootstrap: int = 10000,
    ) -> SelfCorrectionResult:
        """Analyze self-correction for a single system.

        Args:
            system_results: Results for the system
            variant: Specific variant or None for all
            compute_ci: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples

        Returns:
            SelfCorrectionResult with correction metrics
        """
        if variant:
            results = system_results.results_by_variant.get(variant, [])
        else:
            results = system_results.all_results

        if not results:
            return SelfCorrectionResult(
                system_name=system_results.system_name,
                variant=variant,
            )

        # Count iterations
        iteration_counts = Counter()
        single_iter_correct = 0
        single_iter_total = 0
        multi_iter_correct = 0
        multi_iter_total = 0

        for r in results:
            iters = r.iterations if r.iterations > 0 else 1
            iteration_counts[iters] += 1

            if iters == 1:
                single_iter_total += 1
                if r.correct:
                    single_iter_correct += 1
            else:
                multi_iter_total += 1
                if r.correct:
                    multi_iter_correct += 1

        # Calculate metrics
        total = len(results)
        n_correct = sum(1 for r in results if r.correct)
        self_corrected_count = sum(1 for r in results if r.self_corrected)

        iterations_list = [r.iterations if r.iterations > 0 else 1 for r in results]
        avg_iters = np.mean(iterations_list) if iterations_list else 0.0
        max_iters = max(iterations_list) if iterations_list else 0

        result = SelfCorrectionResult(
            system_name=system_results.system_name,
            variant=variant,
            total_problems=total,
            single_iteration=single_iter_total,
            multi_iteration=multi_iter_total,
            self_corrected=self_corrected_count,
            iteration_distribution=dict(iteration_counts),
            avg_iterations=avg_iters,
            max_iterations=max_iters,
            multi_iter_correct=multi_iter_correct,
            multi_iter_incorrect=multi_iter_total - multi_iter_correct,
            correction_success_rate=(
                multi_iter_correct / multi_iter_total if multi_iter_total > 0 else 0.0
            ),
            accuracy=n_correct / total if total > 0 else 0.0,
            first_try_accuracy=(
                single_iter_correct / single_iter_total if single_iter_total > 0 else 0.0
            ),
        )

        # Bootstrap CI for correction success rate
        if compute_ci and multi_iter_total > 0:
            multi_iter_results = [r for r in results if (r.iterations or 1) > 1]
            correction_mask = [r.correct for r in multi_iter_results]
            result.correction_rate_ci = self.stats.bootstrap_accuracy(
                correction_mask, n_bootstrap=n_bootstrap
            )

        return result

    def analyze_all_systems(
        self,
        systems: Dict[str, SystemResults],
        variant: Optional[str] = None,
        compute_ci: bool = True,
        n_bootstrap: int = 10000,
    ) -> SelfCorrectionComparison:
        """Analyze self-correction across all systems.

        Args:
            systems: Dictionary of system key -> SystemResults
            variant: Specific variant or None for all
            compute_ci: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples

        Returns:
            SelfCorrectionComparison with all results
        """
        comparison = SelfCorrectionComparison()

        for sys_key, sys_results in systems.items():
            result = self.analyze_system(
                sys_results,
                variant=variant,
                compute_ci=compute_ci,
                n_bootstrap=n_bootstrap,
            )
            comparison.systems[sys_key] = result

        return comparison

    def analyze_by_variant(
        self,
        system_results: SystemResults,
        compute_ci: bool = True,
    ) -> Dict[str, SelfCorrectionResult]:
        """Analyze self-correction for each variant separately.

        Args:
            system_results: Results for the system

        Returns:
            Dictionary of variant -> SelfCorrectionResult
        """
        results = {}
        for variant in system_results.variants:
            results[variant] = self.analyze_system(
                system_results, variant=variant, compute_ci=compute_ci
            )
        return results

    def generate_narrative(self, comparison: SelfCorrectionComparison) -> str:
        """Generate narrative text for paper discussing self-correction."""
        # Find system with best correction rate
        best = max(
            comparison.systems.values(),
            key=lambda x: x.correction_success_rate,
            default=None,
        )

        if not best:
            return "No self-correction data available."

        highest_usage = max(
            comparison.systems.values(),
            key=lambda x: x.reflection_usage_rate,
        )

        narrative = f"""
**Self-Correction (Reflection Loop) Analysis**

The Generate-Verify-Reflect (GVR) loop is a core mechanism of the NS-MAS architecture.
When the ASP solver returns UNSAT or an error, the system uses the error feedback to
prompt the LLM for a corrected formalization. This analysis evaluates the effectiveness
of this self-correction mechanism.

**Key Findings:**

1. **Reflection Usage Rate:** {highest_usage.reflection_usage_rate:.1%} of problems
   required more than one iteration (system: {highest_usage.system_name}).
   - This indicates that approximately {highest_usage.multi_iteration} problems
     encountered initial solver failures that triggered the reflection loop.

2. **Self-Correction Success Rate:** {best.correction_success_rate:.1%} of multi-iteration
   problems eventually succeeded (system: {best.system_name}).
   - Out of {best.multi_iteration} problems requiring reflection,
     {best.multi_iter_correct} were eventually solved correctly.

3. **Average Iterations:** {best.avg_iterations:.2f} iterations per problem on average.
   - The system typically converges within {best.max_iterations} iterations or
     terminates due to cycle detection.

**Self-Correction Effectiveness Table:**
{comparison.summary_table()}

**Interpretation:**

The self-correction mechanism demonstrates meaningful recovery capability. A success rate
of {best.correction_success_rate:.1%} on initially-failed problems indicates that the
reflection loop provides genuine value—the symbolic solver feedback enables the LLM to
identify and fix errors in its formalization.

However, the relatively high rate of multi-iteration failures ({best.multi_iter_incorrect}
problems) suggests limitations:
- **Cycle detection** prevents infinite loops but also terminates potentially solvable problems
- **Error feedback quality** may be insufficient for certain error classes
- **LLM capability limits** in understanding complex ASP semantics

**Recommendations for Future Work:**
- Implement semantic-aware cycle detection (not just syntactic hash)
- Enhance error feedback with worked examples
- Consider fine-tuning on ASP correction tasks
"""
        return narrative.strip()
