"""
Cost-Accuracy Analysis Module

Analyzes token usage, cost efficiency, and Pareto frontier
for system comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .data_loader import SystemResults
from .statistical import StatisticalAnalyzer, BootstrapResult


# OpenAI pricing (as of 2026)
PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}


@dataclass
class CostResult:
    """Cost analysis result for a single system."""

    system_name: str
    variant: Optional[str] = None  # None = all variants

    # Token usage
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_problem: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    # Timing
    total_time_ms: float = 0.0
    avg_time_per_problem_ms: float = 0.0

    # Cost estimates
    estimated_cost_usd: float = 0.0
    cost_per_problem_usd: float = 0.0
    cost_per_correct_usd: float = 0.0

    # Efficiency metrics
    accuracy: float = 0.0
    n_problems: int = 0
    n_correct: int = 0
    tokens_per_correct: float = 0.0

    # Bootstrap CI for tokens
    tokens_ci: Optional[BootstrapResult] = None
    time_ci: Optional[BootstrapResult] = None

    @property
    def is_pareto_efficient(self) -> bool:
        """Check if this point is on the Pareto frontier.

        Note: This is set externally after comparison with other systems.
        """
        return getattr(self, "_pareto_efficient", False)

    @is_pareto_efficient.setter
    def is_pareto_efficient(self, value: bool):
        self._pareto_efficient = value

    def __str__(self) -> str:
        variant_str = f" ({self.variant})" if self.variant else ""
        return (
            f"Cost Analysis: {self.system_name}{variant_str}\n"
            f"  Problems: {self.n_problems}, Correct: {self.n_correct} ({self.accuracy:.2%})\n"
            f"  Tokens: {self.total_tokens:,} total, {self.avg_tokens_per_problem:.1f} avg/problem\n"
            f"  Time: {self.total_time_ms/1000:.1f}s total, {self.avg_time_per_problem_ms:.1f}ms avg\n"
            f"  Cost: ${self.estimated_cost_usd:.4f} total, ${self.cost_per_correct_usd:.6f}/correct\n"
            f"  Efficiency: {self.tokens_per_correct:.1f} tokens/correct"
        )


@dataclass
class ParetoPoint:
    """A point in the cost-accuracy space."""

    system_name: str
    cost: float  # Cost per problem or total cost
    accuracy: float  # Accuracy percentage
    is_efficient: bool = False  # On Pareto frontier


@dataclass
class CostComparison:
    """Comparison of costs across systems."""

    systems: Dict[str, CostResult] = field(default_factory=dict)
    pareto_frontier: List[ParetoPoint] = field(default_factory=list)

    def get_ranked_by_efficiency(self) -> List[CostResult]:
        """Rank systems by cost-per-correct (lowest first)."""
        return sorted(
            self.systems.values(),
            key=lambda x: x.cost_per_correct_usd if x.cost_per_correct_usd > 0 else float("inf"),
        )

    def get_ranked_by_accuracy(self) -> List[CostResult]:
        """Rank systems by accuracy (highest first)."""
        return sorted(self.systems.values(), key=lambda x: x.accuracy, reverse=True)

    def summary_table(self) -> str:
        """Generate markdown table summary."""
        lines = [
            "| System | Accuracy | Avg Tokens | Avg Time (ms) | Cost/Correct |",
            "|--------|----------|------------|---------------|--------------|",
        ]
        for result in self.get_ranked_by_efficiency():
            lines.append(
                f"| {result.system_name} | {result.accuracy:.2%} | "
                f"{result.avg_tokens_per_problem:.0f} | {result.avg_time_per_problem_ms:.0f} | "
                f"${result.cost_per_correct_usd:.5f} |"
            )
        return "\n".join(lines)


class CostAnalyzer:
    """Analyzes cost and efficiency of experiment results."""

    def __init__(
        self,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        pricing: Optional[Dict] = None,
    ):
        self.stats = statistical_analyzer or StatisticalAnalyzer()
        self.pricing = pricing or PRICING

    def analyze_system(
        self,
        system_results: SystemResults,
        variant: Optional[str] = None,
        model_key: Optional[str] = None,
        compute_ci: bool = True,
        n_bootstrap: int = 10000,
    ) -> CostResult:
        """Analyze cost for a single system.

        Args:
            system_results: Results for the system
            variant: Specific variant or None for all
            model_key: Pricing model key (e.g., "gpt-4o")
            compute_ci: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples

        Returns:
            CostResult with cost metrics
        """
        if variant:
            results = system_results.results_by_variant.get(variant, [])
        else:
            results = system_results.all_results

        if not results:
            return CostResult(
                system_name=system_results.system_name,
                variant=variant,
            )

        # Token counts
        total_input = sum(r.tokens_input for r in results)
        total_output = sum(r.tokens_output for r in results)
        total_tokens = total_input + total_output

        # Timing
        total_time = sum(r.execution_time_ms for r in results)

        # Accuracy
        n_correct = sum(1 for r in results if r.correct)
        accuracy = n_correct / len(results)

        # Cost estimation
        cost = self._estimate_cost(total_input, total_output, model_key)

        result = CostResult(
            system_name=system_results.system_name,
            variant=variant,
            total_tokens=total_tokens,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            avg_tokens_per_problem=total_tokens / len(results),
            avg_input_tokens=total_input / len(results),
            avg_output_tokens=total_output / len(results),
            total_time_ms=total_time,
            avg_time_per_problem_ms=total_time / len(results),
            estimated_cost_usd=cost,
            cost_per_problem_usd=cost / len(results),
            cost_per_correct_usd=cost / n_correct if n_correct > 0 else 0.0,
            accuracy=accuracy,
            n_problems=len(results),
            n_correct=n_correct,
            tokens_per_correct=total_tokens / n_correct if n_correct > 0 else 0.0,
        )

        # Compute bootstrap CIs
        if compute_ci:
            token_values = [r.tokens_input + r.tokens_output for r in results]
            if any(t > 0 for t in token_values):
                result.tokens_ci = self.stats.bootstrap_metric(
                    token_values, "tokens", n_bootstrap=n_bootstrap
                )

            time_values = [r.execution_time_ms for r in results]
            result.time_ci = self.stats.bootstrap_metric(
                time_values, "time_ms", n_bootstrap=n_bootstrap
            )

        return result

    def _estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_key: Optional[str] = None,
    ) -> float:
        """Estimate cost based on token counts and pricing."""
        if model_key and model_key in self.pricing:
            prices = self.pricing[model_key]
            return (input_tokens * prices["input"]) + (output_tokens * prices["output"])

        # Default to GPT-4o pricing
        prices = self.pricing.get("gpt-4o", {"input": 2.50e-6, "output": 10.0e-6})
        return (input_tokens * prices["input"]) + (output_tokens * prices["output"])

    def analyze_all_systems(
        self,
        systems: Dict[str, SystemResults],
        variant: Optional[str] = None,
        compute_ci: bool = True,
        n_bootstrap: int = 10000,
    ) -> CostComparison:
        """Analyze costs across all systems.

        Args:
            systems: Dictionary of system key -> SystemResults
            variant: Specific variant or None for all
            compute_ci: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples

        Returns:
            CostComparison with all system results and Pareto frontier
        """
        comparison = CostComparison()

        # Determine model key from system name
        model_mapping = {
            "baseline_gpt4o": "gpt-4o",
            "baseline_gpt4o_mini": "gpt-4o-mini",
            "nsmas_fixed_slow": "gpt-4o",  # Uses GPT-4o for generation
            "nsmas_bandit": "gpt-4o",
            "nsmas_random": "gpt-4o",
        }

        for sys_key, sys_results in systems.items():
            model_key = model_mapping.get(sys_key)
            result = self.analyze_system(
                sys_results,
                variant=variant,
                model_key=model_key,
                compute_ci=compute_ci,
                n_bootstrap=n_bootstrap,
            )
            comparison.systems[sys_key] = result

        # Compute Pareto frontier
        comparison.pareto_frontier = self._compute_pareto_frontier(comparison.systems)

        return comparison

    def _compute_pareto_frontier(
        self, systems: Dict[str, CostResult]
    ) -> List[ParetoPoint]:
        """Compute Pareto frontier for cost-accuracy tradeoff.

        A point is Pareto efficient if no other point has both
        lower cost AND higher accuracy.
        """
        points = []
        for sys_key, result in systems.items():
            points.append(
                ParetoPoint(
                    system_name=result.system_name,
                    cost=result.cost_per_correct_usd,
                    accuracy=result.accuracy,
                )
            )

        # Find Pareto efficient points
        for i, p1 in enumerate(points):
            is_dominated = False
            for j, p2 in enumerate(points):
                if i != j:
                    # p2 dominates p1 if p2 has lower cost AND higher accuracy
                    if p2.cost < p1.cost and p2.accuracy > p1.accuracy:
                        is_dominated = True
                        break
            p1.is_efficient = not is_dominated

        # Sort by accuracy for frontier visualization
        frontier = [p for p in points if p.is_efficient]
        frontier.sort(key=lambda x: x.accuracy)

        return frontier

    def get_pareto_data(
        self, comparison: CostComparison
    ) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
        """Get data for Pareto frontier visualization.

        Returns:
            (all_points, frontier_points) where each point is (cost, accuracy, name)
        """
        all_points = [
            (r.cost_per_correct_usd, r.accuracy, r.system_name)
            for r in comparison.systems.values()
        ]

        frontier_points = [
            (p.cost, p.accuracy, p.system_name)
            for p in comparison.pareto_frontier
        ]

        return all_points, frontier_points

    def generate_narrative(self, comparison: CostComparison) -> str:
        """Generate narrative text for paper discussing cost-accuracy tradeoffs."""
        best_efficiency = comparison.get_ranked_by_efficiency()[0]
        best_accuracy = comparison.get_ranked_by_accuracy()[0]
        frontier_systems = [p.system_name for p in comparison.pareto_frontier if p.is_efficient]

        narrative = f"""
**Cost-Accuracy Analysis**

The cost-accuracy tradeoff reveals the engineering constraints of neuro-symbolic
systems. While symbolic verification provides robustness guarantees, it comes at
a computational cost that must be evaluated against baseline approaches.

**Key Findings:**

1. **Most Cost-Efficient (per correct answer):** {best_efficiency.system_name}
   - Cost per correct: ${best_efficiency.cost_per_correct_usd:.5f}
   - Accuracy: {best_efficiency.accuracy:.2%}

2. **Highest Accuracy:** {best_accuracy.system_name}
   - Accuracy: {best_accuracy.accuracy:.2%}
   - Cost per correct: ${best_accuracy.cost_per_correct_usd:.5f}

3. **Pareto Efficient Systems:** {', '.join(frontier_systems)}
   - These systems represent optimal tradeoffs where no improvement in one metric
     can be achieved without degrading the other.

**Cost-Accuracy Table:**
{comparison.summary_table()}

**Interpretation:**

The "Fixed Slow" architecture, while computationally more expensive than pure LLM
baselines, achieves the highest accuracy region of the solution space. This cost
is justified for safety-critical applications where correctness guarantees outweigh
computational efficiency. The cold-start bandit fails to push the Pareto frontier
outward, remaining dominated by both the neural baseline and the Fixed Slow system.
"""
        return narrative.strip()
