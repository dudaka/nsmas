"""
Visualization Module

Creates publication-quality plots for Phase 7 analysis.
Priority visualizations as specified in Phase 7 strategy document.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import plotting libraries (optional - graceful degradation)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import PercentFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .robustness import RobustnessComparison, RobustnessResult
from .complexity import ComplexityComparison, ComplexityResult
from .cost import CostComparison, CostResult
from .errors import ErrorComparison, ErrorBreakdown
from .self_correction import SelfCorrectionComparison, SelfCorrectionResult
from .bandit import BanditResult


class Visualizer:
    """Creates publication-quality visualizations for Phase 7 analysis."""

    # Color palette for consistent styling
    COLORS = {
        "ns_mas": "#2ecc71",  # Green for NS-MAS (positive)
        "baseline": "#e74c3c",  # Red for baseline (comparison)
        "gpt4o": "#3498db",  # Blue for GPT-4o
        "gpt4o_mini": "#9b59b6",  # Purple for GPT-4o-mini
        "sc": "#e67e22",  # Dark orange for Self-Consistency
        "fixed_slow": "#27ae60",  # Dark green
        "bandit": "#f39c12",  # Orange
        "random": "#95a5a6",  # Gray
    }

    # Style configuration
    STYLE = {
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    def __init__(self, output_dir: Optional[Path] = None, style: str = "whitegrid"):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            style: Seaborn style name
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_MATPLOTLIB:
            plt.rcParams.update(self.STYLE)
        if HAS_SEABORN:
            sns.set_style(style)

    def _check_deps(self):
        """Check that visualization dependencies are available."""
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

    def plot_robustness_delta(
        self,
        comparison: RobustnessComparison,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot 1: Robustness Delta - The 'Money Plot'.

        Grouped bar chart comparing Base vs NoOp accuracy across systems,
        with explicit delta annotations.

        Args:
            comparison: RobustnessComparison with system results
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        systems = list(comparison.systems.values())
        n_systems = len(systems)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar positions
        x = np.arange(n_systems)
        width = 0.35

        # Data
        base_acc = [s.base_accuracy * 100 for s in systems]
        noop_acc = [s.noop_accuracy * 100 for s in systems]
        names = [s.system_name for s in systems]

        # Bars
        bars1 = ax.bar(x - width / 2, base_acc, width, label="Base (Clean)",
                      color="#3498db", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, noop_acc, width, label="NoOp (Perturbed)",
                      color="#e74c3c", edgecolor="black", linewidth=0.5)

        # Add delta annotations
        for i, (b, n, sys) in enumerate(zip(base_acc, noop_acc, systems)):
            drop = sys.accuracy_drop * 100
            color = "green" if drop < 5 else "red"
            ax.annotate(
                f"Δ={drop:.1f}%",
                xy=(i, max(b, n) + 2),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        # Add RRR values below bars
        for i, sys in enumerate(systems):
            ax.annotate(
                f"RRR={sys.rrr:.2f}",
                xy=(i, min(base_acc[i], noop_acc[i]) - 8),
                ha="center",
                fontsize=8,
                style="italic",
            )

        # Reference lines
        ax.axhline(y=35, color="red", linestyle="--", alpha=0.5, label="SOTA RRR≈0.35 level")

        ax.set_xlabel("System")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Robustness Analysis: Base vs NoOp Accuracy\n(Lower Δ = More Robust)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 100)

        plt.tight_layout()

        # Save
        path = Path(save_path) if save_path else self.output_dir / "robustness_delta.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def plot_pareto_frontier(
        self,
        comparison: CostComparison,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot 2: Cost-Accuracy Pareto Frontier.

        Scatter plot showing cost-accuracy tradeoff with Pareto frontier.

        Args:
            comparison: CostComparison with system results
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points
        for sys_key, result in comparison.systems.items():
            color = self._get_system_color(sys_key)
            marker = "s" if "nsmas" in sys_key else "o"
            size = 200 if any(p.system_name == result.system_name and p.is_efficient
                            for p in comparison.pareto_frontier) else 100

            ax.scatter(
                result.cost_per_correct_usd * 1000,  # Convert to millicents
                result.accuracy * 100,
                s=size,
                c=color,
                marker=marker,
                label=result.system_name,
                edgecolors="black",
                linewidth=1.5 if size == 200 else 0.5,
                zorder=10 if size == 200 else 5,
            )

        # Draw Pareto frontier line
        frontier_points = sorted(
            [(p.cost * 1000, p.accuracy * 100) for p in comparison.pareto_frontier if p.is_efficient],
            key=lambda x: x[0]
        )
        if len(frontier_points) > 1:
            fx, fy = zip(*frontier_points)
            ax.plot(fx, fy, "g--", linewidth=2, alpha=0.7, label="Pareto Frontier")

        ax.set_xlabel("Cost per Correct Answer (millicents)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Cost-Accuracy Tradeoff\n(Upper-left = Better)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Add annotation for dominated region
        ax.annotate(
            "Dominated\nRegion",
            xy=(0.7, 0.3),
            xycoords="axes fraction",
            fontsize=10,
            alpha=0.5,
            ha="center",
        )

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "pareto_frontier.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def plot_complexity_decay(
        self,
        comparison: ComplexityComparison,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot 3: Complexity Decay Curves.

        Line plot showing accuracy decline from Base → P1 → P2.

        Args:
            comparison: ComplexityComparison with system results
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, ax = plt.subplots(figsize=(10, 6))

        tiers = ["Base", "P1", "P2"]
        x = [0, 1, 2]

        for sys_key, result in comparison.systems.items():
            color = self._get_system_color(sys_key)
            accuracies = [
                result.base_accuracy * 100,
                result.p1_accuracy * 100,
                result.p2_accuracy * 100,
            ]

            ax.plot(
                x, accuracies,
                marker="o",
                markersize=10,
                linewidth=2.5,
                label=f"{result.system_name} (slope={result.decay_slope:.3f})",
                color=color,
            )

            # Add error bars if CIs available
            if result.base_ci and result.p1_ci and result.p2_ci:
                yerr = [
                    [result.base_ci.ci_width * 50, result.p1_ci.ci_width * 50, result.p2_ci.ci_width * 50],
                    [result.base_ci.ci_width * 50, result.p1_ci.ci_width * 50, result.p2_ci.ci_width * 50],
                ]
                ax.errorbar(x, accuracies, yerr=yerr, fmt="none", color=color, alpha=0.3, capsize=3)

        ax.set_xlabel("Problem Complexity Tier")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Complexity Decay Analysis\n(Base → P1 → P2)")
        ax.set_xticks(x)
        ax.set_xticklabels(tiers)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "complexity_decay.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def plot_error_distribution(
        self,
        comparison: ErrorComparison,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot 4: Error Distribution Breakdown.

        Stacked bar chart showing error category distribution by system.

        Args:
            comparison: ErrorComparison with error breakdowns
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, ax = plt.subplots(figsize=(12, 6))

        systems = list(comparison.systems.keys())
        categories = comparison.all_categories if comparison.all_categories else []

        if not categories:
            comparison._compute_all_categories()
            categories = comparison.all_categories

        n_systems = len(systems)
        x = np.arange(n_systems)
        width = 0.6

        # Color map for categories
        cat_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

        # Build stacked bars
        bottom = np.zeros(n_systems)
        for i, cat in enumerate(categories):
            values = []
            for sys_key in systems:
                breakdown = comparison.systems[sys_key]
                count = breakdown.error_counts.get(cat, 0)
                pct = count / breakdown.total_problems * 100 if breakdown.total_problems > 0 else 0
                values.append(pct)

            ax.bar(x, values, width, bottom=bottom, label=cat, color=cat_colors[i])
            bottom += np.array(values)

        # Get display names
        display_names = [comparison.systems[s].system_name for s in systems]

        ax.set_xlabel("System")
        ax.set_ylabel("Error Rate (%)")
        ax.set_title("Error Distribution by Category")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=15, ha="right")
        ax.legend(title="Error Category", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "error_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def plot_self_correction_effectiveness(
        self,
        comparison: SelfCorrectionComparison,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot 5: Self-Correction Success Rates.

        Bar chart showing reflection loop effectiveness.

        Args:
            comparison: SelfCorrectionComparison with results
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        systems = list(comparison.systems.values())
        names = [s.system_name for s in systems]
        x = np.arange(len(systems))
        width = 0.35

        # Plot 1: Reflection usage rate
        usage_rates = [s.reflection_usage_rate * 100 for s in systems]
        ax1.bar(x, usage_rates, width, color="#3498db", edgecolor="black")
        ax1.set_xlabel("System")
        ax1.set_ylabel("Problems Requiring Reflection (%)")
        ax1.set_title("Reflection Loop Usage Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha="right")

        # Plot 2: Correction success rate
        success_rates = [s.correction_success_rate * 100 for s in systems]
        colors = ["#27ae60" if r > 50 else "#e74c3c" for r in success_rates]
        ax2.bar(x, success_rates, width, color=colors, edgecolor="black")
        ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
        ax2.set_xlabel("System")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Self-Correction Success Rate\n(Multi-iteration problems that succeeded)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=15, ha="right")
        ax2.legend()

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "self_correction.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def plot_accuracy_comparison(
        self,
        accuracies: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot accuracy comparison across all systems and variants.

        Args:
            accuracies: Dict of system_name -> {variant: accuracy}
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, ax = plt.subplots(figsize=(12, 6))

        systems = list(accuracies.keys())
        variants = ["base", "p1", "p2", "noop"]
        n_systems = len(systems)
        n_variants = len(variants)

        x = np.arange(n_systems)
        width = 0.2
        offsets = np.arange(n_variants) - (n_variants - 1) / 2

        colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

        for i, variant in enumerate(variants):
            values = [accuracies[sys].get(variant, 0) * 100 for sys in systems]
            ax.bar(x + offsets[i] * width, values, width, label=variant.upper(),
                  color=colors[i], edgecolor="black", linewidth=0.5)

        ax.set_xlabel("System")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy Comparison Across Systems and Variants")
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=15, ha="right")
        ax.legend(title="Variant")
        ax.set_ylim(0, 100)

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "accuracy_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def _get_system_color(self, sys_key: str) -> str:
        """Get color for a system based on its key."""
        if "gpt4o_mini" in sys_key or "gpt-4o-mini" in sys_key.lower():
            return self.COLORS["gpt4o_mini"]
        elif "sc" in sys_key or "self_consistency" in sys_key.lower():
            return self.COLORS["sc"]
        elif "gpt4o" in sys_key or "gpt-4o" in sys_key.lower():
            return self.COLORS["gpt4o"]
        elif "fixed_slow" in sys_key:
            return self.COLORS["fixed_slow"]
        elif "bandit" in sys_key:
            return self.COLORS["bandit"]
        elif "random" in sys_key:
            return self.COLORS["random"]
        elif "nsmas" in sys_key:
            return self.COLORS["ns_mas"]
        else:
            return "#7f8c8d"

    def plot_routing_negative_result(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot the negative result: bandit routing fails to outperform random.

        Shows:
        1. Bandit vs Random accuracy comparison
        2. Policy degenerates to constant "always slow"
        3. Features don't discriminate problem difficulty

        Args:
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Bandit vs Random vs Fixed Slow comparison
        systems = ["Fixed Slow\n(Always Verify)", "Random\n(50/50)", "Bandit\n(Cold Start)", "Bandit\n(Warm Start)"]
        accuracies = [70.89, 57.12, 56.76, 95.0]  # Phase 8a warm-start = 95% but all slow
        colors = [self.COLORS["fixed_slow"], self.COLORS["random"],
                  self.COLORS["bandit"], "#d35400"]  # Darker orange for warm

        bars = ax1.bar(systems, accuracies, color=colors, edgecolor="black", linewidth=1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.annotate(
                f"{acc:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # Add annotation about warm-start being "always slow"
        ax1.annotate(
            "* Uses 100%\nslow path",
            xy=(3, 80),
            ha="center",
            fontsize=8,
            style="italic",
            color="#7f8c8d",
        )

        # Highlight that random ≈ bandit (cold start)
        ax1.annotate(
            "",
            xy=(1.5, 58),
            xytext=(2.5, 58),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2),
        )
        ax1.annotate(
            "Δ = 0.36%\n(not significant)",
            xy=(2, 60),
            ha="center",
            fontsize=9,
            color="red",
        )

        ax1.set_ylabel("Overall Accuracy (%)")
        ax1.set_title("Routing Strategy Comparison\n(Bandit fails to beat random)")
        ax1.set_ylim(0, 100)
        ax1.axhline(y=70.89, color=self.COLORS["fixed_slow"], linestyle="--", alpha=0.5)

        # Right plot: Path selection distribution
        strategies = ["Fixed Slow", "Random", "Cold Bandit", "Warm Bandit*"]
        fast_pcts = [0, 50, 48.5, 0]  # Warm bandit goes 100% slow
        slow_pcts = [100, 50, 51.5, 100]

        x = np.arange(len(strategies))
        width = 0.35

        bars1 = ax2.bar(x - width/2, fast_pcts, width, label="Fast Path (Zero-shot)",
                        color="#3498db", edgecolor="black")
        bars2 = ax2.bar(x + width/2, slow_pcts, width, label="Slow Path (GVR)",
                        color="#27ae60", edgecolor="black")

        ax2.set_ylabel("Path Selection (%)")
        ax2.set_title("Path Selection Distribution\n(Learned policy degenerates to constant)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.legend(loc="upper right")
        ax2.set_ylim(0, 110)

        # Add annotation about warm-start failure
        ax2.annotate(
            "Warm-start learns\n'always slow'\n(no feature signal)",
            xy=(3, 85),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#ffecb3", edgecolor="#f39c12"),
        )

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "routing_negative_result.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def plot_main_results_comparison(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot main results comparison including all baselines and NS-MAS variants.

        Highlights:
        1. NS-MAS Fixed Slow dominates all baselines
        2. SC performs worse than zero-temp CoT (counter-intuitive)
        3. Verification > sampling for mathematical reasoning

        Args:
            save_path: Optional custom save path
            show: Whether to display the plot

        Returns:
            Path to saved figure
        """
        self._check_deps()

        fig, ax = plt.subplots(figsize=(14, 7))

        # Systems in order of accuracy
        systems = [
            "GPT-4o-mini\nCoT",
            "GPT-4o\nCoT+SC (k=5)",
            "GPT-4o\nCoT",
            "NS-MAS\nRandom",
            "NS-MAS\nBandit",
            "NS-MAS\nFixed Slow",
        ]

        # Overall accuracies from paper
        overall = [42.86, 38.44, 55.63, 57.12, 56.76, 70.89]

        # Colors
        colors = [
            self.COLORS["gpt4o_mini"],
            self.COLORS["sc"],
            self.COLORS["gpt4o"],
            self.COLORS["random"],
            self.COLORS["bandit"],
            self.COLORS["fixed_slow"],
        ]

        bars = ax.barh(systems, overall, color=colors, edgecolor="black", linewidth=1)

        # Add value labels
        for bar, acc in zip(bars, overall):
            ax.annotate(
                f"{acc:.1f}%",
                xy=(acc + 1, bar.get_y() + bar.get_height() / 2),
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        # Highlight the SC anomaly
        ax.annotate(
            "SC < zero-temp CoT\n(17.19% worse)",
            xy=(38.44, 1),
            xytext=(20, 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            bbox=dict(boxstyle="round", facecolor="#ffcccc", edgecolor="red"),
        )

        # Highlight NS-MAS improvement
        ax.annotate(
            "+15.26% over\nGPT-4o CoT",
            xy=(70.89, 5),
            xytext=(80, 4.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
            bbox=dict(boxstyle="round", facecolor="#ccffcc", edgecolor="green"),
        )

        # Vertical line at GPT-4o CoT baseline
        ax.axvline(x=55.63, color=self.COLORS["gpt4o"], linestyle="--", alpha=0.5,
                   label="GPT-4o CoT baseline")

        ax.set_xlabel("Overall Accuracy (%)")
        ax.set_title("Main Results: Verification Beats Sampling\n(NS-MAS Fixed Slow dominates all baselines)")
        ax.set_xlim(0, 95)
        ax.legend(loc="lower right")

        plt.tight_layout()

        path = Path(save_path) if save_path else self.output_dir / "main_results_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return path

    def generate_all_plots(
        self,
        robustness: Optional[RobustnessComparison] = None,
        complexity: Optional[ComplexityComparison] = None,
        cost: Optional[CostComparison] = None,
        errors: Optional[ErrorComparison] = None,
        self_correction: Optional[SelfCorrectionComparison] = None,
        show: bool = False,
    ) -> Dict[str, Path]:
        """Generate all priority plots.

        Args:
            robustness: RobustnessComparison data
            complexity: ComplexityComparison data
            cost: CostComparison data
            errors: ErrorComparison data
            self_correction: SelfCorrectionComparison data
            show: Whether to display plots

        Returns:
            Dict of plot name -> file path
        """
        paths = {}

        if robustness:
            paths["robustness_delta"] = self.plot_robustness_delta(robustness, show=show)

        if cost:
            paths["pareto_frontier"] = self.plot_pareto_frontier(cost, show=show)

        if complexity:
            paths["complexity_decay"] = self.plot_complexity_decay(complexity, show=show)

        if errors:
            paths["error_distribution"] = self.plot_error_distribution(errors, show=show)

        if self_correction:
            paths["self_correction"] = self.plot_self_correction_effectiveness(
                self_correction, show=show
            )

        return paths
