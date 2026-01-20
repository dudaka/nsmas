"""
Statistical Analysis Module

Implements Bootstrap Confidence Intervals, McNemar's Test, and Cohen's d
for evaluating experiment results.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import stats


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    std_error: float

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_upper - self.ci_lower

    def __str__(self) -> str:
        return (
            f"{self.metric_name}: {self.point_estimate:.4f} "
            f"({self.confidence_level:.0%} CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


@dataclass
class McNemarResult:
    """Result of McNemar's test for paired nominal data."""

    system1_name: str
    system2_name: str
    statistic: float
    p_value: float
    n_both_correct: int  # a: both correct
    n_sys1_only: int  # b: system 1 correct, system 2 wrong
    n_sys2_only: int  # c: system 1 wrong, system 2 correct
    n_both_wrong: int  # d: both wrong
    significant: bool  # at alpha=0.05

    @property
    def total_n(self) -> int:
        return self.n_both_correct + self.n_sys1_only + self.n_sys2_only + self.n_both_wrong

    def __str__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"McNemar's Test: {self.system1_name} vs {self.system2_name}\n"
            f"  χ² = {self.statistic:.4f}, p = {self.p_value:.6f} ({sig_str})\n"
            f"  Contingency: a={self.n_both_correct}, b={self.n_sys1_only}, "
            f"c={self.n_sys2_only}, d={self.n_both_wrong}"
        )


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""

    metric_name: str
    cohens_h: float  # Cohen's h for proportions
    interpretation: str  # "negligible", "small", "medium", "large"

    def __str__(self) -> str:
        return f"Cohen's h = {self.cohens_h:.4f} ({self.interpretation})"


class StatisticalAnalyzer:
    """Performs statistical analysis on experiment results."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def bootstrap_accuracy(
        self,
        correct_mask: List[bool],
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """Compute bootstrap confidence interval for accuracy.

        Args:
            correct_mask: Boolean list where True = correct prediction
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            BootstrapResult with point estimate and CI
        """
        data = np.array(correct_mask, dtype=float)
        n = len(data)
        point_estimate = np.mean(data)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample_idx = self.rng.integers(0, n, size=n)
            bootstrap_sample = data[sample_idx]
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)

        # Percentile method for CI
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        std_error = np.std(bootstrap_means)

        return BootstrapResult(
            metric_name="accuracy",
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            std_error=std_error,
        )

    def bootstrap_metric(
        self,
        values: List[float],
        metric_name: str,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """Compute bootstrap CI for any metric (e.g., tokens, time).

        Args:
            values: List of metric values
            metric_name: Name of the metric
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            BootstrapResult with point estimate and CI
        """
        data = np.array(values, dtype=float)
        n = len(data)
        point_estimate = np.mean(data)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample_idx = self.rng.integers(0, n, size=n)
            bootstrap_sample = data[sample_idx]
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        std_error = np.std(bootstrap_means)

        return BootstrapResult(
            metric_name=metric_name,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            std_error=std_error,
        )

    def mcnemar_test(
        self,
        sys1_correct: List[bool],
        sys2_correct: List[bool],
        sys1_name: str = "System 1",
        sys2_name: str = "System 2",
        alpha: float = 0.05,
    ) -> McNemarResult:
        """Perform McNemar's test for paired nominal data.

        Tests whether two systems have significantly different error rates
        on the same set of problems.

        Args:
            sys1_correct: Boolean mask for system 1 correct predictions
            sys2_correct: Boolean mask for system 2 correct predictions
            sys1_name: Display name for system 1
            sys2_name: Display name for system 2
            alpha: Significance level

        Returns:
            McNemarResult with test statistic and p-value
        """
        assert len(sys1_correct) == len(sys2_correct), "Must have same number of samples"

        # Build contingency table
        # a: both correct, b: sys1 correct only, c: sys2 correct only, d: both wrong
        a = b = c = d = 0
        for s1, s2 in zip(sys1_correct, sys2_correct):
            if s1 and s2:
                a += 1
            elif s1 and not s2:
                b += 1
            elif not s1 and s2:
                c += 1
            else:
                d += 1

        # McNemar's test statistic (with continuity correction)
        if b + c == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            # Chi-squared with continuity correction
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return McNemarResult(
            system1_name=sys1_name,
            system2_name=sys2_name,
            statistic=statistic,
            p_value=p_value,
            n_both_correct=a,
            n_sys1_only=b,
            n_sys2_only=c,
            n_both_wrong=d,
            significant=p_value < alpha,
        )

    def cohens_h(
        self,
        p1: float,
        p2: float,
        metric_name: str = "accuracy",
    ) -> EffectSizeResult:
        """Calculate Cohen's h for comparing two proportions.

        Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

        Interpretation (Cohen, 1988):
        - |h| < 0.2: negligible
        - 0.2 <= |h| < 0.5: small
        - 0.5 <= |h| < 0.8: medium
        - |h| >= 0.8: large

        Args:
            p1: First proportion (e.g., accuracy of system 1)
            p2: Second proportion (e.g., accuracy of system 2)
            metric_name: Name of the metric being compared

        Returns:
            EffectSizeResult with Cohen's h and interpretation
        """
        # Clamp proportions to valid range
        p1 = max(0.0, min(1.0, p1))
        p2 = max(0.0, min(1.0, p2))

        # Calculate Cohen's h
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        h = phi1 - phi2

        # Interpret effect size
        abs_h = abs(h)
        if abs_h < 0.2:
            interpretation = "negligible"
        elif abs_h < 0.5:
            interpretation = "small"
        elif abs_h < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return EffectSizeResult(
            metric_name=metric_name,
            cohens_h=h,
            interpretation=interpretation,
        )

    def bootstrap_difference(
        self,
        values1: List[float],
        values2: List[float],
        metric_name: str = "difference",
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """Compute bootstrap CI for the difference between two means.

        Args:
            values1: Values for group 1
            values2: Values for group 2
            metric_name: Name of the difference metric
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            BootstrapResult for the difference (values1 - values2)
        """
        data1 = np.array(values1, dtype=float)
        data2 = np.array(values2, dtype=float)
        n1, n2 = len(data1), len(data2)

        point_estimate = np.mean(data1) - np.mean(data2)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            idx1 = self.rng.integers(0, n1, size=n1)
            idx2 = self.rng.integers(0, n2, size=n2)
            diff = np.mean(data1[idx1]) - np.mean(data2[idx2])
            bootstrap_diffs.append(diff)

        bootstrap_diffs = np.array(bootstrap_diffs)
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        std_error = np.std(bootstrap_diffs)

        return BootstrapResult(
            metric_name=metric_name,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            std_error=std_error,
        )

    def ci_overlap_test(
        self, ci1: Tuple[float, float], ci2: Tuple[float, float]
    ) -> bool:
        """Check if two confidence intervals overlap.

        Returns True if CIs overlap (not significantly different).
        """
        return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
