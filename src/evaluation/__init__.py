"""
Phase 7: Evaluation and Analysis Module

Provides statistical analysis, visualization, and report generation
for NS-MAS experiment results targeting EXTRAAMAS 2026.
"""

from .data_loader import DataLoader, SystemResults
from .statistical import StatisticalAnalyzer, BootstrapResult, McNemarResult
from .robustness import RobustnessAnalyzer, RobustnessResult
from .complexity import ComplexityAnalyzer, ComplexityResult
from .cost import CostAnalyzer, CostResult
from .errors import ErrorAnalyzer, ErrorBreakdown
from .self_correction import SelfCorrectionAnalyzer, SelfCorrectionResult
from .bandit import BanditAnalyzer, BanditResult

__all__ = [
    # Data loading
    "DataLoader",
    "SystemResults",
    # Statistical
    "StatisticalAnalyzer",
    "BootstrapResult",
    "McNemarResult",
    # Robustness
    "RobustnessAnalyzer",
    "RobustnessResult",
    # Complexity
    "ComplexityAnalyzer",
    "ComplexityResult",
    # Cost
    "CostAnalyzer",
    "CostResult",
    # Errors
    "ErrorAnalyzer",
    "ErrorBreakdown",
    # Self-correction
    "SelfCorrectionAnalyzer",
    "SelfCorrectionResult",
    # Bandit
    "BanditAnalyzer",
    "BanditResult",
]
