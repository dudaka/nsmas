"""
Contextual Bandit Router for NS-MAS.

Phase 5: Adaptive routing between Fast (zero-shot) and Slow (GVR) paths
using Vowpal Wabbit contextual bandits with SquareCB exploration.

Key Components:
- BanditConfig: Configuration dataclass
- Featurizer: Hybrid feature extraction (regex + embeddings)
- BanditRouter: VW wrapper for predict/learn
- FastSolver: Zero-shot LLM solver for fast path
- BanditTrainer: Oracle construction and policy training

Usage:
    from src.bandit import BanditConfig, BanditRouter, FastSolver

    config = BanditConfig()
    router = BanditRouter(config)

    # Predict action for a question
    action, prob = router.predict("John has 5 apples...")

    if action == 0:  # Fast path
        solver = FastSolver(config)
        answer = solver.solve(question)
    else:  # Slow path (GVR agent)
        ...
"""

from .config import BanditConfig
from .featurizer import Featurizer
from .router import BanditRouter
from .fast_solver import FastSolver
from .reward import calculate_loss, LossResult
from .training import BanditTrainer

__all__ = [
    "BanditConfig",
    "Featurizer",
    "BanditRouter",
    "FastSolver",
    "calculate_loss",
    "LossResult",
    "BanditTrainer",
]
