"""
ASP Solver Module for Neuro-Symbolic Mathematical Reasoning.

This module provides the symbolic reasoning engine for the NS-MAS framework,
using Answer Set Programming (ASP) via Clingo to verify and solve mathematical
word problems.

Components:
    - solver: ASPSolver class for executing and validating ASP programs
    - ontology.lp: L1 (Core Physics) and L2 (Semantic Actions) predicates
    - lib_math.lp: Arithmetic wrappers with Python @calc hooks
    - lib_time.lp: Simplified state-transition rules
"""

from .solver import ASPSolver, SolveResult, ErrorType
from .context import MathContext

__all__ = ["ASPSolver", "SolveResult", "ErrorType", "MathContext"]
