"""
LangGraph state definitions for the NS-MAS agent.

Defines AgentState TypedDict used by all nodes in the StateGraph,
along with supporting types like VerificationResult.
"""

import operator
from typing import Annotated, Any, List, Optional, TypedDict


class VerificationResult(TypedDict):
    """
    Result from ASP solver verification.

    Maps directly from SolveResult in src/asp_solver/solver.py.
    """

    status: str  # "SAT", "UNSAT", "ERROR", "TIMEOUT"
    answer: Optional[int]  # Numeric answer if SAT
    error_type: Optional[str]  # ErrorType enum name (e.g., "SYNTAX", "GROUNDING")
    error_message: str  # Human-readable error description
    solve_time_ms: float  # Time spent solving
    models: List[List[str]]  # Raw atoms from model(s)


class AgentState(TypedDict, total=False):
    """
    LangGraph state schema for NS-MAS agent.

    All fields are designed for the Generator -> Verifier -> Reflector loop.
    Uses Annotated with operator.add for append-only fields.

    Required fields:
        question: The original math word problem

    Optional fields (populated during execution):
        expected_answer: Ground truth for evaluation
        parsed_entities: Extracted entities from entity extraction step
        entity_extraction_reasoning: Full CoT reasoning from extraction
        asp_code: Current ASP program
        asp_code_history: Previous ASP attempts (for cycle detection)
        verification_result: Full Clingo output
        solver_feedback: Human-readable feedback for LLM
        critique_history: Append-only list of reflection critiques
        iteration_count: Current loop iteration (0-indexed)
        final_answer: Numeric result when solved
        status: Agent status ("running", "success", "max_retries", "cycle_detected")
        max_retries: Override for max retries (from config by default)
    """

    # Input
    question: str
    expected_answer: Optional[int]

    # Entity Extraction (CoT step)
    parsed_entities: List[str]
    entity_extraction_reasoning: str

    # ASP Generation
    asp_code: str
    asp_code_history: List[str]

    # Verification
    verification_result: VerificationResult
    solver_feedback: str

    # Reflection (append-only via operator.add)
    critique_history: Annotated[List[str], operator.add]

    # Loop Control
    iteration_count: int
    final_answer: Optional[int]
    status: str
    max_retries: int


def create_initial_state(
    question: str,
    expected_answer: Optional[int] = None,
    max_retries: int = 5,
) -> AgentState:
    """
    Create an initial AgentState for a new problem.

    Args:
        question: The math word problem to solve
        expected_answer: Optional ground truth for evaluation
        max_retries: Maximum retry attempts

    Returns:
        Initialized AgentState ready for graph execution
    """
    return AgentState(
        question=question,
        expected_answer=expected_answer,
        parsed_entities=[],
        entity_extraction_reasoning="",
        asp_code="",
        asp_code_history=[],
        verification_result=VerificationResult(
            status="",
            answer=None,
            error_type=None,
            error_message="",
            solve_time_ms=0.0,
            models=[],
        ),
        solver_feedback="",
        critique_history=[],
        iteration_count=0,
        final_answer=None,
        status="running",
        max_retries=max_retries,
    )
