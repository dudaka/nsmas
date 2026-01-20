"""
ASP Verification node for the NS-MAS agent.

Wraps the ASPSolver from Phase 3 to verify generated ASP code.
"""

import logging
from typing import Callable

from ..config import AgentConfig
from ..state import AgentState, VerificationResult
from src.asp_solver import ASPSolver, ErrorType

logger = logging.getLogger(__name__)


def create_verify_asp_node(config: AgentConfig) -> Callable[[AgentState], dict]:
    """
    Create the ASP verification node.

    Args:
        config: Agent configuration with solver settings

    Returns:
        Node function that verifies ASP code and updates state
    """
    # Create solver instance (reused across invocations)
    solver = ASPSolver(
        timeout_ms=config.solver_timeout_ms,
        load_libraries=config.load_asp_libraries,
    )

    def verify_asp(state: AgentState) -> dict:
        """
        Verify the generated ASP code using Clingo.

        Uses the ASPSolver from Phase 3 and converts the result
        to agent state updates.

        Args:
            state: Current agent state with asp_code

        Returns:
            State updates with verification results
        """
        asp_code = state.get("asp_code", "")

        if not asp_code.strip():
            logger.warning("Empty ASP code provided for verification")
            return {
                "verification_result": VerificationResult(
                    status="ERROR",
                    answer=None,
                    error_type="SYNTAX",
                    error_message="Empty ASP code provided",
                    solve_time_ms=0.0,
                    models=[],
                ),
                "solver_feedback": "ERROR: No ASP code was generated. Please provide valid ASP code.",
                "status": "running",
            }

        logger.info(f"Verifying ASP code ({len(asp_code)} chars)")
        logger.debug(f"ASP code:\n{asp_code}")

        # Run solver
        result = solver.solve(asp_code)

        # Map ErrorType to status string
        if result.success:
            status_str = "SAT"
        elif result.error_type == ErrorType.UNSAT:
            status_str = "UNSAT"
        elif result.error_type == ErrorType.TIMEOUT:
            status_str = "TIMEOUT"
        else:
            status_str = "ERROR"

        # Convert to VerificationResult
        verification_result = VerificationResult(
            status=status_str,
            answer=result.answer,
            error_type=result.error_type.name if result.error_type != ErrorType.NONE else None,
            error_message=result.error_message,
            solve_time_ms=result.solve_time_ms,
            models=result.models,
        )

        # Generate feedback for LLM
        solver_feedback = result.to_feedback()

        # Determine agent status
        if result.success:
            agent_status = "success"
            final_answer = result.answer
            logger.info(f"Verification SUCCESS: answer={result.answer}")
        else:
            agent_status = state.get("status", "running")
            final_answer = state.get("final_answer")
            logger.info(f"Verification FAILED: {result.error_type.name} - {result.error_message[:100]}")

        return {
            "verification_result": verification_result,
            "solver_feedback": solver_feedback,
            "status": agent_status,
            "final_answer": final_answer,
        }

    return verify_asp
