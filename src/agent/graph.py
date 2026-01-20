"""
LangGraph StateGraph definition for the NS-MAS agent.

Implements the Generator -> Verifier -> Reflector loop with
conditional routing based on verification results.

Phase 5 Addition: Optional Bandit Router integration for
adaptive Fast/Slow path routing.
"""

import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from .config import AgentConfig
from .state import AgentState, create_initial_state
from .nodes import (
    create_extract_entities_node,
    create_generate_asp_node,
    create_verify_asp_node,
    create_reflect_node,
)

logger = logging.getLogger(__name__)


def route_after_verification(state: AgentState) -> Literal["end", "reflect"]:
    """
    Determine next step after ASP verification.

    Routes to END if:
        - Verification succeeded (SAT with answer)
        - Max retries exceeded
        - Cycle detected (same ASP code generated twice)

    Routes to reflect if:
        - Verification failed but retries remaining

    Args:
        state: Current agent state

    Returns:
        "end" or "reflect" routing decision
    """
    status = state.get("status", "running")
    iteration = state.get("iteration_count", 0)
    max_retries = state.get("max_retries", 5)

    # Check for success
    if status == "success":
        logger.info("Routing to END: verification succeeded")
        return "end"

    # Check for cycle detection
    if status == "cycle_detected":
        logger.warning("Routing to END: cycle detected")
        return "end"

    # Check for max retries
    if iteration >= max_retries:
        logger.warning(f"Routing to END: max retries ({max_retries}) exceeded")
        return "end"

    # Continue with reflection
    logger.info(f"Routing to reflect: attempt {iteration}/{max_retries}")
    return "reflect"


def create_agent_graph(config: AgentConfig) -> StateGraph:
    """
    Create the NS-MAS agent graph.

    Graph topology:
        [extract_entities] -> [generate_asp] -> [verify_asp] -> route
                                   ^                              |
                                   |                              v
                                   +-------- [reflect] <---- (if not SAT)
                                                                  |
                                                                  v
                                                                END

    Args:
        config: Agent configuration

    Returns:
        Compiled LangGraph StateGraph
    """
    logger.info("Creating agent graph")

    # Create graph with state schema
    graph = StateGraph(AgentState)

    # Create nodes with config
    extract_entities_node = create_extract_entities_node(config)
    generate_asp_node = create_generate_asp_node(config)
    verify_asp_node = create_verify_asp_node(config)
    reflect_node = create_reflect_node(config)

    # Add nodes to graph
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("generate_asp", generate_asp_node)
    graph.add_node("verify_asp", verify_asp_node)
    graph.add_node("reflect", reflect_node)

    # Define edges based on config
    if config.enable_entity_extraction:
        graph.set_entry_point("extract_entities")
        graph.add_edge("extract_entities", "generate_asp")
    else:
        graph.set_entry_point("generate_asp")

    graph.add_edge("generate_asp", "verify_asp")

    # Conditional edge from verify_asp
    graph.add_conditional_edges(
        "verify_asp",
        route_after_verification,
        {
            "end": END,
            "reflect": "reflect",
        },
    )

    # Edge from reflect back to generate_asp
    graph.add_edge("reflect", "generate_asp")

    logger.info("Graph created successfully")
    return graph.compile()


class Agent:
    """
    High-level interface for the NS-MAS agent.

    Wraps the LangGraph StateGraph with a simple solve() method.

    Phase 5: Now supports optional Bandit Router for adaptive routing
    between Fast (zero-shot) and Slow (GVR) paths.

    Usage:
        # Without bandit routing (Phase 4 behavior)
        agent = Agent(AgentConfig())
        result = agent.solve("John has 10 apples...")

        # With bandit routing (Phase 5)
        from src.bandit import BanditConfig
        agent = Agent(AgentConfig(), bandit_config=BanditConfig())
        result = agent.solve("John has 10 apples...")
    """

    def __init__(
        self,
        config: AgentConfig = None,
        bandit_config: Optional["BanditConfig"] = None,
    ):
        """
        Initialize the agent.

        Args:
            config: Agent configuration (uses defaults if not provided)
            bandit_config: Optional bandit config for Fast/Slow routing
        """
        self.config = config or AgentConfig()
        self.bandit_config = bandit_config

        self._graph = None
        self._router = None
        self._fast_solver = None

    @property
    def graph(self) -> StateGraph:
        """Lazy initialization of the graph."""
        if self._graph is None:
            self._graph = create_agent_graph(self.config)
        return self._graph

    @property
    def router(self):
        """Lazy initialization of bandit router."""
        if self._router is None and self.bandit_config is not None:
            from src.bandit import BanditRouter

            self._router = BanditRouter(self.bandit_config)
        return self._router

    @property
    def fast_solver(self):
        """Lazy initialization of fast solver."""
        if self._fast_solver is None and self.bandit_config is not None:
            from src.bandit import FastSolver

            self._fast_solver = FastSolver(self.bandit_config)
        return self._fast_solver

    def solve(
        self,
        question: str,
        expected_answer: int = None,
        max_retries: int = None,
        force_path: Optional[Literal["fast", "slow"]] = None,
    ) -> AgentState:
        """
        Solve a math word problem.

        If bandit routing is enabled, routes to Fast (zero-shot) or
        Slow (GVR) path based on learned policy.

        Args:
            question: The math word problem to solve
            expected_answer: Optional ground truth for evaluation
            max_retries: Override for max retry attempts
            force_path: Force "fast" or "slow" path (bypasses bandit)

        Returns:
            Final agent state with solution or error information
        """
        # Determine path based on bandit or force_path
        use_fast_path = False
        routing_prob = None

        if force_path == "fast":
            use_fast_path = True
            logger.info("Forced to fast path")
        elif force_path == "slow":
            use_fast_path = False
            logger.info("Forced to slow path")
        elif self.router is not None:
            # Use bandit to decide
            action, prob = self.router.predict(question)
            routing_prob = prob
            use_fast_path = action == 0  # 0 = fast, 1 = slow
            logger.info(
                f"Bandit routing: {'fast' if use_fast_path else 'slow'} "
                f"(prob={prob:.3f})"
            )

        # Fast path: zero-shot without verification
        if use_fast_path and self.fast_solver is not None:
            answer = self.fast_solver.solve(question)

            # Return state-like dict for consistency
            final_state = AgentState(
                question=question,
                expected_answer=expected_answer,
                final_answer=answer,
                status="success" if answer is not None else "fast_failed",
                iteration_count=0,
                asp_code="",
                asp_code_history=[],
                parsed_entities=[],
                entity_extraction_reasoning="",
                verification_result={
                    "status": "FAST_PATH",
                    "answer": answer,
                    "error_type": None,
                    "error_message": "",
                    "solve_time_ms": 0.0,
                    "models": [],
                },
                solver_feedback="Used fast path (zero-shot)",
                critique_history=[],
                max_retries=0,
            )

            if answer is not None:
                logger.info(f"Fast path succeeded: answer={answer}")
            else:
                logger.warning("Fast path failed to extract answer")

            return final_state

        # Slow path: full GVR loop
        return self._solve_slow(question, expected_answer, max_retries)

    def _solve_slow(
        self,
        question: str,
        expected_answer: int = None,
        max_retries: int = None,
    ) -> AgentState:
        """
        Solve using the slow path (GVR loop).

        This is the original Phase 4 solve implementation.

        Args:
            question: The math word problem to solve
            expected_answer: Optional ground truth
            max_retries: Override for max retries

        Returns:
            Final agent state
        """
        # Create initial state
        retries = max_retries if max_retries is not None else self.config.max_retries
        initial_state = create_initial_state(
            question=question,
            expected_answer=expected_answer,
            max_retries=retries,
        )

        logger.info(f"Starting slow path: question='{question[:50]}...'")

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Log result
        if final_state.get("status") == "success":
            logger.info(f"Slow path succeeded: answer={final_state.get('final_answer')}")
        else:
            logger.warning(
                f"Slow path failed: status={final_state.get('status')}, "
                f"iterations={final_state.get('iteration_count')}"
            )

        return final_state

    def solve_batch(
        self,
        questions: list[str],
        expected_answers: list[int] = None,
    ) -> list[AgentState]:
        """
        Solve multiple math word problems.

        Args:
            questions: List of math word problems
            expected_answers: Optional list of ground truth answers

        Returns:
            List of final agent states
        """
        if expected_answers is None:
            expected_answers = [None] * len(questions)

        results = []
        for i, (q, a) in enumerate(zip(questions, expected_answers)):
            logger.info(f"Solving problem {i + 1}/{len(questions)}")
            result = self.solve(q, expected_answer=a)
            results.append(result)

        # Log summary
        successes = sum(1 for r in results if r.get("status") == "success")
        logger.info(f"Batch complete: {successes}/{len(questions)} succeeded")

        return results
