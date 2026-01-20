"""
NS-MAS Experiment Runner

Runs NS-MAS experiments with different routing strategies:
- Fixed Slow (always GVR)
- Bandit (adaptive routing)
- Random (50/50 baseline)
"""

import asyncio
import logging
import random
import time
from typing import Optional

from .config import ExperimentConfig, RunConfig, RunType
from .runner import ExperimentRunner, Problem
from .metrics import ExperimentResult, PathTaken

logger = logging.getLogger(__name__)


class NSMASRunner(ExperimentRunner):
    """
    Runner for NS-MAS experiments.

    Supports three modes:
    - Fixed Slow: Always use full GVR loop
    - Bandit: Use contextual bandit for routing
    - Random: Random 50/50 routing
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_config: RunConfig,
        agent=None,
        bandit_router=None,
        fast_solver=None,
    ):
        super().__init__(config, run_config)
        self._agent = agent
        self._bandit_router = bandit_router
        self._fast_solver = fast_solver
        self._random = random.Random(config.seed)

    def _get_agent(self):
        """Lazy load agent to avoid import issues."""
        if self._agent is None:
            from src.agent import Agent, AgentConfig, LLMConfig, LLMProvider

            agent_config = AgentConfig(
                generator_llm=LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model=self.run_config.model,
                    temperature=self.run_config.temperature,
                ),
                max_retries=5,
                enable_entity_extraction=False,
                trace_mode="OFF",
            )
            self._agent = Agent(agent_config)
        return self._agent

    def _get_fast_solver(self):
        """Lazy load fast solver."""
        if self._fast_solver is None:
            from src.bandit import FastSolver

            # FastSolver uses BanditConfig which defaults to gpt-4o-mini
            self._fast_solver = FastSolver()
        return self._fast_solver

    def _get_bandit_router(self):
        """Lazy load bandit router."""
        if self._bandit_router is None:
            from src.bandit import BanditRouter, BanditConfig

            bandit_config = BanditConfig()
            self._bandit_router = BanditRouter(bandit_config)

            # Load pre-trained model if specified
            if self.run_config.bandit_model_path:
                self._bandit_router.load(self.run_config.bandit_model_path)
        return self._bandit_router

    def _decide_path(self, problem: Problem) -> tuple[PathTaken, Optional[float]]:
        """
        Decide which path to take based on run type.

        Returns:
            (path, probability) - probability is None for fixed strategies
        """
        run_type = self.run_config.run_type

        if run_type == RunType.NSMAS_FIXED_SLOW:
            return PathTaken.SLOW, None

        elif run_type == RunType.NSMAS_RANDOM:
            if self._random.random() < self.run_config.random_fast_probability:
                return PathTaken.FAST, self.run_config.random_fast_probability
            else:
                return PathTaken.SLOW, 1 - self.run_config.random_fast_probability

        elif run_type == RunType.NSMAS_BANDIT:
            router = self._get_bandit_router()
            action, prob = router.predict(problem.question)
            if action == 0:  # Fast
                return PathTaken.FAST, prob
            else:  # Slow
                return PathTaken.SLOW, prob

        else:
            raise ValueError(f"Unknown run type: {run_type}")

    async def _run_fast_path(self, problem: Problem) -> dict:
        """Run fast path (zero-shot GPT-4o-mini)."""
        fast_solver = self._get_fast_solver()

        # FastSolver is sync, run in executor
        # Use solve_with_metadata to get a dict with answer and error info
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            fast_solver.solve_with_metadata,
            problem.question,
        )
        return result

    async def _run_slow_path(self, problem: Problem) -> dict:
        """Run slow path (full GVR loop)."""
        agent = self._get_agent()

        # Agent.solve is sync, run in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            agent.solve,
            problem.question,
        )
        return result

    async def solve_problem(self, problem: Problem) -> ExperimentResult:
        """Solve a problem using the configured routing strategy."""
        start_time = time.perf_counter()

        # Decide path
        path, probability = self._decide_path(problem)

        # Execute
        if path == PathTaken.FAST:
            result = await self._run_fast_path(problem)
            final_answer = result.get("answer")
            iterations = 0
            solver_feedback = None
            self_corrected = False
            error_type = result.get("error")
        else:
            result = await self._run_slow_path(problem)
            final_answer = result.get("final_answer")
            iterations = result.get("iteration_count", 0)
            solver_feedback = result.get("solver_feedback")
            # Self-corrected if took more than 1 iteration and succeeded
            self_corrected = iterations > 1 and result.get("status") == "success"
            error_type = (
                result.get("status")
                if result.get("status") not in ["success", "running"]
                else None
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Check correctness
        correct = final_answer == problem.answer

        # Update bandit with cost (for online learning)
        if self.run_config.run_type == RunType.NSMAS_BANDIT and probability is not None:
            router = self._get_bandit_router()
            action = 0 if path == PathTaken.FAST else 1
            # Cost is 0 for correct, 1 for incorrect
            cost = 0.0 if correct else 1.0
            router.learn(problem.question, action, cost, probability)

        return ExperimentResult(
            problem_id=problem.id,
            template_id=problem.template_id,
            variant=problem.variant,
            question=problem.question,
            ground_truth=problem.answer,
            system=self.run_config.run_type.value,
            path_taken=path,
            final_answer=final_answer,
            correct=correct,
            iterations=iterations,
            solver_feedback=solver_feedback,
            error_type=error_type,
            self_corrected=self_corrected,
            execution_time_ms=elapsed_ms,
            bandit_action=path.value if self.run_config.run_type == RunType.NSMAS_BANDIT else None,
            bandit_probability=probability,
        )


async def run_nsmas_experiment(
    run_type: str = "fixed_slow",
    data_dir: str = "output",
    results_dir: str = "results",
    limit: Optional[int] = None,
    variants: Optional[list] = None,
    bandit_model_path: Optional[str] = None,
):
    """
    Convenience function to run an NS-MAS experiment.

    Args:
        run_type: One of "fixed_slow", "bandit", "random"
        data_dir: Directory containing dataset files
        results_dir: Directory to save results
        limit: Optional limit on problems (for pilot runs)
        variants: Optional list of variants to run (default: all)
        bandit_model_path: Optional path to pre-trained bandit model
    """
    from pathlib import Path
    from .config import (
        ExperimentConfig,
        NSMAS_FIXED_SLOW_CONFIG,
        NSMAS_BANDIT_CONFIG,
        NSMAS_RANDOM_CONFIG,
        DatasetVariant,
    )

    # Select run config
    if run_type == "fixed_slow":
        run_config = NSMAS_FIXED_SLOW_CONFIG
    elif run_type == "bandit":
        run_config = NSMAS_BANDIT_CONFIG
        if bandit_model_path:
            run_config.bandit_model_path = Path(bandit_model_path)
    elif run_type == "random":
        run_config = NSMAS_RANDOM_CONFIG
    else:
        raise ValueError(f"Unknown run type: {run_type}")

    # Build experiment config
    config = ExperimentConfig(
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
    )

    if variants:
        config.variants = [DatasetVariant(v) for v in variants]

    # Create runner
    runner = NSMASRunner(config, run_config)

    # Run
    if limit:
        # Pilot mode
        variant = config.variants[0]
        metrics = await runner.run(variant, limit=limit)
        summary = metrics.compute_summary(run_config.run_type.value, variant.value)
        return {variant.value: summary}
    else:
        # Full run
        results = await runner.run_all_variants()
        summaries = {}
        for variant, metrics in results.items():
            summaries[variant] = metrics.compute_summary(
                run_config.run_type.value, variant
            )
        return summaries


# CLI entry point
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run NS-MAS experiment")
    parser.add_argument(
        "--run-type",
        choices=["fixed_slow", "bandit", "random"],
        default="fixed_slow",
        help="Routing strategy",
    )
    parser.add_argument(
        "--data-dir",
        default="output",
        help="Dataset directory",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit problems (for pilot)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variants to run",
    )
    parser.add_argument(
        "--bandit-model",
        default=None,
        help="Path to pre-trained bandit model",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results = asyncio.run(
        run_nsmas_experiment(
            run_type=args.run_type,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            limit=args.limit,
            variants=args.variants,
            bandit_model_path=args.bandit_model,
        )
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for variant, summary in results.items():
        print(f"\n{variant.upper()}:")
        print(f"  Accuracy: {summary.accuracy * 100:.2f}%")
        print(f"  Total: {summary.total_problems}")
        print(f"  Correct: {summary.correct_count}")
        print(f"  Fast path: {summary.fast_path_count}")
        print(f"  Slow path: {summary.slow_path_count}")
        if summary.fast_path_count > 0:
            print(f"  Fast accuracy: {summary.fast_path_accuracy * 100:.2f}%")
        if summary.slow_path_count > 0:
            print(f"  Slow accuracy: {summary.slow_path_accuracy * 100:.2f}%")
        print(f"  Self-correction rate: {summary.self_correction_rate * 100:.2f}%")
