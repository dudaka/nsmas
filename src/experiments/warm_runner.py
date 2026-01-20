"""
Warm-Start Bandit Experiment Runner.

Runs NS-MAS with warm-started bandit router using PCA features.

Usage:
    python -m src.experiments.warm_runner \
        --policy models/warm_policy.vw \
        --pca models/pca_64.pkl \
        --data-dir output \
        --results-dir results/phase8a
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WarmExperimentResult:
    """Result for a single problem."""
    problem_id: str
    variant: str
    question: str
    expected_answer: int
    predicted_answer: Optional[int]
    correct: bool
    path_taken: str  # "fast" or "slow"
    routing_prob: float
    latency_ms: float
    error: Optional[str] = None


def extract_answer(solution_str: str) -> Optional[int]:
    """Extract numeric answer from GSM solution string.

    Format: "... #### 724" -> 724
    """
    import re
    match = re.search(r'####\s*(-?\d+)', solution_str)
    if match:
        return int(match.group(1))
    return None


def load_problems(data_dir: Path, variants: list[str]) -> list[dict]:
    """Load problems from JSONL files."""
    problems = []
    for variant in variants:
        file_path = data_dir / f"gsm_{variant}.jsonl"
        if not file_path.exists():
            logger.warning(f"Dataset not found: {file_path}")
            continue

        with open(file_path) as f:
            for line in f:
                problem = json.loads(line)
                problem["variant"] = variant
                # Extract numeric answer from solution string
                problem["answer_num"] = extract_answer(problem.get("answer", ""))
                problems.append(problem)

    logger.info(f"Loaded {len(problems)} problems from {variants}")
    return problems


def run_experiment(
    policy_path: str,
    pca_path: str,
    data_dir: str,
    results_dir: str,
    variants: list[str] = None,
    limit: int = None,
):
    """Run the warm-start experiment."""
    from src.bandit.warmstart import WarmStartRouter
    from src.bandit import FastSolver
    from src.agent import Agent, AgentConfig, LLMConfig, LLMProvider

    if variants is None:
        variants = ["base", "p1", "p2", "noop"]

    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load components
    logger.info(f"Loading warm-start router from {policy_path}")
    router = WarmStartRouter(model_path=policy_path, pca_path=pca_path)

    logger.info("Initializing fast solver")
    fast_solver = FastSolver()

    logger.info("Initializing GVR agent")
    agent_config = AgentConfig(
        generator_llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            temperature=0.0,
        ),
        max_retries=5,
        enable_entity_extraction=False,
        trace_mode="OFF",
    )
    slow_agent = Agent(agent_config)

    # Load problems
    problems = load_problems(data_path, variants)
    if limit:
        problems = problems[:limit]

    results = []
    stats = {v: {"correct": 0, "total": 0, "fast": 0, "slow": 0} for v in variants}

    logger.info(f"Starting experiment with {len(problems)} problems")

    for i, problem in enumerate(problems):
        problem_id = problem.get("id", f"problem_{i}")
        variant = problem["variant"]
        question = problem["question"]
        expected = problem.get("answer_num")  # Use extracted numeric answer

        start_time = time.time()

        try:
            # Get routing decision from warm-start router
            action, prob = router.predict_deterministic(question)
            path = "fast" if action == 0 else "slow"

            # Execute the chosen path
            if path == "fast":
                # Fast path: zero-shot LLM
                answer_str = fast_solver.solve(question)
                try:
                    predicted = int(float(answer_str)) if answer_str else None
                except (ValueError, TypeError):
                    predicted = None
            else:
                # Slow path: GVR loop
                result = slow_agent.solve(question)
                predicted = result.get("final_answer")

            latency = (time.time() - start_time) * 1000
            correct = predicted == expected

            exp_result = WarmExperimentResult(
                problem_id=problem_id,
                variant=variant,
                question=question[:200],
                expected_answer=expected,
                predicted_answer=predicted,
                correct=correct,
                path_taken=path,
                routing_prob=prob,
                latency_ms=latency,
            )

            # Update stats
            stats[variant]["total"] += 1
            if correct:
                stats[variant]["correct"] += 1
            stats[variant][path] += 1

        except Exception as e:
            logger.error(f"Error on problem {problem_id}: {e}")
            latency = (time.time() - start_time) * 1000
            exp_result = WarmExperimentResult(
                problem_id=problem_id,
                variant=variant,
                question=question[:200],
                expected_answer=expected,
                predicted_answer=None,
                correct=False,
                path_taken="error",
                routing_prob=0.0,
                latency_ms=latency,
                error=str(e),
            )
            stats[variant]["total"] += 1

        results.append(exp_result)

        # Progress report every 100 problems
        if (i + 1) % 100 == 0:
            total_correct = sum(s["correct"] for s in stats.values())
            total_done = sum(s["total"] for s in stats.values())
            logger.info(
                f"Progress: {i + 1}/{len(problems)} "
                f"Accuracy: {total_correct}/{total_done} "
                f"({100 * total_correct / total_done:.1f}%)"
            )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"warm_bandit_{timestamp}.jsonl"

    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    logger.info(f"Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("WARM-START BANDIT EXPERIMENT RESULTS")
    print("=" * 60)

    total_correct = 0
    total_problems = 0
    total_fast = 0
    total_slow = 0

    for variant, s in stats.items():
        if s["total"] == 0:
            continue
        acc = 100 * s["correct"] / s["total"]
        fast_pct = 100 * s["fast"] / s["total"]
        print(f"\n{variant.upper()}:")
        print(f"  Accuracy: {s['correct']}/{s['total']} ({acc:.1f}%)")
        print(f"  Fast path: {s['fast']} ({fast_pct:.1f}%)")
        print(f"  Slow path: {s['slow']} ({100 - fast_pct:.1f}%)")

        total_correct += s["correct"]
        total_problems += s["total"]
        total_fast += s["fast"]
        total_slow += s["slow"]

    if total_problems > 0:
        overall_acc = 100 * total_correct / total_problems
        overall_fast = 100 * total_fast / total_problems
        print(f"\nOVERALL:")
        print(f"  Accuracy: {total_correct}/{total_problems} ({overall_acc:.1f}%)")
        print(f"  Fast path: {total_fast} ({overall_fast:.1f}%)")
        print(f"  Slow path: {total_slow} ({100 - overall_fast:.1f}%)")

    print("=" * 60)

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Run warm-start bandit experiment")
    parser.add_argument("--policy", required=True, help="Path to VW policy model")
    parser.add_argument("--pca", required=True, help="Path to PCA transformer")
    parser.add_argument("--data-dir", default="output", help="Dataset directory")
    parser.add_argument("--results-dir", default="results/phase8a", help="Results directory")
    parser.add_argument("--variants", nargs="+", default=["base", "p1", "p2", "noop"])
    parser.add_argument("--limit", type=int, help="Limit number of problems")

    args = parser.parse_args()

    run_experiment(
        policy_path=args.policy,
        pca_path=args.pca,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        variants=args.variants,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
