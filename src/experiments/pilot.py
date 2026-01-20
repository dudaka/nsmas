"""
Pilot Runner

Phase 6a: Pipeline validation with a small subset of problems.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .config import ExperimentConfig, DatasetVariant
from .baseline import BaselineRunner
from .nsmas_runner import NSMASRunner
from .metrics import MetricsSummary

logger = logging.getLogger(__name__)


@dataclass
class PilotResults:
    """Results from a pilot run."""
    baseline_summary: MetricsSummary
    bandit_summary: Optional[MetricsSummary]
    total_cost_estimate: float
    tokens_per_problem: float
    problems_processed: int


async def run_pilot(
    num_problems: int = 500,
    data_dir: str = "output",
    results_dir: str = "results/pilot",
    run_bandit: bool = True,
) -> PilotResults:
    """
    Run pilot validation (Phase 6a).

    1. Run GPT-4o-mini baseline on small subset
    2. Optionally run bandit validation
    3. Calculate cost estimates for full run

    Args:
        num_problems: Number of problems to run
        data_dir: Directory containing dataset files
        results_dir: Directory to save pilot results
        run_bandit: Whether to run bandit validation

    Returns:
        PilotResults with summaries and cost estimates
    """
    from .config import BASELINE_GPT4O_MINI_CONFIG, NSMAS_BANDIT_CONFIG

    config = ExperimentConfig(
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
        variants=[DatasetVariant.BASE],  # Pilot on base only
    )

    # 1. Run GPT-4o-mini baseline
    logger.info("=" * 60)
    logger.info("PILOT: Running GPT-4o-mini baseline")
    logger.info("=" * 60)

    baseline_runner = BaselineRunner(config, BASELINE_GPT4O_MINI_CONFIG)
    baseline_metrics = await baseline_runner.run(
        DatasetVariant.BASE,
        limit=num_problems,
        resume=False,
    )
    baseline_summary = baseline_metrics.compute_summary(
        BASELINE_GPT4O_MINI_CONFIG.run_type.value,
        DatasetVariant.BASE.value,
    )

    logger.info(f"Baseline accuracy: {baseline_summary.accuracy * 100:.2f}%")
    logger.info(f"Avg tokens/problem: {baseline_summary.avg_tokens_per_problem:.1f}")

    # 2. Run bandit validation (optional)
    bandit_summary = None
    if run_bandit:
        logger.info("\n" + "=" * 60)
        logger.info("PILOT: Running Bandit validation")
        logger.info("=" * 60)

        # Use lower concurrency for GPT-4o (30K TPM limit)
        bandit_config = ExperimentConfig(
            data_dir=Path(data_dir),
            results_dir=Path(results_dir),
            variants=[DatasetVariant.BASE],
            max_concurrent=5,  # Reduced from 50 to avoid rate limits
            requests_per_minute=60,  # Conservative rate limit
        )
        bandit_runner = NSMASRunner(bandit_config, NSMAS_BANDIT_CONFIG)
        bandit_metrics = await bandit_runner.run(
            DatasetVariant.BASE,
            limit=min(num_problems, 50),  # Smaller for bandit (more expensive)
            resume=False,
        )
        bandit_summary = bandit_metrics.compute_summary(
            NSMAS_BANDIT_CONFIG.run_type.value,
            DatasetVariant.BASE.value,
        )

        logger.info(f"Bandit accuracy: {bandit_summary.accuracy * 100:.2f}%")
        logger.info(f"Fast path: {bandit_summary.fast_path_count}")
        logger.info(f"Slow path: {bandit_summary.slow_path_count}")

    # 3. Calculate cost estimates
    # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
    # GPT-4o pricing: $2.50/1M input, $10.00/1M output
    avg_tokens = baseline_summary.avg_tokens_per_problem
    avg_input = baseline_summary.total_tokens_input / max(baseline_summary.total_problems, 1)
    avg_output = baseline_summary.total_tokens_output / max(baseline_summary.total_problems, 1)

    # Estimate for full run (13,396 problems, 5 runs)
    total_problems = 13396
    num_runs = 5

    # GPT-4o baseline estimate
    gpt4o_cost_per_problem = (avg_input * 2.50 + avg_output * 10.00) / 1_000_000
    gpt4o_total = gpt4o_cost_per_problem * total_problems

    # GPT-4o-mini baseline estimate
    gpt4o_mini_cost_per_problem = (avg_input * 0.15 + avg_output * 0.60) / 1_000_000
    gpt4o_mini_total = gpt4o_mini_cost_per_problem * total_problems

    # GVR estimate (more tokens due to iterations)
    gvr_multiplier = 3.5  # Avg input tokens, 2.0 output (reflections)
    gvr_cost_per_problem = (avg_input * gvr_multiplier * 2.50 + avg_output * 2.0 * 10.00) / 1_000_000
    gvr_total = gvr_cost_per_problem * total_problems

    total_estimate = gpt4o_total + gpt4o_mini_total + (gvr_total * 3)  # 3 NS-MAS runs

    logger.info("\n" + "=" * 60)
    logger.info("COST ESTIMATES (for full 13,396 problem run)")
    logger.info("=" * 60)
    logger.info(f"Avg tokens per problem: {avg_tokens:.1f}")
    logger.info(f"  Input: {avg_input:.1f}, Output: {avg_output:.1f}")
    logger.info(f"\nPer-run estimates:")
    logger.info(f"  GPT-4o CoT baseline: ${gpt4o_total:.2f}")
    logger.info(f"  GPT-4o-mini CoT baseline: ${gpt4o_mini_total:.2f}")
    logger.info(f"  NS-MAS GVR (each): ${gvr_total:.2f}")
    logger.info(f"\nTotal estimate (5 runs): ${total_estimate:.2f}")

    return PilotResults(
        baseline_summary=baseline_summary,
        bandit_summary=bandit_summary,
        total_cost_estimate=total_estimate,
        tokens_per_problem=avg_tokens,
        problems_processed=num_problems,
    )


# CLI entry point
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Phase 6a pilot validation")
    parser.add_argument(
        "--problems",
        type=int,
        default=500,
        help="Number of problems to run",
    )
    parser.add_argument(
        "--data-dir",
        default="output",
        help="Dataset directory",
    )
    parser.add_argument(
        "--results-dir",
        default="results/pilot",
        help="Results directory",
    )
    parser.add_argument(
        "--skip-bandit",
        action="store_true",
        help="Skip bandit validation",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results = asyncio.run(
        run_pilot(
            num_problems=args.problems,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            run_bandit=not args.skip_bandit,
        )
    )

    print("\n" + "=" * 60)
    print("PILOT COMPLETE")
    print("=" * 60)
    print(f"Processed: {results.problems_processed} problems")
    print(f"Baseline accuracy: {results.baseline_summary.accuracy * 100:.2f}%")
    print(f"Estimated full run cost: ${results.total_cost_estimate:.2f}")
