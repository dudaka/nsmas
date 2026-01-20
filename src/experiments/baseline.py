"""
Baseline Runner

Runs GPT-4o and GPT-4o-mini zero-shot Chain-of-Thought baselines.
"""

import asyncio
import logging
import re
import time
from typing import Optional

from openai import AsyncOpenAI

from .config import ExperimentConfig, RunConfig, RunType
from .runner import ExperimentRunner, Problem
from .metrics import ExperimentResult, PathTaken

logger = logging.getLogger(__name__)


# Zero-Shot CoT prompt template (from Kojima et al.)
COT_PROMPT_TEMPLATE = """Q: {question}
A: Let's think step by step."""


def extract_final_answer(response: str) -> Optional[int]:
    """
    Extract the final numerical answer from a CoT response.

    Looks for patterns like:
    - "#### 42"
    - "the answer is 42"
    - "= 42"
    - Final number in the response
    """
    # Try to find "#### <number>" pattern (GSM format)
    match = re.search(r"####\s*(-?\d+)", response)
    if match:
        return int(match.group(1))

    # Try "the answer is <number>"
    match = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*(-?\d+)", response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try "= <number>" at end of a line
    match = re.search(r"=\s*(-?\d+)\s*$", response, re.MULTILINE)
    if match:
        return int(match.group(1))

    # Try to find last number in the response
    numbers = re.findall(r"(-?\d+)", response)
    if numbers:
        return int(numbers[-1])

    return None


class BaselineRunner(ExperimentRunner):
    """
    Runner for GPT-4o and GPT-4o-mini zero-shot CoT baselines.

    Uses the "Let's think step by step" prompt from Kojima et al.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_config: RunConfig,
        client: Optional[AsyncOpenAI] = None,
    ):
        super().__init__(config, run_config)
        self.client = client or AsyncOpenAI()

    async def solve_problem(self, problem: Problem) -> ExperimentResult:
        """Solve a problem using zero-shot CoT."""
        start_time = time.perf_counter()

        # Build prompt
        prompt = COT_PROMPT_TEMPLATE.format(question=problem.question)

        # Call API
        response = await self.client.chat.completions.create(
            model=self.run_config.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.run_config.temperature,
            max_tokens=self.run_config.max_tokens,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract answer
        content = response.choices[0].message.content or ""
        final_answer = extract_final_answer(content)
        correct = final_answer == problem.answer

        # Token usage
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return ExperimentResult(
            problem_id=problem.id,
            template_id=problem.template_id,
            variant=problem.variant,
            question=problem.question,
            ground_truth=problem.answer,
            system=self.run_config.run_type.value,
            path_taken=PathTaken.BASELINE,
            final_answer=final_answer,
            correct=correct,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            execution_time_ms=elapsed_ms,
        )


async def run_baseline_experiment(
    model: str = "gpt-4o",
    data_dir: str = "output",
    results_dir: str = "results",
    limit: Optional[int] = None,
    variants: Optional[list] = None,
):
    """
    Convenience function to run a baseline experiment.

    Args:
        model: Model to use ("gpt-4o" or "gpt-4o-mini")
        data_dir: Directory containing dataset files
        results_dir: Directory to save results
        limit: Optional limit on problems (for pilot runs)
        variants: Optional list of variants to run (default: all)
    """
    from pathlib import Path
    from .config import (
        ExperimentConfig,
        BASELINE_GPT4O_CONFIG,
        BASELINE_GPT4O_MINI_CONFIG,
        DatasetVariant,
    )

    # Select run config
    if model == "gpt-4o":
        run_config = BASELINE_GPT4O_CONFIG
    elif model == "gpt-4o-mini":
        run_config = BASELINE_GPT4O_MINI_CONFIG
    else:
        raise ValueError(f"Unknown model: {model}")

    # Build experiment config
    config = ExperimentConfig(
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
    )

    if variants:
        config.variants = [DatasetVariant(v) for v in variants]

    # Create runner
    runner = BaselineRunner(config, run_config)

    # Run
    if limit:
        # Pilot mode - run on first variant only
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

    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument(
        "--model",
        choices=["gpt-4o", "gpt-4o-mini"],
        default="gpt-4o",
        help="Model to use",
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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results = asyncio.run(
        run_baseline_experiment(
            model=args.model,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            limit=args.limit,
            variants=args.variants,
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
        print(f"  Avg tokens: {summary.avg_tokens_per_problem:.1f}")
