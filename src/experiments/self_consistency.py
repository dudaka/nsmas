"""
Self-Consistency Baseline Runner

Implements CoT + Self-Consistency (Wang et al., 2022): sample k responses
with temperature > 0 and take majority vote.

This is a stronger baseline than single-shot CoT, trading cost for accuracy.
"""

import asyncio
import logging
import time
from collections import Counter
from typing import Optional, List

from openai import AsyncOpenAI

from .config import ExperimentConfig, RunConfig, RunType, DatasetVariant
from .runner import ExperimentRunner, Problem
from .metrics import ExperimentResult, PathTaken
from .baseline import extract_final_answer, COT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class SelfConsistencyRunner(ExperimentRunner):
    """
    Runner for GPT-4o CoT + Self-Consistency baseline.

    Samples k responses with temperature > 0 and takes majority vote.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_config: RunConfig,
        client: Optional[AsyncOpenAI] = None,
        k_samples: int = 5,
        temperature: float = 0.7,
    ):
        super().__init__(config, run_config)
        self.client = client or AsyncOpenAI()
        self.k_samples = k_samples
        self.temperature = temperature

    async def _sample_once(self, prompt: str) -> tuple[Optional[int], int, int, str]:
        """
        Sample a single response.

        Returns: (answer, tokens_in, tokens_out, raw_response)
        """
        response = await self.client.chat.completions.create(
            model=self.run_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.run_config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        answer = extract_final_answer(content)

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return answer, tokens_in, tokens_out, content

    async def solve_problem(self, problem: Problem) -> ExperimentResult:
        """Solve a problem using CoT + Self-Consistency (majority vote of k samples)."""
        start_time = time.perf_counter()

        # Build prompt
        prompt = COT_PROMPT_TEMPLATE.format(question=problem.question)

        # Sample k responses concurrently
        tasks = [self._sample_once(prompt) for _ in range(self.k_samples)]
        samples = await asyncio.gather(*tasks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract answers and aggregate tokens
        answers = []
        total_tokens_in = 0
        total_tokens_out = 0

        for answer, tokens_in, tokens_out, _ in samples:
            if answer is not None:
                answers.append(answer)
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out

        # Majority vote
        if answers:
            vote_counts = Counter(answers)
            final_answer, vote_count = vote_counts.most_common(1)[0]
        else:
            final_answer = None
            vote_count = 0

        correct = final_answer == problem.answer

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
            tokens_input=total_tokens_in,
            tokens_output=total_tokens_out,
            execution_time_ms=elapsed_ms,
            iterations=self.k_samples,  # Reuse iterations field for k
            solver_feedback=f"k={self.k_samples}, votes={vote_count}/{len(answers)}",
        )


async def run_self_consistency_experiment(
    model: str = "gpt-4o",
    data_dir: str = "output",
    results_dir: str = "results",
    limit: Optional[int] = None,
    variants: Optional[List[str]] = None,
    k_samples: int = 5,
    temperature: float = 0.7,
    sample_seed: Optional[int] = None,
):
    """
    Run CoT + Self-Consistency baseline experiment.

    Args:
        model: Model to use (default: gpt-4o)
        data_dir: Directory containing dataset files
        results_dir: Directory to save results
        limit: Optional limit on problems per variant
        variants: Optional list of variants to run (default: all)
        k_samples: Number of samples for self-consistency (default: 5)
        temperature: Sampling temperature (default: 0.7)
        sample_seed: Optional seed for reproducible problem sampling
    """
    from pathlib import Path
    import random

    # Build run config
    run_config = RunConfig(
        run_type=RunType.BASELINE_SC,
        name=f"GPT-4o CoT + Self-Consistency (k={k_samples})",
        model=model,
        temperature=temperature,
    )

    # Build experiment config
    config = ExperimentConfig(
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
    )

    if variants:
        config.variants = [DatasetVariant(v) for v in variants]

    # Set seed for reproducible sampling
    if sample_seed is not None:
        random.seed(sample_seed)

    # Create runner
    runner = SelfConsistencyRunner(
        config,
        run_config,
        k_samples=k_samples,
        temperature=temperature,
    )

    # Run experiment
    if limit:
        # Run on limited sample (for cost control)
        all_results = {}
        for variant in config.variants:
            logger.info(f"Running {variant.value} with limit={limit}")
            metrics = await runner.run(variant, limit=limit)
            summary = metrics.compute_summary(run_config.run_type.value, variant.value)
            all_results[variant.value] = summary
        return all_results
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

    parser = argparse.ArgumentParser(
        description="Run CoT + Self-Consistency baseline"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use (default: gpt-4o)",
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
        help="Limit problems per variant (for cost control)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variants to run (default: all)",
    )
    parser.add_argument(
        "-k", "--k-samples",
        type=int,
        default=5,
        help="Number of samples for self-consistency (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for problem sampling",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results = asyncio.run(
        run_self_consistency_experiment(
            model=args.model,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            limit=args.limit,
            variants=args.variants,
            k_samples=args.k_samples,
            temperature=args.temperature,
            sample_seed=args.seed,
        )
    )

    print("\n" + "=" * 60)
    print(f"SELF-CONSISTENCY RESULTS (k={args.k_samples})")
    print("=" * 60)
    for variant, summary in results.items():
        print(f"\n{variant.upper()}:")
        print(f"  Accuracy: {summary.accuracy * 100:.2f}%")
        print(f"  Total: {summary.total_problems}")
        print(f"  Correct: {summary.correct_count}")
        print(f"  Avg tokens: {summary.avg_tokens_per_problem:.1f}")
        print(f"  Tokens/problem: {summary.avg_tokens_per_problem:.1f} (k={args.k_samples})")
