"""
Async Experiment Runner

Handles concurrent API calls with rate limiting, checkpointing, and error recovery.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from datetime import datetime

from .config import ExperimentConfig, RunConfig, DatasetVariant
from .metrics import ExperimentResult, MetricsCollector, PathTaken

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """A problem to solve."""
    id: str
    question: str
    answer: int
    template_id: str
    variant: str


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.rpm,
                self.tokens + elapsed * (self.rpm / 60.0)
            )

            if self.tokens < 1:
                # Need to wait
                wait_time = (1 - self.tokens) * (60.0 / self.rpm)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class CheckpointManager:
    """Manages checkpointing for experiment runs."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, processed_ids: set, metrics_summary: Dict[str, Any]):
        """Save checkpoint."""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "processed_count": len(processed_ids),
            "processed_ids": list(processed_ids),
            "metrics_summary": metrics_summary,
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {len(processed_ids)} problems processed")

    def load(self) -> Optional[set]:
        """Load checkpoint. Returns set of processed IDs or None if no checkpoint."""
        if not self.checkpoint_path.exists():
            return None

        with open(self.checkpoint_path) as f:
            checkpoint = json.load(f)

        processed_ids = set(checkpoint.get("processed_ids", []))
        logger.info(f"Checkpoint loaded: {len(processed_ids)} problems already processed")
        return processed_ids

    def clear(self):
        """Remove checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


class ExperimentRunner(ABC):
    """
    Base class for experiment runners.

    Handles async execution, rate limiting, checkpointing, and error recovery.
    Subclasses implement the actual solving logic.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_config: RunConfig,
    ):
        self.config = config
        self.run_config = run_config
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.metrics = MetricsCollector()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    @abstractmethod
    async def solve_problem(self, problem: Problem) -> ExperimentResult:
        """
        Solve a single problem.

        Subclasses must implement this method.
        """
        pass

    async def _solve_with_retry(self, problem: Problem) -> ExperimentResult:
        """Solve a problem with retry logic."""
        last_error = None

        for attempt in range(self.config.retry_on_error):
            try:
                await self.rate_limiter.acquire()

                async with self._semaphore:
                    result = await asyncio.wait_for(
                        self.solve_problem(problem),
                        timeout=self.run_config.timeout_seconds,
                    )
                    return result

            except asyncio.TimeoutError:
                last_error = "TIMEOUT"
                logger.warning(
                    f"Timeout on problem {problem.id} (attempt {attempt + 1})"
                )
                # Exponential backoff
                backoff = min(
                    self.config.backoff_base ** attempt,
                    self.config.backoff_max,
                )
                await asyncio.sleep(backoff)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Error on problem {problem.id} (attempt {attempt + 1}): {e}"
                )
                backoff = min(
                    self.config.backoff_base ** attempt,
                    self.config.backoff_max,
                )
                await asyncio.sleep(backoff)

        # All retries failed
        return ExperimentResult(
            problem_id=problem.id,
            template_id=problem.template_id,
            variant=problem.variant,
            question=problem.question,
            ground_truth=problem.answer,
            system=self.run_config.run_type.value,
            path_taken=PathTaken.BASELINE,
            final_answer=None,
            correct=False,
            error_type=f"MAX_RETRIES: {last_error}",
        )

    def load_problems(self, variant: DatasetVariant) -> List[Problem]:
        """Load problems from dataset file."""
        path = self.config.get_dataset_path(variant)
        problems = []

        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    problems.append(Problem(
                        id=data["id"],
                        question=data["question"],
                        answer=data["final_answer"],
                        template_id=data["template_id"],
                        variant=data["variant"],
                    ))

        return problems

    async def run(
        self,
        variant: DatasetVariant,
        limit: Optional[int] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MetricsCollector:
        """
        Run the experiment on a dataset variant.

        Args:
            variant: Dataset variant to run on
            limit: Optional limit on number of problems (for pilot)
            resume: Whether to resume from checkpoint
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            MetricsCollector with all results
        """
        # Reset metrics for this variant run to avoid cross-contamination
        self.metrics = MetricsCollector()

        # Setup paths
        results_path = self.config.get_results_path(
            self.run_config.run_type, variant
        )
        checkpoint_path = self.config.get_checkpoint_path(
            self.run_config.run_type, variant
        )

        checkpoint_mgr = CheckpointManager(checkpoint_path)

        # Load existing results if resuming
        processed_ids = set()
        if resume:
            loaded = self.metrics.load_results(results_path)
            if loaded > 0:
                processed_ids = self.metrics.get_processed_ids()
                logger.info(f"Resumed with {loaded} existing results")

            # Also check checkpoint
            checkpoint_ids = checkpoint_mgr.load()
            if checkpoint_ids:
                processed_ids.update(checkpoint_ids)

        # Load problems
        all_problems = self.load_problems(variant)
        if limit:
            all_problems = all_problems[:limit]

        # Filter out already processed
        problems = [p for p in all_problems if p.id not in processed_ids]
        total = len(all_problems)
        remaining = len(problems)

        logger.info(
            f"Running {self.run_config.name} on {variant.value}: "
            f"{remaining} remaining of {total} total"
        )

        if remaining == 0:
            logger.info("All problems already processed")
            return self.metrics

        # Process in batches for checkpointing
        batch_size = self.config.checkpoint_interval
        completed = len(processed_ids)

        for i in range(0, len(problems), batch_size):
            batch = problems[i:i + batch_size]

            # Process batch concurrently
            tasks = [self._solve_with_retry(p) for p in batch]
            results = await asyncio.gather(*tasks)

            # Collect results
            for result in results:
                self.metrics.add(result)
                processed_ids.add(result.problem_id)

            completed += len(results)

            # Progress callback
            if progress_callback:
                progress_callback(completed, total)

            # Checkpoint
            summary = self.metrics.compute_summary(
                self.run_config.run_type.value, variant.value
            )
            checkpoint_mgr.save(processed_ids, summary.__dict__)

            # Save results incrementally
            self.metrics.save_results(results_path)

            logger.info(
                f"Progress: {completed}/{total} "
                f"({completed/total*100:.1f}%) - "
                f"Accuracy: {summary.accuracy*100:.1f}%"
            )

        # Final save
        self.metrics.save_results(results_path)
        checkpoint_mgr.clear()  # Remove checkpoint on completion

        return self.metrics

    async def run_all_variants(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, MetricsCollector]:
        """Run on all configured variants."""
        results = {}

        for variant in self.config.variants:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting {variant.value} variant")
            logger.info(f"{'='*60}\n")

            def variant_progress(completed, total):
                if progress_callback:
                    progress_callback(variant.value, completed, total)

            results[variant.value] = await self.run(
                variant,
                progress_callback=variant_progress,
            )

        return results
