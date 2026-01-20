"""
Data Loader for Experiment Results

Loads and organizes experiment results from JSONL files.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.experiments.metrics import ExperimentResult


@dataclass
class SystemResults:
    """Results for a single system across all variants."""

    system_name: str
    results_by_variant: Dict[str, List[ExperimentResult]] = field(default_factory=dict)

    @property
    def all_results(self) -> List[ExperimentResult]:
        """Get all results across variants."""
        all_res = []
        for results in self.results_by_variant.values():
            all_res.extend(results)
        return all_res

    @property
    def variants(self) -> List[str]:
        """Get list of variants."""
        return list(self.results_by_variant.keys())

    def get_accuracy(self, variant: Optional[str] = None) -> float:
        """Get accuracy for a variant or overall."""
        if variant:
            results = self.results_by_variant.get(variant, [])
        else:
            results = self.all_results
        if not results:
            return 0.0
        return sum(1 for r in results if r.correct) / len(results)

    def get_correct_mask(self, variant: str) -> List[bool]:
        """Get boolean mask of correct answers for a variant."""
        results = self.results_by_variant.get(variant, [])
        return [r.correct for r in results]

    def get_results_by_problem_id(self, variant: str) -> Dict[str, ExperimentResult]:
        """Get results indexed by problem_id."""
        results = self.results_by_variant.get(variant, [])
        return {r.problem_id: r for r in results}


class DataLoader:
    """Loads experiment results from the results directory."""

    SYSTEM_DIRS = {
        "baseline_gpt4o": "GPT-4o CoT",
        "baseline_gpt4o_mini": "GPT-4o-mini CoT",
        "nsmas_fixed_slow": "NS-MAS Fixed Slow",
        "nsmas_bandit": "NS-MAS Bandit Cold",
        "nsmas_random": "NS-MAS Random",
    }

    VARIANTS = ["base", "p1", "p2", "noop"]

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.systems: Dict[str, SystemResults] = {}

    def load_all(self) -> Dict[str, SystemResults]:
        """Load all experiment results."""
        for system_dir, display_name in self.SYSTEM_DIRS.items():
            system_path = self.results_dir / system_dir
            if system_path.exists():
                self.systems[system_dir] = self._load_system(system_path, display_name)
        return self.systems

    def _load_system(self, system_path: Path, display_name: str) -> SystemResults:
        """Load results for a single system."""
        system_results = SystemResults(system_name=display_name)

        for variant in self.VARIANTS:
            variant_file = system_path / f"{variant}.jsonl"
            if variant_file.exists():
                results = self._load_jsonl(variant_file)
                system_results.results_by_variant[variant] = results

        return system_results

    def _load_jsonl(self, path: Path) -> List[ExperimentResult]:
        """Load results from JSONL file."""
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(ExperimentResult.from_dict(data))
        return results

    def get_system(self, system_key: str) -> Optional[SystemResults]:
        """Get results for a specific system."""
        return self.systems.get(system_key)

    def get_paired_results(
        self, system1_key: str, system2_key: str, variant: str
    ) -> Tuple[List[ExperimentResult], List[ExperimentResult]]:
        """Get paired results for two systems on the same variant.

        Returns results aligned by problem_id.
        """
        sys1 = self.systems.get(system1_key)
        sys2 = self.systems.get(system2_key)

        if not sys1 or not sys2:
            return [], []

        results1_by_id = sys1.get_results_by_problem_id(variant)
        results2_by_id = sys2.get_results_by_problem_id(variant)

        # Find common problem IDs
        common_ids = set(results1_by_id.keys()) & set(results2_by_id.keys())
        common_ids = sorted(common_ids)

        paired1 = [results1_by_id[pid] for pid in common_ids]
        paired2 = [results2_by_id[pid] for pid in common_ids]

        return paired1, paired2

    def get_all_systems(self) -> List[str]:
        """Get list of loaded system keys."""
        return list(self.systems.keys())

    def summary(self) -> str:
        """Generate a summary of loaded data."""
        lines = ["Loaded Experiment Data Summary", "=" * 40]

        total_records = 0
        for sys_key, sys_results in self.systems.items():
            display_name = self.SYSTEM_DIRS.get(sys_key, sys_key)
            lines.append(f"\n{display_name}:")

            for variant in self.VARIANTS:
                if variant in sys_results.results_by_variant:
                    results = sys_results.results_by_variant[variant]
                    count = len(results)
                    acc = sys_results.get_accuracy(variant)
                    lines.append(f"  {variant}: {count} records, {acc:.2%} accuracy")
                    total_records += count

            overall_acc = sys_results.get_accuracy()
            lines.append(f"  Overall: {overall_acc:.2%} accuracy")

        lines.append(f"\nTotal records: {total_records}")
        return "\n".join(lines)
