"""
GSM-Symbolic Dataset Generation Pipeline

Main orchestration script for Phase 2 data engineering.
Generates Base, P1, P2, and NoOp dataset variants.

Reference: Phase 2 Specification - Operational Roadmap Section 7
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .template_engine import TemplateEngine, GeneratedProblem
from .noise_injector import NoiseInjector, TopicDomain
from .name_provider import NameProvider
from .serializer import DatasetSerializer, DatasetRecord, DatasetValidator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the generation pipeline."""
    templates_dir: Path
    output_dir: Path
    instances_per_template: int = 50
    num_distractors: int = 1  # For NoOp variant
    base_seed: int = 42
    variants: List[str] = None  # ['base', 'p1', 'p2', 'noop']
    validate: bool = True
    verbose: bool = False

    def __post_init__(self):
        if self.variants is None:
            self.variants = ['base', 'p1', 'p2', 'noop']


@dataclass
class GenerationStats:
    """Statistics for a generation run."""
    variant: str
    templates_loaded: int
    instances_requested: int
    instances_generated: int
    instances_failed: int
    validation_errors: int
    duration_seconds: float


class GSMSymbolicPipeline:
    """
    Main pipeline for generating GSM-Symbolic datasets.

    Coordinates template loading, instance generation, NoOp injection,
    and JSONL serialization.
    """

    # Mapping of variant names to template directories
    VARIANT_DIRS = {
        'base': 'symbolic',
        'p1': 'p1',
        'p2': 'p2',
    }

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.template_engine = TemplateEngine(
            templates_dir=config.templates_dir,
            seed=config.base_seed
        )
        self.noise_injector = NoiseInjector(seed=config.base_seed)
        self.serializer = DatasetSerializer()
        self.validator = DatasetValidator()
        self.stats: Dict[str, GenerationStats] = {}

    def run(self) -> Dict[str, GenerationStats]:
        """
        Execute the full pipeline.

        Returns:
            Dictionary of generation statistics per variant
        """
        logger.info("=" * 60)
        logger.info("GSM-Symbolic Dataset Generation Pipeline")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  Templates directory: {self.config.templates_dir}")
        logger.info(f"  Output directory: {self.config.output_dir}")
        logger.info(f"  Instances per template: {self.config.instances_per_template}")
        logger.info(f"  Variants to generate: {self.config.variants}")
        logger.info(f"  Base seed: {self.config.base_seed}")
        logger.info("=" * 60)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate each variant
        for variant in self.config.variants:
            if variant == 'noop':
                # NoOp is derived from base
                self._generate_noop_variant()
            else:
                self._generate_standard_variant(variant)

        # Write summary
        self._write_summary()

        return self.stats

    def _generate_standard_variant(self, variant: str) -> GenerationStats:
        """Generate a standard variant (base, p1, or p2)."""
        start_time = datetime.now()
        logger.info(f"\n--- Generating {variant.upper()} variant ---")

        # Map variant to directory
        template_dir = self.VARIANT_DIRS.get(variant, variant)

        # Load templates
        try:
            num_templates = self.template_engine.load_templates(template_dir)
            logger.info(f"Loaded {num_templates} templates for {variant}")
        except Exception as e:
            logger.error(f"Failed to load templates for {variant}: {e}")
            return self._record_stats(variant, 0, 0, 0, 0, 0, start_time)

        # Generate instances
        all_records: List[DatasetRecord] = []
        instances_failed = 0

        template_ids = self.template_engine.get_all_template_ids()
        total_to_generate = len(template_ids) * self.config.instances_per_template

        for i, template_id in enumerate(template_ids):
            if self.config.verbose:
                logger.info(f"  Processing template {i+1}/{len(template_ids)}: {template_id}")

            # Generate instances for this template
            problems = self.template_engine.generate_batch(
                template_id,
                num_instances=self.config.instances_per_template,
                start_seed=self.config.base_seed + i * 1000
            )

            # Convert to records
            for problem in problems:
                record = self._problem_to_record(problem, variant)
                all_records.append(record)

            instances_failed += self.config.instances_per_template - len(problems)

        # Validate if enabled
        validation_errors = 0
        if self.config.validate:
            report = self.validator.validate_dataset(all_records)
            validation_errors = report['invalid_records']
            if validation_errors > 0:
                logger.warning(f"Validation found {validation_errors} invalid records")

        # Write to JSONL
        output_path = self.config.output_dir / f"gsm_{variant}.jsonl"
        self.serializer.write_jsonl(all_records, output_path)
        logger.info(f"Wrote {len(all_records)} records to {output_path}")

        # Write metadata
        self.serializer.write_metadata(
            self.config.output_dir,
            variant,
            {
                'templates': num_templates,
                'instances_per_template': self.config.instances_per_template,
                'total_records': len(all_records),
                'failed_generations': instances_failed,
                'validation_errors': validation_errors,
            }
        )

        return self._record_stats(
            variant, num_templates, total_to_generate,
            len(all_records), instances_failed, validation_errors, start_time
        )

    def _generate_noop_variant(self) -> GenerationStats:
        """Generate NoOp variant by injecting distractors into base problems."""
        start_time = datetime.now()
        logger.info(f"\n--- Generating NOOP variant ---")

        # Check if base exists
        base_path = self.config.output_dir / "gsm_base.jsonl"
        if not base_path.exists():
            logger.warning("Base variant not found. Generating base first...")
            self._generate_standard_variant('base')

        # Read base records
        base_records = list(self.serializer.read_jsonl(base_path))
        logger.info(f"Loaded {len(base_records)} base records")

        # Inject distractors
        noop_records: List[DatasetRecord] = []
        validation_errors = 0

        for i, base_record in enumerate(base_records):
            # Set seed for reproducibility
            self.noise_injector.set_seed(self.config.base_seed + i)

            # Inject distractors
            injected = self.noise_injector.inject_distractors(
                base_record.question,
                num_distractors=self.config.num_distractors
            )

            # Create NoOp record
            noop_record = DatasetRecord(
                id=f"noop_{base_record.template_id}_{base_record.instance:04d}",
                question=injected.modified_question,
                answer=base_record.answer,
                template_id=base_record.template_id,
                is_noop=True,
                instance=base_record.instance,
                final_answer=base_record.final_answer,
                original_question=base_record.question,
                original_answer=base_record.answer,
                variant='noop'
            )

            # Validate NoOp invariant
            if self.config.validate:
                if not self.validator.validate_noop_invariant(base_record, noop_record):
                    logger.error(f"NoOp invariant violation at {noop_record.id}")
                    validation_errors += 1

            noop_records.append(noop_record)

        # Write to JSONL
        output_path = self.config.output_dir / "gsm_noop.jsonl"
        self.serializer.write_jsonl(noop_records, output_path)
        logger.info(f"Wrote {len(noop_records)} NoOp records to {output_path}")

        # Write metadata
        self.serializer.write_metadata(
            self.config.output_dir,
            'noop',
            {
                'base_records': len(base_records),
                'noop_records': len(noop_records),
                'distractors_per_problem': self.config.num_distractors,
                'validation_errors': validation_errors,
            }
        )

        return self._record_stats(
            'noop', len(base_records), len(base_records),
            len(noop_records), 0, validation_errors, start_time
        )

    def _problem_to_record(self, problem: GeneratedProblem, variant: str) -> DatasetRecord:
        """Convert a GeneratedProblem to a DatasetRecord."""
        return DatasetRecord(
            id=f"{variant}_{problem.template_id}_{problem.instance:04d}",
            question=problem.question,
            answer=problem.answer,
            template_id=problem.template_id,
            is_noop=problem.is_noop,
            instance=problem.instance,
            final_answer=problem.final_answer,
            original_question=problem.original_question,
            original_answer=problem.original_answer,
            variant=variant
        )

    def _record_stats(self, variant: str, templates: int, requested: int,
                      generated: int, failed: int, errors: int,
                      start_time: datetime) -> GenerationStats:
        """Record statistics for a generation run."""
        duration = (datetime.now() - start_time).total_seconds()
        stats = GenerationStats(
            variant=variant,
            templates_loaded=templates,
            instances_requested=requested,
            instances_generated=generated,
            instances_failed=failed,
            validation_errors=errors,
            duration_seconds=duration
        )
        self.stats[variant] = stats
        return stats

    def _write_summary(self):
        """Write pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)

        total_generated = 0
        total_failed = 0
        total_errors = 0

        for variant, stats in self.stats.items():
            logger.info(f"\n{variant.upper()}:")
            logger.info(f"  Templates: {stats.templates_loaded}")
            logger.info(f"  Generated: {stats.instances_generated}/{stats.instances_requested}")
            logger.info(f"  Failed: {stats.instances_failed}")
            logger.info(f"  Validation Errors: {stats.validation_errors}")
            logger.info(f"  Duration: {stats.duration_seconds:.2f}s")

            total_generated += stats.instances_generated
            total_failed += stats.instances_failed
            total_errors += stats.validation_errors

        logger.info("\n" + "-" * 40)
        logger.info(f"TOTAL GENERATED: {total_generated}")
        logger.info(f"TOTAL FAILED: {total_failed}")
        logger.info(f"TOTAL ERRORS: {total_errors}")
        logger.info("=" * 60)

        # Write summary JSON
        summary_path = self.config.output_dir / "pipeline_summary.json"
        summary = {
            'generated_at': datetime.utcnow().isoformat(),
            'config': {
                'instances_per_template': self.config.instances_per_template,
                'base_seed': self.config.base_seed,
                'variants': self.config.variants,
            },
            'results': {
                variant: {
                    'templates': s.templates_loaded,
                    'generated': s.instances_generated,
                    'failed': s.instances_failed,
                    'errors': s.validation_errors,
                    'duration_seconds': s.duration_seconds,
                }
                for variant, s in self.stats.items()
            },
            'totals': {
                'generated': total_generated,
                'failed': total_failed,
                'errors': total_errors,
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate GSM-Symbolic dataset variants'
    )
    parser.add_argument(
        '--templates-dir', '-t',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'external' / 'ml-gsm-symbolic' / 'templates',
        help='Directory containing GSM-Symbolic templates'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'generated',
        help='Output directory for generated datasets'
    )
    parser.add_argument(
        '--instances', '-n',
        type=int,
        default=50,
        help='Number of instances per template (default: 50)'
    )
    parser.add_argument(
        '--variants', '-v',
        nargs='+',
        default=['base', 'p1', 'p2', 'noop'],
        help='Variants to generate (default: all)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Base random seed (default: 42)'
    )
    parser.add_argument(
        '--distractors', '-d',
        type=int,
        default=1,
        help='Number of distractors for NoOp variant (default: 1)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation checks'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    config = PipelineConfig(
        templates_dir=args.templates_dir,
        output_dir=args.output_dir,
        instances_per_template=args.instances,
        num_distractors=args.distractors,
        base_seed=args.seed,
        variants=args.variants,
        validate=not args.no_validate,
        verbose=args.verbose
    )

    pipeline = GSMSymbolicPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
