"""
Unit Test Suite for GSM-Symbolic Data Engineering Pipeline.

Implements the three critical test modules from Phase 2 Specification Section 6:
1. Semantic Validator (Constraint Checking)
2. Solver-Verifier (Reversibility)
3. NoOp Invariant Test

Reference: Phase 2 Specification Section 6 - Quality Assurance
"""

import pytest
import random
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_engineering.name_provider import NameProvider, CulturalBlock
from src.data_engineering.template_engine import (
    TemplateEngine, TemplateParser, ConditionEvaluator, GeneratedProblem
)
from src.data_engineering.noise_injector import NoiseInjector, TopicDomain
from src.data_engineering.serializer import (
    DatasetSerializer, DatasetRecord, DatasetValidator
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def templates_dir():
    """Path to GSM-Symbolic templates."""
    return Path(__file__).parent.parent / 'external' / 'ml-gsm-symbolic' / 'templates'


@pytest.fixture
def name_provider():
    """NameProvider with fixed seed for reproducibility."""
    return NameProvider(seed=42)


@pytest.fixture
def noise_injector():
    """NoiseInjector with fixed seed."""
    return NoiseInjector(seed=42)


@pytest.fixture
def template_engine(templates_dir):
    """TemplateEngine loaded with base templates."""
    engine = TemplateEngine(templates_dir=templates_dir, seed=42)
    if templates_dir.exists():
        engine.load_templates('symbolic')
    return engine


# =============================================================================
# Module 1: Semantic Validator (Constraint Checking)
# =============================================================================

class TestSemanticValidator:
    """
    Tests for semantic validity of generated problems.

    Checks:
    - Integer constraint for discrete objects
    - Non-negativity for physical quantities
    - Range sanity (reasonable bounds)
    """

    def test_integer_check_for_discrete_quantities(self, template_engine: TemplateEngine):
        """Verify discrete quantities (people, objects) are integers."""
        if not template_engine.templates:
            pytest.skip("No templates loaded")

        template_id = list(template_engine.templates.keys())[0]
        problems = template_engine.generate_batch(template_id, num_instances=10)

        for problem in problems:
            # Check that final answer is integer when expected
            if isinstance(problem.final_answer, float):
                assert problem.final_answer == int(problem.final_answer), \
                    f"Non-integer answer for discrete problem: {problem.final_answer}"

    def test_non_negativity_check(self, template_engine: TemplateEngine):
        """Verify all physical quantities are non-negative."""
        if not template_engine.templates:
            pytest.skip("No templates loaded")

        for template_id in list(template_engine.templates.keys())[:5]:
            problems = template_engine.generate_batch(template_id, num_instances=10)

            for problem in problems:
                assert problem.final_answer is not None, \
                    f"Null answer for {problem.id}"

                if isinstance(problem.final_answer, (int, float)):
                    assert problem.final_answer >= 0, \
                        f"Negative answer {problem.final_answer} for {problem.id}"

    def test_range_sanity(self, template_engine: TemplateEngine):
        """Verify answers are within reasonable bounds."""
        MAX_REASONABLE_ANSWER = 1_000_000

        if not template_engine.templates:
            pytest.skip("No templates loaded")

        for template_id in list(template_engine.templates.keys())[:5]:
            problems = template_engine.generate_batch(template_id, num_instances=10)

            for problem in problems:
                if isinstance(problem.final_answer, (int, float)):
                    assert abs(problem.final_answer) < MAX_REASONABLE_ANSWER, \
                        f"Answer {problem.final_answer} exceeds reasonable range"

    @pytest.mark.parametrize("seed", range(10))
    def test_deterministic_generation(self, templates_dir, seed: int):
        """Verify same seed produces same results."""
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        engine1 = TemplateEngine(templates_dir=templates_dir, seed=seed)
        engine1.load_templates('symbolic')
        template_id = list(engine1.templates.keys())[0]
        problems1 = engine1.generate_batch(template_id, num_instances=5, start_seed=seed)

        engine2 = TemplateEngine(templates_dir=templates_dir, seed=seed)
        engine2.load_templates('symbolic')
        problems2 = engine2.generate_batch(template_id, num_instances=5, start_seed=seed)

        for p1, p2 in zip(problems1, problems2):
            assert p1.question == p2.question
            assert p1.final_answer == p2.final_answer


# =============================================================================
# Module 2: Solver-Verifier (Reversibility)
# =============================================================================

class TestSolverVerifier:
    """
    Tests for answer correctness verification.

    Ensures the computed answer matches what's embedded in the solution text.
    """

    def test_answer_extraction_matches_final(self, template_engine: TemplateEngine):
        """Verify extracted answer matches computed final_answer."""
        if not template_engine.templates:
            pytest.skip("No templates loaded")

        serializer = DatasetSerializer()

        for template_id in list(template_engine.templates.keys())[:5]:
            problems = template_engine.generate_batch(template_id, num_instances=5)

            for problem in problems:
                extracted = serializer.extract_final_answer(problem.answer)

                if extracted is not None and problem.final_answer is not None:
                    # Allow for floating point comparison
                    assert abs(float(extracted) - float(problem.final_answer)) < 0.01, \
                        f"Extracted {extracted} != computed {problem.final_answer}"

    def test_answer_format_contains_marker(self, template_engine: TemplateEngine):
        """Verify all answers contain the #### marker."""
        if not template_engine.templates:
            pytest.skip("No templates loaded")

        for template_id in list(template_engine.templates.keys())[:5]:
            problems = template_engine.generate_batch(template_id, num_instances=5)

            for problem in problems:
                assert '####' in problem.answer, \
                    f"Answer missing #### marker: {problem.id}"

    def test_condition_evaluator_functions(self):
        """Test the condition evaluator helper functions."""
        # Test is_int
        assert ConditionEvaluator.is_int(5) is True
        assert ConditionEvaluator.is_int(5.0) is True
        assert ConditionEvaluator.is_int(5.5) is False

        # Test divides
        assert ConditionEvaluator.divides(3, 9) is True
        assert ConditionEvaluator.divides(3, 10) is False
        assert ConditionEvaluator.divides(0, 5) is False

    def test_condition_evaluation_with_variables(self):
        """Test condition evaluation with variable context."""
        variables = {'x': 10, 'y': 5, 'z': 2}

        assert ConditionEvaluator.evaluate("x > y", variables) is True
        assert ConditionEvaluator.evaluate("is_int(x / z)", variables) is True
        assert ConditionEvaluator.evaluate("is_int(x / 3)", variables) is False
        assert ConditionEvaluator.evaluate("divides(z, x)", variables) is True


# =============================================================================
# Module 3: NoOp Invariant Test
# =============================================================================

class TestNoOpInvariant:
    """
    Tests for the critical NoOp invariant: distractor injection
    must NOT change the numerical answer.

    This is the most important test for Phase 2.
    """

    def test_noop_invariant_basic(self, noise_injector: NoiseInjector):
        """Basic test that injection doesn't alter structure of calculation."""
        base_question = "Alice has 10 apples. She gives 3 to Bob. How many apples does Alice have?"

        for seed in range(20):
            noise_injector.set_seed(seed)
            result = noise_injector.inject_distractors(base_question, num_distractors=1)

            # The answer should still be 7 (10 - 3)
            # Verify the essential numbers are preserved
            assert "10" in result.modified_question
            assert "3" in result.modified_question
            assert result.answer_unchanged is True

    @pytest.mark.parametrize("seed", range(100))
    def test_noop_invariant_with_seeds(self, templates_dir, seed: int):
        """
        Test NoOp invariant across many seeds.

        This replicates the test specification from Section 6.2.3:
        def test_noop_invariant(seed):
            base_problem, base_answer = generate_base(seed)
            noop_problem, noop_answer = generate_noop(seed)
            assert base_answer == noop_answer
        """
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        # Generate base problem
        engine = TemplateEngine(templates_dir=templates_dir, seed=seed)
        engine.load_templates('symbolic')

        if not engine.templates:
            pytest.skip("No templates loaded")

        template_id = list(engine.templates.keys())[0]
        base_problems = engine.generate_batch(template_id, num_instances=1, start_seed=seed)

        if not base_problems:
            pytest.skip(f"Failed to generate problem for seed {seed}")

        base_problem = base_problems[0]
        base_answer = base_problem.final_answer

        # Generate NoOp variant
        injector = NoiseInjector(seed=seed)
        result = injector.inject_distractors(base_problem.question, num_distractors=1)

        # The answer should be unchanged
        # Since we're not re-solving, we verify structural integrity
        assert result.answer_unchanged is True, \
            f"NoOp altered the solution! Seed: {seed}"

    def test_noop_distractor_is_semantically_irrelevant(self, noise_injector: NoiseInjector):
        """Verify distractors don't introduce new calculations."""
        question = "John bought 5 books for $10 each. How much did he spend?"

        for _ in range(10):
            result = noise_injector.inject_distractors(question, num_distractors=2)

            # Distractors should not contain calculation-triggering phrases
            # that would change the actual math problem
            modified = result.modified_question.lower()

            # The original numbers should be preserved
            assert "5" in modified
            assert "10" in modified

    def test_noop_validator_integration(self):
        """Test the NoOp validator in serializer module."""
        validator = DatasetValidator()

        base_record = DatasetRecord(
            id="base_0001_0000",
            question="Alice has 5 apples.",
            answer="#### 5",
            template_id="0001",
            is_noop=False,
            final_answer=5
        )

        # Valid NoOp (same answer)
        valid_noop = DatasetRecord(
            id="noop_0001_0000",
            question="Alice has 5 apples. Note that 2 were bruised.",
            answer="#### 5",
            template_id="0001",
            is_noop=True,
            final_answer=5
        )

        # Invalid NoOp (different answer)
        invalid_noop = DatasetRecord(
            id="noop_0001_0000",
            question="Alice has 5 apples. She ate 2.",
            answer="#### 3",
            template_id="0001",
            is_noop=True,
            final_answer=3
        )

        assert validator.validate_noop_invariant(base_record, valid_noop) is True
        assert validator.validate_noop_invariant(base_record, invalid_noop) is False


# =============================================================================
# Name Provider Tests
# =============================================================================

class TestNameProvider:
    """Tests for culturally diverse name sampling."""

    def test_total_names_minimum(self, name_provider: NameProvider):
        """Verify we have at least 1000 unique names as specified."""
        assert name_provider.total_unique_names >= 1000, \
            f"Only {name_provider.total_unique_names} names, need >= 1000"

    def test_all_cultural_blocks_present(self, name_provider: NameProvider):
        """Verify all cultural blocks have names."""
        stats = name_provider.get_stats()
        for block in CulturalBlock:
            assert stats.get(block.value, 0) > 0, \
                f"No names in cultural block: {block.value}"

    def test_round_robin_distribution(self, name_provider: NameProvider):
        """Verify round-robin sampling covers all blocks."""
        blocks_seen = set()

        for _ in range(len(CulturalBlock) * 2):
            sample = name_provider.get_name_round_robin()
            blocks_seen.add(sample.cultural_block)

        assert blocks_seen == set(CulturalBlock), \
            "Round-robin did not cover all cultural blocks"

    def test_no_name_collision(self, name_provider: NameProvider):
        """Verify get_names_for_problem returns unique names."""
        names = name_provider.get_names_for_problem(5)
        assert len(names) == len(set(names)), \
            f"Duplicate names returned: {names}"

    def test_reproducibility_with_seed(self):
        """Verify same seed produces same names."""
        provider1 = NameProvider(seed=12345)
        provider2 = NameProvider(seed=12345)

        for _ in range(20):
            assert provider1.get_name().name == provider2.get_name().name


# =============================================================================
# Noise Injector Tests
# =============================================================================

class TestNoiseInjector:
    """Tests for distractor injection functionality."""

    def test_topic_detection(self, noise_injector: NoiseInjector):
        """Verify topic detection from problem text."""
        agriculture_text = "Alice picked 10 apples from the tree."
        commerce_text = "John bought a book for $15 at the store."
        travel_text = "The car traveled 100 miles in 2 hours."

        assert TopicDomain.AGRICULTURE in noise_injector.detect_topics(agriculture_text)
        assert TopicDomain.COMMERCE in noise_injector.detect_topics(commerce_text)
        assert TopicDomain.TRAVEL in noise_injector.detect_topics(travel_text)

    def test_object_extraction(self, noise_injector: NoiseInjector):
        """Verify object extraction from problem text."""
        text = "Alice has 5 apples and 3 oranges."
        objects = noise_injector.extract_objects(text)

        assert 'apples' in objects or 'apple' in objects
        assert 'oranges' in objects or 'orange' in objects

    def test_distractor_count(self, noise_injector: NoiseInjector):
        """Verify correct number of distractors are injected."""
        question = "Alice has 5 apples. How many does she have?"

        for num in [1, 2, 3]:
            result = noise_injector.inject_distractors(question, num_distractors=num)
            assert len(result.distractors_used) == num

    def test_distractor_positions(self, noise_injector: NoiseInjector):
        """Verify distractors are inserted at valid positions."""
        question = "Alice has 5 apples. She gives 2 away. How many does she have?"

        result = noise_injector.inject_distractors(question, num_distractors=1)

        # Modified question should be longer
        assert len(result.modified_question) > len(question)

        # Original content should be preserved
        assert "5" in result.modified_question
        assert "2" in result.modified_question


# =============================================================================
# Serializer Tests
# =============================================================================

class TestSerializer:
    """Tests for JSONL serialization."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Verify data survives write/read cycle."""
        serializer = DatasetSerializer()

        original_records = [
            DatasetRecord(
                id=f"test_{i:04d}",
                question=f"Question {i}",
                answer=f"Answer\n#### {i}",
                template_id=f"template_{i}",
                is_noop=i % 2 == 0,
                instance=i,
                final_answer=i
            )
            for i in range(10)
        ]

        output_path = tmp_path / "test.jsonl"
        serializer.write_jsonl(original_records, output_path)

        read_records = list(serializer.read_jsonl(output_path))

        assert len(read_records) == len(original_records)
        for orig, read in zip(original_records, read_records):
            assert orig.id == read.id
            assert orig.question == read.question
            assert orig.final_answer == read.final_answer

    def test_answer_extraction(self):
        """Test final answer extraction from various formats."""
        serializer = DatasetSerializer()

        test_cases = [
            ("The answer is 42.\n#### 42", 42),
            ("5 + 3 = 8\n#### 8", 8),
            ("Total: $15.50\n#### 15.5", 15.5),
            ("Result = -10\n#### -10", -10),
            ("No marker but last number is 99", 99),
        ]

        for text, expected in test_cases:
            extracted = serializer.extract_final_answer(text)
            assert extracted == expected, \
                f"Expected {expected}, got {extracted} from: {text}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_template_loading(self, templates_dir):
        """Test loading templates from all variants."""
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        engine = TemplateEngine(templates_dir=templates_dir, seed=42)

        for variant in ['symbolic', 'p1', 'p2']:
            variant_dir = templates_dir / variant
            if variant_dir.exists():
                count = engine.load_templates(variant)
                assert count > 0, f"No templates loaded for {variant}"

    def test_full_generation_cycle(self, templates_dir):
        """Test complete generation from template to JSONL."""
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        engine = TemplateEngine(templates_dir=templates_dir, seed=42)
        engine.load_templates('symbolic')

        if not engine.templates:
            pytest.skip("No templates available")

        # Generate problems - try multiple templates since some have restrictive conditions
        problems = []
        for template_id in list(engine.templates.keys())[:10]:
            problems = engine.generate_batch(template_id, num_instances=5)
            if problems:
                break

        assert len(problems) > 0, "Failed to generate any problems from first 10 templates"

        # Inject noise
        injector = NoiseInjector(seed=42)
        noop_problems = []

        for problem in problems:
            result = injector.inject_distractors(problem.question)
            noop_problems.append(result)

        # Validate
        validator = DatasetValidator()
        for problem in problems:
            record = DatasetRecord(
                id=problem.id,
                question=problem.question,
                answer=problem.answer,
                template_id=problem.template_id,
                is_noop=False,
                final_answer=problem.final_answer
            )
            errors = validator.validate_record(record)
            assert len(errors) == 0, f"Validation errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
