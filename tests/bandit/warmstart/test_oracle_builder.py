"""
Tests for OracleBuilder.
"""

import json
import pytest

from src.bandit.warmstart.oracle_builder import OracleBuilder, OracleLabel, OracleStats


class TestOracleLabel:
    """Test OracleLabel dataclass."""

    def test_create_oracle_label(self):
        """Test creating an oracle label."""
        label = OracleLabel(
            problem_id="gsm_001",
            question="Test question?",
            ground_truth=42,
            variant="base",
            fast_answer=42,
            fast_correct=True,
            slow_answer=42,
            slow_correct=True,
            oracle_action="fast",
            oracle_reason="both_correct",
        )

        assert label.problem_id == "gsm_001"
        assert label.oracle_action == "fast"
        assert label.oracle_reason == "both_correct"

    def test_to_dict(self, sample_oracle_labels):
        """Test converting label to dictionary."""
        label = sample_oracle_labels[0]
        d = label.to_dict()

        assert d["problem_id"] == "gsm_001"
        assert d["oracle_action"] == "fast"
        assert isinstance(d, dict)

    def test_from_dict(self):
        """Test creating label from dictionary."""
        d = {
            "problem_id": "gsm_001",
            "question": "Test?",
            "ground_truth": 42,
            "variant": "base",
            "fast_answer": 42,
            "fast_correct": True,
            "slow_answer": 42,
            "slow_correct": True,
            "oracle_action": "fast",
            "oracle_reason": "both_correct",
        }

        label = OracleLabel.from_dict(d)

        assert label.problem_id == "gsm_001"
        assert label.oracle_action == "fast"

    def test_roundtrip_dict(self, sample_oracle_labels):
        """Test dict conversion roundtrip."""
        original = sample_oracle_labels[0]
        d = original.to_dict()
        restored = OracleLabel.from_dict(d)

        assert original.problem_id == restored.problem_id
        assert original.oracle_action == restored.oracle_action
        assert original.oracle_reason == restored.oracle_reason


class TestOracleStats:
    """Test OracleStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        stats = OracleStats()

        assert stats.total == 0
        assert stats.both_correct == 0
        assert stats.fast_rate == 0.0
        assert stats.oracle_accuracy == 0.0

    def test_fast_rate_calculation(self):
        """Test fast rate calculation."""
        stats = OracleStats(
            total=100,
            both_correct=60,  # -> fast
            only_slow_correct=10,  # -> slow
            only_fast_correct=15,  # -> fast
            both_wrong=15,  # -> fast
        )

        # Fast rate = (both_correct + only_fast_correct + both_wrong) / total
        # = (60 + 15 + 15) / 100 = 0.90
        assert stats.fast_rate == 0.90

    def test_oracle_accuracy_calculation(self):
        """Test oracle accuracy calculation."""
        stats = OracleStats(
            total=100,
            both_correct=60,
            only_slow_correct=10,
            only_fast_correct=15,
            both_wrong=15,
        )

        # Oracle accuracy = (both_correct + only_slow_correct + only_fast_correct) / total
        # = (60 + 10 + 15) / 100 = 0.85
        assert stats.oracle_accuracy == 0.85

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = OracleStats(total=100, both_correct=60)
        d = stats.to_dict()

        assert d["total"] == 100
        assert d["both_correct"] == 60
        assert "fast_rate" in d
        assert "oracle_accuracy" in d


class TestOracleBuilder:
    """Test OracleBuilder class."""

    def test_init_default_variants(self):
        """Test initialization with default variants."""
        builder = OracleBuilder()

        assert builder.variants == ["base", "p1", "p2", "noop"]

    def test_init_custom_variants(self):
        """Test initialization with custom variants."""
        builder = OracleBuilder(variants=["base", "noop"])

        assert builder.variants == ["base", "noop"]

    def test_determine_oracle_action_both_correct(self):
        """Test oracle decision: both correct -> fast."""
        builder = OracleBuilder()
        action, reason = builder.determine_oracle_action(
            fast_correct=True, slow_correct=True
        )

        assert action == "fast"
        assert reason == "both_correct"

    def test_determine_oracle_action_only_slow_correct(self):
        """Test oracle decision: only slow correct -> slow."""
        builder = OracleBuilder()
        action, reason = builder.determine_oracle_action(
            fast_correct=False, slow_correct=True
        )

        assert action == "slow"
        assert reason == "only_slow_correct"

    def test_determine_oracle_action_only_fast_correct(self):
        """Test oracle decision: only fast correct -> fast."""
        builder = OracleBuilder()
        action, reason = builder.determine_oracle_action(
            fast_correct=True, slow_correct=False
        )

        assert action == "fast"
        assert reason == "only_fast_correct"

    def test_determine_oracle_action_both_wrong(self):
        """Test oracle decision: both wrong -> fast (minimize cost)."""
        builder = OracleBuilder()
        action, reason = builder.determine_oracle_action(
            fast_correct=False, slow_correct=False
        )

        assert action == "fast"
        assert reason == "both_wrong"

    def test_load_results(self, sample_results_dirs):
        """Test loading results from JSONL file."""
        fast_dir, _ = sample_results_dirs
        builder = OracleBuilder()

        results = builder.load_results(fast_dir, "base")

        assert len(results) == 4
        assert "gsm_001" in results
        assert results["gsm_001"]["correct"] is True

    def test_load_results_missing_file(self, temp_dir):
        """Test loading non-existent results file."""
        builder = OracleBuilder()

        results = builder.load_results(temp_dir, "missing")

        assert results == {}

    def test_build_oracle_for_variant(self, sample_results_dirs):
        """Test building oracle labels for single variant."""
        fast_dir, slow_dir = sample_results_dirs
        builder = OracleBuilder()

        fast_results = builder.load_results(fast_dir, "base")
        slow_results = builder.load_results(slow_dir, "base")

        labels, stats = builder.build_oracle_for_variant(
            fast_results, slow_results, "base"
        )

        assert len(labels) == 4
        assert stats.total == 4
        assert stats.both_correct == 1  # gsm_001
        assert stats.only_slow_correct == 1  # gsm_002
        assert stats.only_fast_correct == 1  # gsm_003
        assert stats.both_wrong == 1  # gsm_004

    def test_build_oracle(self, sample_results_dirs):
        """Test building complete oracle labels."""
        fast_dir, slow_dir = sample_results_dirs
        builder = OracleBuilder(variants=["base"])

        labels, stats = builder.build_oracle(fast_dir, slow_dir)

        assert len(labels) == 4
        assert "base" in stats
        assert "overall" in stats

    def test_save_and_load(self, sample_results_dirs, temp_dir):
        """Test saving and loading oracle labels."""
        fast_dir, slow_dir = sample_results_dirs
        builder = OracleBuilder(variants=["base"])

        labels, stats = builder.build_oracle(fast_dir, slow_dir)

        # Save
        output_path = temp_dir / "oracle_labels.json"
        builder.save(labels, output_path, stats)

        assert output_path.exists()

        # Load
        loaded_labels, loaded_stats = OracleBuilder.load(output_path)

        assert len(loaded_labels) == len(labels)
        assert loaded_labels[0].problem_id == labels[0].problem_id

    def test_save_creates_directories(self, temp_dir):
        """Test that save creates parent directories."""
        builder = OracleBuilder()
        labels = []
        output_path = temp_dir / "nested" / "path" / "labels.json"

        builder.save(labels, output_path)

        assert output_path.exists()

    def test_oracle_stats_match_labels(self, sample_results_dirs):
        """Test that stats accurately reflect label distribution."""
        fast_dir, slow_dir = sample_results_dirs
        builder = OracleBuilder(variants=["base"])

        labels, stats = builder.build_oracle(fast_dir, slow_dir)

        # Count labels by oracle_action
        fast_count = sum(1 for l in labels if l.oracle_action == "fast")
        slow_count = sum(1 for l in labels if l.oracle_action == "slow")

        # Verify consistency
        overall = stats["overall"]
        expected_fast = overall.both_correct + overall.only_fast_correct + overall.both_wrong
        expected_slow = overall.only_slow_correct

        assert fast_count == expected_fast
        assert slow_count == expected_slow
