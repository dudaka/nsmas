"""
Tests for agent configuration.
"""

import os
import pytest

from src.agent.config import AgentConfig, LLMConfig, LLMProvider


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.api_key is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=2048,
            api_key="test-key",
        )
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.api_key == "test-key"

    def test_get_api_key_from_config(self):
        """Test getting API key from config."""
        config = LLMConfig(api_key="my-api-key")
        assert config.get_api_key() == "my-api-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """Test getting API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        config = LLMConfig()
        assert config.get_api_key() == "env-api-key"

    def test_get_api_key_missing(self, monkeypatch):
        """Test error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig()
        with pytest.raises(ValueError, match="No API key found"):
            config.get_api_key()


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.max_retries == 5
        assert config.enable_entity_extraction is True
        assert config.enable_cycle_detection is True
        assert config.solver_timeout_ms == 30000
        assert config.trace_mode == "LOCAL"
        assert config.include_few_shot_examples is True
        assert config.num_few_shot_examples == 3

    def test_reflector_llm_defaults_to_generator(self):
        """Test that reflector_llm defaults to generator_llm."""
        generator = LLMConfig(model="gpt-4o")
        config = AgentConfig(generator_llm=generator)
        assert config.reflector_llm == generator

    def test_reflector_llm_can_be_different(self):
        """Test that reflector_llm can be set independently."""
        generator = LLMConfig(model="gpt-4o")
        reflector = LLMConfig(model="gpt-3.5-turbo")
        config = AgentConfig(generator_llm=generator, reflector_llm=reflector)
        assert config.reflector_llm == reflector
        assert config.reflector_llm != config.generator_llm

    def test_get_trace_mode_from_env(self, monkeypatch):
        """Test getting trace mode from environment."""
        monkeypatch.setenv("TRACE_MODE", "CLOUD")
        config = AgentConfig(trace_mode="LOCAL")
        assert config.get_trace_mode() == "CLOUD"

    def test_get_trace_mode_from_config(self, monkeypatch):
        """Test getting trace mode from config when env not set."""
        monkeypatch.delenv("TRACE_MODE", raising=False)
        config = AgentConfig(trace_mode="OFF")
        assert config.get_trace_mode() == "OFF"
