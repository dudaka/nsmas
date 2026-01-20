"""
Configuration dataclasses for the NS-MAS agent.

Provides AgentConfig and LLMConfig following the pattern from
data_engineering/pipeline.py.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """
    Configuration for an LLM provider.

    Attributes:
        provider: The LLM provider (OpenAI or Anthropic)
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens in response
        api_key: API key (falls back to environment variable if not set)
    """

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 4096
    api_key: Optional[str] = None

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key

        env_var = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }.get(self.provider)

        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key

        raise ValueError(
            f"No API key found for {self.provider.value}. "
            f"Set {env_var} environment variable or pass api_key to LLMConfig."
        )


@dataclass
class AgentConfig:
    """
    Configuration for the NS-MAS agent.

    Follows the pattern from PipelineConfig in data_engineering/pipeline.py.

    Attributes:
        generator_llm: LLM config for ASP code generation
        reflector_llm: LLM config for reflection (defaults to generator_llm)
        max_retries: Maximum retry attempts before giving up
        enable_entity_extraction: Whether to run entity extraction step
        enable_cycle_detection: Whether to detect repeated code generation
        solver_timeout_ms: Timeout for ASP solver in milliseconds
        load_asp_libraries: Whether to load lib_math.lp, lib_time.lp, ontology.lp
        trace_mode: Tracing mode (CLOUD, LOCAL, or OFF)
        trace_dir: Directory for local traces
        include_ontology_reference: Include ontology in generation prompt
        include_few_shot_examples: Include few-shot examples in prompt
        num_few_shot_examples: Number of few-shot examples to include
    """

    # LLM Configuration
    generator_llm: LLMConfig = field(default_factory=LLMConfig)
    reflector_llm: Optional[LLMConfig] = None

    # Agent Behavior
    max_retries: int = 5
    enable_entity_extraction: bool = True
    enable_cycle_detection: bool = True

    # ASP Solver Configuration
    solver_timeout_ms: int = 30000
    load_asp_libraries: bool = True

    # Tracing Configuration
    trace_mode: Literal["CLOUD", "LOCAL", "OFF"] = "LOCAL"
    trace_dir: str = ".traces"

    # Prompt Configuration
    include_ontology_reference: bool = True
    include_few_shot_examples: bool = True
    num_few_shot_examples: int = 3

    def __post_init__(self):
        """Set defaults for optional configs."""
        if self.reflector_llm is None:
            self.reflector_llm = self.generator_llm

    def get_trace_mode(self) -> str:
        """Get trace mode from config or TRACE_MODE env var."""
        env_mode = os.environ.get("TRACE_MODE")
        if env_mode:
            return env_mode.upper()
        return self.trace_mode
