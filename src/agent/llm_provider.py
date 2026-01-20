"""
LLM Provider abstraction for the NS-MAS agent.

Implements a Protocol-based interface with concrete implementations
for OpenAI and Anthropic, enabling configurable model backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, runtime_checkable

from .config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol defining the LLM provider interface."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System message setting context
            user_prompt: User message with the task
            temperature: Override for sampling temperature
            max_tokens: Override for max response tokens

        Returns:
            Generated text response
        """
        ...


class BaseLLMProvider(ABC):
    """Base class for LLM providers with common functionality."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the provider.

        Args:
            config: LLM configuration with model and API settings
        """
        self.config = config
        self._client = None

    @abstractmethod
    def _create_client(self):
        """Create the underlying API client. Called lazily on first use."""
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @property
    def client(self):
        """Lazy initialization of the API client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def _create_client(self):
        """Create OpenAI client."""
        from openai import OpenAI

        api_key = self.config.get_api_key()
        return OpenAI(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using OpenAI API.

        Args:
            system_prompt: System message setting context
            user_prompt: User message with the task
            temperature: Override for sampling temperature
            max_tokens: Override for max response tokens

        Returns:
            Generated text response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        logger.debug(f"OpenAI request: model={self.config.model}, temp={temp}")

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )

        content = response.choices[0].message.content
        logger.debug(f"OpenAI response: {len(content)} chars")

        return content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""

    def _create_client(self):
        """Create Anthropic client."""
        from anthropic import Anthropic

        api_key = self.config.get_api_key()
        return Anthropic(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using Anthropic API.

        Args:
            system_prompt: System message setting context
            user_prompt: User message with the task
            temperature: Override for sampling temperature
            max_tokens: Override for max response tokens

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        logger.debug(f"Anthropic request: model={self.config.model}, temp={temp}")

        response = self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temp,
            max_tokens=tokens,
        )

        content = response.content[0].text
        logger.debug(f"Anthropic response: {len(content)} chars")

        return content


def create_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        config: LLM configuration specifying provider and settings

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If the provider is not supported
    """
    providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
    }

    provider_class = providers.get(config.provider)
    if provider_class is None:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    return provider_class(config)
