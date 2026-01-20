"""
Fast Path Solver for Bandit Router.

Implements zero-shot LLM solving without ASP verification.
Used for simple problems that don't require the full GVR loop.

Design Decisions:
- Uses GPT-4o-mini by default (cheap, fast)
- Single prompt, no self-correction
- Extracts numeric answer from response
- Abstracted to allow swapping in local models later
"""

import logging
import re
from typing import Optional, Protocol

from .config import BanditConfig

logger = logging.getLogger(__name__)


# System prompt for fast path
FAST_SOLVER_PROMPT = """You are a math problem solver. Solve the following problem step by step, then provide your final answer.

IMPORTANT: At the end of your response, write your final numeric answer on a new line in this exact format:
ANSWER: <number>

For example:
ANSWER: 42

Only provide a single integer as the answer. Do not include units, fractions, or decimal points unless the answer requires it."""


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        ...


class OpenAIClient:
    """OpenAI API client for fast path."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required for FastSolver. "
                    "Install with: pip install openai"
                )
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate response from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FAST_SOLVER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class AnthropicClient:
    """Anthropic API client for fast path."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required for FastSolver with Anthropic. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate response from Anthropic."""
        response = self.client.messages.create(
            model=self.model,
            system=FAST_SOLVER_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text


class FastSolver:
    """
    Fast path solver for simple math problems.

    Uses zero-shot LLM without ASP verification.
    Designed for problems the bandit routes to the fast path.

    Usage:
        config = BanditConfig()
        solver = FastSolver(config)

        answer = solver.solve("John has 5 apples. He eats 2. How many left?")
        # Returns: 3
    """

    # Pattern to extract answer from response
    ANSWER_PATTERN = re.compile(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)

    # Fallback patterns if ANSWER: format not found
    FALLBACK_PATTERNS = [
        re.compile(r"(?:final answer|the answer|answer is)[:\s]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"(?:=\s*)(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
        re.compile(r"(-?\d+(?:\.\d+)?)\s*$"),  # Last number in text
    ]

    def __init__(self, config: Optional[BanditConfig] = None):
        """
        Initialize the fast solver.

        Args:
            config: Bandit configuration
        """
        self.config = config or BanditConfig()
        self._client: Optional[LLMClient] = None

    @property
    def client(self) -> LLMClient:
        """Lazy initialization of LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> LLMClient:
        """Create appropriate LLM client based on config."""
        from src.agent.config import LLMProvider

        api_key = self.config.get_fast_api_key()

        if self.config.fast_provider == LLMProvider.OPENAI:
            return OpenAIClient(
                api_key=api_key,
                model=self.config.fast_model,
                temperature=self.config.fast_temperature,
                max_tokens=self.config.fast_max_tokens,
            )
        elif self.config.fast_provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(
                api_key=api_key,
                model=self.config.fast_model,
                temperature=self.config.fast_temperature,
                max_tokens=self.config.fast_max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.fast_provider}")

    def solve(self, question: str) -> Optional[int]:
        """
        Solve a math problem using zero-shot prompting.

        Args:
            question: The math word problem

        Returns:
            Integer answer or None if extraction failed
        """
        try:
            # Generate response
            response = self.client.generate(question)
            logger.debug(f"Fast solver response: {response[:200]}...")

            # Extract answer
            answer = self._extract_answer(response)

            if answer is not None:
                logger.info(f"Fast solver extracted answer: {answer}")
            else:
                logger.warning("Fast solver failed to extract answer")

            return answer

        except Exception as e:
            logger.error(f"Fast solver error: {e}")
            return None

    def _extract_answer(self, response: str) -> Optional[int]:
        """
        Extract numeric answer from LLM response.

        Tries multiple patterns in order of specificity.

        Args:
            response: LLM response text

        Returns:
            Integer answer or None
        """
        # Try primary ANSWER: pattern first
        match = self.ANSWER_PATTERN.search(response)
        if match:
            return self._parse_number(match.group(1))

        # Try fallback patterns
        for pattern in self.FALLBACK_PATTERNS:
            match = pattern.search(response)
            if match:
                return self._parse_number(match.group(1))

        return None

    def _parse_number(self, text: str) -> Optional[int]:
        """Parse a number string to integer."""
        try:
            # Handle decimals by rounding
            value = float(text)
            return int(round(value))
        except (ValueError, TypeError):
            return None

    def solve_with_metadata(self, question: str) -> dict:
        """
        Solve and return full metadata.

        Useful for logging and debugging.

        Args:
            question: The math word problem

        Returns:
            Dictionary with answer and metadata
        """
        result = {
            "question": question,
            "answer": None,
            "raw_response": None,
            "success": False,
            "error": None,
        }

        try:
            response = self.client.generate(question)
            result["raw_response"] = response

            answer = self._extract_answer(response)
            result["answer"] = answer
            result["success"] = answer is not None

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Fast solver error: {e}")

        return result
