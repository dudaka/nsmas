"""
Mock LLM provider for deterministic testing.

Returns pre-configured responses based on prompt patterns,
enabling reliable unit tests without API calls.
"""

from typing import Dict, List, Optional, Tuple


class MockLLMProvider:
    """
    Deterministic mock LLM for testing.

    Returns pre-configured responses based on prompt patterns.
    Tracks all calls for assertions.
    """

    def __init__(self, responses: Dict[str, str] = None):
        """
        Initialize the mock provider.

        Args:
            responses: Dict mapping pattern strings to response strings
        """
        self.responses: Dict[str, str] = responses or {}
        self.call_history: List[Tuple[str, str]] = []
        self.default_response = "```asp\nfinal_answer(42).\n```"

    def add_response(self, pattern: str, response: str) -> None:
        """
        Add a pattern-response mapping.

        Args:
            pattern: Substring to match in prompts
            response: Response to return when pattern matches
        """
        self.responses[pattern] = response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response based on configured patterns.

        Args:
            system_prompt: System message
            user_prompt: User message
            temperature: Ignored
            max_tokens: Ignored

        Returns:
            Matched response or default
        """
        # Record the call
        self.call_history.append((system_prompt, user_prompt))

        # Combine prompts for matching
        combined = f"{system_prompt}\n{user_prompt}".lower()

        # Find matching pattern
        for pattern, response in self.responses.items():
            if pattern.lower() in combined:
                return response

        # Return default response
        return self.default_response

    def reset(self) -> None:
        """Clear call history."""
        self.call_history.clear()

    @property
    def call_count(self) -> int:
        """Number of generate calls made."""
        return len(self.call_history)

    def get_last_call(self) -> Optional[Tuple[str, str]]:
        """Get the most recent call's prompts."""
        if self.call_history:
            return self.call_history[-1]
        return None


# Pre-configured mock responses for common test scenarios
MOCK_ENTITY_EXTRACTION_RESPONSE = """Entities:
- John (person with apples)
- Mary (recipient)
- apples (countable item)

Quantities:
- John starts with 10 apples
- Mary has 5 apples

Operations:
- Addition: combine quantities

Target:
- Total number of apples"""


MOCK_ASP_GENERATION_RESPONSE = """```asp
% Facts
quantity(john, apples, 10).
quantity(mary, apples, 5).

% Calculate total
total(T) :- quantity(john, apples, A), quantity(mary, apples, B), T = @add(A, B).

final_answer(T) :- total(T).
```"""


MOCK_REFLECTION_RESPONSE = """The error occurs because the variable T in the rule head is not bound in the body.
To fix this, ensure T appears in a positive literal that binds it, such as using @add to compute the result."""


def create_mock_for_simple_problem() -> MockLLMProvider:
    """
    Create a mock configured for simple addition problems.

    Returns:
        MockLLMProvider with pre-configured responses
    """
    mock = MockLLMProvider()
    mock.add_response("extract", MOCK_ENTITY_EXTRACTION_RESPONSE)
    mock.add_response("asp", MOCK_ASP_GENERATION_RESPONSE)
    mock.add_response("reflect", MOCK_REFLECTION_RESPONSE)
    return mock
