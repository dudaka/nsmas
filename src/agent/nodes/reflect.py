"""
Reflection node for the NS-MAS agent.

Generates error-specific coaching to guide the LLM's next
ASP code generation attempt.
"""

import logging
from typing import Callable

from ..config import AgentConfig
from ..state import AgentState
from ..llm_provider import create_llm_provider
from ..prompts.system_prompts import REFLECTION_SYSTEM_PROMPT
from ..prompts.error_coaching import get_error_coaching

logger = logging.getLogger(__name__)


def create_reflect_node(config: AgentConfig) -> Callable[[AgentState], dict]:
    """
    Create the reflection node.

    Args:
        config: Agent configuration with LLM settings

    Returns:
        Node function that generates reflection/coaching
    """
    # Create LLM provider (use reflector_llm if different from generator)
    provider = create_llm_provider(config.reflector_llm)

    def reflect(state: AgentState) -> dict:
        """
        Reflect on the verification failure and generate guidance.

        Uses error-specific coaching templates to help the LLM
        understand and fix the issue.

        Args:
            state: Current agent state with verification result

        Returns:
            State updates with critique and enhanced feedback
        """
        verification_result = state.get("verification_result", {})
        error_type = verification_result.get("error_type", "UNKNOWN")
        error_message = verification_result.get("error_message", "")
        asp_code = state.get("asp_code", "")
        iteration = state.get("iteration_count", 0)

        logger.info(f"Reflecting on error: {error_type} (iteration {iteration})")

        # Get error-specific coaching
        coaching = get_error_coaching(error_type, error_message, asp_code)

        # Build reflection prompt
        user_prompt = f"""## Error Information
Type: {error_type}
Message: {error_message}

## Current ASP Code
```asp
{asp_code}
```

## Error-Specific Guidance
{coaching}

## Task
Analyze what went wrong and explain in 2-3 sentences:
1. The specific issue in the code
2. How to fix it

Do NOT regenerate the code - just explain the fix needed."""

        # Generate reflection
        response = provider.generate(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        logger.info(f"Generated reflection ({len(response)} chars)")
        logger.debug(f"Reflection response:\n{response}")

        # Build critique entry for history
        critique = f"[Attempt {iteration}] {error_type}: {response}"

        # Enhance solver feedback with reflection
        original_feedback = state.get("solver_feedback", "")
        enhanced_feedback = f"{original_feedback}\n\n## Analysis\n{response}"

        return {
            "critique_history": [critique],  # Will be appended via operator.add
            "solver_feedback": enhanced_feedback,
        }

    return reflect
