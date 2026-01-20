"""
ASP code generation node for the NS-MAS agent.

Generates ASP code from the math word problem, using extracted
entities and any previous feedback for self-correction.
"""

import hashlib
import logging
import re
from typing import Callable

from ..config import AgentConfig
from ..state import AgentState
from ..llm_provider import create_llm_provider
from ..prompts.system_prompts import ASP_GENERATION_SYSTEM_PROMPT
from ..prompts.few_shot_examples import get_few_shot_examples

logger = logging.getLogger(__name__)


def create_generate_asp_node(config: AgentConfig) -> Callable[[AgentState], dict]:
    """
    Create the ASP code generation node.

    Args:
        config: Agent configuration with LLM and prompt settings

    Returns:
        Node function that generates ASP code
    """
    # Create LLM provider
    provider = create_llm_provider(config.generator_llm)

    def generate_asp(state: AgentState) -> dict:
        """
        Generate ASP code for the math problem.

        Uses parsed entities and any previous feedback to generate
        or improve ASP code.

        Args:
            state: Current agent state with question and optional feedback

        Returns:
            State updates with generated ASP code
        """
        question = state.get("question", "")
        iteration = state.get("iteration_count", 0)

        logger.info(f"Generating ASP code (iteration {iteration})")

        # Build system prompt with optional few-shot examples
        system_prompt = ASP_GENERATION_SYSTEM_PROMPT

        if config.include_few_shot_examples:
            examples = get_few_shot_examples(config.num_few_shot_examples)
            system_prompt += f"\n\n## Examples\n\n{examples}"

        # Build user prompt with context
        user_parts = [f"## Problem\n{question}"]

        # Include extracted entities if available
        parsed_entities = state.get("parsed_entities", [])
        if parsed_entities:
            entities_str = "\n".join(f"- {e}" for e in parsed_entities)
            user_parts.append(f"\n## Extracted Information\n{entities_str}")

        # Include previous feedback if this is a retry
        solver_feedback = state.get("solver_feedback", "")
        previous_code = state.get("asp_code", "")

        if solver_feedback and iteration > 0 and previous_code:
            user_parts.append(f"\n## Previous Attempt Feedback\n{solver_feedback}")
            user_parts.append(f"\n## Previous ASP Code (Fix This)\n```asp\n{previous_code}\n```")
            user_parts.append("\nAnalyze the error and generate CORRECTED ASP code.")
        else:
            user_parts.append("\nGenerate ASP code to solve this problem.")

        user_parts.append("\nOutput ONLY the ASP code in a ```asp code block.")

        user_prompt = "\n".join(user_parts)

        # Generate response
        response = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Extract ASP code from response
        asp_code = extract_asp_code(response)

        logger.info(f"Generated ASP code ({len(asp_code)} chars)")
        logger.debug(f"Generated ASP:\n{asp_code}")

        # Track code history for cycle detection
        code_history = list(state.get("asp_code_history", []))
        code_hash = hashlib.md5(asp_code.encode()).hexdigest()

        # Check for cycle (exact code repetition)
        history_hashes = [
            hashlib.md5(c.encode()).hexdigest() for c in code_history
        ]

        status = state.get("status", "running")
        if config.enable_cycle_detection and code_hash in history_hashes:
            logger.warning("Cycle detected: same ASP code generated twice")
            status = "cycle_detected"

        # Add current code to history
        code_history.append(asp_code)

        return {
            "asp_code": asp_code,
            "asp_code_history": code_history,
            "iteration_count": iteration + 1,
            "status": status,
        }

    return generate_asp


def extract_asp_code(response: str) -> str:
    """
    Extract ASP code from LLM response.

    Looks for code blocks with asp, prolog, or no language tag.
    Falls back to treating entire response as ASP code.

    Args:
        response: Raw LLM response text

    Returns:
        Extracted ASP code
    """
    # Try to find code blocks with various language tags
    patterns = [
        r"```asp\n(.*?)```",
        r"```prolog\n(.*?)```",
        r"```clingo\n(.*?)```",
        r"```\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: if response looks like ASP code, use it directly
    # (contains predicates ending with periods)
    if re.search(r"\w+\([^)]*\)\s*\.", response):
        # Remove any markdown-style text before/after
        lines = response.strip().split("\n")
        asp_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            # Skip obvious non-code lines
            if stripped.startswith("#") or stripped.startswith("**"):
                continue
            if re.match(r"^[a-z_]\w*\(", stripped) or stripped.startswith("%"):
                in_code = True
            if in_code:
                asp_lines.append(line)

        if asp_lines:
            return "\n".join(asp_lines).strip()

    # Last resort: return entire response stripped
    logger.warning("Could not extract ASP code block, using raw response")
    return response.strip()
