"""
Entity extraction node for the NS-MAS agent.

Performs Chain-of-Thought entity extraction before ASP generation
to help ground the problem and identify relevant quantities.
"""

import logging
from typing import Callable

from ..config import AgentConfig
from ..state import AgentState
from ..llm_provider import create_llm_provider
from ..prompts.system_prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_extract_entities_node(config: AgentConfig) -> Callable[[AgentState], dict]:
    """
    Create the entity extraction node.

    Args:
        config: Agent configuration with LLM settings

    Returns:
        Node function that extracts entities from the question
    """
    # Create LLM provider
    provider = create_llm_provider(config.generator_llm)

    def extract_entities(state: AgentState) -> dict:
        """
        Extract entities, quantities, and relationships from the question.

        This is a mandatory CoT step that helps the LLM identify what's
        mathematically relevant before attempting ASP generation.

        Args:
            state: Current agent state with question

        Returns:
            State updates with parsed entities and reasoning
        """
        question = state.get("question", "")

        if not question.strip():
            logger.warning("Empty question provided for entity extraction")
            return {
                "parsed_entities": [],
                "entity_extraction_reasoning": "No question provided.",
            }

        logger.info(f"Extracting entities from question ({len(question)} chars)")

        # Build user prompt
        user_prompt = f"""Question: {question}

Extract all mathematically relevant entities, quantities, operations, relationships, and the target.
Identify any irrelevant details that should be ignored."""

        # Generate response
        response = provider.generate(
            system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Parse entities from response
        # Simple line-based parsing - extract bullet points
        entities = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Look for bullet points or list items
            if line.startswith("- "):
                entities.append(line[2:].strip())
            elif line.startswith("* "):
                entities.append(line[2:].strip())
            elif line.startswith("• "):
                entities.append(line[2:].strip())

        logger.info(f"Extracted {len(entities)} entity items")
        logger.debug(f"Entity extraction response:\n{response}")

        return {
            "parsed_entities": entities,
            "entity_extraction_reasoning": response,
        }

    return extract_entities
