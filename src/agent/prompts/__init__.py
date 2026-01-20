"""
Prompt templates for the NS-MAS agent.

Contains system prompts, error-specific coaching, and few-shot examples.
"""

from .system_prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ASP_GENERATION_SYSTEM_PROMPT,
)
from .error_coaching import get_error_coaching, ERROR_COACHING
from .few_shot_examples import get_few_shot_examples, FEW_SHOT_EXAMPLES

__all__ = [
    "ENTITY_EXTRACTION_SYSTEM_PROMPT",
    "ASP_GENERATION_SYSTEM_PROMPT",
    "get_error_coaching",
    "ERROR_COACHING",
    "get_few_shot_examples",
    "FEW_SHOT_EXAMPLES",
]
