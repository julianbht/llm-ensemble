"""Utilities for deriving filenames from entity class names.

Provides a consistent, DRY approach to naming persisted entity files by
deriving filenames from Pydantic model class names instead of hardcoding strings.

Examples:
    LLMJudgement + json → llm_judgements.json
    InferRunInfo + json → infer_run_info.json
    JudgingSample + ndjson → judging_samples.ndjson
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case.

    Args:
        name: CamelCase string (e.g., "LLMJudgement")

    Returns:
        snake_case string (e.g., "llm_judgement")

    Examples:
        >>> to_snake_case("LLMJudgement")
        'llm_judgement'
        >>> to_snake_case("InferRunInfo")
        'infer_run_info'
        >>> to_snake_case("JudgingSample")
        'judging_sample'
    """
    # Insert underscore before uppercase letters (except at start)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letters preceded by lowercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def pluralize(word: str) -> str:
    """Simple pluralization for entity names.

    Args:
        word: Singular word (e.g., "judgement")

    Returns:
        Plural word (e.g., "judgements")

    Note:
        Uses simple rules - add 's' for most words, or keep as-is for
        words ending in 'info' or 'data'.
    """
    # Don't pluralize words like "info", "data", "metadata"
    if word.endswith(('info', 'data')):
        return word

    # Simple pluralization - just add 's'
    # (Could be extended with more rules if needed)
    return word + 's'


def get_entity_filename(
    entity_class: type[BaseModel],
    file_format: str,
    *,
    plural: bool = True
) -> str:
    """Derive filename from entity class name.

    Converts the Pydantic model class name to snake_case, optionally pluralizes,
    and appends the file extension. This ensures filenames are consistent and
    derived from the source of truth (the entity class).

    Args:
        entity_class: Pydantic model class (e.g., LLMJudgement)
        file_format: File extension without dot (e.g., "json", "ndjson")
        plural: Whether to pluralize the entity name (default: True)
                Set to False for singleton entities like run metadata

    Returns:
        Filename string (e.g., "llm_judgements.json")

    Examples:
        >>> from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
        >>> get_entity_filename(LLMJudgement, "json")
        'llm_judgements.json'

        >>> from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
        >>> get_entity_filename(InferRunInfo, "json", plural=False)
        'infer_run_info.json'

        >>> from llm_ensemble.infer.schemas.llm_judgement import JudgingSample
        >>> get_entity_filename(JudgingSample, "ndjson")
        'judging_samples.ndjson'
    """
    # Get class name and convert to snake_case
    class_name = entity_class.__name__
    snake_name = to_snake_case(class_name)

    # Optionally pluralize
    if plural:
        snake_name = pluralize(snake_name)

    # Add extension
    return f"{snake_name}.{file_format}"
