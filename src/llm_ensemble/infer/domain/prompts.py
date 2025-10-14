"""Pure domain logic for prompt construction.

No I/O, no API calls — just string formatting from domain models.
"""
from __future__ import annotations
from typing import Protocol


class JudgingExampleProtocol(Protocol):
    """Protocol for the input record (allows duck-typing with ingest.domain.models.JudgingExample)."""
    query_id: str
    query_text: str
    docid: str
    doc: str


def build_relevance_prompt(example: JudgingExampleProtocol, system_prompt: str | None = None) -> str:
    """Build a prompt for relevance judging.

    Args:
        example: The query-document pair to judge
        system_prompt: Optional system prompt override

    Returns:
        Formatted prompt string ready for LLM inference
    """
    default_system = (
        "You are a relevance judge. Given a query and a document, "
        "you must determine if the document is relevant, partially relevant, or irrelevant to the query. "
        "Respond with your judgement and a brief explanation."
    )

    sys_prompt = system_prompt or default_system

    user_prompt = f"""Query: {example.query_text}

Document: {example.doc}

Is this document relevant to the query? Respond with:
- "relevant" if the document fully answers the query
- "partially" if the document is somewhat related but incomplete
- "irrelevant" if the document does not address the query

Format your response as:
LABEL: <relevant|partially|irrelevant>
REASONING: <your explanation>"""

    return f"{sys_prompt}\n\n{user_prompt}"


def build_chat_messages(example: JudgingExampleProtocol, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Build chat-formatted messages for chat-based models.

    Args:
        example: The query-document pair to judge
        system_prompt: Optional system prompt override

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    default_system = (
        "You are a relevance judge. Given a query and a document, "
        "you must determine if the document is relevant, partially relevant, or irrelevant to the query. "
        "Respond with your judgement and a brief explanation."
    )

    sys_prompt = system_prompt or default_system

    user_content = f"""Query: {example.query_text}

Document: {example.doc}

Is this document relevant to the query? Respond with:
- "relevant" if the document fully answers the query
- "partially" if the document is somewhat related but incomplete
- "irrelevant" if the document does not address the query

Format your response as:
LABEL: <relevant|partially|irrelevant>
REASONING: <your explanation>"""

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
