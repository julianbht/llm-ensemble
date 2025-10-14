"""Pure domain logic for parsing LLM outputs into structured judgements.

No I/O, no API calls — just string parsing and validation.
"""
from __future__ import annotations
import re
from typing import Literal


ParsedJudgement = tuple[
    Literal["relevant", "partially", "irrelevant"],  # label
    float,  # score
    str | None,  # rationale
    list[str],  # warnings
]


def parse_judgement(raw_text: str) -> ParsedJudgement:
    """Parse raw LLM output into structured judgement components.

    Expected format:
        LABEL: <relevant|partially|irrelevant>
        REASONING: <explanation>

    Args:
        raw_text: The raw text output from the LLM

    Returns:
        Tuple of (label, score, rationale, warnings)
        - label: One of the three valid labels
        - score: Normalized confidence score [0,1]
        - rationale: Extracted reasoning text or None
        - warnings: List of parsing issues encountered

    Raises:
        ValueError: If no valid label can be extracted
    """
    warnings: list[str] = []
    raw_text = raw_text.strip()

    # Extract label
    label_match = re.search(
        r"LABEL:\s*(relevant|partially|irrelevant)",
        raw_text,
        re.IGNORECASE
    )

    if not label_match:
        # Fallback: look for standalone keywords
        text_lower = raw_text.lower()
        if "relevant" in text_lower and "irrelevant" not in text_lower:
            if "partially" in text_lower or "partial" in text_lower:
                label = "partially"
                warnings.append("Fallback: extracted 'partially' from unstructured text")
            else:
                label = "relevant"
                warnings.append("Fallback: extracted 'relevant' from unstructured text")
        elif "irrelevant" in text_lower:
            label = "irrelevant"
            warnings.append("Fallback: extracted 'irrelevant' from unstructured text")
        else:
            raise ValueError(f"Could not extract valid label from output: {raw_text[:100]}")
    else:
        label = label_match.group(1).lower()

    # Normalize label
    if label not in {"relevant", "partially", "irrelevant"}:
        raise ValueError(f"Invalid label: {label}")

    # Extract reasoning
    reasoning_match = re.search(
        r"REASONING:\s*(.+)",
        raw_text,
        re.IGNORECASE | re.DOTALL
    )

    if reasoning_match:
        rationale = reasoning_match.group(1).strip()
    else:
        # Fallback: use everything after the label
        if label_match:
            rationale = raw_text[label_match.end():].strip()
            if rationale:
                warnings.append("Fallback: used text after label as rationale")
        else:
            rationale = None
            warnings.append("No rationale found in output")

    # Compute score based on label (simple heuristic)
    # In a real system, this might come from logprobs or explicit confidence
    score_map = {
        "relevant": 0.9,
        "partially": 0.5,
        "irrelevant": 0.1,
    }
    score = score_map[label]

    return (label, score, rationale, warnings)


def normalize_confidence(score: float, label: Literal["relevant", "partially", "irrelevant"]) -> float:
    """Compute confidence from score.

    For now, this is just a passthrough. In future iterations, this could
    incorporate logprobs, calibration curves, etc.

    Args:
        score: The raw score [0,1]
        label: The predicted label

    Returns:
        Confidence value [0,1]
    """
    # Partially relevant judgements have inherently lower confidence
    if label == "partially":
        return score * 0.8

    return score
