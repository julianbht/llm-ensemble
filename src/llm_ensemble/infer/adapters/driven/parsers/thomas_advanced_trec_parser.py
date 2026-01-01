"""Thomas et al. advanced TREC prompt parser.

Parses JSON responses from the thomas-advanced-trec prompt which outputs {"M": N, "T": N, "O": N} format.
Uses cascading strategy for robust parsing of LLM responses.
Supports 4-level TREC scale (0-3).
"""

from __future__ import annotations
import json
import re
import uuid
from typing import Optional

from llm_ensemble.infer.application.ports.driven.for_parsing_responses import ForParsingResponses
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue, ParserIssueCode
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class ThomasAdvancedTrecParser(ForParsingResponses):
    """Parser for thomas-advanced-trec prompt responses.

    Expects JSON output: {"M": N, "T": N, "O": N} where:
    - M: Match score (how well content matches query intent)
    - T: Trust score (trustworthiness of the web page)
    - O: Overall relevance score (0-3)

    Uses cascading strategy:
    1. Try clean JSON parsing
    2. Try embedded JSON extraction (markdown blocks, text)
    3. Try regex pattern matching
    4. Try fuzzy text matching

    Supports 4-level TREC scale:
    - 0 = Irrelevant
    - 1 = Relevant
    - 2 = Highly Relevant
    - 3 = Perfectly Relevant
    """

    PARSER_NAME = "thomas-advanced-trec"
    PARSER_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-ensemble.thomas-advanced-trec-parser-v1")

    def __init__(self):
        """Initialize parser and create cached parser entity."""
        self._parser = ResponseParser(
            id=self.PARSER_ID,
            name=self.PARSER_NAME,
            version="1.0"
        )

    def parse(self, raw_text: str) -> tuple[Optional[LLMScore], list[ParserIssue]]:
        """Parse JSON response using cascading strategy.

        Tries multiple extraction strategies in order of reliability.
        Returns None if all strategies fail.
        Stops at first successful extraction (early return).

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (LLMScore or None, issues):
            - LLMScore: parsed score, or None if extraction failed
            - issues: List of parser issues encountered
        """
        issues: list[ParserIssue] = []

        # Cascading strategy: try each method until one succeeds
        strategies = [
            self._try_clean_json,
            self._try_embedded_json,
            self._try_regex_extract,
            self._try_fuzzy_extract,
        ]

        for strategy in strategies:
            # Each strategy gets its own issue list to avoid accumulation
            strategy_issues: list[ParserIssue] = []
            score = strategy(raw_text, strategy_issues)
            if score is not None:
                # Success - return with only this strategy's issues
                issues.extend(strategy_issues)
                return score, issues
            # Strategy failed - continue to next strategy without accumulating issues

        # All strategies failed
        issues.append(ParserIssue(
            code=ParserIssueCode.MALFORMED_RESPONSE,
            message="All parsing strategies failed to extract score",
            metadata={"strategies_tried": len(strategies)}
        ))
        return None, issues

    def get_parser(self) -> ResponseParser:
        """Get Parser metadata for this adapter.

        Returns:
            Parser entity with id and name
        """
        return self._parser

    # ========================================================================
    # Cascading parsing strategies
    # ========================================================================

    def _try_clean_json(self, raw_text: str, issues: list[ParserIssue]) -> Optional[LLMScore]:
        """Strategy 1: Try parsing response as clean JSON at root level.

        Args:
            raw_text: Raw LLM response
            issues: List to append issues to

        Returns:
            LLMScore if successful, None otherwise
        """
        try:
            data = json.loads(raw_text.strip())
            if isinstance(data, dict):
                return self._extract_score_from_dict(data, issues)
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_embedded_json(self, raw_text: str, issues: list[ParserIssue]) -> Optional[LLMScore]:
        """Strategy 2: Try extracting JSON from markdown blocks or embedded text.

        Looks for:
        - ```json ... ```
        - ``` ... ```
        - JSON objects embedded in explanation text

        Args:
            raw_text: Raw LLM response
            issues: List to append issues to

        Returns:
            LLMScore if successful, None otherwise
        """
        # Try markdown code blocks first
        markdown_patterns = [
            r'```json\s*(\{[^`]+\})\s*```',
            r'```\s*(\{[^`]+\})\s*```',
        ]

        for pattern in markdown_patterns:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict):
                        score = self._extract_score_from_dict(data, issues)
                        if score:
                            issues.append(ParserIssue(
                                code=ParserIssueCode.NON_STANDARD_FORMAT,
                                message="Extracted JSON from markdown code block",
                                metadata={"extraction_method": "markdown"}
                            ))
                            return score
                except (json.JSONDecodeError, ValueError):
                    continue

        # Try finding JSON object with required fields anywhere in text
        json_pattern = r'\{[^}]*"O"\s*:\s*\d+[^}]*\}'
        match = re.search(json_pattern, raw_text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    score = self._extract_score_from_dict(data, issues)
                    if score:
                        issues.append(ParserIssue(
                            code=ParserIssueCode.NON_STANDARD_FORMAT,
                            message="Extracted JSON object from text",
                            metadata={"extraction_method": "embedded_json"}
                        ))
                        return score
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _try_regex_extract(self, raw_text: str, issues: list[ParserIssue]) -> Optional[LLMScore]:
        """Strategy 3: Try extracting score using regex patterns.

        Looks for patterns like:
        - "O": 2
        - O: 2
        - "O" = 2

        Args:
            raw_text: Raw LLM response
            issues: List to append issues to

        Returns:
            LLMScore if successful, None otherwise
        """
        patterns = [
            r'"O"\s*:\s*(\d+)',
            r'O\s*:\s*(\d+)',
            r'"O"\s*=\s*(\d+)',
            r'O\s*=\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_text)
            if match:
                try:
                    score_value = int(match.group(1))
                    label = self._validate_score_value(score_value, issues)
                    if label is not None:
                        issues.append(ParserIssue(
                            code=ParserIssueCode.NON_STANDARD_FORMAT,
                            message="Extracted score using regex pattern matching",
                            metadata={"extraction_method": "regex", "pattern": pattern}
                        ))
                        return LLMScore(label=label, confidence=None, rationale=None)
                except ValueError:
                    continue

        return None

    def _try_fuzzy_extract(self, raw_text: str, issues: list[ParserIssue]) -> Optional[LLMScore]:
        """Strategy 4: Try fuzzy matching based on keywords.

        Last resort strategy looking for relevance keywords in text.
        Records PARTIAL_PARSE_ISSUE as this is least reliable.

        Args:
            raw_text: Raw LLM response
            issues: List to append issues to

        Returns:
            LLMScore if successful, None otherwise
        """
        text_lower = raw_text.lower()

        # Check for explicit relevance keywords
        if any(word in text_lower for word in ["perfectly relevant", "perfect match", "exact answer"]):
            issues.append(ParserIssue(
                code=ParserIssueCode.LOW_CONFIDENCE_EXTRACTION,
                message="Extracted score using fuzzy keyword matching (perfectly relevant)",
                metadata={"extraction_method": "fuzzy", "confidence": "low"}
            ))
            return LLMScore(label=RelevanceScore.PERFECTLY_RELEVANT, confidence=None, rationale=None)

        if any(word in text_lower for word in ["highly relevant", "very helpful", "vital information"]):
            issues.append(ParserIssue(
                code=ParserIssueCode.LOW_CONFIDENCE_EXTRACTION,
                message="Extracted score using fuzzy keyword matching (highly relevant)",
                metadata={"extraction_method": "fuzzy", "confidence": "low"}
            ))
            return LLMScore(label=RelevanceScore.HIGHLY_RELEVANT, confidence=None, rationale=None)

        if any(word in text_lower for word in ["relevant", "related", "partly helpful"]):
            issues.append(ParserIssue(
                code=ParserIssueCode.LOW_CONFIDENCE_EXTRACTION,
                message="Extracted score using fuzzy keyword matching (relevant)",
                metadata={"extraction_method": "fuzzy", "confidence": "low"}
            ))
            return LLMScore(label=RelevanceScore.RELEVANT, confidence=None, rationale=None)

        if any(word in text_lower for word in ["not relevant", "irrelevant", "nothing to do"]):
            issues.append(ParserIssue(
                code=ParserIssueCode.LOW_CONFIDENCE_EXTRACTION,
                message="Extracted score using fuzzy keyword matching (irrelevant)",
                metadata={"extraction_method": "fuzzy", "confidence": "low"}
            ))
            return LLMScore(label=RelevanceScore.IRRELEVANT, confidence=None, rationale=None)

        return None

    # ========================================================================
    # Helper methods for extraction and validation
    # ========================================================================

    def _extract_score_from_dict(self, data: dict, issues: list[ParserIssue]) -> Optional[LLMScore]:
        """Extract and validate O field from parsed JSON dict.

        Args:
            data: Parsed JSON dictionary
            issues: List to append issues to

        Returns:
            LLMScore if valid O field found, None otherwise
        """
        if "O" not in data:
            issues.append(ParserIssue(
                code=ParserIssueCode.MISSING_REQUIRED_FIELD,
                message="Missing 'O' field in parsed JSON",
                metadata={"field_name": "O", "available_fields": list(data.keys())}
            ))
            return None

        score_value = data["O"]
        label = self._validate_score_value(score_value, issues)
        if label is None:
            return None

        return LLMScore(label=label, confidence=None, rationale=None)

    def _validate_score_value(self, score_value: any, issues: list[ParserIssue]) -> Optional[RelevanceScore]:
        """Validate score value and convert to RelevanceScore enum.

        Handles type conversion (string to int) and range validation.
        Valid range: 0-3 (4-level TREC scale).

        Args:
            score_value: Raw score value from JSON or text
            issues: List to append issues to

        Returns:
            RelevanceScore enum if valid, None otherwise
        """
        # Try to convert to int if it's a string
        if isinstance(score_value, str):
            try:
                score_value = int(score_value)
                issues.append(ParserIssue(
                    code=ParserIssueCode.TYPE_COERCION_APPLIED,
                    message="Converted string score to integer",
                    metadata={"original_type": "str", "converted_value": score_value}
                ))
            except ValueError:
                issues.append(ParserIssue(
                    code=ParserIssueCode.INVALID_FIELD_VALUE,
                    message=f"Score value is not a valid integer: {score_value}",
                    metadata={"field_name": "O", "actual_value": str(score_value)}
                ))
                return None

        # Validate type and range
        if not isinstance(score_value, int) or score_value not in [0, 1, 2, 3]:
            issues.append(ParserIssue(
                code=ParserIssueCode.INVALID_FIELD_VALUE,
                message=f"Invalid O score: {score_value} (expected 0, 1, 2, or 3)",
                metadata={"field_name": "O", "actual_value": str(score_value), "valid_range": "0-3", "actual_type": type(score_value).__name__}
            ))
            return None

        try:
            return RelevanceScore(score_value)
        except ValueError:
            issues.append(ParserIssue(
                code=ParserIssueCode.INVALID_FIELD_VALUE,
                message=f"Failed to convert to RelevanceScore: {score_value}",
                metadata={"value": score_value}
            ))
            return None
