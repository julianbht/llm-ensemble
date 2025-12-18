"""JSON response parser adapter.

Generic parser for JSON-formatted LLM judge outputs to extract relevance labels.
Supports both simple and multi-aspect formats with configurable score field.
"""

from __future__ import annotations
import json
import re
import uuid
from typing import Optional

from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.schemas.entities.llm_score import LLMScore
from llm_ensemble.infer.schemas.entities.parser import Parser
from llm_ensemble.infer.schemas.warnings import ParserWarning, ParserWarningCode
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.schemas import RelevanceScore


class JsonResponseParser(ResponseParserPort):
    """Generic parser for JSON-formatted LLM responses.

    Parses JSON output with configurable score field.
    Default format: {"O": N} where N is 0, 1, 2, or 3.

    Can also handle multi-aspect formats like {"M": 2, "T": 1, "O": 1}.
    """

    PARSER_NAME = "json-generic"
    PARSER_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-ensemble.json-generic-parser-v1")

    def __init__(self, score_field: str = "O"):
        """Initialize JSON response parser.

        Args:
            score_field: Name of the JSON field containing the relevance score (default: "O")
        """
        self._parser = Parser(
            id=self.PARSER_ID,
            name=self.PARSER_NAME,
        )
        self.score_field = score_field
        self.logger = get_logger(component="json_response_parser")

    def parse(self, raw_text: str) -> tuple[LLMScore, list[ParserWarning]]:
        """Parse JSON response and create LLMScore domain entity.

        Extracts the configured score field from JSON response and constructs
        an LLMScore with extracted label. Returns warnings separately.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (LLMScore, warnings):
            - LLMScore: parsed fields (no parser metadata or response_text)
            - warnings: List of parser warnings from the parsing process
        """
        warnings: list[ParserWarning] = []
        label: Optional[RelevanceScore] = None

        # Extract and validate score using testable helper methods
        json_data = self._extract_json(raw_text, warnings)
        if json_data is not None:
            score_value = self._extract_score_field(json_data, warnings)
            if score_value is not None:
                label = self._validate_score(score_value, warnings)

        score = LLMScore(
            label=label,
            confidence=None,
            rationale=None,
        )

        return score, warnings

    def get_parser(self) -> Parser:
        """Get Parser metadata for this adapter.

        Returns:
            Parser entity with id and name
        """
        return self._parser

    def _extract_json(self, raw_text: str, warnings: list[ParserWarning]) -> Optional[dict]:
        """Extract JSON object with score field from raw text.

        Pure, testable function for JSON extraction logic.

        Args:
            raw_text: Raw text response from the LLM
            warnings: List to append warnings to

        Returns:
            Parsed JSON dict if successful, None otherwise
        """
        json_pattern = r'\{[^}]*"' + self.score_field + r'"\s*:\s*\d+[^}]*\}'
        json_match = re.search(json_pattern, raw_text)

        if not json_match:
            warning = ParserWarning(
                code=ParserWarningCode.PARSE_ERROR,
                message=f"No JSON object with '{self.score_field}' field found in response",
                metadata={"score_field": self.score_field}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return None

        json_str = json_match.group(0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            warning = ParserWarning(
                code=ParserWarningCode.PARSE_ERROR,
                message=f"Failed to parse JSON: {e}",
                metadata={"error_type": type(e).__name__}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return None

    def _extract_score_field(self, json_data: dict, warnings: list[ParserWarning]) -> Optional[int]:
        """Extract the score field from parsed JSON.

        Pure, testable function for field extraction logic.

        Args:
            json_data: Parsed JSON dict
            warnings: List to append warnings to

        Returns:
            Score value if present, None otherwise
        """
        score = json_data.get(self.score_field)

        if score is None:
            warning = ParserWarning(
                code=ParserWarningCode.FIELD_ERROR,
                message=f"Missing '{self.score_field}' field in parsed JSON",
                metadata={"field_name": self.score_field}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return None

        return score

    def _validate_score(self, score_value: int, warnings: list[ParserWarning]) -> Optional[RelevanceScore]:
        """Validate score value and convert to RelevanceScore enum.

        Pure, testable function for validation logic.

        Args:
            score_value: Raw score value from JSON
            warnings: List to append warnings to

        Returns:
            RelevanceScore enum if valid, None otherwise
        """
        if not isinstance(score_value, int) or score_value not in [0, 1, 2, 3]:
            warning = ParserWarning(
                code=ParserWarningCode.VALIDATION_ERROR,
                message=f"Invalid {self.score_field} score: {score_value} (expected 0, 1, 2, or 3)",
                metadata={"field_name": self.score_field, "actual_value": str(score_value)}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return None

        try:
            return RelevanceScore(score_value)
        except ValueError:
            warning = ParserWarning(
                code=ParserWarningCode.VALIDATION_ERROR,
                message=f"Invalid score value: {score_value}",
                metadata={"value": score_value}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return None
