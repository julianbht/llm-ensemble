"""JSON response parser adapter.

Parses JSON-formatted LLM judge outputs to extract relevance labels.
Supports both simple and multi-aspect formats.
Handles its own logging.
"""

from __future__ import annotations
import json
import re

from llm_ensemble.infer.ports import ResponseParser
from llm_ensemble.infer.schemas.llm_judgement import LLMScore
from llm_ensemble.infer.schemas.warnings import ParserWarning, ParserWarningCode
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.schemas import RelevanceScore


class JsonResponseParser(ResponseParser):
    """Parser for JSON-formatted LLM responses.

    Expects JSON output in one of two formats:
    - Simple: {"O": 2}
    - Multi-aspect: {"M": 2, "T": 1, "O": 1}

    Where O is the final relevance score (0, 1, or 2).

    This implementation is based on the Thomas et al. prompt format but can
    work with any JSON format that includes an "O" field for the overall score.
    """

    def __init__(self, parser_name: str, score_field: str = "O"):
        """Initialize JSON response parser with identity from config.

        Args:
            parser_name: Natural key for Parser entity (from config)
            score_field: Name of the JSON field containing the relevance score (from config, default: "O")
        """
        super().__init__(parser_name, score_field)
        self.logger = get_logger(component="json_response_parser")

    def parse(self, raw_text: str) -> LLMScore:
        """Parse JSON response to extract relevance label.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            LLMScore with llm_response_text, extracted label, and any parsing warnings
        """
        warnings: list[ParserWarning] = []

        # Try to find JSON object in the response
        # Look for patterns like {"O": N} or {"M": N, "T": N, "O": N}
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
            return LLMScore(llm_response_text=raw_text, warnings=warnings)

        json_str = json_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            warning = ParserWarning(
                code=ParserWarningCode.PARSE_ERROR,
                message=f"Failed to parse JSON: {e}",
                metadata={"error_type": type(e).__name__}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return LLMScore(llm_response_text=raw_text, warnings=warnings)

        # Extract the score
        score = data.get(self.score_field)

        if score is None:
            warning = ParserWarning(
                code=ParserWarningCode.FIELD_ERROR,
                message=f"Missing '{self.score_field}' field in parsed JSON",
                metadata={"field_name": self.score_field}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return LLMScore(llm_response_text=raw_text, warnings=warnings)

        # Validate the score is 0, 1, 2, or 3
        if not isinstance(score, int) or score not in [0, 1, 2, 3]:
            warning = ParserWarning(
                code=ParserWarningCode.VALIDATION_ERROR,
                message=f"Invalid {self.score_field} score: {score} (expected 0, 1, 2, or 3)",
                metadata={"field_name": self.score_field, "actual_value": str(score)}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return LLMScore(llm_response_text=raw_text, warnings=warnings)

        # Convert to RelevanceScore enum
        try:
            relevance_label = RelevanceScore(score)
        except ValueError:
            warning = ParserWarning(
                code=ParserWarningCode.VALIDATION_ERROR,
                message=f"Invalid score value: {score}",
                metadata={"value": score}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return LLMScore(llm_response_text=raw_text, warnings=warnings)

        return LLMScore(llm_response_text=raw_text, label=relevance_label, warnings=warnings)
