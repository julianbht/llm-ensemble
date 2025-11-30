"""Thomas et al. simple prompt parser.

Parses JSON responses from the thomas-simple prompt which outputs {"O": N} format.
Parser knows exactly what to look for - tightly coupled to the thomas-simple prompt.
"""

from __future__ import annotations
import json
import re

from llm_ensemble.infer.ports import ResponseParser
from llm_ensemble.infer.schemas.llm_judgement import LLMScore
from llm_ensemble.infer.schemas.warnings import ParserWarning, ParserWarningCode
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.schemas import RelevanceScore


class ThomasSimpleParser(ResponseParser):
    """Parser for thomas-simple prompt responses.

    Expects JSON output: {"O": N} where N is 0, 1, 2, or 3.
    The "O" field represents the overall relevance score.
    """

    def __init__(self):
        self.logger = get_logger(component="thomas_simple_parser")

    def parse(self, raw_text: str) -> LLMScore:
        """Parse JSON response to extract relevance label from "O" field.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            LLMScore with llm_response_text, extracted label, and any parsing warnings
        """
        warnings: list[ParserWarning] = []

        # Look for JSON object with "O" field
        json_pattern = r'\{[^}]*"O"\s*:\s*\d+[^}]*\}'
        json_match = re.search(json_pattern, raw_text)

        if not json_match:
            warning = ParserWarning(
                code=ParserWarningCode.PARSE_ERROR,
                message="No JSON object with 'O' field found in response",
                metadata={"expected_format": '{"O": N}'}
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

        # Extract the "O" score
        score = data.get("O")

        if score is None:
            warning = ParserWarning(
                code=ParserWarningCode.FIELD_ERROR,
                message="Missing 'O' field in parsed JSON",
                metadata={"field_name": "O"}
            )
            warnings.append(warning)
            self.logger.warning("parser_warning", code=warning.code.value, message=warning.message)
            return LLMScore(llm_response_text=raw_text, warnings=warnings)

        # Validate the score is 0, 1, 2, or 3
        if not isinstance(score, int) or score not in [0, 1, 2, 3]:
            warning = ParserWarning(
                code=ParserWarningCode.VALIDATION_ERROR,
                message=f"Invalid O score: {score} (expected 0, 1, 2, or 3)",
                metadata={"field_name": "O", "actual_value": str(score)}
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
