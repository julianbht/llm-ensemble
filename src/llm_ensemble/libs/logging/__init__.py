"""Structlog-based logging for LLM Ensemble CLIs."""

from llm_ensemble.libs.logging.structlog_logger import configure_logger

__all__ = ["configure_logger","IngestLogEvent","IngestWriteEvent"]
