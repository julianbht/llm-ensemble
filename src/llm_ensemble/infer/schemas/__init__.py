"""Pydantic schemas for the infer CLI.

Exports configuration schemas, DTOs, and run metadata for convenient importing.

Run Entity Structure (Refactored):
- InferRunInfo: Run metadata (git info, timestamps, run_type, notes)
- InferRunConfig: Configuration bundle (model_config, adapter_config, retry_config)
- InferRunContext: Execution context (input_run_name, start_idx, end_idx, io_name)
- InferRunOutput: Judgements and metrics produced (llm_judgements, sample_fingerprint, aggregate metrics)
"""

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.infer_run_config import InferRunConfig
from llm_ensemble.infer.schemas.ingest_run_context import IngestRunContext
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.schemas.warnings import BaseWarning, ParserWarning, ParserWarningCode

__all__ = [
    "ModelConfig",
    "RetryConfig",
    "InferRunInfo",
    "InferRunConfig",
    "IngestRunContext",
    "InferRunSummary",
    "WriteSummary",
    "BaseWarning",
    "ParserWarning",
    "ParserWarningCode",
]
