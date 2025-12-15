"""Pydantic schemas for the infer CLI.

Run Entity Structure (Refactored):
- InferRunInfo: Run metadata (git info, timestamps, run_type, notes)
- InferRunConfig: Configuration bundle (model_config, adapter_config, retry_config)
- InferRunContext: Execution context (input_run_name, start_idx, end_idx, io_name)
- InferRunOutput: Judgements and metrics produced (llm_judgements, sample_fingerprint, aggregate metrics)
"""
