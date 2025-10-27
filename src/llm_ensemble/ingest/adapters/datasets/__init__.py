"""Dataset adapters for the ingest CLI.

This package contains concrete implementations of the DatasetAdapter port
for different IR dataset formats.
"""

from llm_ensemble.ingest.adapters.datasets.llm_judge_adapter import LlmJudgeAdapter

__all__ = ["LlmJudgeAdapter"]
