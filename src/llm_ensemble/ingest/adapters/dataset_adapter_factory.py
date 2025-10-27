"""Factory for instantiating dataset adapters.

Provides configuration-driven adapter selection following the factory pattern.
"""

from __future__ import annotations

from llm_ensemble.ingest.ports import DatasetAdapter
from llm_ensemble.ingest.adapters.datasets import LlmJudgeAdapter


def get_dataset_adapter(adapter_name: str, dataset_id: str) -> DatasetAdapter:
    """Factory function to instantiate dataset adapters.

    Args:
        adapter_name: Adapter identifier from dataset config (e.g., 'llm_judge')
        dataset_id: Dataset identifier to embed in JudgingExample records

    Returns:
        DatasetAdapter: Concrete adapter instance

    Raises:
        ValueError: If adapter_name is not recognized

    Example:
        >>> adapter = get_dataset_adapter('llm_judge', 'llm-judge-2024')
        >>> examples = adapter.read(Path('/data'))
    """
    if adapter_name == "llm_judge":
        return LlmJudgeAdapter(dataset_id=dataset_id)
    else:
        raise ValueError(
            f"Unknown dataset adapter: '{adapter_name}'. "
            f"Available adapters: ['llm_judge']"
        )
