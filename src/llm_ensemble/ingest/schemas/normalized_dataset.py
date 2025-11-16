"""NormalizedDataset - container for dataset metadata and samples.

This is the output of the DatasetReader - a complete, normalized dataset
ready for persistence. The reader handles all normalization and UUID computation.
"""

from __future__ import annotations
from dataclasses import dataclass

from llm_ensemble.ingest.schemas import Dataset, JudgingSample


@dataclass(frozen=True)
class NormalizedDataset:
    """Container for complete normalized dataset.

    Combines dataset metadata with fully-formed judging samples.
    This is what DatasetReader returns after normalizing raw IR data.

    The reader is responsible for:
    - Extracting dataset metadata from data
    - Creating Query and Document entities with computed UUIDs
    - Creating complete JudgingSample objects
    - Returning everything as a cohesive unit

    Attributes:
        dataset: Dataset metadata (name, description, id)
        samples: List of complete JudgingSample objects
    """

    dataset: Dataset
    samples: list[JudgingSample]

    @property
    def sample_count(self) -> int:
        """Get number of samples in this dataset."""
        return len(self.samples)
