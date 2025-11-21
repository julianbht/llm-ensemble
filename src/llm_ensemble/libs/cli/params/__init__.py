"""Typed CLI parameter definitions shared across commands."""

from .shared_params import (
    InputPath,
    RunName,
    LogCfg,
    Official,
    Notes,
    Override,
    StartIdx,
    EndIdx,
    ModelCfg,
    PromptCfg,
    EnsembleCfg,
    RetryCfg,
    Tag,
)
from .aggregate import AggregateIoCfg, InferRunInput
from .ingest import IngestIoCfg
from .infer import InferIoCfg, IngestRunInput as InferIngestRunInput

__all__ = [
    "InputPath",
    "RunName",
    "LogCfg",
    "Official",
    "Notes",
    "Override",
    "StartIdx",
    "EndIdx",
    "ModelCfg",
    "PromptCfg",
    "EnsembleCfg",
    "RetryCfg",
    "Tag",
    "AggregateIoCfg",
    "IngestIoCfg",
    "InferIoCfg",
    "InferIngestRunInput",
    "InferRunInput",
]
