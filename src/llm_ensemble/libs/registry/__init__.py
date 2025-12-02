"""Registry infrastructure for adapter selection."""

from llm_ensemble.libs.registry.base_registry import BaseRegistry, RegistryMetadata
from llm_ensemble.libs.registry.adapter_wrapper import AdapterWithMetadata

__all__ = ["BaseRegistry", "RegistryMetadata", "AdapterWithMetadata"]
