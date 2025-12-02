"""Wrapper for adapters with identity metadata."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterWithMetadata:
    """Pairs an adapter instance with its identity.

    Adapters are pure implementations (no identity fields).
    Identity tracked separately for domain objects and persistence.
    """
    adapter: Any
    name: str
