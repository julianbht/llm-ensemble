"""Database utilities for LLM Ensemble.

Provides SQLAlchemy ORM base, engine factory, session management,
and UUID computation helpers for deterministic entity identification.
"""

from llm_ensemble.libs.db.base import (
    Base,
    get_engine,
    create_schemas,
    create_all_tables,
)

from llm_ensemble.libs.db.session import (
    get_session,
    session_context,
)

from llm_ensemble.libs.db.uuid_helpers import (
    compute_judged_dataset_fingerprint,
    compute_judged_dataset_uuid,
)

from llm_ensemble.libs.db.utcnow import utcnow

__all__ = [
    # Base and engine
    "Base",
    "get_engine",
    "create_schemas",
    "create_all_tables",
    # Session management
    "get_session",
    "session_context",
    # UUID helpers (still used for JudgedDataset)
    "compute_judged_dataset_fingerprint",
    "compute_judged_dataset_uuid",
    # Other db helpers
    "utcnow",
]
