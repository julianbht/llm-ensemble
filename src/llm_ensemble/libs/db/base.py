"""SQLAlchemy metadata, engine factory, and namespace UUIDs for deterministic ID generation."""

import uuid
import os
from typing import Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base

# SQLAlchemy declarative base for ORM models
Base = declarative_base()


# ========================================================================
# Namespace UUIDs for deterministic UUIDv5 generation
# ========================================================================
# Each entity type has its own namespace UUID to ensure no collisions
# between different entity types even if they have the same natural key.

NAMESPACE_DATASET = uuid.UUID('f0e1d2c3-b4a5-9687-7654-321fedcba098')
NAMESPACE_QUERY = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
NAMESPACE_DOCUMENT = uuid.UUID('b2c3d4e5-f678-90ab-cdef-123456789012')
NAMESPACE_JUDGING_SAMPLE = uuid.UUID('c3d4e5f6-7890-abcd-ef12-34567890abcd')
NAMESPACE_INGEST_RUN = uuid.UUID('d4e5f678-90ab-cdef-1234-567890abcdef')
NAMESPACE_INFER_RUN = uuid.UUID('e5f67890-abcd-ef12-3456-7890abcdef12')
NAMESPACE_AGGREGATE_RUN = uuid.UUID('f6789012-3456-7890-abcd-ef1234567890')
NAMESPACE_LLM_REQUEST = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
NAMESPACE_LLM_RESPONSE = uuid.UUID('b1c2d3e4-f567-890a-bcde-f12345678901')
NAMESPACE_LLM_SCORE = uuid.UUID('c2d3e4f5-6789-0abc-def1-234567890abc')
NAMESPACE_LLM_JUDGEMENT = uuid.UUID('d3e4f567-890a-bcde-f123-4567890abcde')
NAMESPACE_AGGREGATED_SCORE = uuid.UUID('e4f56789-0abc-def1-2345-67890abcdef1')
NAMESPACE_AGGREGATED_JUDGEMENT = uuid.UUID('f5678901-abcd-ef12-3456-7890abcdef12')


# ========================================================================
# Engine Factory
# ========================================================================

def get_engine(database_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for the database.
    
    Args:
        database_url: Database connection URL. If None, reads from DATABASE_URL
                     environment variable. Defaults to SQLite if not set.
        echo: If True, log all SQL statements (useful for debugging)
    
    Returns:
        Configured SQLAlchemy engine
    
    Example:
        >>> engine = get_engine()  # Uses env var or default SQLite
        >>> engine = get_engine("postgresql://user:pass@localhost/db")
        >>> engine = get_engine(echo=True)  # Enable SQL logging
    """
    if database_url is None:
        # Read from environment or use default SQLite
        database_url = os.getenv(
            "DATABASE_URL",
            "sqlite:///artifacts/llm_ensemble.db"
        )
    
    # Create engine with appropriate settings
    engine = create_engine(
        database_url,
        echo=echo,
        # SQLite-specific: enable foreign keys
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {}
    )
    
    return engine


def create_all_tables(engine: Engine) -> None:
    """Create all database tables from SQLAlchemy metadata.
    
    This creates tables for all SQLAlchemy ORM models that inherit from Base.
    Safe to call multiple times (idempotent).
    
    Args:
        engine: SQLAlchemy engine to use
    
    Example:
        >>> engine = get_engine()
        >>> create_all_tables(engine)
    """
    Base.metadata.create_all(engine)
