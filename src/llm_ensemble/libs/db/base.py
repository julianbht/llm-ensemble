"""SQLAlchemy metadata, engine factory, and namespace UUIDs for deterministic ID generation."""

import os
from typing import Optional
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import declarative_base

# SQLAlchemy declarative base for ORM models
Base = declarative_base()


# ========================================================================
# Engine Factory
# ========================================================================

def get_engine(database_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL database.

    Args:
        database_url: PostgreSQL connection URL. If None, reads from DATABASE_URL
                     environment variable (required).
        echo: If True, log all SQL statements (useful for debugging)

    Returns:
        Configured SQLAlchemy engine

    Raises:
        ValueError: If database_url is None and DATABASE_URL env var is not set

    Example:
        >>> engine = get_engine()  # Uses DATABASE_URL env var
        >>> engine = get_engine("postgresql://user:pass@localhost:5432/llm_ensemble")
        >>> engine = get_engine(echo=True)  # Enable SQL logging
    """
    if database_url is None:
        # Read from environment (required)
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Example: postgresql://user:password@localhost:5432/llm_ensemble"
            )

    # Create PostgreSQL engine with default connection pooling
    engine = create_engine(
        database_url,
        echo=echo,
    )

    return engine


def create_schemas(engine: Engine) -> None:
    """Create PostgreSQL schemas if they don't exist.

    Creates the schemas used by the application: ingest, infer, aggregate, evaluate.
    Also ensures shared ENUM types (RunType, RelevanceScore) exist in public schema
    so they can be referenced across multiple schemas.

    Safe to call multiple times (idempotent).

    Args:
        engine: SQLAlchemy engine to use

    Example:
        >>> engine = get_engine()
        >>> create_schemas(engine)
    """
    schemas = ["public", "ingest", "infer", "aggregate", "evaluate"]

    with engine.connect() as conn:
        # Create schemas (including public for shared ENUMs)
        for schema in schemas:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

        # Note: Shared ENUMs (RunType, RelevanceScore) will be created in public schema
        # by SQLAlchemy when create_all_tables() is called. We don't create them manually
        # here because SQLAlchemy handles the proper type creation based on the Python enums.

        conn.commit()


def create_all_tables(engine: Engine) -> None:
    """Create all database tables from SQLAlchemy metadata.

    This creates tables for all SQLAlchemy ORM models that inherit from Base.
    Safe to call multiple times (idempotent).

    IMPORTANT: You must call create_schemas() first to create the schemas.

    Args:
        engine: SQLAlchemy engine to use

    Example:
        >>> engine = get_engine()
        >>> create_schemas(engine)  # Create schemas first
        >>> create_all_tables(engine)
    """
    Base.metadata.create_all(engine)
