"""SQLAlchemy metadata, engine factory, and namespace UUIDs for deterministic ID generation."""

import os
from typing import Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base

# SQLAlchemy declarative base for ORM models
Base = declarative_base()


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
