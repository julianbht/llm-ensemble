"""Session management utilities for SQLAlchemy/SQLModel."""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_session(engine: Engine) -> Session:
    """Create a new database session from an engine.
    
    Args:
        engine: SQLAlchemy engine
    
    Returns:
        New database session
    
    Example:
        >>> engine = get_engine()
        >>> session = get_session(engine)
        >>> try:
        ...     # Do work with session
        ...     session.commit()
        ... finally:
        ...     session.close()
    """
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    return SessionLocal()


@contextmanager
def session_context(engine: Engine) -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic cleanup.
    
    Automatically commits on success and rolls back on exception.
    Always closes the session.
    
    Args:
        engine: SQLAlchemy engine
    
    Yields:
        Database session
    
    Example:
        >>> engine = get_engine()
        >>> with session_context(engine) as session:
        ...     query = session.query(Query).filter_by(external_id="q1")
        ...     # Automatically commits on success, rolls back on exception
    """
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
