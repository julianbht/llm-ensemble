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
