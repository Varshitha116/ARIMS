#!/usr/bin/env python3
"""
ARIMS — Database Session Management

Provides engine creation, session factory, and table initialisation.
Uses DATABASE_URL from config; if empty, the database layer is silently
disabled and ARIMS runs in memory-only mode (dashboard + API still work).
"""

import sys
from pathlib import Path
from typing import Optional, Generator
from contextlib import contextmanager

# Ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base

# Module-level singletons (initialised lazily)
_engine = None
_SessionFactory = None


def get_engine(database_url: Optional[str] = None):
    """
    Create or return the SQLAlchemy engine.

    Args:
        database_url: Override for the connection string.
                      If None, reads from config.settings.DATABASE_URL.

    Returns:
        Engine instance, or None if no DATABASE_URL is configured.
    """
    global _engine
    if _engine is not None:
        return _engine

    if database_url is None:
        from config import settings
        database_url = settings.DATABASE_URL

    if not database_url:
        return None

    from config import settings
    _engine = create_engine(
        database_url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        echo=settings.DB_ECHO,
        pool_pre_ping=True,         # reconnect on stale connections
    )
    return _engine


def get_session_factory():
    """Return the session factory, creating the engine if needed."""
    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory

    engine = get_engine()
    if engine is None:
        return None

    _SessionFactory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Optional[Session], None, None]:
    """
    Context-managed database session.

    Usage:
        with get_session() as session:
            if session is None:
                return  # DB not configured
            session.add(road)
            session.commit()
    """
    factory = get_session_factory()
    if factory is None:
        yield None
        return

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(database_url: Optional[str] = None):
    """
    Create all tables if they don't exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS semantics.
    """
    engine = get_engine(database_url)
    if engine is None:
        print("ℹ️  DATABASE_URL not set — running without persistent database.")
        return False

    Base.metadata.create_all(engine)
    print("✅ Database tables created / verified.")
    return True


def check_db_connection() -> dict:
    """
    Quick health-check on the database connection.

    Returns:
        dict with "connected" (bool) and optional "error" (str).
    """
    engine = get_engine()
    if engine is None:
        return {"connected": False, "error": "DATABASE_URL not configured"}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True}
    except Exception as e:
        return {"connected": False, "error": str(e)}
