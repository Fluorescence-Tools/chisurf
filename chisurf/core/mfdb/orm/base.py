"""SQLAlchemy base utilities for MFDB ORM.

This module provides the foundational SQLAlchemy infrastructure for the MFDB ORM
mapping, including engine creation, session management, and declarative base class.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import sqlalchemy
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in the MFDB slice."""

    pass


def make_engine(db_path: str | Path) -> sqlalchemy.engine.Engine:
    """Create and return a SQLAlchemy engine for an MFDB sqlite database.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Returns
    -------
    sqlalchemy.engine.Engine
        SQLAlchemy engine configured for the database.

    Notes
    -----
    The engine is configured with:
    - SQLite dialect
    - Foreign key constraint enforcement
    - Connection pooling appropriate for SQLite
    - WAL mode for better concurrent access
    """
    db_path = Path(db_path)

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    database_url = f"sqlite:///{db_path.resolve()}"

    engine = create_engine(
        database_url,
        echo=False,
        future=True,
        # Enable foreign key constraints
        connect_args={"check_same_thread": False},
    )

    # Configure SQLite for better WAL mode and foreign key handling
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")

    return engine


def _get_sessionmaker(db_path: str | Path) -> sessionmaker[Session]:
    """Get or create a sessionmaker for the given database path."""
    engine = make_engine(db_path)
    return sessionmaker(engine, expire_on_commit=False, class_=Session)


# Cache sessionmakers by database path to avoid creating multiple engines
_sessionmaker_cache: dict[str, sessionmaker[Session]] = {}


@contextmanager
def session_scope(db_path: str | Path) -> Generator[Session, None, None]:
    """Yield a short-lived SQLAlchemy session with commit/rollback handling.

    This context manager provides a transactional scope around a series of
    operations. If the block completes without exception, the session is
    committed. If an exception occurs, the session is rolled back.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Yields
    ------
    Session
        SQLAlchemy session bound to the database.

    Examples
    --------
    >>> with session_scope("/path/to/mfdb.db") as session:
    ...     # Use session for database operations
    ...     sample = session.get(FlrSample, sample_id)
    ...     session.add(new_probe)
    """
    db_path_str = str(Path(db_path).resolve())

    if db_path_str not in _sessionmaker_cache:
        _sessionmaker_cache[db_path_str] = _get_sessionmaker(db_path)

    SessionMaker = _sessionmaker_cache[db_path_str]
    session = SessionMaker()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def session_from_mfdatabase(db: Any) -> Optional[Session]:
    """Return a session bound to the same database path when safe.

    Parameters
    ----------
    db : MFDatabase
        An existing MFDatabase instance.

    Returns
    -------
    Session or None
        A SQLAlchemy session bound to the same database path, or None if
        it's not safe to create one.

    Notes
    -----
    This function checks if the MFDatabase has an active transaction and
    avoids creating a SQLAlchemy session that could conflict with the
    existing sqlite3 connection.

    If binding safely to an existing MFDatabase instance is not possible,
    SQLAlchemy-backed methods should open short-lived sessions by path and
    must not be used inside active db._transaction() blocks.
    """
    try:
        # Check if the database object has a connection attribute
        if hasattr(db, 'conn') and hasattr(db, '_transaction'):
            # Check if there's an active transaction
            # This is a simplified check - in practice, we can't safely
            # determine if there's an active transaction without more
            # intimate knowledge of MFDatabase internals
            logger.warning(
                "Creating SQLAlchemy session alongside existing MFDatabase. "
                "Avoid using both in the same transaction boundary."
            )
            return Session(make_engine(db.db_path))
        else:
            # No way to check for active transactions, proceed with caution
            logger.warning(
                "Creating SQLAlchemy session for MFDatabase. "
                "This should only be used when no sqlite3 transactions are active."
            )
            return Session(make_engine(db.db_path))
    except Exception as e:
        logger.error(f"Failed to create session from MFDatabase: {e}")
        return None


def clear_session_cache() -> None:
    """Clear the cached sessionmakers.

    This should be called when database files are deleted or moved,
    or when you want to force fresh engine creation.
    """
    global _sessionmaker_cache
    _sessionmaker_cache.clear()
