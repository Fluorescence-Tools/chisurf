from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager

import uuid

logger = logging.getLogger(__name__)


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Context manager ensuring transaction safety and atomicity using SAVEPOINTs.

    If any error occurs within the block, the transaction is rolled back to the savepoint.
    Otherwise, the savepoint is released (committed).

    Parameters
    ----------
    conn : sqlite3.Connection
        The SQLite database connection.
    """
    if conn.in_transaction:
        sp_name = f"sp_{uuid.uuid4().hex}"
        conn.execute(f"SAVEPOINT {sp_name}")
        try:
            yield conn
            conn.execute(f"RELEASE SAVEPOINT {sp_name}")
        except Exception as exc:
            conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            conn.execute(f"RELEASE SAVEPOINT {sp_name}")
            logger.error("Database transaction failed and was rolled back: %s", exc)
            raise
    else:
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("Database transaction failed and was rolled back: %s", exc)
            raise


