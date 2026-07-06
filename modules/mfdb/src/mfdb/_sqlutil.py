"""Pure JSON / time helpers shared by the MFDB repository and its query mixins.

These were defined inside ``repository.py``. They are extracted here so the
per-concern query mixins (``mfdb.queries.*``) can use them without importing back
into ``repository.py`` — which imports the mixins for its class bases and would
otherwise form an import cycle. The helpers are pure (standard library only).

``repository.py`` re-exports these names (including the underscore-prefixed
aliases) so existing imports such as ``from mfdb.repository import _utc_now``
keep working.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a :class:`sqlite3.Row` to a plain dict, or pass through ``None``.

    Parameters
    ----------
    row : sqlite3.Row or None
        A row fetched with ``row_factory = sqlite3.Row``.

    Returns
    -------
    dict or None
        ``{column: value}`` for the row, or ``None`` when *row* is ``None``.
    """
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def row_exists(conn: sqlite3.Connection, table: str, column: str, value: Any) -> bool:
    """Return whether any row in *table* has *column* equal to *value*.

    The *table* and *column* identifiers are caller/code-controlled (never user
    input); *value* is always a bound parameter.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    table : str
        Table name (trusted identifier).
    column : str
        Column name (trusted identifier).
    value : Any
        Value to match, bound as a parameter.

    Returns
    -------
    bool
        ``True`` if at least one matching row exists.
    """
    row = conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone()
    return row is not None


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns
    -------
    str
        Current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str | None:
    """Serialize *value* to canonical (sorted, compact) JSON.

    Parameters
    ----------
    value : Any
        Value to serialize; ``None`` passes through as ``None``.

    Returns
    -------
    str or None
        Canonical JSON text, or ``None`` when *value* is ``None``.
    """
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def json_loads(value: str | None) -> Any:
    """Deserialize a JSON string, returning ``None`` for empty input.

    Parameters
    ----------
    value : str or None
        JSON text to decode, or ``None``/empty.

    Returns
    -------
    Any
        The decoded object, or ``None`` when *value* is falsy.
    """
    if not value:
        return None
    return json.loads(value)


def json_hash(value: Any) -> str | None:
    """Return the SHA-256 hex digest of *value*'s canonical JSON.

    Parameters
    ----------
    value : Any
        Value to hash via its canonical JSON encoding.

    Returns
    -------
    str or None
        Hex-encoded SHA-256 digest, or ``None`` when *value* is ``None``.
    """
    text = json_dumps(value)
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Backwards-compatible private aliases. Callers historically import the
# underscore-prefixed names from ``mfdb.repository``, which re-exports these.
_row_to_dict = row_to_dict
_exists = row_exists
_utc_now = utc_now
_json_dumps = json_dumps
_json_loads = json_loads
_json_hash = json_hash
