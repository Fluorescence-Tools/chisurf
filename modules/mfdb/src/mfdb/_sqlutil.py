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
from datetime import datetime, timezone
from typing import Any


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
_utc_now = utc_now
_json_dumps = json_dumps
_json_loads = json_loads
_json_hash = json_hash
