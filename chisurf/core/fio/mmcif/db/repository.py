"""Legacy repository — deprecated shim.

All functionality has moved to :mod:`chisurf.core.mfdb.repository`.
This module re-exports the canonical ``FluorescenceDatabase`` and
``FluorophoreDatabase`` classes and a few private helpers for
backward compatibility.  New code should import from
``chisurf.core.mfdb.repository`` directly.
"""

from __future__ import annotations

import warnings
from typing import Any

from chisurf.core.mfdb.repository import (
    MFDatabase,
    _json_hash,
    _json_dumps,
    _json_loads,
    _row_to_dict,
    _utc_now,
)

warnings.warn(
    "chisurf.core.fio.mmcif.db.repository is deprecated. "
    "Use chisurf.core.mfdb.repository instead.",
    DeprecationWarning,
    stacklevel=2,
)


class FluorescenceDatabase(MFDatabase):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FluorescenceDatabase is deprecated. Use chisurf.core.mfdb.repository.MFDatabase instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class FluorophoreDatabase(MFDatabase):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FluorophoreDatabase is deprecated. Use chisurf.core.mfdb.repository.MFDatabase instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "FluorescenceDatabase",
    "FluorophoreDatabase",
    "_json_hash",
    "_json_dumps",
    "_json_loads",
    "_row_to_dict",
    "_utc_now",
]

