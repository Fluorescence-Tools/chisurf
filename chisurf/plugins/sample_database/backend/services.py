"""Compatibility wrapper for MFDB admin services."""

from chisurf.plugins.core.mfdb_admin.backend.services import *  # noqa: F403
from chisurf.plugins.core.mfdb_admin.backend.services import (  # noqa: F401
    _validate_fdb_methods_in_manifest,
)

__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
]
__all__.append("_validate_fdb_methods_in_manifest")
