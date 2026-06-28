"""Fluorophore Database Plugin.

Manages a centralized database of fluorophore spectra and optical
properties. Provides a curation GUI, CLI commands, and an import
pipeline for reference spectral data.

**Status:** experimental — data imported from the reference set is
unverified until approved by a human curator.
"""

from __future__ import annotations

from pathlib import Path

from .mfdb_adapter import DEFAULT_DATABASE_PATH, FluorophoreDatabase

name = "Spectroscopy:Fluorophore DB"
cli_entrypoint = "fluorophore=chisurf.plugins.fluorophore_db.cli:cli"

_SHARED_DB_CACHE: dict[str, FluorophoreDatabase] = {}


def get_db(db_path: str | None = None) -> FluorophoreDatabase:
    """Get a shared FluorophoreDatabase instance.

    Parameters
    ----------
    db_path : str, optional
        Path to the MFDB database. Defaults to the resolved user database.
    """
    key = db_path or "default"
    if key not in _SHARED_DB_CACHE:
        if db_path:
            _SHARED_DB_CACHE[key] = FluorophoreDatabase(db_path)
        else:
            _SHARED_DB_CACHE[key] = FluorophoreDatabase()
    return _SHARED_DB_CACHE[key]


if __name__ == "plugin":
    from .db_manager_widget import FluorophoreDBWidget as _Widget
    window = _Widget()
    window.show()
