"""Spectra Downloader Plugin.

Downloads fluorophore spectra from external sources (FPbase, ATTO-TEC,
Chroma, Thorlabs, etc.) into the fluorophore database.

**Status:** experimental — downloaded data is unverified until approved.
"""

from __future__ import annotations

from chisurf.plugins.fluorophore_db.mfdb_adapter import FluorophoreDatabase

name = "Spectroscopy:Spectra Downloader"
cli_entrypoint = "spectra-download=chisurf.plugins.spectra_downloader.cli:cli"


def get_db() -> FluorophoreDatabase:
    from chisurf.plugins.fluorophore_db import get_db as _get_db
    return _get_db()


if __name__ == "plugin":
    from .download_manager import DownloadManagerDialog
    db = get_db()
    window = DownloadManagerDialog(db)
    window.show()
