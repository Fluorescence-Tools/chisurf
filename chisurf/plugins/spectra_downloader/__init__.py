"""Spectra Downloader Plugin.

Downloads fluorophore spectra from external sources (FPbase, ATTO-TEC,
Chroma, Thorlabs, etc.) into the fluorophore database.

**Status:** experimental — downloaded data is unverified until approved.
"""

from __future__ import annotations

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)

name = "Spectroscopy:Spectra Downloader"
cli_entrypoint = "spectra-download=chisurf.plugins.spectra_downloader.cli:cli"


def get_db() -> FluorophoreDatabase:
    # The reference spectra database + adapter live with the _dev tooling (the
    # same source the download/* scripts write to); mfdb-admin's
    # `csc fluorophore import-reference-set` pulls it into the live MFDB.
    return FluorophoreDatabase(DEFAULT_DATABASE_PATH)


if __name__ == "plugin":
    from .download_manager import DownloadManagerDialog
    db = get_db()
    window = DownloadManagerDialog(db)
    window.show()
