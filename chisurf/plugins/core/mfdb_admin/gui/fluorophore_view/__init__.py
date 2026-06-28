"""Fluorophore curation view, migrated from the fluorophore_db plugin into
mfdb-admin (PRD-06 integration). The scraper (spectra_downloader) stays a
separate plugin that populates the MFDB."""

from .fluorophore_dock import FluorophoreDock

__all__ = ["FluorophoreDock"]
