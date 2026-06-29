"""Spectra Downloader Plugin.

Downloads optical-component spectra (fluorophores, filters, dichroics,
detectors, light sources) from external sources (FPbase, ATTO-TEC, Chroma,
Thorlabs, PhotochemCAD, 3DOptix, Omega, …).

Three-stage pipeline — every ``download/*`` scraper MUST follow it and align:

1. **Download & sort.** Fetch from the source and classify each item into a
   canonical *kind* (e.g. ``bandpass``, ``dichroic``, ``apd``,
   ``fluorescent_protein``) from the shared taxonomy
   ``mfdb_adapter.COMPONENT_KINDS``.
2. **Save to spectra.db.** Persist through the single canonical entry point
   ``FluorophoreDatabase.register_component(...)`` into the staging reference
   database (``spectra.db``). This guarantees a consistent ``category``,
   provenance (``source``/``source_ref``/``retrieved_at``), the granular
   ``component_kind`` and a uniform spectrum-type vocabulary. The staging DB is
   **not** the live MFDB.
3. **Integrate into MFDB.** ``MFDatabase.import_reference_set`` pulls the staging
   DB into the live, dictionary-driven MFDB (carrying category + provenance
   through) and runs de-duplication (``consolidate_probes``: simple exact-name
   merge within a category, metadata merged, proteins never fuzzy-merged).

Scrapers must never call ``add_probe``/``add_spectrum`` directly — always go
through ``register_component`` so all sources populate the DB identically.

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
    from .gui.tool import SpectraTool
    db = get_db()
    db.connect()
    window = SpectraTool(db)
    window.show()
