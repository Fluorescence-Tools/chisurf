"""End-to-end spectra pipeline test (no network, no GUI).

Exercises the real pipeline functions the scrapers/GUI use, skipping the
expensive network scrapes by substituting per-source staging DBs built through
the canonical ``register_component`` contract:

    download+sort (register_component)  →  per-source DBs
    merge_all (parallel-merge + dedup)  →  one staging spectra.db
    import_reference_set(replace)       →  the live MFDB (import ALL)

and asserts the end state: every optical category present, provenance populated,
cross-source duplicates merged (metadata unioned), spectra carried through, and
the admin/session gate honoured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import FluorophoreDatabase
from chisurf.plugins.spectra_downloader.download.merge import merge_all


def _spec(n=12):
    wl = np.linspace(400.0, 700.0, n)
    return wl, np.ones_like(wl)


def _source_db(path: Path, items: list[dict]) -> str:
    db = FluorophoreDatabase(str(path))
    with db:
        for kw in items:
            db.register_component(**kw)
        db.conn.commit()
    db.close()
    return str(path)


def test_pipeline_scrape_merge_import_all(tmp_path):
    # -- stage 1+2: per-source scrapes (substituted for the expensive network) --
    thorlabs = _source_db(tmp_path / "thorlabs.db", [
        dict(name="FB340-10", source="thorlabs", kind="bandpass", source_ref="FB340-10",
             properties={"Center Wavelength (nm)": "340", "Bandwidth (nm)": "10"},
             spectra={"transmission": _spec()}),
        dict(name="APD120A2", source="thorlabs", kind="apd",
             spectra={"responsivity": _spec()}),
    ])
    chroma = _source_db(tmp_path / "chroma.db", [
        dict(name="T495lpxr", source="chroma", kind="dichroic", spectra={"transmission": _spec()}),
        dict(name="SOLA", source="chroma", kind="light_source", spectra={"emission": _spec()}),
        # SAME bandpass as Thorlabs, punctuation variant → must dedup + union source
        dict(name="FB340 10", source="chroma", kind="bandpass", source_ref="ET340",
             properties={"Coating": "hard"}, spectra={"transmission": _spec()}),
    ])
    fpbase = _source_db(tmp_path / "fpbase.db", [
        dict(name="EGFP", source="fpbase", kind="fluorescent_protein", source_ref="egfp",
             properties={"Quantum Yield": "0.6"},
             spectra={"absorption": _spec(), "emission": _spec()}),
        dict(name="Alexa 488", source="fpbase", kind="organic_dye", cas="247144-90-7",
             spectra={"absorption": _spec()}),
    ])

    # -- stage 2.5: merge per-source DBs into one staging DB + dedup ----------
    staging = tmp_path / "spectra.db"
    summary = merge_all(str(staging), [thorlabs, chroma, fpbase], consolidate=True)
    assert summary[thorlabs] == 2 and summary[chroma] == 3 and summary[fpbase] == 2
    # the duplicate bandpass collapsed (7 ingested − 1 merged = 6)
    staged = FluorophoreDatabase(str(staging)); staged.connect()
    n_staged = staged.conn.execute(
        "SELECT COUNT(*) FROM probes WHERE deleted_at IS NULL"
    ).fetchone()[0]
    staged.close()
    assert n_staged == 6

    # -- stage 3: import ALL into a fresh MFDB (admin) with replace -----------
    mfdb_path = tmp_path / "live.mfdb"
    with MFDatabase(str(mfdb_path)) as d:
        d.add_user("user_default", "Default User", is_admin=1)
        # seed some junk to prove `replace` purges it
        d.conn.execute(
            "INSERT INTO probes (chromophore_name, category, verification_status, "
            "created_at, updated_at) VALUES ('JUNK', 'other', 'unverified', '', '')"
        )
        d.conn.commit()

    # session/permission gate: the active admin may add without a password
    from chisurf.plugins.core.mfdb_admin.gui.session import local_admin_status
    is_admin, _ = local_admin_status(str(mfdb_path), "user_default")
    assert is_admin

    with MFDatabase(str(mfdb_path)) as db:
        counts = db.import_reference_set(source_path=str(staging), replace=True)

    assert counts["purged"]["probes"] == 1  # JUNK removed
    assert counts["skipped"] == 0

    # -- verify the end state -------------------------------------------------
    with MFDatabase(str(mfdb_path)) as db:
        cats = {
            r["category"]: r["n"]
            for r in db.conn.execute(
                "SELECT category, COUNT(*) n FROM probes WHERE deleted_at IS NULL GROUP BY category"
            )
        }
        # every optical class made it through
        assert {"filter", "detector", "dichroic", "light_source", "protein", "organic_dye"} <= set(cats)
        # provenance populated for all, none empty
        empty = db.conn.execute(
            "SELECT COUNT(*) FROM probes WHERE (source IS NULL OR source='') AND deleted_at IS NULL"
        ).fetchone()[0]
        assert empty == 0
        # the cross-source duplicate carries BOTH sources
        bp = db.conn.execute(
            "SELECT source FROM probes WHERE chromophore_name LIKE 'FB340%' AND deleted_at IS NULL"
        ).fetchone()
        assert set(bp["source"].split(",")) == {"thorlabs", "chroma"}
        # spectra carried through (protein has abs+em, detector its responsivity)
        stypes = {
            r["spectrum_type"]
            for r in db.conn.execute("SELECT DISTINCT spectrum_type FROM spectra WHERE deleted_at IS NULL")
        }
        assert {"absorption", "emission", "transmission", "responsivity"} <= stypes
        # CAS canonicalized onto the dye
        cas = db.conn.execute(
            "SELECT property_value FROM optical_properties WHERE property_name='cas' "
            "AND deleted_at IS NULL LIMIT 1"
        ).fetchone()
        assert cas and cas[0] == "247144-90-7"
