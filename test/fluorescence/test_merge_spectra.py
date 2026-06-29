"""Parallel-scrape merge step (``download.merge.merge_database`` / ``merge_all``).

Scrapers run in parallel into per-source DBs; the final step merges them into one
canonical ``spectra.db`` and de-duplicates across sources. These tests pin that
the merge keeps every record canonical (category / source / spectra) and that a
catalogue part scraped by two sources collapses to one probe carrying both
sources.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import FluorophoreDatabase
from chisurf.plugins.spectra_downloader.download.merge import merge_all


def _spec(n=12):
    wl = np.linspace(400.0, 700.0, n)
    return wl, np.ones_like(wl)


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp()
    yield Path(d)


def _source_db(path, items):
    db = FluorophoreDatabase(str(path))
    with db:
        for kw in items:
            db.register_component(spectra=_spec(), **kw)
        db.conn.commit()
    db.close()
    return str(path)


def test_merge_all_unions_and_dedups(workdir):
    # Source A (thorlabs): a bandpass filter
    a = _source_db(workdir / "thorlabs.db", [
        dict(name="FB340-10", source="thorlabs", kind="bandpass", source_ref="FB340-10"),
        dict(name="APD120A2", source="thorlabs", kind="apd"),
    ])
    # Source B (chroma): a dye + the SAME filter under a punctuation variant
    b = _source_db(workdir / "chroma.db", [
        dict(name="Alexa 488", source="chroma", kind="organic_dye"),
        dict(name="FB340 10", source="chroma", kind="bandpass", source_ref="ET340"),
    ])

    target = workdir / "spectra.db"
    summary = merge_all(str(target), [a, b], consolidate=True)
    assert summary[a] == 2 and summary[b] == 2

    db = FluorophoreDatabase(str(target))
    try:
        rows = {
            r["chromophore_name"]: r
            for r in db.conn.execute(
                "SELECT chromophore_name, category, source FROM probes WHERE deleted_at IS NULL"
            )
        }
        # the duplicate bandpass collapsed to one probe carrying BOTH sources
        assert "FB340-10" in rows or "FB340 10" in rows
        bp = rows.get("FB340-10") or rows.get("FB340 10")
        assert bp["category"] == "filter"
        assert set(bp["source"].split(",")) == {"thorlabs", "chroma"}
        # distinct components preserved with their canonical categories
        assert rows["APD120A2"]["category"] == "detector"
        assert rows["Alexa 488"]["category"] == "organic_dye"
        # 4 ingested − 1 duplicate merged = 3
        assert len(rows) == 3
    finally:
        db.close()
