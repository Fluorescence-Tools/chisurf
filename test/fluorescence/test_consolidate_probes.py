"""Probe de-duplication (``MFDatabase.consolidate_probes``).

The maintainer's rule: prefer *simple* dedups (the same catalogue part scraped
from two sources) over *complex* ones (fuzzy-merging similar-but-distinct
fluorescent proteins). These tests pin that behaviour:

- exact-name duplicates within a category merge, and their metadata is *merged*
  (union of optical properties + sources), never dropped;
- merges never cross a category boundary;
- distinct fluorescent proteins with lookalike names are left alone;
- reactive-group suffix merging is opt-in (``aggressive=True``) and never
  touches proteins.
"""
from __future__ import annotations

import pytest

from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def db():
    database = MFDatabase(":memory:")
    yield database
    database.close()


def _add(db, name, category, source, source_ref=None, props=None):
    pid = db.conn.execute(
        "INSERT INTO probes (chromophore_name, category, source, source_ref, "
        "verification_status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'unverified', '', '')",
        (name, category, source, source_ref),
    ).lastrowid
    for k, v in (props or {}).items():
        db.conn.execute(
            "INSERT INTO optical_properties (probe_id, property_name, property_value, created_at, updated_at) "
            "VALUES (?, ?, ?, '', '')",
            (pid, k, v),
        )
    db.conn.commit()
    return pid


def _names(db, category=None):
    q = "SELECT chromophore_name FROM probes WHERE deleted_at IS NULL"
    if category:
        q += f" AND category = '{category}'"
    return sorted(r[0] for r in db.conn.execute(q))


def _props(db, pid):
    return {
        r[0]: r[1]
        for r in db.conn.execute(
            "SELECT property_name, property_value FROM optical_properties "
            "WHERE probe_id = ? AND deleted_at IS NULL",
            (pid,),
        )
    }


def test_simple_duplicate_filter_merges_and_merges_metadata(db):
    """Same filter from two sources → one probe carrying both sources + all props."""
    _add(db, "FB340-10", "filter", "thorlabs", "FB340-10",
         props={"center_wavelength": "340", "bandwidth": "10"})
    _add(db, "FB340-10", "filter", "3doptix", "http://x",
         props={"Coating": "hard", "bandwidth": "10"})

    res = db.consolidate_probes()
    assert res["deleted_probes"] == 1
    rows = db.conn.execute(
        "SELECT probe_id, source FROM probes WHERE deleted_at IS NULL"
    ).fetchall()
    assert len(rows) == 1
    # both sources preserved (union)
    assert set(rows[0]["source"].split(",")) == {"thorlabs", "3doptix"}
    # union of optical properties from both copies
    props = _props(db, rows[0]["probe_id"])
    assert props["center_wavelength"] == "340"
    assert props["bandwidth"] == "10"
    assert props["Coating"] == "hard"


def test_merge_never_crosses_category(db):
    """A filter and a dye that normalize the same must NOT merge."""
    _add(db, "X100", "filter", "chroma")
    _add(db, "X100", "organic_dye", "atto")
    res = db.consolidate_probes()
    assert res["deleted_probes"] == 0
    assert len(_names(db)) == 2


def test_similar_proteins_not_merged(db):
    """Distinct proteins with lookalike names are preserved (even aggressive)."""
    _add(db, "mCherry", "protein", "fpbase")
    _add(db, "mCherry2", "protein", "fpbase")
    _add(db, "EGFP", "protein", "fpbase")
    db.consolidate_probes(aggressive=True)
    assert _names(db, "protein") == ["EGFP", "mCherry", "mCherry2"]


def test_case_and_punctuation_insensitive_simple_merge(db):
    """'Alexa 488' and 'alexa-488' are the same dye."""
    _add(db, "Alexa 488", "organic_dye", "fpbase")
    _add(db, "alexa-488", "organic_dye", "photochemcad")
    res = db.consolidate_probes()
    assert res["deleted_probes"] == 1
    assert len(_names(db, "organic_dye")) == 1


def test_aggressive_merges_reactive_conjugates_for_dyes(db):
    """Opt-in: 'ATTO 647N NHS ester' collapses onto 'ATTO 647N' (dye only)."""
    _add(db, "ATTO 647N", "organic_dye", "atto")
    _add(db, "ATTO 647N NHS ester", "organic_dye", "atto")
    # simple pass leaves them separate
    assert db.consolidate_probes(aggressive=False)["deleted_probes"] == 0
    assert len(_names(db, "organic_dye")) == 2
    # aggressive pass merges the conjugate, keeping the alternate name as a synonym
    res = db.consolidate_probes(aggressive=True)
    assert res["deleted_probes"] == 1
    survivors = _names(db, "organic_dye")
    assert survivors == ["ATTO 647N"]
    pid = db.conn.execute(
        "SELECT probe_id FROM probes WHERE deleted_at IS NULL"
    ).fetchone()[0]
    assert _props(db, pid).get("synonym") == "ATTO 647N NHS ester"
