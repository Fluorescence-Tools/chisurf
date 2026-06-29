"""The canonical ingestion contract every spectra scraper must use.

``register_component`` is the single entry point through which all scrapers
(fpbase, chroma, thorlabs, 3doptix, atto, photochemcad, …) write optical
components. These tests pin the guarantees that make every scraper populate the
database *the same way*:

- a consistent canonical ``category`` (fluorophore subtypes + filter / dichroic
  / detector / light_source) derived from the shared ``COMPONENT_KINDS``
  taxonomy — so filters/dichroics/detectors no longer collapse into "other";
- provenance (``source`` / ``source_ref`` / ``retrieved_at``) on every probe;
- the granular ``component_kind`` preserved as an optical property;
- canonical optical-property keys (``cut_on``/``center_wavelength``/…);
- spectra recorded under a uniform spectrum-type vocabulary.

No network access — all data is synthetic.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    COMPONENT_KINDS,
    FluorophoreDatabase,
    resolve_component_kind,
)


@pytest.fixture
def db():
    """A throwaway file-backed adapter database."""
    fd = tempfile.mkdtemp()
    path = Path(fd) / "spectra.db"
    database = FluorophoreDatabase(str(path))
    database.connect()
    yield database
    database.close()


def _spec(n: int = 16):
    wl = np.linspace(400.0, 700.0, n)
    return wl, np.ones_like(wl)


def _probe_row(db, name):
    return db.conn.execute(
        "SELECT * FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
        (name,),
    ).fetchone()


def _props(db, probe_id):
    return {
        r["property_name"]: r["property_value"]
        for r in db.conn.execute(
            "SELECT property_name, property_value FROM optical_properties "
            "WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        )
    }


# -- taxonomy ---------------------------------------------------------------

def test_taxonomy_categories_are_canonical():
    """Every kind resolves to one of the GUI radio categories."""
    allowed = {
        "protein", "organic_dye", "quantum_dot", "nanoparticle",
        "filter", "dichroic", "detector", "light_source", "other",
    }
    for kind, (category, spectrum, label) in COMPONENT_KINDS.items():
        assert category in allowed, f"{kind} -> bad category {category}"
        assert spectrum and label


def test_resolve_unknown_kind_falls_back_to_other():
    assert resolve_component_kind("not-a-real-kind")[0] == "other"
    assert resolve_component_kind(None)[0] == "other"


# -- per-class ingestion ----------------------------------------------------

@pytest.mark.parametrize(
    "kind, expected_category, expected_spectrum",
    [
        ("fluorescent_protein", "protein", "absorption"),
        ("organic_dye", "organic_dye", "absorption"),
        ("bandpass", "filter", "transmission"),
        ("longpass", "filter", "transmission"),
        ("dichroic", "dichroic", "transmission"),
        ("mirror", "dichroic", "reflectance"),
        ("apd", "detector", "responsivity"),
        ("light_source", "light_source", "emission"),
    ],
)
def test_register_component_sets_category_and_default_spectrum(
    db, kind, expected_category, expected_spectrum
):
    """A single (x, y) spectrum is stored under the kind's default type."""
    db.register_component(
        name=f"item-{kind}",
        source="unit",
        kind=kind,
        source_ref="ref-1",
        spectra=_spec(),
    )
    row = _probe_row(db, f"item-{kind}")
    assert row["category"] == expected_category
    assert row["source"] == "unit"
    assert row["source_ref"] == "ref-1"
    assert row["retrieved_at"]  # provenance timestamp recorded

    stypes = [
        r["spectrum_type"]
        for r in db.conn.execute(
            "SELECT spectrum_type FROM spectra WHERE probe_id = ?",
            (row["probe_id"],),
        )
    ]
    assert stypes == [expected_spectrum]

    # granular kind preserved as a property
    assert _props(db, row["probe_id"]).get("component_kind") == kind


def test_register_component_preserves_explicit_spectrum_types(db):
    """A {type: (x,y)} mapping keeps each declared spectrum type."""
    db.register_component(
        name="EGFP",
        source="fpbase",
        kind="fluorescent_protein",
        source_ref="egfp",
        spectra={"absorption": _spec(), "emission": _spec()},
    )
    row = _probe_row(db, "EGFP")
    stypes = {
        r["spectrum_type"]
        for r in db.conn.execute(
            "SELECT spectrum_type FROM spectra WHERE probe_id = ?",
            (row["probe_id"],),
        )
    }
    assert stypes == {"absorption", "emission"}


def test_register_component_canonicalizes_filter_properties(db):
    """Heterogeneous scraped property names collapse to canonical keys."""
    db.register_component(
        name="FB340-10",
        source="thorlabs",
        kind="bandpass",
        properties={
            "Center Wavelength (nm)": "340",
            "Bandwidth (nm)": "10",
            "Cut-On Wavelength (nm)": "335",
        },
        spectra=_spec(),
    )
    props = _props(db, _probe_row(db, "FB340-10")["probe_id"])
    assert props["center_wavelength"] == "340"
    assert props["bandwidth"] == "10"
    assert props["cut_on"] == "335"


def test_register_component_skips_empty_properties_and_spectra(db):
    """None/empty values must not create rows."""
    db.register_component(
        name="sparse",
        source="unit",
        kind="organic_dye",
        properties={"qy": "", "abs_max": None, "em_max": "509"},
        spectra={"absorption": None, "emission": _spec()},
    )
    row = _probe_row(db, "sparse")
    props = _props(db, row["probe_id"])
    assert "qy" not in props and "abs_max" not in props
    assert props["em_max"] == "509"
    stypes = [
        r["spectrum_type"]
        for r in db.conn.execute(
            "SELECT spectrum_type FROM spectra WHERE probe_id = ?",
            (row["probe_id"],),
        )
    ]
    assert stypes == ["emission"]


def test_reregister_keeps_provenance_and_refreshes_spectra(db):
    """A second pass without provenance must not wipe the stored source."""
    db.register_component(
        name="dup", source="chroma", kind="emission_filter",
        source_ref="EM-1", spectra=_spec(),
    )
    pid1 = _probe_row(db, "dup")["probe_id"]
    # re-run via plain add_probe (no provenance args) — source must persist
    db.add_probe(chromophore_name="dup", type_id=_probe_row(db, "dup")["type_id"])
    row = _probe_row(db, "dup")
    assert row["probe_id"] == pid1
    assert row["source"] == "chroma"
    assert row["source_ref"] == "EM-1"


def test_cas_is_canonicalized_and_looked_up(db):
    """Foundation for PRD-45: CAS aliases collapse to `cas` and are queryable."""
    db.register_component(name="Benzene", source="photochemcad", kind="organic_dye",
                          properties={"CAS": "71-43-2"}, spectra=_spec())
    db.register_component(name="Fluorescein", source="atto", kind="organic_dye",
                          cas="2321-07-5", spectra=_spec())
    db.conn.commit()
    # the non-canonical "CAS" key collapsed to "cas"
    pid = _probe_row(db, "Benzene")["probe_id"]
    assert _props(db, pid).get("cas") == "71-43-2"
    # whitespace-tolerant lookup
    assert [p["chromophore_name"] for p in db.find_probes_by_cas(" 71-43-2 ")] == ["Benzene"]
    assert [p["chromophore_name"] for p in db.find_probes_by_cas("2321-07-5")] == ["Fluorescein"]


def test_categories_match_gui_registry(db):
    """Every category produced here is selectable by some GUI radio tab."""
    import json

    registry_path = (
        Path(__file__).resolve().parents[3]
        / "core" / "mfdb_admin" / "gui" / "optical_components" / "components.json"
    )
    if not registry_path.exists():
        pytest.skip("optical-components GUI registry not present")
    registry = json.loads(registry_path.read_text())
    gui_categories = {c for entry in registry for c in entry["categories"]}
    produced = {cat for (cat, _s, _l) in COMPONENT_KINDS.values()}
    # everything a scraper can emit is reachable from a radio tab
    unreachable = produced - gui_categories - {"other"}
    assert not unreachable, f"categories with no GUI tab: {unreachable}"
