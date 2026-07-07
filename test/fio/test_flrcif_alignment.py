"""Tests for PRD-02c flrCIF alignment of ChiSurf's parameter registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfdb.schema.pdbx_metadata import MmcifDictionary
from mfdb.chinet_adapter import (
    _load_parameter_registry,
    _lookup_flrcif_name,
)


REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "chisurf"
    / "core"
    / "settings"
    / "constants"
    / "parameter_registry.json"
)

from mfdb.schema.pdbx_metadata import MmcifDictionary as _MmcifDictionary

DIC_PATH = _MmcifDictionary.DATA_DIR / "mfdb_flr_ext.dic"


def test_registry_file_exists():
    """The renamed parameter registry file exists."""
    assert REGISTRY_PATH.is_file(), (
        f"parameter_registry.json not found at {REGISTRY_PATH}"
    )


def test_registry_has_no_fitting_parameters_reference():
    """The registry should use 'parameter_registry' not 'fitting_parameters'."""
    with open(REGISTRY_PATH, "r") as fh:
        data = json.load(fh)
    assert "version" in data
    assert "parameters" in data


def test_all_parameters_have_flrcif_item_id():
    """Every parameter entry in the registry has an flrcif_item_id mapping."""
    with open(REGISTRY_PATH, "r") as fh:
        data = json.load(fh)
    params = data.get("parameters", {})
    missing = [
        key for key, entry in params.items()
        if isinstance(entry, dict) and "flrcif_item_id" not in entry
    ]
    assert not missing, (
        f"{len(missing)} parameter(s) missing flrcif_item_id: {missing[:10]}"
    )


def test_flrcif_item_ids_are_unique():
    """No two parameters share the same flrcif_item_id."""
    with open(REGISTRY_PATH, "r") as fh:
        data = json.load(fh)
    params = data.get("parameters", {})
    seen = {}
    for key, entry in params.items():
        if not isinstance(entry, dict):
            continue
        flrcif = entry.get("flrcif_item_id")
        if flrcif:
            if flrcif in seen:
                pytest.fail(
                    f"Duplicate flrcif_item_id '{flrcif}' for "
                    f"'{key}' and '{seen[flrcif]}'"
                )
            seen[flrcif] = key


def test_dic_file_exists():
    """The extension dictionary file exists."""
    assert DIC_PATH.is_file(), f"mfdb_flr_ext.dic not found at {DIC_PATH}"


def test_dic_parses_correctly():
    """The extended dictionary parses without errors."""
    d = MmcifDictionary(DIC_PATH)
    assert d.get_category("flr_chisurf_parameter") is not None


def test_dic_contains_flr_chisurf_parameter_items():
    """The dictionary contains items from the flr_chisurf_parameter category."""
    d = MmcifDictionary(DIC_PATH)
    cat = d.get_category("flr_chisurf_parameter")
    assert cat is not None
    assert len(cat.items) > 0, "flr_chisurf_parameter category has no items"


def test_all_registry_ids_mapped_to_dic_items():
    """Every flrcif_item_id in the registry has a matching item in the .dic."""
    with open(REGISTRY_PATH, "r") as fh:
        data = json.load(fh)
    params = data.get("parameters", {})
    d = MmcifDictionary.load_bundled()
    missing = []
    for key, entry in params.items():
        if not isinstance(entry, dict):
            continue
        flrcif = entry.get("flrcif_item_id")
        if not isinstance(flrcif, str):
            continue
        item = d.get_item(flrcif)
        if item is None:
            missing.append((key, flrcif))
    assert not missing, (
        f"{len(missing)} flrcif_item_id(s) not found in bundled dictionaries: "
        f"{missing[:10]}"
    )


def test_dic_item_metadata_matches_registry():
    """Type codes on dictionary items should be float for chisurf parameters."""
    d = MmcifDictionary.load_bundled()
    cat = d.get_category("flr_chisurf_parameter")
    assert cat is not None
    for item in cat.items.values():
        assert item.type_code == "float", (
            f"Expected float type_code for {item.name}, got {item.type_code}"
        )


def test_lookup_flrcif_name_resolves_known_parameters():
    """Known parameter short names resolve to canonical identifiers."""
    assert _lookup_flrcif_name("E_FRET") == "_flr_chisurf_parameter.E_FRET"
    assert _lookup_flrcif_name("bg") == "_flr_chisurf_parameter.bg"
    assert _lookup_flrcif_name("R0") == "_flr_chisurf_parameter.R0"


def test_lookup_flrcif_name_returns_none_for_unknown():
    """Unknown parameter names return None."""
    assert _lookup_flrcif_name("__nonexistent__") is None
    assert _lookup_flrcif_name("") is None


def test_lookup_flrcif_name_resolves_family_prefixed():
    """Family-prefixed parameter names also resolve correctly."""
    result = _lookup_flrcif_name("fcs.N")
    assert result == "_flr_chisurf_parameter.fcs_N"


def test_dic_items_have_schema_bindings():
    """Every flr_chisurf_parameter item has table/column schema bindings."""
    d = MmcifDictionary.load_bundled()
    cat = d.get_category("flr_chisurf_parameter")
    assert cat is not None
    for item in cat.items.values():
        assert item.schema_table == "flr_chisurf_parameter", (
            f"{item.name} missing schema_table"
        )
        assert item.schema_column, (
            f"{item.name} missing schema_column"
        )
