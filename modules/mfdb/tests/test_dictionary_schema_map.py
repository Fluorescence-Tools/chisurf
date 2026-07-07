"""Tests for dictionary-authoritative MFDB schema mapping."""

from __future__ import annotations

import inspect

import pytest

from mfdb.repository import MFDatabase
from mfdb.schema import dictionary_schema_map as mapping_module
from mfdb.schema.dictionary_schema_map import (
    DictionarySchemaMap,
    MappedColumn,
    build_dictionary_schema_map,
    map_dictionary_item,
    unsupported_flr_items,
)


@pytest.fixture()
def mfdb_path(tmp_path):
    """Return a temporary MFDB path with the canonical schema installed."""
    path = tmp_path / "test.db"
    db = MFDatabase(str(path))
    db.close()
    return path


@pytest.fixture()
def mapper(mfdb_path):
    """Return a mapper built from dictionaries and live schema."""
    return build_dictionary_schema_map(mfdb_path)


def test_no_python_override_tables_exist():
    """Schema-name differences must live in .dic metadata, not Python maps."""
    assert not hasattr(mapping_module, "OVERRIDE_MAP")
    assert not hasattr(mapping_module, "UNSUPPORTED_FLR_CATEGORIES")
    source = inspect.getsource(mapping_module)
    assert "OVERRIDE_MAP" not in source
    assert "UNSUPPORTED_FLR_CATEGORIES" not in source


def test_build_dictionary_schema_map_returns_registry(mapper):
    """Mapping registry builds from dictionary and live schema."""
    assert isinstance(mapper, DictionarySchemaMap)
    assert mapper.schema
    assert mapper.dictionary.get_item("_flr_sample.id") is not None


@pytest.mark.parametrize(
    ("dictionary_item", "table_name", "column_name", "source"),
    [
        ("_flr_sample.id", "flr_sample", "sample_id", "dictionary"),
        ("_flr_sample.sample_description", "flr_sample", "description", "dictionary"),
        ("_flr_sample.solvent_phase", "flr_sample", "solvent_phase", "direct"),
        ("_flr_sample_condition.id", "flr_sample_condition", "condition_id", "dictionary"),
        ("_flr_sample_condition.ph", "flr_sample_condition", "ph", "dictionary"),
        ("_flr_sample_condition.temperature", "flr_sample_condition", "temperature", "dictionary"),
        ("_flr_fret_forster_radius.id", "flr_fret_forster_radius", "forster_radius_id", "dictionary"),
        ("_flr_fret_forster_radius.kappa_squared", "flr_fret_forster_radius", "kappa_squared", "dictionary"),
        ("_flr_fret_forster_radius.index_of_refraction", "flr_fret_forster_radius", "index_of_refraction", "dictionary"),
        ("_flr_poly_probe_position.seq_id", "flr_poly_probe_position", "residue_number", "dictionary"),
        ("_flr_poly_probe_position.comp_id", "flr_poly_probe_position", "residue_name", "dictionary"),
        ("_flr_probe_list.probe_origin", "probes", "probe_origin", "dictionary"),
        ("_flr_probe_list.probe_link_type", "probes", "probe_link_type", "dictionary"),
        ("_flr_probe_list.reactive_probe_flag", "probes", "reactive_probe_flag", "dictionary"),
        ("_flr_sample_probe_details.fluorophore_type", "flr_sample_probe", "fluorophore_type", "dictionary"),
        ("_ihm_chemical_component_descriptor.smiles", "chem_descriptors", "descriptor", "dictionary"),
    ],
)
def test_dictionary_declares_schema_bindings(
    mapper, dictionary_item, table_name, column_name, source
):
    """Important dictionary items map to live schema columns programmatically."""
    mapped = mapper.map_dictionary_item(dictionary_item)
    assert mapped == MappedColumn(
        dictionary_name=dictionary_item,
        table_name=table_name,
        column_name=column_name,
        category=dictionary_item[1:].split(".", 1)[0],
        attribute=dictionary_item.split(".", 1)[1],
        type_code=mapped.type_code,
        description=mapped.description,
        source=source,
    )
    ok, message = mapper.validate_mapping(dictionary_item)
    assert ok, message


@pytest.mark.parametrize(
    ("dictionary_item", "default_value", "enum_values"),
    [
        ("_flr_probe_list.probe_origin", "extrinsic", {"intrinsic", "extrinsic"}),
        ("_flr_probe_list.probe_link_type", "covalent", {"covalent", "ligand"}),
        ("_flr_probe_list.reactive_probe_flag", "no", {"yes", "no"}),
        ("_flr_sample_probe_details.fluorophore_type", "unspecified", {"donor", "acceptor", "unspecified"}),
        ("_flr_poly_probe_position.mutation_flag", "no", {"yes", "no"}),
        ("_flr_poly_probe_position.modification_flag", "no", {"yes", "no"}),
    ],
)
def test_dictionary_declares_prd02_defaults_and_enums(
    mapper, dictionary_item, default_value, enum_values
):
    """PRD-02 defaults and enums are dictionary metadata, not Python literals."""
    item = mapper.dictionary.get_item(dictionary_item)
    assert item is not None
    assert item.default_value == default_value
    assert enum_values.issubset(set(item.enumerations))


def test_dictionary_item_lookup_accepts_non_underscored_form(mapper):
    """Callers may omit the leading underscore."""
    mapped = mapper.map_dictionary_item("flr_sample.id")
    assert mapped is not None
    assert mapped.table_name == "flr_sample"
    assert mapped.column_name == "sample_id"


def test_every_mapped_item_points_to_live_column(mapper):
    """Generated mappings cannot point at nonexistent database columns."""
    for mapped in mapper.get_mapped_flr_items():
        assert mapped.table_name in mapper.schema, mapped
        assert mapped.column_name in mapper.schema[mapped.table_name], mapped


def test_every_flr_category_is_live_or_reported_unsupported(mapper):
    """Absent flrCIF categories are derived from schema, not hard-coded."""
    missing_categories = set(mapper.categories_without_live_table())
    for category in mapper.dictionary.flr_categories():
        if category in mapper.schema:
            continue
        category_items = mapper.dictionary.get_category(category).items.values()
        has_live_bound_item = any(
            item.schema_table in mapper.schema for item in category_items
        )
        assert has_live_bound_item or category in missing_categories


def test_unsupported_items_are_derived_from_live_schema(mfdb_path):
    """Unsupported item reporting is computed from the live schema."""
    mapper = build_dictionary_schema_map(mfdb_path)
    unsupported = unsupported_flr_items(mfdb_path)
    assert isinstance(unsupported, list)
    assert unsupported
    live_tables = set(mapper.schema)
    assert any(
        item.candidate_table not in live_tables
        for item in mapper.get_unmapped_flr_items()
    )


def test_convenience_function_uses_same_mapping(mfdb_path):
    """Module-level lookup uses the dictionary-authoritative mapper."""
    mapped = map_dictionary_item("_flr_sample_condition.ph", mfdb_path)
    assert mapped is not None
    assert mapped.table_name == "flr_sample_condition"
    assert mapped.column_name == "ph"


def test_unknown_item_does_not_map(mapper):
    """Unknown dictionary items do not produce fabricated mappings."""
    assert mapper.map_dictionary_item("_nonexistent.item") is None
    ok, message = mapper.validate_mapping("_nonexistent.item")
    assert not ok
    assert "Unknown dictionary item" in message
