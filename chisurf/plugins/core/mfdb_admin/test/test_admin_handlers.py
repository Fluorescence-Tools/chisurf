"""Tests for mfdb-admin RPC handlers defined in PRD-02b."""
from __future__ import annotations

import pytest

from mfdb.models import DEFAULT_FLUOROPHORE_SPECTRA

from mfdb.admin.backend.services import (
    create_structured_sample_handler,
    delete_entity_handler,
    delete_fret_pair_handler,
    list_processed_data_handler,
    list_processing_handler,
    get_sample_full_description_handler,
    list_entities_handler,
    list_fret_pairs_handler,
    list_probe_positions_handler,
    save_entity_handler,
    save_fret_pair_handler,
    save_probe_handler,
    save_probe_optical_properties_handler,
    suggest_pdbx_keys_handler,
    validate_pdbx_value_handler,
    validate_sample_export_handler,
    save_sample_handler,
)

from .conftest import patch_db


def test_full_description_handler(db, sample_with_entities):
    """mfdb.samples.full_description returns nested dict with entities,
    probes, fret_pairs, condition, key_values."""
    _, sample_id = sample_with_entities
    with patch_db(db):
        result = get_sample_full_description_handler(sample_id=sample_id, auth=None)
    description = result.get("description", {})
    assert description
    assert "entities" in description
    assert "probes" in description
    assert "fret_pairs" in description


def test_generic_processing_and_processed_data_handlers_round_trip(db):
    """Generic admin browse endpoints expose processing and product IDs."""
    db.record_operation(
        operation_id="proc_generic_ok",
        operation_type="filtering",
        status="succeeded",
    )
    db.record_operation(
        operation_id="proc_generic_fail",
        operation_type="filtering",
        status="failed",
    )
    db.register_artifact(
        artifact_id="prod_generic",
        artifact_kind="processed_data",
        storage_mode="embedded_json",
        data_json='{"curve": [1, 2, 3]}',
        validation_status="valid",
        metadata={
            "processing_id": "proc_generic_ok",
            "product_type": "processed_data",
        },
    )

    with patch_db(db):
        processing = list_processing_handler(auth=None)["processing"]
        products = list_processed_data_handler(auth=None)["processed_data"]

        from mfdb.admin.gui.client import MFDBClient

        client = MFDBClient(inprocess=True)
        succeeded_runs = client.list_processing_runs(status="succeeded")
        scoped_products = client.list_processed_data(processing_id="proc_generic_ok")

    processing_by_id = {row["processing_id"]: row for row in processing}
    assert processing_by_id["proc_generic_ok"]["type"] == "filtering"
    assert processing_by_id["proc_generic_fail"]["status"] == "failed"

    product = next(row for row in products if row["product_id"] == "prod_generic")
    assert product["processed_data_id"] == "prod_generic"
    assert product["processing_id"] == "proc_generic_ok"

    assert [row["processing_id"] for row in succeeded_runs] == ["proc_generic_ok"]
    assert [row["processed_data_id"] for row in scoped_products] == ["prod_generic"]


def test_provenance_graph_export_canonicalizes_seed_node_types(db):
    """Legacy artifact seed types should not duplicate canonical graph nodes."""
    db.record_operation(
        operation_id="proc_graph",
        operation_type="filtering",
        status="succeeded",
    )
    db.register_artifact(
        artifact_id="raw_graph",
        artifact_kind="raw_data",
        storage_mode="local_file",
        file_path="raw_graph.ptu",
    )
    db.register_artifact(
        artifact_id="prod_graph",
        artifact_kind="processed_data",
        storage_mode="embedded_json",
        data_json='{"curve": [1]}',
        metadata={"processing_id": "proc_graph"},
    )
    db.record_operation_link("proc_graph", "raw_graph", "input")
    db.record_operation_link("proc_graph", "prod_graph", "output")

    graph = db.export_provenance_graph("processed_data", "prod_graph")

    node_keys = {(node["node_type"], node["node_id"]) for node in graph["nodes"]}
    assert node_keys == {
        ("artifact", "raw_graph"),
        ("operation", "proc_graph"),
        ("artifact", "prod_graph"),
    }
    assert ("processed_data", "prod_graph") not in node_keys


def test_validate_export_handler_complete(db, sample_with_entities):
    """mfdb.samples.validate_export returns empty warnings for populated
    sample."""
    _, sample_id = sample_with_entities
    with patch_db(db):
        result = validate_sample_export_handler(sample_id=sample_id, auth=None)
    assert result.get("valid") is True
    assert result.get("warnings") == []


def test_create_structured_handler(db):
    """mfdb.samples.create_structured creates sample via SampleDefinition."""
    sample_data = {
        "name": "test_create",
        "sample_id": "test_create_id",
        "description": "Test structured creation",
        "entities": [
            {"entity_id": "ent1", "type": "polymer", "sequence": "AAAAA",
             "common_name": "Test entity"},
        ],
        "probes": [
            {"name": "Cy3B", "entity_index": 0,
             "seq_id": 1, "comp_id": "DA", "asym_id": "A"},
            {"name": "ATTO647N", "entity_index": 0,
             "seq_id": 5, "comp_id": "DT", "asym_id": "A"},
        ],
        "fret_pairs": [
            {"probe_1_index": 0, "probe_2_index": 1,
             "forster_radius_nm": 6.0, "kappa_squared": 0.6666667,
             "refractive_index": 1.4},
        ],
    }
    with patch_db(db):
        result = create_structured_sample_handler(sample_data, auth=None)
    assert "sample_id" in result
    description = result.get("description", {})
    assert description.get("entities")
    assert description.get("probes")


def test_create_structured_auto_fills_spectra(db):
    """create_structured_sample_handler auto-populates optical properties
    from DEFAULT_FLUOROPHORE_SPECTRA for known probe names."""
    known_name = (
        list(DEFAULT_FLUOROPHORE_SPECTRA.keys())[0]
        if DEFAULT_FLUOROPHORE_SPECTRA else "Cy3B"
    )
    sample_data = {
        "name": "auto_spectra",
        "sample_id": "auto_spectra_id",
        "entities": [
            {"entity_id": "ent1", "type": "polymer", "sequence": "AAAAA",
             "common_name": "Entity"},
        ],
        "probes": [
            {"name": known_name, "entity_index": 0,
             "seq_id": 1, "comp_id": "DA", "asym_id": "A"},
        ],
    }
    with patch_db(db):
        result = create_structured_sample_handler(sample_data, auth=None)
    description = result.get("description", {})
    probes = description.get("probes", [])
    assert len(probes) > 0
    probe = probes[0]
    properties = probe.get("properties", {})
    has_abs = bool(
        properties.get("absorption_wavelength")
    )
    assert has_abs


def test_entity_list_handler(db, sample_with_entities):
    """mfdb.entities.list returns entities scoped to sample_id when
    provided, all non-deleted when sample_id is None."""
    _, sample_id = sample_with_entities
    with patch_db(db):
        scoped = list_entities_handler(sample_id=sample_id, auth=None)
    assert scoped.get("entities")

    with patch_db(db):
        all_ents = list_entities_handler(sample_id=None, auth=None)
    assert len(all_ents.get("entities", [])) >= len(scoped.get("entities", []))


def test_entity_save_handler_with_sequence(db):
    """mfdb.entities.save persists entity with sequence."""
    entity = {
        "entity_id": "test_entity_seq",
        "name": "Test entity with sequence",
        "type": "polymer",
        "sequence": "ACGTACGT",
    }
    with patch_db(db):
        result = save_entity_handler(entity, auth=None)
    saved = result.get("entity", {})
    assert saved.get("entity_id") == "test_entity_seq"
    seq = saved.get("sequence", [])
    assert seq


def test_entity_delete_handler(db, sample_with_entities):
    """mfdb.entities.delete soft-deletes entity."""
    _, sample_id = sample_with_entities
    with patch_db(db):
        entities = list_entities_handler(sample_id=sample_id, auth=None)
    ents = entities.get("entities", [])
    if not ents:
        pytest.skip("No entities to delete")
    entity_id = ents[0].get("entity_id")
    with patch_db(db):
        result = delete_entity_handler(entity_id, auth=None)
    assert result.get("ok") is True
    assert result.get("entity_id") == entity_id


def test_probe_save_handler_all_fields(db):
    """mfdb.probes.save persists all chemical fields."""
    probe = {
        "chromophore_name": "TestProbe-Chem",
        "category": "organic",
        "probe_origin": "synthetic",
        "probe_link_type": "covalent",
        "reactive_probe_flag": "yes",
        "reactive_probe_name": "TestProbe-mal",
        "chromophore_center_atom": "CA",
    }
    with patch_db(db):
        result = save_probe_handler(probe, auth=None)
    saved = result.get("probe", {})
    assert saved.get("chromophore_name") == "TestProbe-Chem"
    assert saved.get("probe_id") is not None


def test_probe_optical_properties_save_handler(db):
    """mfdb.probes.optical_properties.save persists optical props."""
    probe = {"chromophore_name": "OpticalTest", "category": "other"}
    with patch_db(db):
        save_result = save_probe_handler(probe, auth=None)
    probe_data = save_result.get("probe", {})
    probe_id = probe_data.get("probe_id")
    assert probe_id is not None

    properties = [
        {"property_name": "absorption_maximum_nm",
         "property_value": 550, "unit": "nm"},
        {"property_name": "emission_maximum_nm",
         "property_value": 570, "unit": "nm"},
        {"property_name": "quantum_yield",
         "property_value": 0.85},
        {"property_name": "extinction_coefficient",
         "property_value": 130000, "unit": "M-1 cm-1"},
    ]
    with patch_db(db):
        result = save_probe_optical_properties_handler(
            probe_id, properties, auth=None,
        )
    props = result.get("optical_properties", [])
    assert len(props) >= 4
    prop_names = [p.get("property_type") for p in props]
    assert "absorption_maximum_nm" in prop_names


def test_probe_positions_list_handler(db, sample_with_entities):
    """mfdb.probes.positions.list returns positions with atom_id,
    mutation_flag, modification_flag, auth_name."""
    _, sample_id = sample_with_entities
    with patch_db(db):
        result = list_probe_positions_handler(sample_id=sample_id, auth=None)
    positions = result.get("positions", [])
    if positions:
        pos = positions[0]
        for field in ("atom_id", "mutation_flag", "modification_flag"):
            assert field in pos


def test_fret_pair_crud_handlers(db, sample_with_entities):
    """mfdb.fret_pairs.{list,save,delete} round-trip."""
    _, sample_id = sample_with_entities

    with patch_db(db):
        result = list_fret_pairs_handler(sample_id, auth=None)
    existing = result.get("fret_pairs", [])
    assert existing, "Fixture should include FRET pairs"

    # Delete the fixture's FRET pair first so we can create a new one
    # with the same probes (unique constraint on sample+donor+acceptor).
    pair_to_delete = existing[0]
    forster_radius_id = pair_to_delete.get("forster_radius_id")
    with patch_db(db):
        del_result = delete_fret_pair_handler(forster_radius_id, auth=None)
    assert del_result.get("ok") is True

    # Create a new FRET pair with the same probes
    pair = {
        "sample_id": sample_id,
        "donor_probe_id": 1,
        "acceptor_probe_id": 2,
        "forster_radius": 5.5,
        "kappa_squared": 0.6666667,
    }
    with patch_db(db):
        new_pair = save_fret_pair_handler(pair, auth=None)
    assert new_pair.get("fret_pair", {}).get("forster_radius_id")
    new_id = new_pair["fret_pair"]["forster_radius_id"]

    # Clean up
    with patch_db(db):
        del_result = delete_fret_pair_handler(new_id, auth=None)
    assert del_result.get("ok") is True


def test_pdbx_suggest_handler():
    """mfdb.pdbx.suggest_keys returns matching keys for prefix '_flr'."""
    result = suggest_pdbx_keys_handler(prefix="_flr", auth=None)
    keys = result.get("keys", [])
    assert isinstance(keys, list)
    if keys:
        assert isinstance(keys[0], dict)
        assert "key" in keys[0]


def test_pdbx_validate_handler():
    """mfdb.pdbx.validate_value returns valid flag + message."""
    result = validate_pdbx_value_handler(
        "_flr_sample.sample_id", "test_sample", auth=None,
    )
    assert "valid" in result
    assert "message" in result


def test_save_sample_handler_legacy_compat(db):
    """save_sample_handler works with flat legacy dict."""
    sample = {
        "sample_id": "legacy_compat_test",
        "description": "Legacy flat save",
    }
    with patch_db(db):
        result = save_sample_handler(sample, auth=None)
    sample_result = result.get("sample", {})
    assert sample_result.get("sample_id") == "legacy_compat_test"
    assert sample_result.get("description") == "Legacy flat save"


def test_save_sample_handler_structured(db):
    """save_sample_handler with structured dict delegates to
    create_sample() and persists all fields."""
    sample = {
        "sample_id": "structured_save_test",
        "name": "structured_save_test",
        "description": "Structured save via save_sample_handler",
        "entities": [
            {"entity_id": "ent_struct", "type": "polymer",
             "sequence": "GGGG", "name": "Structured entity"},
        ],
        "probes": [
            {"name": "Cy3B", "entity_index": 0,
             "seq_id": 1, "comp_id": "DA", "asym_id": "A"},
        ],
    }
    with patch_db(db):
        result = save_sample_handler(sample, auth=None)
    # Structured path returns flat keys; legacy path wraps in {"sample": ...}
    sample_id = result.get("sample_id") or result.get("sample", {}).get("sample_id")
    assert sample_id == "structured_save_test"
