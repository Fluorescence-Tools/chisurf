"""Tests for the MFDB result registry."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mfdb.base import MFDBClientBase
from mfdb.models import SampleDefinition
from mfdb.payload_codec import PayloadSchemaError, encode_payload
from mfdb.payload_models import BurstSelection, FcsCorrelation
from mfdb.repository import MFDatabase
from mfdb.result_registry import (
    LinkValidationError,
    read_result,
    register_calibration,
    register_fit_result,
    register_processed_data,
    register_raw_measurement,
    register_result,
    set_global_db,
)
from mfdb.sample_manager import create_sample, get_artifacts_for_sample


@pytest.fixture
def db():
    """Create a temporary MFDatabase for each result registry test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        database = MFDatabase(os.path.join(tmpdir, "test.db"))
        try:
            yield database
        finally:
            database.close()


def test_register_result_with_dict_data(db):
    """Dictionary payloads create msgpack object-backed artifacts."""
    art_id = register_result(
        kind="processed_data",
        data={"x": [1, 2, 3], "y": [4, 5, 6]},
        operation_type="analysis",
        db=db,
    )

    assert art_id
    row = db.conn.execute(
        "SELECT artifact_kind, data_format, object_uuid FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,),
    ).fetchone()
    assert row is not None
    assert row["artifact_kind"] == "processed_data"
    assert row["data_format"] == "msgpack"
    assert row["object_uuid"]
    payload = read_result(db, art_id)
    assert payload.KIND == "generic_curve"
    assert payload.x.tolist() == [1.0, 2.0, 3.0]


def test_register_result_with_file(db, tmp_path):
    """File payloads infer the data format from the file suffix."""
    file_path = tmp_path / "test.ptu"
    file_path.write_bytes(b"fake ptu data")

    art_id = register_result(
        kind="raw_measurement",
        data=str(file_path),
        operation_type="measurement_import",
        db=db,
    )

    assert art_id
    row = db.conn.execute(
        "SELECT data_format FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,),
    ).fetchone()
    assert row["data_format"] == "ptu"


def test_mfdatabase_implements_client_base(db, tmp_path):
    """MFDatabase should satisfy the core MFDB client contract."""
    assert isinstance(db, MFDBClientBase)

    sample_id = create_sample(db, SampleDefinition(name="contract sample"))
    file_path = tmp_path / "raw.spc"
    file_path.write_bytes(b"contract raw data")
    art_id = register_raw_measurement(str(file_path), sample_id=sample_id, db=db)
    artifact = db.get_artifact(art_id)

    assert db.sample_exists(sample_id) is True
    assert artifact is not None
    assert db.lookup_sample_by_md5(artifact["checksum"]) is None
    db.set_object_sample_id(artifact["object_uuid"], sample_id)
    assert db.lookup_sample_by_md5(artifact["checksum"]) == sample_id
    assert db.find_raw_artifact_by_md5(artifact["checksum"]) == art_id


def test_register_result_creates_derived_from_edge(db):
    """Child results receive a derived_from edge to their parent artifact."""
    parent_id = register_result(
        kind="raw_measurement",
        data=b"raw data",
        operation_type="measurement_import",
        db=db,
    )
    child_id = register_result(
        kind="processed_data",
        data=b"processed",
        parent_artifact_id=parent_id,
        operation_type="analysis",
        db=db,
    )

    row = db.conn.execute(
        """SELECT relationship_type FROM mfdb_edge
           WHERE source_node_id = ? AND target_node_id = ? AND deleted_at IS NULL""",
        (child_id, parent_id),
    ).fetchone()
    assert row is not None
    assert row["relationship_type"] == "derived_from"


def test_register_result_input_link(db):
    """Child operations receive an input link to the parent artifact."""
    parent_id = register_result(
        kind="raw_measurement",
        data=b"raw",
        operation_type="measurement_import",
        db=db,
    )
    child_id = register_result(
        kind="processed_data",
        data=b"proc",
        parent_artifact_id=parent_id,
        operation_type="analysis",
        db=db,
    )

    row = db.conn.execute(
        """SELECT input_link.direction FROM mfdb_operation_artifact input_link
           JOIN mfdb_operation_artifact output_link
             ON output_link.operation_id = input_link.operation_id
           WHERE input_link.artifact_id = ?
             AND input_link.direction = 'input'
             AND output_link.artifact_id = ?
             AND output_link.direction = 'output'""",
        (parent_id, child_id),
    ).fetchone()
    assert row is not None


def test_register_result_with_sample(db):
    """Results can be linked to an existing sample."""
    sample_id = create_sample(db, SampleDefinition(name="test_sample"))

    art_id = register_result(
        kind="raw_measurement",
        data=b"data",
        sample_id=sample_id,
        operation_type="measurement_import",
        db=db,
    )

    assert art_id in get_artifacts_for_sample(db, sample_id)


def test_register_result_with_parameters(db):
    """Scalar and structured parameters are written to mfdb_parameter."""
    art_id = register_result(
        kind="fit_result",
        data={"chi2": 1.05},
        operation_type="local_fit",
        parameters={
            "tau1": {"value": 4.0, "error": 0.1, "fixed": False, "bounds": [0, 20], "units": "ns"},
            "amplitude": 0.85,
        },
        db=db,
    )

    rows = db.conn.execute(
        """SELECT p.name, p.value FROM mfdb_parameter p
           JOIN mfdb_operation_artifact oa ON oa.operation_id = p.operation_id
           WHERE oa.artifact_id = ? AND oa.direction = 'output'""",
        (art_id,),
    ).fetchall()
    values = {row["name"]: row["value"] for row in rows}
    assert values["tau1"] == 4.0
    assert values["amplitude"] == 0.85


def test_register_result_missing_parent_rolls_back_all_rows(db):
    """Invalid provenance links fail loudly before partial result rows are committed."""
    # register_result is fail-loud (PRD-10/PRD-25): a bad link raises rather than
    # silently dropping data, and nothing is persisted.
    with pytest.raises(LinkValidationError):
        register_result(
            kind="processed_data",
            data={"x": [1, 2], "y": [3, 4]},
            parent_artifact_id="missing_parent",
            operation_type="analysis",
            db=db,
        )

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation_artifact").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_object").fetchone()[0] == 0


def test_register_result_missing_sample_rolls_back_all_rows(db):
    """Invalid sample links raise and do not leave orphaned artifacts."""
    with pytest.raises(LinkValidationError):
        register_result(
            kind="processed_data",
            data={"x": [1, 2], "y": [3, 4]},
            sample_id="missing_sample",
            operation_type="analysis",
            db=db,
        )

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_object").fetchone()[0] == 0


def test_register_result_rejects_unreadable_object_dtype_table(db):
    """Generic table fallback must not write object-dtype msgpack artifacts."""
    with pytest.raises(ValueError):
        register_result(
            kind="fit_result",
            data={"model": {"a": 1}},
            operation_type="analysis",
            db=db,
        )

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_object").fetchone()[0] == 0


def test_register_result_rejects_artifact_payload_kind_mismatch(db):
    """Known payload dataclasses cannot be stored under a different payload artifact kind."""
    payload = FcsCorrelation(
        lag=np.array([1e-6, 2e-6], dtype=np.float64),
        correlation=np.array([1.0, 0.9], dtype=np.float64),
    )

    with pytest.raises(ValueError):
        register_result(kind="spectra", data=payload, operation_type="analysis", db=db)

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0


def test_register_result_accepts_burst_selection_payload_kind(db):
    """Burst selections are first-class artifact and payload kinds."""
    payload = BurstSelection(burst_ids=np.array([1, 2], dtype=np.int64))

    art_id = register_result(kind="burst_selection", data=payload, operation_type="burst_selection", db=db)

    assert art_id
    row = db.conn.execute(
        "SELECT artifact_kind, data_format FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,),
    ).fetchone()
    assert row["artifact_kind"] == "burst_selection"
    assert row["data_format"] == "msgpack"
    np.testing.assert_array_equal(read_result(db, art_id).burst_ids, payload.burst_ids)


def test_register_result_burst_selection_dataframe_parses_string_mask_values(db):
    """String mask values from CSV/table widgets must preserve false entries."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"mask": ["False", "True", "0", "1", "no", "yes"]})

    art_id = register_result(kind="burst_selection", data=df, operation_type="burst_selection", db=db)

    assert art_id
    payload = read_result(db, art_id)
    assert payload.KIND == "burst_selection"
    np.testing.assert_array_equal(
        payload.mask,
        np.array([False, True, False, True, False, True], dtype=bool),
    )


def test_register_result_burst_selection_dataframe_rejects_ambiguous_mask_value(db):
    """Ambiguous mask strings fail without writing artifacts."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"mask": ["True", "maybe"]})

    with pytest.raises(ValueError):
        register_result(kind="burst_selection", data=df, operation_type="burst_selection", db=db)

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_object").fetchone()[0] == 0


def test_register_result_fcs_dataframe_preserves_payload_kind(db):
    """Known DataFrame layouts are coerced to matching typed payloads."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"lag": [1e-6, 2e-6], "correlation": [1.0, 0.9]})

    art_id = register_result(kind="fcs_correlation", data=df, operation_type="fcs_correlation", db=db)

    assert art_id
    artifact = db.get_artifact(art_id)
    payload = read_result(db, art_id)
    assert artifact["artifact_kind"] == "fcs_correlation"
    assert payload.KIND == "fcs_correlation"
    np.testing.assert_array_equal(payload.lag, np.array([1e-6, 2e-6], dtype=np.float64))


@pytest.mark.parametrize("normalized_value", ["False", "false", "0", "no", 0, False])
def test_register_result_spectrum_dataframe_parses_false_normalized_values(db, normalized_value):
    """String false values from CSV/table widgets must not become True."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "wavelength": [500.0, 510.0],
            "intensity": [0.2, 1.0],
            "spectrum_type": ["emission", "emission"],
            "normalized": [normalized_value, normalized_value],
        }
    )

    art_id = register_result(kind="spectra", data=df, operation_type="analysis", db=db)

    assert art_id
    payload = read_result(db, art_id)
    assert payload.KIND == "spectra"
    assert payload.normalized is False


@pytest.mark.parametrize("normalized_value", ["True", "true", "1", "yes", 1, True])
def test_register_result_spectrum_dataframe_parses_true_normalized_values(db, normalized_value):
    """Common true spellings from tables are parsed explicitly."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "wavelength": [500.0, 510.0],
            "intensity": [0.2, 1.0],
            "spectrum_type": ["emission", "emission"],
            "normalized": [normalized_value, normalized_value],
        }
    )

    art_id = register_result(kind="spectra", data=df, operation_type="analysis", db=db)

    assert art_id
    payload = read_result(db, art_id)
    assert payload.normalized is True


def test_register_result_spectrum_dataframe_rejects_ambiguous_normalized_value(db):
    """Ambiguous boolean strings fail without writing artifacts."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "wavelength": [500.0, 510.0],
            "intensity": [0.2, 1.0],
            "spectrum_type": ["emission", "emission"],
            "normalized": ["maybe", "maybe"],
        }
    )

    with pytest.raises(ValueError):
        register_result(kind="spectra", data=df, operation_type="analysis", db=db)

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0


def test_register_result_rejects_unsupported_known_kind_dataframe(db):
    """Known artifact kinds cannot silently store GenericTable envelopes."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})

    with pytest.raises(ValueError):
        register_result(kind="pda_histogram", data=df, operation_type="analysis", db=db)

    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM mfdb_object").fetchone()[0] == 0


def test_read_result_rejects_known_artifact_payload_kind_mismatch(db):
    """Existing bad rows with known artifact/payload mismatches fail loudly."""
    blob, data_format = encode_payload("fcs_correlation", {
        "lag": np.array([1e-6, 2e-6], dtype=np.float64),
        "correlation": np.array([1.0, 0.9], dtype=np.float64),
    })
    ref = db.put_object(data=blob, filename="bad.msgpack", mime_type="application/msgpack")
    artifact_id = "bad-mismatch"
    db.register_artifact(
        artifact_id=artifact_id,
        artifact_kind="spectra",
        data_format=data_format,
        storage_mode="embedded_blob",
        object_uuid=ref["object_uuid"],
    )

    with pytest.raises(PayloadSchemaError, match="does not match"):
        read_result(db, artifact_id)


def test_object_store_dedup(db):
    """Identical payloads share one content-addressed object."""
    first = register_result(kind="processed_data", data=b"same", operation_type="analysis", db=db)
    second = register_result(kind="processed_data", data=b"same", operation_type="analysis", db=db)

    first_uuid = db.conn.execute(
        "SELECT object_uuid FROM mfdb_artifact WHERE artifact_id = ?",
        (first,),
    ).fetchone()["object_uuid"]
    second_uuid = db.conn.execute(
        "SELECT object_uuid FROM mfdb_artifact WHERE artifact_id = ?",
        (second,),
    ).fetchone()["object_uuid"]
    assert first_uuid == second_uuid


def test_metadata_only_artifact(db):
    """Metadata-only artifacts are allowed and do not create object rows."""
    art_id = register_result(kind="processed_data", data=None, metadata={"note": "x"}, db=db)

    assert art_id
    row = db.conn.execute(
        "SELECT object_uuid, metadata_json FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,),
    ).fetchone()
    assert row["object_uuid"] is None
    assert "note" in (row["metadata_json"] or "")


def test_no_db_returns_empty(monkeypatch):
    """No available database returns an empty artifact ID and does not raise."""
    import mfdb.result_registry as result_registry

    set_global_db(None)
    monkeypatch.setattr(result_registry, "_get_global_db", lambda: None)

    assert register_result(kind="processed_data", data=b"data") == ""


def test_set_global_db_override(db):
    """The explicit global database override is used when db is omitted."""
    set_global_db(db)
    try:
        art_id = register_result(kind="processed_data", data=b"global-db")
    finally:
        set_global_db(None)

    assert art_id
    assert db.get_artifact(art_id) is not None


def test_convenience_wrappers(db, tmp_path):
    """Convenience wrappers use canonical artifact and operation vocabulary."""
    file_path = tmp_path / "m.ptu"
    file_path.write_bytes(b"raw")

    raw = register_raw_measurement(str(file_path), db=db, metadata={"k": "v"})
    processed = register_processed_data({"x": [1]}, parent_artifact_id=raw, db=db)
    fit = register_fit_result({"chi2": 1.0}, parent_artifact_id=processed, db=db)
    calibration = register_calibration(
        {"g_factor": 1.02},
        calibration_type="g_factor",
        method="user_provided",
        notes="Hellenkamp 2018",
        db=db,
    )

    assert raw and processed and fit and calibration
    row = db.conn.execute(
        "SELECT artifact_kind, data_format, metadata_json FROM mfdb_artifact WHERE artifact_id = ?",
        (calibration,),
    ).fetchone()
    assert row["artifact_kind"] == "calibration_data"
    assert row["data_format"] == "msgpack"
    assert "g_factor" in (row["metadata_json"] or "")
    assert read_result(db, calibration).KIND == "generic_table"
