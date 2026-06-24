"""Tests for v18 vocabulary validation, atomic audit rollback, manifest
consistency, and migration report contents.

These tests verify the quality-of-life improvements added in the v18
consolidation phase of the FDB architecture migration.
"""

from __future__ import annotations

import pathlib
import sqlite3
from unittest.mock import patch

import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb import schema
from chisurf.core.mfdb import (
    ARTIFACT_KINDS,
    OPERATION_TYPES,
    DIRECTIONS,
    STATUS_VALUES,
    VALIDATION_STATUS_VALUES,
    STORAGE_MODES,
    PARAMETER_TYPES,
)
from chisurf.plugins.core.mfdb_admin.backend.services import (
    _validate_mfdb_methods_in_manifest,
)


# ── 1. Vocabulary validation ──────────────────────────────────────────────

def test_register_artifact_rejects_invalid_artifact_type(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="artifact_type|artifact_kind"):
            db.register_artifact("bad_art", artifact_type="invalid_type", storage_mode="local")


def test_register_artifact_rejects_invalid_storage_mode(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="storage_mode"):
            db.register_artifact("bad_stor", artifact_type="raw_data", storage_mode="nonsense")


def test_register_artifact_rejects_invalid_validation_status(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="validation_status"):
            db.register_artifact(
                "bad_val", artifact_type="raw_data", storage_mode="local",
                validation_status="maybe",
            )


def test_record_operation_rejects_invalid_operation_type(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="operation_type"):
            db.record_operation("bad_op", operation_type="quantum_computing")


def test_record_operation_rejects_invalid_status(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="status"):
            db.record_operation("bad_st", operation_type="import", status="potato")


def test_record_operation_link_rejects_invalid_direction(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="direction"):
            db.record_operation_link(
                operation_id="op_1", artifact_id="art_1", direction="sideways",
            )


def test_add_provenance_edge_rejects_invalid_relationship(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="relationship_type"):
            db.add_provenance_edge(
                source_artifact_id="a",
                target_artifact_id="b",
                relationship_type="nonsense_rel",
            )


def test_get_operations_filters_workflow_id_from_metadata(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_workflow.db"
    with MFDatabase(db_path) as db:
        db.record_operation(
            "op_one",
            operation_type="fitting",
            metadata={"workflow_id": "wf-1"},
        )
        db.record_operation(
            "op_two",
            operation_type="fitting",
            metadata={"workflow_id": "wf-2"},
        )
        assert {row["operation_id"] for row in db.get_operations(workflow_id="wf-1")} == {"op_one"}
        assert {row["operation_id"] for row in db.get_operations(workflow_id="missing")} == set()


def test_record_parameter_rejects_invalid_parameter_type(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_vocab.db"
    with MFDatabase(db_path) as db:
        db.record_operation("op_for_param", operation_type="fitting")
        with pytest.raises(ValueError, match="parameter_type"):
            db.record_parameter(
                parameter_uuid="p1", operation_id="op_for_param",
                name="k", value=1.0, parameter_type="imaginary",
            )


def test_record_parameter_writes_audit_log(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_parameter_audit.db"
    with MFDatabase(db_path) as db:
        db.record_operation("op_for_param", operation_type="fitting")
        db.record_parameter(
            parameter_uuid="p1", operation_id="op_for_param",
            name="k", value=1.0,
        )
        audit = db.conn.execute(
            "SELECT action, target_type, target_id FROM mfdb_audit_log WHERE target_type = 'parameter'"
        ).fetchone()
        assert audit is not None
        assert audit["action"] == "create"
        assert audit["target_id"] == "p1"


# ── 2. Atomic audit rollback ─────────────────────────────────────────────

def test_register_artifact_rolls_back_on_audit_failure(tmp_path: pathlib.Path) -> None:
    """When add_audit_log raises inside the transaction, the INSERT is rolled back."""
    db_path = tmp_path / "test_atomic.db"
    with MFDatabase(db_path) as db:
        def failing_add_audit(*args, **kwargs):
            raise RuntimeError("audit failure simulated")
        monkey = patch.object(db, "add_audit_log", failing_add_audit)
        monkey.start()
        try:
            with pytest.raises(RuntimeError, match="audit failure simulated"):
                db.register_artifact("atomic_art", artifact_type="raw_data", storage_mode="local")
        finally:
            monkey.stop()
        # Verify artifact was NOT inserted (transaction rolled back)
        assert db.get_artifact("atomic_art") is None


def test_record_operation_rolls_back_on_audit_failure(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_atomic2.db"
    with MFDatabase(db_path) as db:
        def failing_add_audit(*args, **kwargs):
            raise RuntimeError("audit failure simulated")
        monkey = patch.object(db, "add_audit_log", failing_add_audit)
        monkey.start()
        try:
            with pytest.raises(RuntimeError, match="audit failure simulated"):
                db.record_operation("atomic_op", operation_type="import")
        finally:
            monkey.stop()
        # Verify operation was NOT inserted
        row = db.conn.execute(
            "SELECT 1 FROM mfdb_operation WHERE operation_id = ?", ("atomic_op",)
        ).fetchone()
        assert row is None


def test_record_operation_link_rolls_back_on_audit_failure(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_atomic3.db"
    with MFDatabase(db_path) as db:
        db.register_artifact("link_art", artifact_type="raw_data", storage_mode="local")
        db.record_operation("link_op", operation_type="import")
        def failing_add_audit(*args, **kwargs):
            raise RuntimeError("audit failure simulated")
        monkey = patch.object(db, "add_audit_log", failing_add_audit)
        monkey.start()
        try:
            with pytest.raises(RuntimeError, match="audit failure simulated"):
                db.record_operation_link("link_op", "link_art", direction="input")
        finally:
            monkey.stop()
        # Verify link was NOT inserted
        row = db.conn.execute(
            "SELECT 1 FROM mfdb_operation_artifact WHERE operation_id = ? AND artifact_id = ?",
            ("link_op", "link_art"),
        ).fetchone()
        assert row is None


def test_record_operation_with_artifacts_rolls_back_on_output_write_failure(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_atomic_workflow.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.record_operation_with_artifacts(
                operation_id="op_partial",
                operation_type="burst_selection",
                input_artifacts=[{"artifact_id": "input_ok", "artifact_kind": "raw_measurement"}],
                output_artifacts=[
                    {
                        "artifact_id": "output_missing_experiment",
                        "artifact_kind": "processed_data",
                        "experiment_id": "missing_exp",
                    },
                ],
            )
        for table in (
            "mfdb_operation",
            "mfdb_artifact",
            "mfdb_operation_artifact",
            "mfdb_parameter",
            "mfdb_audit_log",
        ):
            assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_record_operation_with_artifacts_returns_actual_counts(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_counts.db"
    with MFDatabase(db_path) as db:
        result = db.record_operation_with_artifacts(
            operation_id="op_counts",
            operation_type="burst_selection",
            input_artifacts=[
                {"artifact_id": "input_ok", "artifact_kind": "raw_measurement"},
                {"artifact_id": "input_ok", "artifact_kind": "raw_measurement"},
            ],
            output_artifacts=[{"artifact_id": "output_ok", "artifact_kind": "processed_data"}],
            parameters=[{"parameter_uuid": "p1", "name": "k", "value": 1.0}],
        )
        assert result["operation_inserted"] == 1
        assert result["input_artifact_inserted"] == 1
        assert result["input_link_inserted"] == 1
        assert result["output_artifact_inserted"] == 1
        assert result["output_link_inserted"] == 1
        assert result["parameter_inserted"] == 1

        repeat = db.record_operation_with_artifacts(
            operation_id="op_counts",
            operation_type="burst_selection",
            input_artifacts=[{"artifact_id": "input_ok", "artifact_kind": "raw_measurement"}],
            output_artifacts=[{"artifact_id": "output_ok", "artifact_kind": "processed_data"}],
            parameters=[{"parameter_uuid": "p1", "name": "k", "value": 2.0}],
        )
        assert repeat["operation_inserted"] == 0
        assert repeat["input_artifact_inserted"] == 0
        assert repeat["input_link_inserted"] == 0
        assert repeat["output_artifact_inserted"] == 0
        assert repeat["output_link_inserted"] == 0
        assert repeat["parameter_inserted"] == 0


def test_record_operation_with_artifacts_rejects_missing_artifact_id(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_atomic_missing_art.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="artifact_id"):
            db.record_operation_with_artifacts(
                operation_id="op_missing",
                operation_type="burst_selection",
                input_artifacts=[{"artifact_kind": "raw_measurement"}],
            )
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation").fetchone()[0] == 0


def test_add_artifact_md5_uses_md5_checksum_algorithm(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_md5_artifact.db"
    with MFDatabase(db_path) as db:
        db.add_artifact("artifact_md5", "raw_data", md5="0" * 32)
        row = db.get_artifact("artifact_md5")
        assert row["checksum"] == "0" * 32
        assert row["checksum_algorithm"] == "md5"


def test_register_artifact_checksum_defaults_to_sha256(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_sha256_artifact.db"
    with MFDatabase(db_path) as db:
        db.register_artifact("artifact_sha256", artifact_type="raw_data", checksum="0" * 64)
        row = db.get_artifact("artifact_sha256")
        assert row["checksum"] == "0" * 64
        assert row["checksum_algorithm"] == "sha256"


def test_register_artifact_rejects_invalid_checksum_length(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_bad_checksum_artifact.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="sha256 checksum"):
            db.register_artifact("artifact_bad_checksum", artifact_type="raw_data", checksum="0" * 32)
        assert db.get_artifact("artifact_bad_checksum") is None


def test_record_operation_with_artifacts_rolls_back_on_parameter_failure(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_atomic_param.db"
    with MFDatabase(db_path) as db:
        with patch.object(
            db,
            "record_parameter",
            side_effect=RuntimeError("parameter failure simulated"),
        ):
            with pytest.raises(RuntimeError, match="parameter failure simulated"):
                db.record_operation_with_artifacts(
                    operation_id="op_param",
                    operation_type="burst_selection",
                    input_artifacts=[{"artifact_id": "input_ok", "artifact_kind": "raw_measurement"}],
                    output_artifacts=[{"artifact_id": "output_ok", "artifact_kind": "processed_data"}],
                    parameters=[{"parameter_uuid": "p1", "name": "k", "value": 1.0}],
                )
        for table in (
            "mfdb_operation",
            "mfdb_artifact",
            "mfdb_operation_artifact",
            "mfdb_parameter",
            "mfdb_audit_log",
        ):
            assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


# ── 3. Schema invariants ──────────────────────────────────────────────

def test_fresh_mfdb_edge_rejects_operation_relationships(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_edge_constraint.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                """INSERT INTO mfdb_edge (
                    source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
                ) VALUES ('artifact', 'a', 'operation', 'op', 'input_to')"""
            )


def test_fresh_database_uses_flr_sample_as_sample_source_of_truth(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_sample_source.db"
    with MFDatabase(db_path) as db:
        db.add_sample("sample1")
        assert db.conn.execute("SELECT COUNT(*) FROM flr_sample").fetchone()[0] == 1
        tables = {
            row[0]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "mfdb_sample" not in tables


# ── 4. Manifest consistency ──────────────────────────────────────────────

def test_manifest_contains_all_fdb_methods() -> None:
    missing = _validate_mfdb_methods_in_manifest()
    assert missing == [], f"Manifest is missing {len(missing)} method(s): {missing}"


def test_manifest_missing_methods_detected(tmp_path: pathlib.Path) -> None:
    """With a minimal manifest, missing methods are reported."""
    import json
    manifest_path = tmp_path / "empty_manifest.json"
    manifest_path.write_text(json.dumps({"rpc_methods": []}))
    missing = _validate_mfdb_methods_in_manifest(str(manifest_path))
    assert len(missing) > 0
    assert "mfdb.v1.artifacts.register" in missing


# ── 4. Migration report contents ─────────────────────────────────────────


def test_fresh_mfdb_edge_enforces_active_vocabulary(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_edge_vocabulary.db"
    with MFDatabase(db_path) as db:
        for rel_type in ("contains", "parameter_depends_on"):
            db.conn.execute(
                """INSERT INTO mfdb_edge (
                    source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
                ) VALUES ('artifact', 'a', 'artifact', 'b', ?)""",
                (rel_type,),
            )

        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                """INSERT INTO mfdb_edge (
                    source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
                ) VALUES ('artifact', 'a', 'artifact', 'c', 'nonsense_rel')"""
            )

        db.register_vocabulary_value("relationship_type", "custom_rel")
        db.conn.execute(
            """INSERT INTO mfdb_edge (
                source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
            ) VALUES ('artifact', 'a', 'artifact', 'c', 'custom_rel')"""
        )
        db.conn.execute(
            "UPDATE mfdb_vocabulary SET is_active = 0 WHERE field_name = 'relationship_type' AND value = 'custom_rel'"
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                """INSERT INTO mfdb_edge (
                    source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
                ) VALUES ('artifact', 'a', 'artifact', 'd', 'custom_rel')"""
            )


def test_add_processing_run_rolls_back_missing_raw_input(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_processing_atomic.db"
    with MFDatabase(db_path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.add_processing_run(
                processing_id="proc_bad",
                processing_type="burst_selection",
                input_raw_data_ids=["missing_raw"],
            )
        for table in (
            "mfdb_operation",
            "mfdb_operation_artifact",
            "mfdb_audit_log",
        ):
            assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_add_processing_run_rolls_back_audit_failure(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "test_processing_audit_atomic.db"
    with MFDatabase(db_path) as db:
        db.register_artifact("raw_ok", artifact_type="raw_data", storage_mode="local_file")
        with patch.object(db, "add_audit_log", side_effect=RuntimeError("audit failure simulated")):
            with pytest.raises(RuntimeError, match="audit failure simulated"):
                db.add_processing_run(
                    processing_id="proc_audit",
                    processing_type="burst_selection",
                    input_raw_data_ids=["raw_ok"],
                )
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_audit_log").fetchone()[0] == 1


# Note: the former test_migrated_mfdb_edge_* tests exercised the removed
# version-chain migration (schema.migrate_schema stepping v19→v23). PRD-19 deleted
# the version chain (pre-PRD-19 DBs are disposable, no forward migration), so those
# tests asserted legacy behaviour. The same edge vocab/constraint guarantees are
# covered on a fresh database by test_fresh_mfdb_edge_enforces_active_vocabulary and
# test_fresh_mfdb_edge_rejects_operation_relationships above.


def test_fresh_database_has_no_migration_report(tmp_path: pathlib.Path) -> None:
    """A fresh database (created at current schema) has no migration report."""
    db_path = tmp_path / "test_mig_report.db"
    with MFDatabase(db_path) as db:
        assert db.migration_report is None


def test_migration_report_dataclass_works() -> None:
    """Verify the MigrationReport dataclass and its summary property."""
    from chisurf.core.mfdb.schema import MigrationReport
    report = MigrationReport(
        from_version=16,
        to_version=17,
        tables_added=["fdb_artifact", "fdb_operation"],
        backfill={
            "fdb_artifact (from raw_data)": {
                "source": 5, "inserted": 5, "skipped": 0, "warnings": [],
            },
        },
    )
    assert report.from_version == 16
    assert report.to_version == 17
    assert report.tables_added == ["fdb_artifact", "fdb_operation"]
    assert "Schema migrated from" in report.summary
    assert "fdb_artifact" in report.summary
    assert "5 source" in report.summary


# ── 5. Vocabulary constants are non-empty (sanity) ───────────────────────

def test_vocabulary_constants_are_populated() -> None:
    assert len(ARTIFACT_KINDS) > 0
    assert len(OPERATION_TYPES) > 0
    assert len(DIRECTIONS) > 0
    assert len(STATUS_VALUES) > 0
    assert len(VALIDATION_STATUS_VALUES) > 0
    assert len(STORAGE_MODES) > 0
    assert len(PARAMETER_TYPES) > 0
