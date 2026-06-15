from __future__ import annotations

import sqlite3
import json
import pytest
import pathlib
from unittest.mock import patch
from chisurf.core.mfdb.schema import migrate_schema, set_schema_version, get_schema_version, SCHEMA_VERSION
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb import api as fdb_api
from chisurf.core.mfdb.graph import traverse_canonical_graph, traverse_legacy_provenance_graph


def test_migration_and_backfill(tmp_path: pathlib.Path) -> None:
    """Verify legacy database schema upgrades to v17 and backfills all records."""
    db_path = tmp_path / "test_migration_v17.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Manually execute CREATE_TABLES_SQL and CREATE_INDICES_SQL to set up legacy schema
    from chisurf.core.mfdb.schema import CREATE_TABLES_SQL, CREATE_INDICES_SQL
    for sql in CREATE_TABLES_SQL + CREATE_INDICES_SQL:
        conn.execute(sql)
    set_schema_version(conn, 16)

    # Seed legacy data
    conn.execute("INSERT INTO flr_experiment (experiment_id) VALUES ('exp1')")

    conn.execute("""
        INSERT INTO fdb_raw_data (
            raw_data_id, experiment_id, data_type, storage_mode, file_path, acquired_at, validation_status
        ) VALUES ('raw1', 'exp1', 'PTU', 'local_file', '/path/to/raw1.ptu', '2026-06-12T12:00:00Z', 'valid')
    """)

    conn.execute("""
        INSERT INTO fdb_processing_run (
            processing_id, processing_type, experiment_id, settings_json, operator_user_id, status
        ) VALUES ('proc1', 'burst_selection', 'exp1', '{"threshold": 10}', 'user1', 'succeeded')
    """)

    conn.execute("""
        INSERT INTO fdb_processing_input (
            processing_id, raw_data_id, ordinal
        ) VALUES ('proc1', 'raw1', 0)
    """)

    conn.execute("""
        INSERT INTO fdb_processed_data (
            processed_data_id, processing_id, product_type, storage_mode, file_path, row_count, validation_status
        ) VALUES ('prod1', 'proc1', 'bur', 'local_file', '/path/to/prod1.bur', 100, 'valid')
    """)

    conn.execute("""
        INSERT INTO fdb_provenance_edge (
            edge_id, source_node_type, source_node_id, target_node_type, target_node_id, relationship_type, processing_id
        ) VALUES (1, 'raw_data', 'raw1', 'processing_run', 'proc1', 'input_to', 'proc1')
    """)
    conn.execute("""
        INSERT INTO fdb_provenance_edge (
            edge_id, source_node_type, source_node_id, target_node_type, target_node_id, relationship_type, processing_id
        ) VALUES (2, 'processing_run', 'proc1', 'processed_data', 'prod1', 'produced', 'proc1')
    """)

    conn.execute("""
        INSERT INTO fdb_setup_definition (
            setup_id, name, version, description
        ) VALUES ('setup1', 'Test Setup', 1, 'My Test Setup')
    """)

    conn.execute("""
        INSERT INTO fdb_analysis_run (
            analysis_id, model_name, model_type, goodness_of_fit_json
        ) VALUES ('proc1', 'GMM', 'fit', '{"chi2": 1.0}')
    """)

    conn.execute("""
        INSERT INTO fdb_analysis_parameter (
            parameter_id, parameter_uuid, analysis_id, name, value, bounds_on
        ) VALUES (1, 'param-uuid-1', 'proc1', 'mu1', 0.5, 0)
    """)

    conn.commit()

    # Migrate database schema to version 17
    migrate_schema(conn)

    assert get_schema_version(conn) == SCHEMA_VERSION

    # Verify backfilled data in canonical tables
    art_raw = conn.execute("SELECT * FROM fdb_artifact WHERE artifact_id = 'raw1'").fetchone()
    assert art_raw is not None
    assert art_raw["artifact_type"] == "raw_data"
    assert art_raw["storage_mode"] == "local_file"
    assert art_raw["file_path"] == "/path/to/raw1.ptu"
    assert art_raw["validation_status"] == "valid"
    meta_raw = json.loads(art_raw["metadata_json"])
    assert meta_raw["acquired_at"] == "2026-06-12T12:00:00Z"

    art_prod = conn.execute("SELECT * FROM fdb_artifact WHERE artifact_id = 'prod1'").fetchone()
    assert art_prod is not None
    assert art_prod["artifact_type"] == "bur"
    assert art_prod["storage_mode"] == "local_file"
    assert art_prod["file_path"] == "/path/to/prod1.bur"
    assert art_prod["row_count"] == 100
    assert art_prod["validation_status"] == "valid"

    setup = conn.execute("SELECT * FROM fdb_setup WHERE setup_id = 'setup1'").fetchone()
    assert setup is not None
    assert setup["name"] == "Test Setup"
    assert setup["version"] == 1
    assert setup["description"] == "My Test Setup"

    op = conn.execute("SELECT * FROM fdb_operation WHERE operation_id = 'proc1'").fetchone()
    assert op is not None
    assert op["operation_type"] == "analysis"
    assert op["operator_user_id"] == "user1"
    assert op["status"] == "succeeded"
    meta_op = json.loads(op["metadata_json"])
    assert meta_op["model_name"] == "GMM"
    assert meta_op["model_type"] == "fit"
    assert meta_op["goodness_of_fit"]["chi2"] == 1.0

    links = conn.execute("SELECT * FROM fdb_operation_artifact").fetchall()
    assert len(links) == 2

    edges = conn.execute("SELECT * FROM fdb_edge").fetchall()
    assert len(edges) == 2
    mfdb_edges = conn.execute("SELECT relationship_type FROM mfdb_edge").fetchall()
    assert mfdb_edges == []
    mfdb_links = conn.execute("SELECT * FROM mfdb_operation_artifact").fetchall()
    assert len(mfdb_links) == 2

    param = conn.execute("SELECT * FROM fdb_parameter WHERE parameter_uuid = 'param-uuid-1'").fetchone()
    assert param is not None
    assert param["name"] == "mu1"
    assert param["value"] == 0.5
    assert param["operation_id"] == "proc1"

    conn.close()


def test_v17_migration_uses_flr_sample_model_and_constrained_migrated_schema(tmp_path: pathlib.Path) -> None:
    """Migrated canonical schema matches fresh FLR sample model and edge constraints."""
    db_path = tmp_path / "test_migration_constraints.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    from chisurf.core.mfdb.schema import CREATE_TABLES_SQL
    for sql in CREATE_TABLES_SQL:
        if not any(sql.lstrip().startswith(f"CREATE TABLE IF NOT EXISTS {name} ") for name in ("mfdb_sample", "mfdb_experiment")):
            conn.execute(sql)
    set_schema_version(conn, 17)
    conn.execute("INSERT INTO flr_sample (sample_id) VALUES ('sample1')")
    conn.execute("INSERT INTO fdb_edge (source_node_type, source_node_id, target_node_type, target_node_id, relationship_type) VALUES ('artifact', 'a', 'operation', 'op', 'input_to')")
    conn.commit()

    migrate_schema(conn)

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mfdb_sample" not in tables
    assert "mfdb_experiment" not in tables
    assert "flr_sample" in tables
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO mfdb_edge (source_node_type, source_node_id, target_node_type, target_node_id, relationship_type) VALUES (?, ?, ?, ?, ?)",
            ("artifact", "b", "operation", "op2", "produced"),
        )
    conn.close()


def test_idempotent_updates(tmp_path: pathlib.Path) -> None:
    """Verify re-saving an operation updates the record without orphaning links."""
    db_path = tmp_path / "test_idempotent.db"
    with MFDatabase(db_path) as db:
        db.add_sample("sample1")
        db.add_experiment("exp1", sample_id="sample1")

        db.record_operation("op1", "burst_selection", experiment_id="exp1", status="pending")
        op1 = db.get_operation("op1")
        assert op1["status"] == "pending"

        db.register_artifact("art1", "bur", "local", experiment_id="exp1")
        db.record_operation_link("op1", "art1", "output")

        assert len(db.conn.execute("SELECT * FROM mfdb_operation_artifact WHERE operation_id = 'op1'").fetchall()) == 1

        db.record_operation("op1", "burst_selection", experiment_id="exp1", status="succeeded")
        op2 = db.get_operation("op1")
        assert op2["status"] == "succeeded"

        # Verify operation links are preserved
        assert len(db.conn.execute("SELECT * FROM mfdb_operation_artifact WHERE operation_id = 'op1'").fetchall()) == 1


def test_transaction_safety(tmp_path: pathlib.Path) -> None:
    """Verify that a failure in a transaction rolls back all modifications."""
    db_path = tmp_path / "test_transaction.db"
    with MFDatabase(db_path) as db:
        db.add_sample("sample1")
        db.add_experiment("exp1", sample_id="sample1")

        with pytest.raises(Exception):
            with db.transaction():
                db.register_artifact("art1", "bur", "local", experiment_id="exp1")
                # Intentionally trigger constraint failure
                db.conn.execute("INSERT INTO mfdb_artifact (artifact_id, artifact_kind) VALUES ('art1', 'invalid')")

        assert db.get_artifact("art1") is None


def test_cycle_safe_graph_traversal(tmp_path: pathlib.Path) -> None:
    """Verify recursive graph query cycle safety."""
    db_path = tmp_path / "test_cycle.db"
    with MFDatabase(db_path) as db:
        # Canonical graph cycle: A -[input_to]-> B -[produced]-> C -[derived_from]-> A
        # input_to and produced go into mfdb_operation_artifact (canonical source of truth)
        db.conn.execute("""
            INSERT INTO mfdb_artifact (artifact_id, artifact_kind, storage_mode)
            VALUES ('A', 'raw_data', 'local'), ('C', 'processed_data', 'local')
        """)
        db.conn.execute("""
            INSERT INTO mfdb_operation (operation_id, operation_type, status)
            VALUES ('B', 'burst_selection', 'succeeded')
        """)
        db.conn.execute("""
            INSERT INTO mfdb_operation_artifact (operation_id, artifact_id, direction, role)
            VALUES ('B', 'A', 'input', 'raw_data')
        """)
        db.conn.execute("""
            INSERT INTO mfdb_operation_artifact (operation_id, artifact_id, direction, role)
            VALUES ('B', 'C', 'output', 'processed_data')
        """)
        # derived_from goes into mfdb_edge (non-operation relationship)
        db.conn.execute("""
            INSERT INTO mfdb_edge (
                source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
            ) VALUES ('artifact', 'C', 'artifact', 'A', 'derived_from')
        """)

        edges = traverse_canonical_graph(db.conn, "artifact", "A", direction="downstream")
        assert len(edges) == 3
        assert {edge["source_node_type"] for edge in edges} == {"operation", "artifact"}
        assert {edge["target_node_type"] for edge in edges} == {"operation", "artifact"}
        assert any(edge.get("source_operation_type") == "burst_selection" for edge in edges)
        assert any(edge.get("target_artifact_kind") == "processed_data" for edge in edges)

        # Legacy provenance graph cycle
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS fdb_provenance_edge (
                source_node_type TEXT,
                source_node_id TEXT,
                target_node_type TEXT,
                target_node_id TEXT,
                relationship_type TEXT,
                processing_id TEXT
            )
        """)
        db.conn.execute("""
            INSERT INTO fdb_provenance_edge (
                source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
            ) VALUES ('artifact', 'A', 'operation', 'B', 'input_to')
        """)
        db.conn.execute("""
            INSERT INTO fdb_provenance_edge (
                source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
            ) VALUES ('operation', 'B', 'artifact', 'C', 'produced')
        """)
        db.conn.execute("""
            INSERT INTO fdb_provenance_edge (
                source_node_type, source_node_id, target_node_type, target_node_id, relationship_type
            ) VALUES ('artifact', 'C', 'artifact', 'A', 'derived_from')
        """)

        legacy_edges = traverse_legacy_provenance_graph(db.conn, "artifact", "A", direction="downstream")
        assert len(legacy_edges) == 3


def test_json_rpc_versioned_services(tmp_path: pathlib.Path) -> None:
    """Verify versioned JSON-RPC compatible service handlers in api.py."""
    db_path = tmp_path / "test_rpc.db"

    with patch("chisurf.core.mfdb.api.resolve_database_path", return_value=db_path):
        # 1. Save artifact
        res = fdb_api.register_artifact("art1", "ptu", "local")
        assert res["ok"] is True
        assert res["artifact_id"] == "art1"

        # 2. Get artifact
        res_get = fdb_api.get_artifact("art1")
        assert res_get["artifact"]["artifact_id"] == "art1"

        # 3. List artifacts
        res_list = fdb_api.list_artifacts()
        assert len(res_list["artifacts"]) == 1

        # 4. Record operation
        res_op = fdb_api.record_operation("op1", "burst_selection")
        assert res_op["ok"] is True
        assert res_op["operation_id"] == "op1"

        # 5. Get operation
        res_op_get = fdb_api.get_operation("op1")
        assert res_op_get["operation"]["operation_id"] == "op1"

        # 6. List operations
        res_op_list = fdb_api.list_operations()
        assert len(res_op_list["operations"]) == 1

        # 7. Record operation link
        res_link = fdb_api.record_operation_link("op1", "art1", "output")
        assert res_link["ok"] is True

        # 8. Record parameter
        res_param = fdb_api.record_parameter("param1", "op1", "tau", value=3.2)
        assert res_param["ok"] is True
        assert res_param["parameter_uuid"] == "param1"

        # 9. Get parameter
        res_param_get = fdb_api.get_parameter("param1")
        assert res_param_get["parameter"]["value"] == 3.2

        # 10. Save setup
        res_setup = fdb_api.save_setup("setup1", "My Setup")
        assert res_setup["ok"] is True

        # 11. Get setup
        res_setup_get = fdb_api.get_setup("setup1")
        assert res_setup_get["setup"]["name"] == "My Setup"

        # 12. List audit logs
        res_audit = fdb_api.list_audit_logs()
        assert len(res_audit["logs"]) > 0
