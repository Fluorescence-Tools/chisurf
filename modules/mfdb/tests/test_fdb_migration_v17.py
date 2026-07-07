from __future__ import annotations

import sqlite3
import json
import pytest
import pathlib
from unittest.mock import patch
from mfdb.schema import migrate_schema, set_schema_version, get_schema_version, SCHEMA_VERSION
from mfdb.repository import MFDatabase
from mfdb import api as fdb_api
from mfdb.graph import traverse_canonical_graph


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


def test_json_rpc_versioned_services(tmp_path: pathlib.Path) -> None:
    """Verify versioned JSON-RPC compatible service handlers in api.py."""
    db_path = tmp_path / "test_rpc.db"

    with patch("mfdb.api.resolve_database_path", return_value=db_path):
        # A fresh DB seeds an admin user, so the api.py write handlers require an
        # authenticated session token; mint one for the seeded default admin.
        from mfdb.auth import create_session
        from mfdb.repository import MFDatabase as _MFDatabase

        with _MFDatabase(str(db_path)) as _db:
            auth = {"token": create_session(_db.conn, "user_default")["token"]}
            _db.conn.commit()

        # 1. Save artifact
        res = fdb_api.register_artifact("art1", "ptu", "local", auth=auth)
        assert res["ok"] is True
        assert res["artifact_id"] == "art1"

        # 2. Get artifact
        res_get = fdb_api.get_artifact("art1", auth=auth)
        assert res_get["artifact"]["artifact_id"] == "art1"

        # 3. List artifacts
        res_list = fdb_api.list_artifacts(auth=auth)
        assert len(res_list["artifacts"]) == 1

        # 4. Record operation
        res_op = fdb_api.record_operation("op1", "burst_selection", auth=auth)
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
