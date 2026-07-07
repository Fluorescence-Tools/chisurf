from __future__ import annotations

import json

import chinet as cn
import pytest

from mfdb.adapters.chinet import (
    clear_mfdb_backend,
    load_chinet_session,
    store_chinet_session,
)
from mfdb.repository import MFDatabase


def _json(value: str | None):
    return json.loads(value) if value else None


def _connected_session() -> cn.Session:
    source = cn.Node(name="source")
    out = cn.Port(1.25, is_output=True, name="out")
    source.add_output_port("out", out)

    follower = cn.Node(name="follower")
    inp = cn.Port(0.0, is_bounded=True, lb=0.1, ub=10.0, name="in")
    follower.add_input_port("in", inp)
    inp.link = out

    return cn.Session({"source": source, "follower": follower})


def test_store_chinet_session_writes_session_node_parameter_and_dependency_rows(tmp_path) -> None:
    db_path = tmp_path / "chinet.db"
    session = _connected_session()
    source_out = session.nodes["source"].outputs["out"]
    follower_in = session.nodes["follower"].inputs["in"]

    with MFDatabase(db_path) as db:
        result = store_chinet_session(
            db,
            session,
            operation_id="op_chinet",
            experiment_id="exp_chinet",
            fit_refs=[{"fit_uid": "fit-1", "operation_id": "op_chinet"}],
        )

        assert result["session_artifact_id"].startswith("chinet_session:")
        assert len(result["node_artifact_ids"]) == 2
        assert result["parameter_count"] == 2
        assert result["dependency_edge_count"] == 1

        artifacts = db.conn.execute(
            "SELECT artifact_id, artifact_kind, data_format, storage_mode "
            "FROM mfdb_artifact ORDER BY artifact_id"
        ).fetchall()
        artifact_kinds = {row["artifact_kind"] for row in artifacts}
        assert {"chinet_session", "chinet_node"}.issubset(artifact_kinds)
        assert len([row for row in artifacts if row["artifact_kind"] == "chinet_session"]) == 1
        assert len([row for row in artifacts if row["artifact_kind"] == "chinet_node"]) == 2

        op_links = db.conn.execute(
            "SELECT artifact_id, direction, role "
            "FROM mfdb_operation_artifact WHERE operation_id = ? "
            "ORDER BY role",
            ("op_chinet",),
        ).fetchall()
        assert any(
            row["direction"] == "output" and row["role"] == "chinet_session"
            for row in op_links
        )
        assert (
            sum(
                row["direction"] == "output" and row["role"] == "chinet_node"
                for row in op_links
            )
            == 2
        )

        parameters = db.list_parameters(operation_id="op_chinet")
        assert len(parameters) == 2
        metadata = [_json(row["metadata_json"]) for row in parameters]
        assert {item["chinet_port_id"] for item in metadata} == {source_out.oid, follower_in.oid}
        follower_param = next(
            item for item in metadata if item["chinet_port_id"] == follower_in.oid
        )
        assert follower_param["port_direction"] == "input"
        assert follower_param["schema_name"] == "chinet.parameter_ref.v1"

        edges = db.conn.execute(
            "SELECT source_node_id, target_node_id, relationship_type "
            "FROM mfdb_edge WHERE relationship_type = 'parameter_depends_on'"
        ).fetchall()
        assert len(edges) == 1
        assert edges[0]["source_node_id"] == source_out.oid
        assert edges[0]["target_node_id"] == follower_in.oid

        restored = load_chinet_session(db, result["session_artifact_id"])
        assert restored.oid == session.oid
        assert (
            restored.nodes["follower"].inputs["in"].link.oid
            == restored.nodes["source"].outputs["out"].oid
        )


def test_store_chinet_session_rejects_invalid_parameter_before_writes(tmp_path) -> None:
    db_path = tmp_path / "chinet_invalid_param.db"
    session = _connected_session()

    with MFDatabase(db_path) as db:
        with pytest.raises(ValueError, match="parameter_type"):
            store_chinet_session(
                db,
                session,
                operation_id="op_invalid_param",
                parameters=[
                    {
                        "parameter_uuid": "bad_param",
                        "name": "bad",
                        "value": 0.0,
                        "parameter_type": "nonsense",
                    }
                ],
            )

        assert db.get_operation("op_invalid_param") is None
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_parameter").fetchone()[0] == 0


def test_store_chinet_session_rolls_back_on_artifact_failure(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "chinet_artifact_failure.db"
    session = _connected_session()

    with MFDatabase(db_path) as db:
        def fail_register_artifact(*args, **kwargs):
            raise RuntimeError("artifact failure")

        monkeypatch.setattr(db, "register_artifact", fail_register_artifact)
        with pytest.raises(RuntimeError, match="artifact failure"):
            store_chinet_session(db, session, operation_id="op_artifact_failure")

        assert db.get_operation("op_artifact_failure") is None
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation_artifact").fetchone()[0] == 0


def test_store_chinet_session_rolls_back_on_link_failure(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "chinet_link_failure.db"
    session = _connected_session()

    with MFDatabase(db_path) as db:
        def fail_record_operation_link(*args, **kwargs):
            raise RuntimeError("link failure")

        monkeypatch.setattr(db, "record_operation_link", fail_record_operation_link)
        with pytest.raises(RuntimeError, match="link failure"):
            store_chinet_session(db, session, operation_id="op_link_failure")

        assert db.get_operation("op_link_failure") is None
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation_artifact").fetchone()[0] == 0


def test_store_chinet_session_rolls_back_on_parameter_failure(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "chinet_parameter_failure.db"
    session = _connected_session()

    with MFDatabase(db_path) as db:
        def fail_record_parameter(*args, **kwargs):
            raise RuntimeError("parameter failure")

        monkeypatch.setattr(db, "record_parameter", fail_record_parameter)
        with pytest.raises(RuntimeError, match="parameter failure"):
            store_chinet_session(db, session, operation_id="op_parameter_failure")

        assert db.get_operation("op_parameter_failure") is None
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_operation_artifact").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM mfdb_parameter").fetchone()[0] == 0


def test_store_chinet_session_accepts_explicit_parameter_uuid_for_link_mapping(tmp_path) -> None:
    db_path = tmp_path / "chinet_explicit.db"
    session = _connected_session()
    follower_port = session.nodes["follower"].inputs["in"]

    with MFDatabase(db_path) as db:
        store_chinet_session(
            db,
            session,
            operation_id="op_explicit",
            parameters=[
                {
                    "parameter_uuid": "follower_param",
                    "name": "follower",
                    "value": 0.0,
                    "metadata": {
                        "schema_name": "chinet.parameter_ref.v1",
                        "chinet_port_id": follower_port.oid,
                    },
                }
            ],
        )
        edges = db.conn.execute(
            "SELECT source_node_id, target_node_id "
            "FROM mfdb_edge WHERE relationship_type = 'parameter_depends_on'"
        ).fetchall()
        assert edges[0]["target_node_id"] == "follower_param"


def test_store_chinet_session_fast_mode_skips_node_artifacts(tmp_path) -> None:
    db_path = tmp_path / "chinet_fast.db"
    session = _connected_session()

    with MFDatabase(db_path) as db:
        result = store_chinet_session(
            db,
            session,
            operation_id="op_chinet_fast",
            store_node_artifacts=False,
        )

        assert result["node_artifact_ids"] == []
        artifacts = db.list_artifacts(experiment_id=None)
        assert len([row for row in artifacts if row["artifact_kind"] == "chinet_session"]) == 1
        assert len([row for row in artifacts if row["artifact_kind"] == "chinet_node"]) == 0
        assert len(db.list_parameters(operation_id="op_chinet_fast")) == 2


@pytest.mark.skipif(
    not hasattr(cn.DB, "set_backend"),
    reason="chinet >=0.3 removed the pluggable DB backend (DB.set_backend); the "
    "transparent MFDB-as-chinet-backend integration needs a redesign against the new "
    "chinet DB registry. The explicit store_/load_chinet_session path is covered above.",
)
def test_connect_to_db_can_use_mfdb_backend(tmp_path) -> None:
    db_path = tmp_path / "chinet_mfdb_backend.db"
    session = _connected_session()
    source_out = session.nodes["source"].outputs["out"]
    follower_in = session.nodes["follower"].inputs["in"]

    try:
        assert session.connect_to_db(
            "mfdb",
            db_path=str(db_path),
            operation_id="op_chinet_backend",
            experiment_id="exp_chinet_backend",
            store_node_artifacts=False,
        )
        assert session.write_to_db()

        with MFDatabase(db_path) as db:
            artifacts = db.list_artifacts(experiment_id="exp_chinet_backend")
            assert len([row for row in artifacts if row["artifact_kind"] == "chinet_session"]) == 1
            assert len([row for row in artifacts if row["artifact_kind"] == "chinet_node"]) == 0
            assert len(db.list_parameters(operation_id="op_chinet_backend")) == 2
            edges = db.conn.execute(
                "SELECT source_node_id, target_node_id FROM mfdb_edge "
                "WHERE operation_id = ? AND relationship_type = 'parameter_depends_on'",
                ("op_chinet_backend",),
            ).fetchall()
            assert edges[0]["source_node_id"] == source_out.oid
            assert edges[0]["target_node_id"] == follower_in.oid

            restored = cn.Session()
            assert restored.read_from_db(session.oid)
        assert (
            restored.nodes["follower"].inputs["in"].link.oid
            == restored.nodes["source"].outputs["out"].oid
        )

    finally:
        clear_mfdb_backend(close=True)
