"""Metadata-reference edge materialization + resolvable audit labels.

A metadata value of the form ``mfdb://<node_type>/<node_id>`` materializes a
provenance edge from the owning node to the referenced node (the ELN
"a metadata field that references another entity is a link" pattern).
"""

from __future__ import annotations

from pathlib import Path

from mfdb.queries.artifacts import parse_node_ref
from mfdb.repository import MFDatabase


def _db(tmp_path: Path) -> MFDatabase:
    return MFDatabase(tmp_path / "meta_edges.db")


def _edges(db: MFDatabase) -> list[dict]:
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM mfdb_edge WHERE deleted_at IS NULL"
    ).fetchall()]


def test_parse_node_ref() -> None:
    assert parse_node_ref("mfdb://artifact/art_1") == ("artifact", "art_1")
    assert parse_node_ref("  mfdb://sample/s 1  ") == ("sample", "s 1")
    assert parse_node_ref("just a string") is None
    assert parse_node_ref("http://example.org/x") is None
    assert parse_node_ref(42) is None


def test_sample_metadata_materializes_edge(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        db.add_sample("s1", description="donor")
        db.register_artifact(artifact_id="art_1", artifact_kind="raw_data")
        db.set_sample_key_value("s1", "related_dataset", "mfdb://artifact/art_1")

        edges = _edges(db)
        assert len(edges) == 1
        e = edges[0]
        assert (e["source_node_type"], e["source_node_id"]) == ("sample", "s1")
        assert (e["target_node_type"], e["target_node_id"]) == ("artifact", "art_1")
        assert e["relationship_type"] == "linked_to"
        assert "related_dataset" in (e["metadata_json"] or "")

        # idempotent: re-setting the same reference adds no duplicate edge
        db.set_sample_key_value("s1", "related_dataset", "mfdb://artifact/art_1")
        assert len(_edges(db)) == 1

        # the edge creation is audited with a resolvable target label
        logs = db.get_audit_logs(target_type="edge")
        assert any("art_1" in (log.get("target_id") or "") for log in logs)


def test_non_reference_and_self_reference_do_not_link(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        db.add_sample("s1", description="x")
        # a plain value → no edge
        db.set_sample_key_value("s1", "note", "just text")
        # a self-reference → no edge
        db.set_sample_key_value("s1", "self", "mfdb://sample/s1")
        assert _edges(db) == []


def test_experiment_metadata_materializes_edge(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        db.add_experiment("e1")
        db.add_sample("s1", description="linked")
        db.set_experiment_key_value("e1", "on_sample", "mfdb://sample/s1")
        edges = _edges(db)
        assert len(edges) == 1
        assert (edges[0]["source_node_type"], edges[0]["target_node_id"]) == ("experiment", "s1")


def test_operation_link_audit_has_target_label(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        db.add_operation("op1", "analysis")
        db.register_artifact(artifact_id="art_lbl", artifact_kind="raw_data")
        db.record_operation_link("op1", "art_lbl", direction="output", role="fret_fit")
        logs = db.get_audit_logs(target_type="operation_artifact")
        assert logs
        details = logs[0].get("details")
        # details is stored as JSON text; the resolvable artifact label (kind) is present
        assert "raw_data" in str(details) and "target_label" in str(details)
