"""PRD-21 Task 1: the lineage query API over the operation graph.

Builds a real derivation chain (sample → raw → microtime_shift → burst_selection)
through the registration path and asserts ancestors/descendants/what_used and the
provenance-graph projection — so call sites can drop bespoke edge SQL.
"""

from __future__ import annotations

import os

import pytest

from chisurf.core.mfdb.lineage import Lineage
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import (
    register_raw_measurement,
    register_result,
    set_global_db,
)


@pytest.fixture
def chain(tmp_path):
    """A 3-hop chain; yields (db, ids dict)."""
    db = MFDatabase(os.path.join(tmp_path, "lin.db"))
    raw_file = tmp_path / "m.ptu"
    raw_file.write_bytes(b"\x00\x01\x02\x03")
    raw = register_raw_measurement(str(raw_file), db=db)
    shifted = register_result(
        kind="processed_data",
        data={"x": [0, 1], "y": [1, 0]},
        parent_artifact_id=raw,
        operation_type="microtime_shift",
        parameters={"global_shift": 5},
        db=db,
    )
    burst = register_result(
        kind="analysis_result",
        data={"x": [0], "y": [1]},
        parent_artifact_id=shifted,
        operation_type="burst_selection",
        parameters={"min_photons": 60},
        db=db,
    )
    try:
        yield db, {"raw": raw, "shifted": shifted, "burst": burst}
    finally:
        set_global_db(None)
        db.close()


def test_ancestors_walks_full_chain(chain):
    db, ids = chain
    lin = Lineage.from_db(db)
    anc = lin.ancestors(ids["burst"])
    assert set(anc) == {ids["shifted"], ids["raw"]}
    # derivation order: nearest parent first
    assert anc[0] == ids["shifted"]
    assert anc[-1] == ids["raw"]


def test_descendants_walks_full_chain(chain):
    db, ids = chain
    lin = Lineage.from_db(db)
    desc = lin.descendants(ids["raw"])
    assert set(desc) == {ids["shifted"], ids["burst"]}
    assert desc[0] == ids["shifted"]


def test_parents_and_children_are_one_hop(chain):
    db, ids = chain
    lin = Lineage.from_db(db)
    assert lin.parents(ids["burst"]) == [ids["shifted"]]
    assert lin.children(ids["raw"]) == [ids["shifted"]]
    assert lin.parents(ids["raw"]) == []
    assert lin.children(ids["burst"]) == []


def test_lineage_to_root_includes_self_first(chain):
    db, ids = chain
    lin = Lineage.from_db(db)
    chain_ids = lin.lineage_to_root(ids["burst"])
    assert chain_ids[0] == ids["burst"]
    assert set(chain_ids) == {ids["burst"], ids["shifted"], ids["raw"]}


def test_what_used_is_downstream_impact(chain):
    db, ids = chain
    lin = Lineage.from_db(db)
    # changing the raw measurement impacts both downstream artifacts
    assert set(lin.what_used(ids["raw"])) == {ids["shifted"], ids["burst"]}


def test_provenance_graph_has_artifacts_operations_and_edges(chain):
    db, ids = chain
    lin = Lineage.from_db(db)
    graph = lin.provenance_graph(ids["shifted"])
    art_ids = {n["node_id"] for n in graph["nodes"] if n["node_type"] == "artifact"}
    op_labels = {n["label"] for n in graph["nodes"] if n["node_type"] == "operation"}
    # all three artifacts reachable from the middle node
    assert {ids["raw"], ids["shifted"], ids["burst"]} <= art_ids
    # both operations present and labelled by operation_type
    assert {"microtime_shift", "burst_selection"} <= op_labels
    # edges are produced/input_to only
    assert {e["relationship"] for e in graph["edges"]} <= {"produced", "input_to"}
    assert graph["edges"], "expected at least one edge"


def test_repository_lineage_accessors_delegate(chain):
    """MFDatabase exposes lineage so call sites use it instead of bespoke SQL."""
    db, ids = chain
    # the lazy property is the lineage service over the live connection
    assert db.lineage.ancestors(ids["burst"]) == db.get_artifact_ancestors(ids["burst"])
    assert set(db.get_artifact_ancestors(ids["burst"])) == {ids["shifted"], ids["raw"]}
    assert set(db.get_artifact_descendants(ids["raw"])) == {ids["shifted"], ids["burst"]}
    # impact == downstream artifacts (PRD-05 data-side)
    assert set(db.get_artifact_impact(ids["raw"])) == {ids["shifted"], ids["burst"]}
    graph = db.get_artifact_provenance_graph(ids["shifted"])
    assert {"nodes", "edges"} == set(graph)


def test_no_lineage_for_isolated_artifact(chain, tmp_path):
    db, ids = chain
    lin = Lineage.from_db(db)
    # a freshly registered raw with no operations has empty lineage
    iso_file = tmp_path / "iso.ptu"
    iso_file.write_bytes(b"\x09\x09")
    iso = register_raw_measurement(str(iso_file), db=db)
    assert lin.ancestors(iso) == []
    assert lin.descendants(iso) == []
