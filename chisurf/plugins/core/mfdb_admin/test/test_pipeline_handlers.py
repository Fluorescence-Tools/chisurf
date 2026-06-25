"""PRD-22: mfdb-admin pipeline RPC handlers.

Stored pipeline definitions (and their grouped runs) are reachable through the admin
backend (and thus MFDBClient): list pipelines, get a pipeline's structure (nodes +
typed wiring), and list runs with operation counts. An unknown id returns an ``error``.
"""

from __future__ import annotations

from chisurf.core.pipeline import (
    Pipeline,
    PipelineEdge,
    PipelineNode,
    PipelineRun,
    record_pipeline_run,
    save_pipeline,
)
from chisurf.plugins.core.mfdb_admin.backend.services import (
    get_pipeline_handler,
    list_pipeline_runs_handler,
    list_pipelines_handler,
)

from .conftest import patch_db


def _demo_pipeline() -> Pipeline:
    return Pipeline(
        name="raw->shift->burst",
        nodes=(
            PipelineNode("shift", "microtime_shift", {"global_shift": 5}),
            PipelineNode("burst", "burst_selection", {}),
        ),
        edges=(PipelineEdge("shift", "shifted", "burst", "raw"),),
    )


def test_list_and_get_pipeline(db):
    with patch_db(db):
        pid = save_pipeline(db, _demo_pipeline(), description="demo")
        pipelines = list_pipelines_handler()["pipelines"]
        detail = get_pipeline_handler(pid)
    assert any(p["pipeline_id"] == pid for p in pipelines)
    assert [n["name"] for n in detail["nodes"]] == ["shift", "burst"]
    assert detail["nodes"][0]["operation_type"] == "microtime_shift"
    assert detail["edges"][0]["source"] == "shift"
    assert detail["edges"][0]["target_port"] == "raw"


def test_get_unknown_pipeline_returns_error(db):
    with patch_db(db):
        res = get_pipeline_handler("does-not-exist")
    assert "error" in res


def test_runs_listed_with_operation_count(db):
    with patch_db(db):
        pid = save_pipeline(db, _demo_pipeline())
        run = PipelineRun(
            pipeline_name="raw->shift->burst",
            node_outputs={"shift": ["a1"], "burst": ["a2"]},
            operation_ids=["op-1", "op-2"],
        )
        record_pipeline_run(db, run, pipeline_id=pid, status="succeeded", name="run-1")
        runs = list_pipeline_runs_handler(pid)["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["operation_count"] == 2
    assert runs[0]["name"] == "run-1"


def test_via_inprocess_client(db):
    with patch_db(db):
        from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

        pid = save_pipeline(db, _demo_pipeline())
        client = MFDBClient(inprocess=True)
        pipelines = client.list_pipelines()
        detail = client.get_pipeline(pid)
    assert any(p["pipeline_id"] == pid for p in pipelines)
    assert detail["name"] == "raw->shift->burst"
