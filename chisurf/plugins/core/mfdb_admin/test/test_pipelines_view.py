"""PRD-22: PipelinesView Qt widget (smoke).

Drives the thin read-only view over a real in-process MFDBClient: list pipelines, and on
selection show the pipeline's nodes, typed wiring, and recorded runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from chisurf.core.pipeline import (
    Pipeline,
    PipelineEdge,
    PipelineNode,
    PipelineRun,
    record_pipeline_run,
    save_pipeline,
)
from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
from chisurf.plugins.core.mfdb_admin.gui.pipelines_view import PipelinesView

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _demo_pipeline() -> Pipeline:
    return Pipeline(
        name="raw->shift->burst",
        nodes=(
            PipelineNode("shift", "microtime_shift", {"global_shift": 5}),
            PipelineNode("burst", "burst_selection", {}),
        ),
        edges=(PipelineEdge("shift", "shifted", "burst", "raw"),),
    )


def _pipeline_names(view) -> set[str]:
    return {
        view.pipeline_table.item(r, 0).text()
        for r in range(view.pipeline_table.rowCount())
    }


def test_view_lists_pipelines(db, qapp):
    with patch_db(db):
        save_pipeline(db, _demo_pipeline())
        client = MFDBClient(inprocess=True)
        view = PipelinesView(client)
        assert _pipeline_names(view) == {"raw->shift->burst"}


def test_view_shows_structure_and_runs_on_select(db, qapp):
    with patch_db(db):
        pid = save_pipeline(db, _demo_pipeline())
        run = PipelineRun(
            pipeline_name="raw->shift->burst",
            node_outputs={"shift": ["a1"], "burst": ["a2"]},
            operation_ids=["op-1", "op-2"],
        )
        record_pipeline_run(db, run, pipeline_id=pid, status="succeeded", name="run-1")
        client = MFDBClient(inprocess=True)
        view = PipelinesView(client)
        view.pipeline_table.selectRow(0)

        nodes = {view.node_table.item(r, 0).text() for r in range(view.node_table.rowCount())}
        assert nodes == {"shift", "burst"}
        # one wiring edge: shift.shifted -> burst.raw
        assert view.edge_table.rowCount() == 1
        assert view.edge_table.item(0, 0).text() == "shift.shifted"
        assert view.edge_table.item(0, 1).text() == "burst.raw"
        # one recorded run with two operations
        assert view.run_table.rowCount() == 1
        assert view.run_table.item(0, 1).text() == "succeeded"
        assert view.run_table.item(0, 2).text() == "2"
