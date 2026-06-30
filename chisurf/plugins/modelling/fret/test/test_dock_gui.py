"""Headless test for the AutoForm docking GUI.

Builds the widget under the Qt ``offscreen`` platform and checks the view
renders and the model maps to the right api.operations request.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_docking_tool_builds(qapp):
    from chisurf.plugins.modelling.fret.gui.dock_tool import FretDockingTool

    w = FretDockingTool()
    w.resize(420, 560)
    # AutoForm rendered some field widgets
    from chisurf.gui.autoform.sections.builtin import ValueWidget

    assert w._form.findChildren(ValueWidget)
    # the grab path used for visual verification works
    assert w.grab().width() > 0


def test_build_params_dock():
    from chisurf.plugins.modelling.fret.gui.dock_tool import _DockingModel

    m = _DockingModel()
    m.pdb_paths = "a.pdb, b.pdb"
    m.fps_json = "x.fps.json"
    m.output_dir = "out"
    m.operation = "dock"
    m.sigma_da = 8.0
    op, params = m.build_params()
    assert op == "dock"
    assert params["pdb_paths"] == ["a.pdb", "b.pdb"]
    assert params["fps_json"] == "x.fps.json"
    assert params["sigma_da"] == 8.0


def test_build_params_screen():
    from chisurf.plugins.modelling.fret.gui.dock_tool import _DockingModel

    m = _DockingModel()
    m.pdb_paths = "lib/"
    m.fps_json = "x.fps.json"
    m.operation = "screen"
    op, params = m.build_params()
    assert op == "screen"
    assert params["pdb_inputs"] == ["lib/"]


def test_repeats_route_to_errors():
    """dock + n_repeats>1 becomes an 'errors' (repeated-docking) request."""
    from chisurf.plugins.modelling.fret.gui.dock_tool import _DockingModel

    m = _DockingModel()
    m.pdb_paths = "a.pdb"
    m.fps_json = "x.fps.json"
    m.output_dir = "out"
    m.operation = "dock"
    m.n_repeats = 5
    op, params = m.build_params()
    assert op == "errors"
    assert params["n_trials"] == 5


def test_results_table_and_plot(qapp, tmp_path):
    """A repeated-docking result populates the table and one curve per trial."""
    from chisurf.plugins.modelling.fret.gui.dock_tool import FretDockingTool

    w = FretDockingTool()
    w._model.output_dir = str(tmp_path)
    w._model.n_repeats = 2
    w._model.method = "mc"  # these fixtures are PMI stat.0.out files
    w._run_op = "errors"
    # two trial stat files so _update_plot has data to draw
    for t, score in ((0, 12.0), (1, 8.0)):
        d = tmp_path / f"trial_{t:03d}"
        d.mkdir()
        (d / "stat.0.out").write_text(
            "HEADER\n{1: '%f', 4: '0'}\n{1: '%f', 4: '1'}\n" % (score + 5, score)
        )
    result = {
        "status": "ok",
        "data": {
            "n_trials": 2, "score_mean": 10.0, "score_std": 2.0, "best_trial": 1,
            "trial_details": [
                {"trial": 0, "score": 12.0, "n_distances": 4,
                 "best_pdb": str(tmp_path / "trial_000/pdbs/model.0.pdb")},
                {"trial": 1, "score": 8.0, "n_distances": 4,
                 "best_pdb": str(tmp_path / "trial_001/pdbs/model.0.pdb")},
            ],
        },
    }
    w._on_finished(result)
    assert w._table.rowCount() == 2
    # columns: Trial(0), Type(1), Score(2), Distances(3), Best PDB(4)
    # default sort is best (lowest) score first
    assert w._table.item(0, 2).text() == "8.00"
    assert w._table.item(1, 2).text() == "12.00"
    assert w._table.item(0, 1).text() == "mc"  # Type column reflects the method
    w._update_plot()
    assert len(w._curves) == 2
    assert "trials" in w._model.status
