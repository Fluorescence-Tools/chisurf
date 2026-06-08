from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
from qtpy import QtWidgets

from chisurf.plugins.modelling.proteinmc.model import ProteinMCProgress


def test_proteinmc_widget_registers_structure_and_trajectory_plots():
    import chisurf.gui.widgets.models.proteinmc as proteinmc_widget

    names = [plot_class.name for plot_class, _ in proteinmc_widget.ProteinMCModelWidget.plot_classes]
    assert names == ["Structure", "Distance Network", "Trajectory-Plot"]


def test_fitting_controller_finds_proteinmc_group_model(qapp):
    from chisurf.gui.widgets.fitting.fit_controller import FittingControllerWidget

    class ProteinMCModel:
        name = "ProteinMC"

        def run_sampling(self):
            return None

    controller = FittingControllerWidget.__new__(FittingControllerWidget)
    controller.fit = SimpleNamespace(model=ProteinMCModel())
    controller.comboBox = QtWidgets.QComboBox()
    controller.comboBox.addItem("ProteinMC")

    assert controller._proteinmc_model_widget() is controller.fit.model


class FakeChimolView(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args[:1] if args else [])
        self.frames = []
        self.object_id = None

    def add_coordinates(self, coords, *, name=None, source_path=None):
        self.object_id = "obj1"
        self.coords = np.asarray(coords)
        return self.object_id

    def add_structure(self, structure, *, name=None, source_path=None):
        self.object_id = "obj1"
        self.structure = structure
        self.coords = np.asarray(getattr(structure, "xyz", np.zeros((0, 3))))
        return self.object_id

    def set_frames(self, frames, *, object_id=None):
        self.frames = list(np.asarray(frames))

    def set_active_frame(self, index, *, object_id=None):
        self.active_frame = int(index)

    def append_frame(self, frame, *, object_id=None):
        self.frames.append(np.asarray(frame))
        return len(self.frames)


def test_proteinmc_widget_updates_arrays_and_chimol(qapp, qtbot, monkeypatch):
    import chisurf.gui.widgets.models.proteinmc as proteinmc_widget

    monkeypatch.setattr(proteinmc_widget, "ChimolView", FakeChimolView)
    fit = SimpleNamespace(name="fit", data=None, plots=[])
    widget = proteinmc_widget.ProteinMCModelWidget(fit=fit)
    qtbot.addWidget(widget)

    progress = ProteinMCProgress(
        frame_index=1,
        target_frames=2,
        iteration=1,
        accepted=1,
        rejected=0,
        energy=2.0,
        labeling_energy=1.0,
        rmsd=[0.0],
        drmsd=[0.0],
        energies=[2.0],
        labeling_energies=[1.0],
        xyz=np.zeros((3, 3)),
        output_file="out.rmf3",
    )
    widget.on_progress(progress)

    assert widget.energy == [2.0]
    assert widget.chi2r == [1.0]
    assert len(widget.viewer.frames) == 1


def test_proteinmc_widget_global_frame_updates_plots(qapp, qtbot, monkeypatch):
    import chisurf.gui.widgets.models.proteinmc as proteinmc_widget

    class PlotStub:
        def __init__(self):
            self.updated = 0

        def update(self):
            self.updated += 1

    monkeypatch.setattr(proteinmc_widget, "ChimolView", FakeChimolView)
    fit = SimpleNamespace(name="fit", data=None, plots=[])
    widget = proteinmc_widget.ProteinMCModelWidget(fit=fit)
    qtbot.addWidget(widget)
    plot = PlotStub()
    fit.plots = [plot]
    widget.trajectory_frames = [np.zeros((3, 3)), np.ones((3, 3))]

    widget.set_current_frame(1)

    assert widget.current_frame_index == 1
    qtbot.waitUntil(lambda: plot.updated == 1, timeout=1000)


def test_distance_network_plot_constructs(qapp, qtbot, tmp_path):
    from chisurf.gui.plots.proteinMC import ProteinMCDistanceNetworkPlot, _agreement_color

    atoms = np.zeros(
        2,
        dtype=[("xyz", float, (3,)), ("atom_name", "S2"), ("res_id", int), ("chain", "S1")],
    )
    atoms["xyz"] = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    atoms["atom_name"] = [b"CA", b"CA"]
    atoms["res_id"] = [1, 2]
    atoms["chain"] = [b"A", b"A"]
    labeling = tmp_path / "fps.json"
    labeling.write_text(
        '{"Positions":{"p1":{"chain_identifier":"A","residue_seq_number":1,"atom_name":"CA"},'
        '"p2":{"chain_identifier":"A","residue_seq_number":2,"atom_name":"CA"}},'
        '"Distances":{"d":{"position1_name":"p1","position2_name":"p2","distance":10,"error_neg":1,"error_pos":1}}}',
        encoding="utf-8",
    )
    model = SimpleNamespace(
        proteinmc_structure=SimpleNamespace(atoms=atoms),
        trajectory_frames=[atoms["xyz"], atoms["xyz"] + 1.0, atoms["xyz"] + 2.0],
        current_frame_index=0,
        frame_count=3,
        labeling_edit=SimpleNamespace(text=lambda: str(labeling)),
    )
    model.set_current_frame = lambda value: setattr(model, "current_frame_index", int(value))
    plot = ProteinMCDistanceNetworkPlot(SimpleNamespace(model=model))
    qtbot.addWidget(plot)

    plot.update_all()

    assert plot.plot_controller.frame_spin.maximum() == 2
    assert plot.plot_controller.start_btn.text() == "|<"
    assert plot.plot_controller.play_btn.text() == "▶"
    assert plot.plot_controller.stop_btn.text() == "■"
    plot.plot_controller.step_spin.setValue(2)
    plot.plot_controller._next_frame()
    assert model.current_frame_index == 2
    assert _agreement_color(-3.0) == _agreement_color(3.0)


def test_proteinmc_widget_start_runs_worker(qapp, qtbot, monkeypatch, tmp_path):
    import chisurf.gui.widgets.models.proteinmc as proteinmc_widget

    class FakeRunner:
        def __init__(self, *, progress_callback=None, output_file=None, **kwargs):
            self.progress_callback = progress_callback
            self.output_file = output_file
            self.structure = SimpleNamespace(atoms=None, xyz=np.zeros((3, 3)))

        def run(self):
            progress = ProteinMCProgress(
                frame_index=1,
                target_frames=1,
                iteration=1,
                accepted=1,
                rejected=0,
                energy=3.0,
                labeling_energy=2.0,
                rmsd=[0.0],
                drmsd=[0.0],
                energies=[3.0],
                labeling_energies=[2.0],
                xyz=np.zeros((3, 3)),
                output_file=self.output_file,
            )
            if self.progress_callback is not None:
                self.progress_callback(progress)
            return SimpleNamespace(output_file=self.output_file)

        def stop(self):
            return None

    monkeypatch.setattr(proteinmc_widget, "ChimolView", FakeChimolView)
    monkeypatch.setattr(proteinmc_widget, "ProteinMCRunner", FakeRunner)
    widget = proteinmc_widget.ProteinMCModelWidget(fit=SimpleNamespace(name="fit", data=None, plots=[]))
    qtbot.addWidget(widget)
    widget.structure_edit.setText(str(Path("test/data/atomic_coordinates/pdb_files/148l.pdb")))
    widget.n_iter_spin.setValue(1)
    widget.n_out_spin.setValue(1)
    widget.n_written_spin.setValue(1)

    widget.start_proteinmc(output_directory=tmp_path)
    assert not widget.structure_edit.isEnabled()

    import time
    start_time = time.perf_counter()
    while widget._thread is not None:
        qapp.processEvents()
        time.sleep(0.01)
        if time.perf_counter() - start_time > 5.0:
            raise TimeoutError("Test timed out waiting for thread to finish")
    assert widget.energy == [3.0]
    assert widget.chi2r == [2.0]
    assert widget.start_button.isEnabled()
    assert widget.structure_edit.isEnabled()


def test_score_set_combobox_in_dialog(qapp, qtbot, monkeypatch, tmp_path):
    """Test the score_set combobox dynamic population and fallback resolution.

    Parameters
    ----------
    qapp : QApplication
        Qt application fixture.
    qtbot : QtBot
        Qt bot fixture.
    monkeypatch : MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : Path
        Temporary directory path.
    """
    import chisurf.gui.widgets.models.proteinmc as proteinmc_widget
    from types import SimpleNamespace
    from qtpy import QtWidgets

    # 1. Create a dummy fps.json file
    labeling = tmp_path / "fps.json"
    labeling.write_text(
        '{"Positions": {"p1": {"chain_identifier": "A", "residue_seq_number": 1}},'
        '"Distances": {"d1": {"position1_name": "p1", "position2_name": "p1", "distance": 5}},'
        '"χ²": {"chi2_C1_20p": {"distances": ["d1"]}, "chi2_C2_33p": {"distances": []}}}',
        encoding="utf-8"
    )

    # 2. Setup widget
    monkeypatch.setattr(proteinmc_widget, "ChimolView", FakeChimolView)
    fit = SimpleNamespace(name="fit", data=None, plots=[])
    widget = proteinmc_widget.ProteinMCModelWidget(fit=fit)
    qtbot.addWidget(widget)

    # Set labeling file
    widget.labeling_edit.setText(str(labeling))

    # Verify score_set_combo contains the options
    items = [widget.score_set_combo.itemText(i) for i in range(widget.score_set_combo.count())]
    assert "chi2_C1_20p" in items
    assert "chi2_C2_33p" in items

    # 3. Test substring matching fallback on load
    state = {
        "proteinmc": {
            "labeling_file": str(labeling),
            "score_set": "c1",
            "settings": {
                "potentials": [
                    {
                        "name": "dye",
                        "weight": 1.0,
                        "settings": {
                            "labeling_file": str(labeling),
                            "score_set": "c1"
                        }
                    }
                ]
            }
        }
    }
    widget.set_state(state)
    assert widget.score_set_combo.currentText() == "chi2_C1_20p"

    # Verify only C1 distance parameters widgets are shown
    assert "d1" in widget._distance_parameters
    assert len(widget._distance_parameters) == 1

    # 4. Test dialog opening and score_set combobox
    dialog_exec_called = False
    def fake_exec(self_dialog):
        nonlocal dialog_exec_called
        dialog_exec_called = True
        combos = self_dialog.findChildren(QtWidgets.QComboBox)
        assert len(combos) >= 1
        combo_items = [combos[0].itemText(i) for i in range(combos[0].count())]
        assert "chi2_C1_20p" in combo_items
        assert "chi2_C2_33p" in combo_items
        assert combos[0].currentText() == "chi2_C1_20p"

        # Simulate user changing score set to "chi2_C2_33p"
        combos[0].setCurrentText("chi2_C2_33p")
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec_", fake_exec)

    # Open dialog for row of "fps" potential
    widget.edit_potential_details(2)

    assert dialog_exec_called
    # The selected score set should now be "chi2_C2_33p" in the widget
    assert widget.score_set_combo.currentText() == "chi2_C2_33p"

    # 5. Verify distances are updated (chi2_C2_33p has 0 distances)
    assert len(widget._distance_parameters) == 0

    # 6. Verify setting score_set to "" shows no distance parameter widgets
    widget.score_set_combo.setCurrentIndex(0) # ""
    assert len(widget._distance_parameters) == 0


def test_update_distance_widgets_integration(qapp, qtbot, monkeypatch, tmp_path):
    """Test that update_distance_widgets updates parameters from trajectory and structure.

    Parameters
    ----------
    qapp : QApplication
        Qt application fixture.
    qtbot : QtBot
        Qt bot fixture.
    monkeypatch : MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : Path
        Temporary directory path.
    """
    import chisurf.gui.widgets.models.proteinmc as proteinmc_widget
    from types import SimpleNamespace
    from qtpy import QtWidgets

    labeling = tmp_path / "fps.json"
    labeling.write_text(
        '{"Positions": {"p1": {"chain_identifier": "A", "residue_seq_number": 1, "atom_name": "CA"},'
        '"p2": {"chain_identifier": "A", "residue_seq_number": 2, "atom_name": "CA"}},'
        '"Distances": {"d1": {"position1_name": "p1", "position2_name": "p2", "distance": 10}},'
        '"χ²": {"chi2_C1": {"distances": ["d1"]}}}',
        encoding="utf-8"
    )

    atoms = np.zeros(
        2,
        dtype=[("xyz", float, (3,)), ("atom_name", "S2"), ("res_id", int), ("chain", "S1")],
    )
    atoms["xyz"] = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    atoms["atom_name"] = [b"CA", b"CA"]
    atoms["res_id"] = [1, 2]
    atoms["chain"] = [b"A", b"A"]
    structure = SimpleNamespace(atoms=atoms, xyz=atoms["xyz"])

    monkeypatch.setattr(proteinmc_widget, "ChimolView", FakeChimolView)
    fit = SimpleNamespace(name="fit", data=structure, plots=[])
    widget = proteinmc_widget.ProteinMCModelWidget(fit=fit)
    qtbot.addWidget(widget)

    widget.labeling_edit.setText(str(labeling))
    widget.score_set_combo.setCurrentText("chi2_C1")

    # Set structural data and verify update_distance_widgets resolves the coordinates
    widget.proteinmc_structure = structure
    widget.update_distance_widgets()
    assert widget._distance_parameters["d1"].value == 10.0

    # Add a frame with a different distance (e.g. 5.0 Å)
    frame = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    widget.trajectory_frames = [frame]
    widget.current_frame_index = 0

    widget.update_distance_widgets()
    assert widget._distance_parameters["d1"].value == 5.0


def test_progress_dialog_minimize_and_restore(qapp, qtbot, monkeypatch):
    """Test progress dialog hide to status bar and double-click to restore."""
    from chisurf.gui.widgets.progress import EnhancedProgressDialog, MinimisedProgressWidget
    from qtpy import QtWidgets, QtCore, QtGui

    # Create dummy main window with a status bar
    main_win = QtWidgets.QMainWindow()
    main_win.setStatusBar(QtWidgets.QStatusBar(main_win))
    main_win.show()
    qtbot.addWidget(main_win)

    # Mock _find_main_window to return our main_win
    monkeypatch.setattr(EnhancedProgressDialog, "_find_main_window", lambda self: main_win)

    dialog = EnhancedProgressDialog(
        title="Test Progress",
        label_text="Running test...",
        min_value=0,
        max_value=100,
        parent=main_win,
        window_modality=QtCore.Qt.NonModal
    )
    dialog.show()
    qtbot.addWidget(dialog)

    # 1. Verify "Hide" button is created
    assert dialog._hide_btn is not None
    assert dialog._hide_btn.text() == "Hide"

    # 2. Hide to status bar
    dialog.hide_to_statusbar()
    assert dialog.isHidden()
    assert dialog._statusbar_widget is not None
    assert isinstance(dialog._statusbar_widget, MinimisedProgressWidget)

    # Verify widget is added to status bar
    statusbar_widgets = main_win.statusBar().findChildren(MinimisedProgressWidget)
    assert len(statusbar_widgets) == 1

    # 3. Restore by double-clicking status bar widget
    # Directly invoke mouseDoubleClickEvent to simulate double click
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonDblClick,
        QtCore.QPointF(5, 5),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier
    ) if hasattr(QtGui, "QMouseEvent") else None
    
    if event:
        dialog._statusbar_widget.mouseDoubleClickEvent(event)
    else:
        dialog.restore_from_statusbar()

    assert dialog.isVisible()
    assert dialog._statusbar_widget is None
    assert len(main_win.statusBar().findChildren(MinimisedProgressWidget)) == 0

    # 4. Hide again and finish to ensure cleanup
    dialog.hide_to_statusbar()
    dialog.finish(close_delay_ms=0)
    assert len(main_win.statusBar().findChildren(MinimisedProgressWidget)) == 0



