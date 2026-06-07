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

    qtbot.waitUntil(lambda: widget._thread is None, timeout=3000)
    assert widget.energy == [3.0]
    assert widget.chi2r == [2.0]
    assert widget.start_button.isEnabled()
    assert "finished" in widget.status_label.text()
