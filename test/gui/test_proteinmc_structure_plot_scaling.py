"""Tests for ProteinMC structure plot coordinate scaling."""

from types import SimpleNamespace

import numpy as np
from qtpy import QtWidgets


def test_structure_plot_uses_angstrom_scale_factor(qapp, qtbot, monkeypatch):
    """ProteinMCStructurePlot must tell Chimol not to scale Angstrom coordinates.

    ChiSurf structure coordinates are in Angstrom. Chimol's default scale
    factor of 10 assumes nanometer input, which would make the structure
    appear 10x too large. The plot should pass scale_factor=1.0.
    """
    import chisurf.gui.plots.proteinMC as proteinMC_module
    from chisurf.gui.plots.proteinMC import ProteinMCStructurePlot

    calls = []

    class CapturingChimolView(QtWidgets.QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args[:1] if args else [])
            calls.append(kwargs)

        def add_structure(self, structure, *, name=None, source_path=None):
            return "obj1"

        def set_representation(self, mode, *, object_id=None):
            pass

        def set_frames(self, frames, *, object_id=None, active_frame=None):
            pass

    monkeypatch.setattr(proteinMC_module, "ChimolView", CapturingChimolView)

    atoms = np.zeros(
        3,
        dtype=[("xyz", float, (3,)), ("atom_name", "S2"), ("res_id", int), ("chain", "S1")],
    )
    atoms["xyz"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms["atom_name"] = [b"CA", b"CA", b"CA"]
    atoms["res_id"] = [1, 2, 3]
    atoms["chain"] = [b"A", b"A", b"A"]
    structure = SimpleNamespace(atoms=atoms)
    model = SimpleNamespace(
        proteinmc_structure=structure,
        trajectory_frames=[],
        current_frame_index=0,
    )
    fit = SimpleNamespace(model=model, name="fit")

    plot = ProteinMCStructurePlot(fit=fit)
    qtbot.addWidget(plot)

    assert len(calls) == 1, f"Expected one ChimolView construction, got {calls}"
    assert calls[0].get("scale_factor") == 1.0, (
        f"Expected scale_factor=1.0 for Angstrom coordinates, got {calls[0]}"
    )
