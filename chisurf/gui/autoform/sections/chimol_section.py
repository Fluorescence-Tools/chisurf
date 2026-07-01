"""AutoForm ``chimol`` section — a 3D molecular viewer with a frame slider.

A reusable custom section that renders the ChiMol viewer and, when the bound
model attribute yields more than one structure, a slider to step through them
(like the proteinMC trajectory viewer). Declared in a ``.view.json`` as::

    {"type": "custom", "key": "chimol", "target": "preview_models",
     "title": "Structure", "height": 320}

The ``target`` attribute may be a single PDB path (str), a list of PDB paths
(each becomes a frame — they must share the same atom count), or a callable
returning either. The ChiMol plugin is imported lazily so AutoForm/core do not
depend on it; a placeholder is shown if it is unavailable.
"""

from __future__ import annotations

import pathlib

import numpy as np
from qtpy import QtCore, QtWidgets

from .registry import register_section


def _pdb_xyz(path: str) -> np.ndarray:
    """Read ATOM/HETATM coordinates (N, 3) from a PDB file, in file order."""
    xyz = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(xyz, dtype=float)


class ChiMolSectionWidget(QtWidgets.QWidget):
    """ChiMol 3D viewer + frame slider bound to an AutoForm model attribute."""

    AUTOFORM_REFRESH = True  # AutoForm.refresh_plots() will call refresh()

    def __init__(self, model, target: str, *, height: int = 320, **options) -> None:
        super().__init__()
        self._model = model
        self._target = target
        self._viewer = None
        self._object_id = None
        self._loaded_key = None       # dedupe signature of the last load
        self._n_frames = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._host = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self._host).setContentsMargins(0, 0, 0, 0)
        self._host.setMinimumHeight(int(height))
        layout.addWidget(self._host, 1)

        row = QtWidgets.QHBoxLayout()
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        self._label = QtWidgets.QLabel("—")
        row.addWidget(QtWidgets.QLabel("Model"))
        row.addWidget(self._slider, 1)
        row.addWidget(self._label)
        self._slider_bar = QtWidgets.QWidget()
        self._slider_bar.setLayout(row)
        self._slider_bar.setVisible(False)
        layout.addWidget(self._slider_bar)

        self.refresh()

    # -- data --------------------------------------------------------------
    def _paths(self):
        obj = getattr(self._model, self._target, None)
        val = obj() if callable(obj) else obj
        if not val:
            return []
        if isinstance(val, (str, pathlib.Path)):
            val = [val]
        return [str(p) for p in val if p and pathlib.Path(str(p)).exists()]

    def _ensure_viewer(self):
        if self._viewer is not None:
            return self._viewer or None
        try:
            from chisurf.plugins.chimol.chimol.renderer.view import MolView
            self._viewer = MolView(self._host)
            self._host.layout().addWidget(self._viewer)
        except Exception:
            placeholder = QtWidgets.QLabel(
                "3D preview needs the ChiMol plugin (install via the package manager).")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            placeholder.setWordWrap(True)
            self._host.layout().addWidget(placeholder)
            self._viewer = False
        return self._viewer or None

    # -- AutoForm hook -----------------------------------------------------
    def refresh(self) -> None:
        paths = self._paths()
        key = tuple(paths)
        if key == self._loaded_key:
            return
        self._loaded_key = key
        if not paths:
            return
        viewer = self._ensure_viewer()
        if viewer is None:
            return
        try:
            self._load(viewer, paths)
        except Exception:
            pass

    def _load(self, viewer, paths) -> None:
        from chisurf.plugins.chimol.chimol.io.structure import load_structure_payload
        try:
            from chisurf.core.structure import Structure as _Struct
        except Exception:
            _Struct = None

        structure, coords0 = load_structure_payload(
            pathlib.Path(paths[0]), structure_factory=_Struct)
        name = pathlib.Path(paths[0]).stem
        if self._object_id is not None:
            try:
                viewer.remove_object(self._object_id)
            except Exception:
                pass
        if structure is not None:
            oid = viewer.add_structure(structure, name=name, source_path=paths[0])
        elif coords0 is not None:
            oid = viewer.add_coordinates(
                np.asarray(coords0, dtype=float), name=name, source_path=paths[0])
        else:
            return
        self._object_id = oid

        # Multiple models -> load them as trajectory frames + enable the slider.
        frames = []
        if len(paths) > 1:
            ref = _pdb_xyz(paths[0])
            for p in paths:
                xyz = _pdb_xyz(p)
                if xyz.shape == ref.shape:
                    frames.append(xyz)
        if len(frames) > 1:
            try:
                viewer.set_frames(np.stack(frames), object_id=oid, active_frame=0)
            except Exception:
                frames = []
        self._n_frames = len(frames) if len(frames) > 1 else 1

        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._slider_bar.setVisible(self._n_frames > 1)
        self._label.setText(f"1 / {self._n_frames}" if self._n_frames > 1 else "—")

    def _on_slider(self, idx: int) -> None:
        viewer = self._viewer or None
        if not viewer or self._n_frames <= 1:
            return
        try:
            viewer.set_current_frame(int(idx))
        except Exception:
            pass
        self._label.setText(f"{int(idx) + 1} / {self._n_frames}")


@register_section("chimol")
def _chimol_section_factory(model, target: str, **options):
    """Custom-section factory rendering a ChiMol 3D viewer with a frame slider."""
    return ChiMolSectionWidget(model, target, **options)


__all__ = ["ChiMolSectionWidget"]
