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

    def __init__(self, model, target: str, *, height: int = 320,
                 scale_factor: float = 1.0, **options) -> None:
        super().__init__()
        self._model = model
        self._target = target
        # PDB / ChiSurf coordinates are in Angstrom; ChiMol's default scale of 10
        # assumes nanometer input, so pass 1.0 (matches the proteinMC viewer).
        self._scale_factor = float(scale_factor)
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

        # Movie navigator: prev / play / next / slider / counter. Always shown
        # (disabled when there is a single frame) so the control is discoverable.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)
        self._btn_prev = QtWidgets.QToolButton()
        self._btn_prev.setText("⏮")
        self._btn_prev.setToolTip("Previous model")
        self._btn_prev.clicked.connect(lambda: self._step(-1))
        self._btn_play = QtWidgets.QToolButton()
        self._btn_play.setText("▶")
        self._btn_play.setCheckable(True)
        self._btn_play.setToolTip("Play / pause through the models")
        self._btn_play.toggled.connect(self._on_play)
        self._btn_next = QtWidgets.QToolButton()
        self._btn_next.setText("⏭")
        self._btn_next.setToolTip("Next model")
        self._btn_next.clicked.connect(lambda: self._step(1))
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        self._label = QtWidgets.QLabel("0 / 0")
        self._label.setMinimumWidth(60)
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        for wdg in (self._btn_prev, self._btn_play, self._btn_next):
            row.addWidget(wdg)
        row.addWidget(self._slider, 1)
        row.addWidget(self._label)
        self._nav_bar = QtWidgets.QWidget()
        self._nav_bar.setLayout(row)
        layout.addWidget(self._nav_bar)

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(400)
        self._play_timer.timeout.connect(lambda: self._step(1, wrap=True))

        self._set_nav_enabled(False)
        self.refresh()

    def _set_nav_enabled(self, enabled: bool) -> None:
        for wdg in (self._btn_prev, self._btn_play, self._btn_next, self._slider):
            wdg.setEnabled(enabled)

    def _step(self, delta: int, wrap: bool = False) -> None:
        if self._n_frames <= 1:
            return
        idx = self._slider.value() + delta
        if wrap:
            idx %= self._n_frames
        idx = max(0, min(idx, self._n_frames - 1))
        self._slider.setValue(idx)

    def _on_play(self, playing: bool) -> None:
        self._btn_play.setText("⏸" if playing else "▶")
        if playing and self._n_frames > 1:
            self._play_timer.start()
        else:
            self._play_timer.stop()

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
            try:
                self._viewer = MolView(self._host, scale_factor=self._scale_factor)
            except TypeError:  # older MolView without the kwarg
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
        self._set_nav_enabled(self._n_frames > 1)
        self._label.setText(f"1 / {self._n_frames}")

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
