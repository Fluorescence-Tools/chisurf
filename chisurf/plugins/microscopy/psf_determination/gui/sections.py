"""Custom AutoForm section for the PSF Determination tool.

The action bar (file load + detection/fit/export buttons + the detected-bead
navigator) is registered here as the ``psf_controls`` custom section. The image
canvas itself reuses AutoForm's enhanced ``image`` section (3D stack browsing +
click-to-pick), configured in ``psf.view.json``; the parameter controls are
plain ``value`` sections. Mirrors the CLSM tool's ``clsm_controls`` pattern.

The widget owns only Qt concerns (buttons, file dialogs, the bead spinner); all
logic lives on the Qt-free :class:`~..view_model.PsfViewModel`. It is imported
(and thus registered) by ``gui.tool``.
"""

from __future__ import annotations

import logging
import pathlib

from qtpy import QtWidgets

from chisurf.gui.autoform.sections.registry import register_section

logger = logging.getLogger(__name__)


def _tool_button(text: str, tooltip: str, slot) -> QtWidgets.QToolButton:
    """Build a configured ``QToolButton`` in one call."""
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    btn.setToolTip(tooltip)
    btn.clicked.connect(slot)
    return btn


@register_section("psf_controls")
def psf_controls(model, target=None, **options):
    """File / detection / fit / export action bar for the PSF tool."""
    return _ControlBar(model)


class _ControlBar(QtWidgets.QWidget):
    """Action bar: load stack, detect, fit selected/all, export, bead navigator."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        form = QtWidgets.QVBoxLayout(self)
        form.setContentsMargins(2, 2, 2, 2)
        form.setSpacing(2)

        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(2)
        row1.addWidget(
            _tool_button(
                "Load Stack", "Load a 3D image stack (TIFF) for PSF inspection.", self._load
            )
        )
        row1.addWidget(
            _tool_button(
                "Detect",
                "Automatic bead detection using an adaptive quantile threshold.",
                self._detect,
            )
        )
        row1.addStretch(1)
        form.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(2)
        row2.addWidget(
            _tool_button(
                "Fit Selected", "Fit a 3D Gaussian PSF to the selected bead.", self._fit_selected
            )
        )
        row2.addWidget(
            _tool_button(
                "Fit All",
                "Fit a 3D Gaussian PSF to every detected bead and summarize.",
                self._fit_all,
            )
        )
        row2.addWidget(
            _tool_button(
                "Export CSV", "Export the batch PSF fit summary to a CSV file.", self._export
            )
        )
        row2.addStretch(1)
        form.addLayout(row2)

        nav = QtWidgets.QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(4)
        nav.addWidget(QtWidgets.QLabel("Bead #"))
        self._bead_spin = QtWidgets.QSpinBox()
        self._bead_spin.setRange(0, 0)
        self._bead_spin.setEnabled(False)
        self._bead_spin.setToolTip("Navigate detected beads; selecting one fits it.")
        self._bead_spin.valueChanged.connect(self._on_bead_index)
        nav.addWidget(self._bead_spin)
        self._info = QtWidgets.QLabel("No stack loaded")
        self._info.setStyleSheet("color: #888888;")
        nav.addWidget(self._info, 1)
        form.addLayout(nav)

        self._model.add_observer(self._on_model_event)

    # ── model wiring ────────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        n = len(self._model.detected_beads)
        if event == "stack":
            self._bead_spin.blockSignals(True)
            self._bead_spin.setRange(0, 0)
            self._bead_spin.setValue(0)
            self._bead_spin.setEnabled(False)
            self._bead_spin.blockSignals(False)
            self._info.setText(pathlib.Path(self._model.filename).name or "stack loaded")
        elif event == "beads":
            self._bead_spin.blockSignals(True)
            self._bead_spin.setRange(0, max(0, n - 1))
            self._bead_spin.setValue(0)
            self._bead_spin.setEnabled(n > 0)
            self._bead_spin.blockSignals(False)
            self._info.setText(f"Detected beads: {n}")

    # ── actions ─────────────────────────────────────────────────────────
    def _working_dir(self) -> str:
        if self._model.filename:
            return str(pathlib.Path(self._model.filename).parent)
        return str(pathlib.Path.home())

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Image Stack",
            self._working_dir(),
            "TIFF Files (*.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        try:
            self._model.load_stack(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "Load Error", f"Failed to read image stack:\n{exc}"
            )

    def _detect(self) -> None:
        if self._model.stack is None:
            QtWidgets.QMessageBox.information(self, "No stack", "Load a stack first.")
            return
        self._model.detect_beads()

    def _fit_selected(self) -> None:
        if self._model.selected_bead is None:
            QtWidgets.QMessageBox.information(
                self, "No bead", "Click a bead in the image or detect beads first."
            )
            return
        self._model.fit_selected()

    def _fit_all(self) -> None:
        if not self._model.detected_beads:
            QtWidgets.QMessageBox.information(self, "No beads", "Detect beads first.")
            return
        self._model.fit_all()

    def _export(self) -> None:
        if not self._model.detected_beads:
            QtWidgets.QMessageBox.information(self, "No beads", "Detect beads first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export PSF batch results to CSV",
            str(pathlib.Path(self._working_dir()) / "psf_batch_results.csv"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            self._model.export_csv(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "CSV export error", f"Failed to export CSV:\n{exc}"
            )

    def _on_bead_index(self, value: int) -> None:
        self._model.select_bead_index(int(value))


__all__ = ["psf_controls"]
