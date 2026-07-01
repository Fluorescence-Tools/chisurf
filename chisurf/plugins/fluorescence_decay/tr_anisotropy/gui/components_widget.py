"""Lifetime + rotation spectrum editor (embedded in the wizard).

Two small tables — one for the fluorescence lifetime spectrum, one for the
rotation (anisotropy) spectrum — each a list of ``[amplitude, value]`` rows, with
add/remove controls and Save/Load buttons backed by
:mod:`...core.spectra`. Bound to ``model.lifetime_spectrum`` /
``model.rotation_spectrum``.

Both tables use a single, consistent column order — **amplitude, value** — which
also matches how the spectra are stored and how the fit consumes them. (The
legacy hand-written wizard used the opposite order in its *add* buttons versus its
*load* path, so manually added components were silently swapped.)
"""

from __future__ import annotations

from qtpy import QtCore, QtWidgets


class _SpectrumTable(QtWidgets.QWidget):
    """One labelled table of ``[amplitude, value]`` rows with add/remove."""

    def __init__(self, title: str, value_header: str, get_rows, set_rows, parent=None):
        super().__init__(parent)
        self._get_rows = get_rows
        self._set_rows = set_rows

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        header = QtWidgets.QLabel(f"<b>{title}</b>")
        layout.addWidget(header)

        self._table = QtWidgets.QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Amplitude", value_header])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(self._commit)
        layout.addWidget(self._table)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("a"))
        self._amp = QtWidgets.QDoubleSpinBox()
        self._amp.setDecimals(3)
        self._amp.setRange(0.0, 1e6)
        self._amp.setValue(0.5)
        bar.addWidget(self._amp)
        bar.addWidget(QtWidgets.QLabel(value_header[0].lower()))
        self._val = QtWidgets.QDoubleSpinBox()
        self._val.setDecimals(3)
        self._val.setRange(0.0, 1e6)
        self._val.setValue(1.0)
        bar.addWidget(self._val)
        add = QtWidgets.QToolButton()
        add.setText("➕")
        add.setToolTip("Add a component.")
        add.clicked.connect(self._add)
        bar.addWidget(add)
        rem = QtWidgets.QToolButton()
        rem.setText("➖")
        rem.setToolTip("Remove the selected component.")
        rem.clicked.connect(self._remove)
        bar.addWidget(rem)
        bar.addStretch(1)
        layout.addLayout(bar)

        self.reload()

    def reload(self) -> None:
        """Rebuild the table from the model rows."""
        rows = self._get_rows()
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for amp, val in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{float(amp):.3f}"))
            self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{float(val):.3f}"))
        self._table.blockSignals(False)

    def _read(self) -> list[list[float]]:
        rows = []
        for r in range(self._table.rowCount()):
            a = self._table.item(r, 0)
            v = self._table.item(r, 1)
            if a is None or v is None:
                continue
            try:
                rows.append([float(a.text()), float(v.text())])
            except ValueError:
                continue
        return rows

    def _commit(self, *_a) -> None:
        self._set_rows(self._read())

    def _add(self) -> None:
        r = self._table.rowCount()
        self._table.blockSignals(True)
        self._table.insertRow(r)
        self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{self._amp.value():.3f}"))
        self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{self._val.value():.3f}"))
        self._table.blockSignals(False)
        self._commit()

    def _remove(self) -> None:
        idx = self._table.currentRow()
        if idx < 0:
            idx = self._table.rowCount() - 1
        if idx >= 0:
            self._table.removeRow(idx)
            self._commit()


class ComponentsWidget(QtWidgets.QWidget):
    """Lifetime + rotation spectrum editor bound to the view-model."""

    _autoform_expanding = True

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tables = QtWidgets.QHBoxLayout()
        self._lifetimes = _SpectrumTable(
            "Lifetime spectrum",
            "Lifetime (ns)",
            lambda: self._model.lifetime_spectrum,
            self._set_lifetimes,
        )
        self._rotations = _SpectrumTable(
            "Rotation spectrum",
            "ρ (ns)",
            lambda: self._model.rotation_spectrum,
            self._set_rotations,
        )
        tables.addWidget(self._lifetimes)
        tables.addWidget(self._rotations)
        layout.addLayout(tables)

        bar = QtWidgets.QHBoxLayout()
        save = QtWidgets.QToolButton()
        save.setText("💾 Save spectrum")
        save.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        save.setToolTip("Save both spectra to a .spk.json file.")
        save.clicked.connect(self._save)
        load = QtWidgets.QToolButton()
        load.setText("📂 Load spectrum")
        load.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        load.setToolTip("Load both spectra from a .spk.json file.")
        load.clicked.connect(self._load)
        bar.addWidget(save)
        bar.addWidget(load)
        bar.addStretch(1)
        layout.addLayout(bar)

    def _set_lifetimes(self, rows) -> None:
        self._model.lifetime_spectrum = rows
        self._model.update()

    def _set_rotations(self, rows) -> None:
        self._model.rotation_spectrum = rows
        self._model.update()

    def _save(self) -> None:
        self._model.save_spectra()
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{self._model.spk_path}")

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load spectrum", "", "Spectra (*.spk.json);;All Files (*)"
        )
        if not path:
            return
        self._model.load_spectra(path)
        self._lifetimes.reload()
        self._rotations.reload()


__all__ = ["ComponentsWidget"]
