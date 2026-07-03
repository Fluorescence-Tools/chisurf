"""FCS Merger panel — model and custom AutoForm sections."""

from __future__ import annotations

import pathlib
import re
import typing

import numpy as np
from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import register_section

_GUI_DIR = pathlib.Path(__file__).resolve().parent


class MergerSettingsModel:
    """Settings model + plot data sources for the merger AutoForm panel."""

    def __init__(self):
        self.folder_path = ""
        self._correlations: typing.List[dict] = []
        self._selected_index: int = -1
        self._form: typing.Any = None

    def view_spec(self):
        return load_view_spec(_GUI_DIR / "merger.view.json")

    # -- plot data sources ---------------------------------------------------

    def curves_series(self):
        result = []
        for i, cor in enumerate(self._correlations):
            included = cor.get("use_curve", True)
            highlighted = i == self._selected_index
            try:
                color = cs.core.settings.colors[i % len(cs.core.settings.colors)][
                    "hex"
                ]
            except Exception:
                color = "y"
            # The row selected in the table is drawn white and thick so it
            # stands out from the other curves; excluded curves are dashed.
            result.append(
                {
                    "x": cor["x"],
                    "y": cor["y"],
                    "name": f"chunk {i}",
                    "color": "w" if highlighted else color,
                    "width": 3 if highlighted else 1,
                    "style": "solid" if included else "dash",
                }
            )
        return result

    def mean_series(self):
        mean = self.compute_mean()
        if mean is not None:
            return [
                {
                    "x": mean["x"].tolist(),
                    "y": mean["y"].tolist(),
                    "name": "merged",
                    "color": "w",
                    "width": 2,
                }
            ]
        return []

    # -- merge logic ---------------------------------------------------------

    def compute_mean(self) -> dict | None:
        from chisurf.core.fluorescence.fcs.merge import (
            compute_average_correlations,
        )
        selected = [c for c in self._correlations if c.get("use_curve", True)]
        if not selected:
            selected = self._correlations
        if not selected:
            return None
        result = compute_average_correlations(selected)
        n = len(selected)
        return {
            "x": np.asarray(result["x"], dtype=float),
            "y": np.asarray(result["y"], dtype=float),
            "ey": np.asarray(result.get("ey", np.zeros_like(result["x"])), dtype=float),
            "duration": float(result["duration"]),
            "count_rate": float(result["count_rate"]),
            "n_curves": n,
        }

    def load_correlations(self, folder: pathlib.Path | None = None) -> None:
        if folder is None:
            folder = pathlib.Path(self.folder_path) if self.folder_path else None
        if folder is None or not folder.is_dir():
            return
        self._correlations.clear()
        self.folder_path = str(folder.resolve())
        from chisurf.core.fluorescence.fcs.merge import (
            parse_correlation_folder,
        )
        raw = parse_correlation_folder(folder)
        for cor in raw:
            cor["use_curve"] = True
            self._correlations.append(cor)
        if self._form is not None:
            self._form.refresh_plots()
            self._refresh_table()

    def set_correlations(
        self,
        correlations: typing.List[dict],
        source_folder: pathlib.Path | None = None,
    ) -> None:
        self._correlations.clear()
        for cor in correlations:
            c = dict(cor)
            c["use_curve"] = True
            self._correlations.append(c)
        if source_folder is not None:
            self.folder_path = str(source_folder.resolve())
        if self._form is not None:
            self._form.refresh_plots()
            self._refresh_table()

    def target_filepath(self) -> pathlib.Path:
        folder = pathlib.Path(self.folder_path) if self.folder_path else pathlib.Path()
        stem = folder.stem
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        if not safe_stem or safe_stem in {".", "..", "_"}:
            safe_stem = "correlation"
        return folder.parent / (safe_stem + ".cor")

    def save_mean(self, filename: pathlib.Path | None = None) -> None:
        from chisurf.core.fluorescence.fcs.merge import save_mean_correlation
        mean = self.compute_mean()
        if mean is None:
            QtWidgets.QMessageBox.warning(None, "No Data", "No correlations to save.")
            return
        if filename is None:
            filename = self.target_filepath()
        save_mean_correlation(mean, filename)

    def add_to_chisurf(self) -> None:
        filepath = self.target_filepath()
        if not filepath.exists():
            self.save_mean(filepath)
        if not filepath.exists():
            QtWidgets.QMessageBox.warning(
                None,
                "No Correlation File",
                "No correlation file available.",
            )
            return
        cs.core.actions.dispatch(
            name="experiment.set",
            payload={"name": "FCS"},
        )
        cs.core.actions.dispatch(
            name="setup.select",
            payload={"name": "Seidel Kristine"},
        )
        cs.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": filepath.as_posix(), "experiment_reader": None},
        )

    # -- table helpers -------------------------------------------------------

    def _refresh_table(self) -> None:
        for w in self._form.findChildren(_MergerTable):
            try:
                w.refresh()
            except Exception:
                pass

    def _toggle_curve(self, index: int) -> None:
        if 0 <= index < len(self._correlations):
            current = self._correlations[index].get("use_curve", True)
            self._correlations[index]["use_curve"] = not current
            if self._form is not None:
                self._form.refresh_plots()


# ---- Custom AutoForm sections ----------------------------------------------


@register_section("merger_folder_picker")
class _MergerFolderPicker(QtWidgets.QWidget):
    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)
        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.clicked.connect(self._on_open)
        layout.addWidget(self.open_btn)
        layout.addStretch(1)

    def _on_browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Correlation Folder"
        )
        if folder:
            self._model.folder_path = folder
            self._model.load_correlations(pathlib.Path(folder))

    def _on_open(self) -> None:
        folder = pathlib.Path(self._model.folder_path) if self._model.folder_path else None
        if folder is None or not folder.is_dir():
            self._on_browse()
            return
        self._model.load_correlations(folder)


@register_section("merger_table")
class _MergerTable(QtWidgets.QWidget):
    AUTOFORM_REFRESH = True

    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Use", "File", "CR A (kHz)", "CR B (kHz)", "Duration (s)"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == 0:
            row = item.row()
            self._model._toggle_curve(row)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        self._model._selected_index = rows[0].row() if rows else -1
        if self._model._form is not None:
            try:
                self._model._form.refresh_plots()
            except Exception:
                pass

    def refresh(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, cor in enumerate(self._model._correlations):
            rc = self.table.rowCount()
            self.table.insertRow(rc)

            checkbox = QtWidgets.QTableWidgetItem()
            checkbox.setFlags(
                QtCore.Qt.ItemIsUserCheckable
                | QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemIsSelectable
            )
            checkbox.setCheckState(
                QtCore.Qt.Checked if cor.get("use_curve", True) else QtCore.Qt.Unchecked
            )
            self.table.setItem(rc, 0, checkbox)

            try:
                fn = cor.get("_filename", f"chunk {i}")
            except Exception:
                fn = f"chunk {i}"
            fn_item = QtWidgets.QTableWidgetItem(str(fn))
            fn_item.setFlags(fn_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(rc, 1, fn_item)

            duration = float(cor.get("duration", 0.0))
            try:
                ca = float(cor["channel_a"]["counts"])
                cb = float(cor["channel_b"]["counts"])
                cr_a = ca / duration / 1000.0 if duration > 0 else 0.0
                cr_b = cb / duration / 1000.0 if duration > 0 else 0.0
            except Exception:
                total_cr = float(cor.get("count_rate", 0.0))
                cr_a = total_cr / 2.0
                cr_b = total_cr / 2.0

            self.table.setItem(rc, 2, QtWidgets.QTableWidgetItem(f"{cr_a:.2f}"))
            self.table.setItem(rc, 3, QtWidgets.QTableWidgetItem(f"{cr_b:.2f}"))
            self.table.setItem(
                rc, 4, QtWidgets.QTableWidgetItem(f"{duration:.2f}")
            )

        self.table.resizeRowsToContents()
        self.table.blockSignals(False)


@register_section("merger_controls")
class _MergerControls(QtWidgets.QWidget):
    AUTOFORM_REFRESH = True

    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.save_btn = QtWidgets.QPushButton("Save Merged")
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #1f7a1f; color: white; "
            "border: 1px solid #166016; border-radius: 4px; "
            "padding: 4px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #249124; }"
        )
        self.save_btn.clicked.connect(lambda: self._model.save_mean())
        layout.addWidget(self.save_btn)

        self.add_btn = QtWidgets.QPushButton("Add to ChiSurf")
        self.add_btn.setStyleSheet(
            "QPushButton { background-color: #1a5f8a; color: white; "
            "border: 1px solid #13446a; border-radius: 4px; "
            "padding: 4px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1e75a8; }"
        )
        self.add_btn.clicked.connect(lambda: self._model.add_to_chisurf())
        layout.addWidget(self.add_btn)

        self.info = QtWidgets.QLabel("")
        layout.addWidget(self.info, 1)

    def refresh(self) -> None:
        n = len(self._model._correlations)
        mean = self._model.compute_mean()
        if mean is not None:
            self.info.setText(
                f"{n} curve(s), "
                f"dur={mean['duration']:.1f}s, "
                f"CR={mean['count_rate']:.1f} kHz"
            )
        else:
            self.info.setText(f"{n} curve(s)")


class _CorrPlot(QtWidgets.QWidget):
    """Minimal log-x FCS plot that redraws a model source on refresh."""

    AUTOFORM_REFRESH = True

    _STYLES = {
        "solid": QtCore.Qt.SolidLine,
        "dash": QtCore.Qt.DashLine,
        "dot": QtCore.Qt.DotLine,
    }

    def __init__(self, model, source: str, title: str, *, legend: bool = True):
        super().__init__()
        import pyqtgraph as pg

        self._model = model
        self._source = source
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Correlation time (ms)")
        self.plot.setLabel("left", "G")
        self.plot.setTitle(title)
        try:
            self.plot.setLogMode(True, False)
        except Exception:
            pass
        if legend:
            try:
                self.plot.addLegend()
            except Exception:
                pass
        try:
            self.plot.getPlotItem().getViewBox().setMenuEnabled(False)
        except Exception:
            pass
        layout.addWidget(self.plot)
        self.refresh()

    def refresh(self) -> None:
        import pyqtgraph as pg

        source = getattr(self._model, self._source, None)
        if not callable(source):
            return
        try:
            series = source() or []
        except Exception:
            return
        self.plot.clear()
        for s in series:
            pen = pg.mkPen(
                s.get("color", "y"),
                width=int(s.get("width", 1)),
                style=self._STYLES.get(s.get("style", "solid"), QtCore.Qt.SolidLine),
            )
            self.plot.plot(s.get("x", []), s.get("y", []), pen=pen, name=s.get("name", ""))


@register_section("merger_workspace")
class _MergerWorkspace(QtWidgets.QWidget):
    """Dockable merger workspace: correlation list (left) + FCS plots (right).

    Uses the shared :class:`DockArea` so the panes can be dragged, resized,
    tabbed or re-split by the user; the arrangement is persisted.
    """

    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        from chisurf.gui.widgets.dock_area.dock_area import DockArea

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: correlation list + save/add controls.
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.table = _MergerTable(model)
        self.controls = _MergerControls(model)
        left_layout.addWidget(self.table, 1)
        left_layout.addWidget(self.controls)

        # Right: individual chunk curves (top) + merged curve (bottom).
        self.curves_plot = _CorrPlot(model, "curves_series", "Individual FCS Curves")
        self.mean_plot = _CorrPlot(model, "mean_series", "Merged FCS Curve", legend=False)

        dock = DockArea(self)
        dock.addTab(left, "Correlations")
        tw_left = dock.find_main_tab_widget()
        tw_ind = self._split(dock, self.curves_plot, "Individual", tw_left, "right")
        self._split(dock, self.mean_plot, "Merged", tw_ind, "bottom")

        root = getattr(dock, "_root_widget", None)
        if isinstance(root, QtWidgets.QSplitter):
            try:
                root.setSizes([320, 700])
            except Exception:
                pass
        try:
            dock.enable_persistence("fcs_merger_dock")
        except Exception:
            pass
        layout.addWidget(dock)
        self._dock = dock

    @staticmethod
    def _split(dock, widget, name, target_tw, zone):
        new_tw = dock._create_tab_widget()
        new_tw.addTab(widget, name)
        dock._all_widgets.append(widget)
        dock._tab_names[widget] = name
        try:
            dock.setTabCloseMode(widget, "hide")
        except Exception:
            pass
        dock.split_tab_widget(target_tw, new_tw, zone)
        return new_tw
