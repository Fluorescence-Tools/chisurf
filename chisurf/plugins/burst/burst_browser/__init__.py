from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from chisurf import logging
from chisurf.core.fio.fluorescence import burst as burstio

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



name = "Spectroscopy:Single-Molecule:Burst Browser"


class BurstTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._rows: np.ndarray | None = None

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        if df is None:
            self._rows = None
        else:
            self._rows = np.arange(len(df), dtype=int)
        self.endResetModel()

    def set_mask(self, mask: np.ndarray):
        if self._df is None:
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(self._df):
            return
        self.beginResetModel()
        self._rows = np.where(mask)[0]
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid() or self._df is None or self._rows is None:
            return 0
        return int(self._rows.size)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid() or self._df is None:
            return 0
        return int(self._df.shape[1])

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):  # type: ignore[override]
        if (
            not index.isValid()
            or self._df is None
            or self._rows is None
            or role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole)
        ):
            return None
        r = int(self._rows[index.row()])
        c = index.column()
        try:
            value = self._df.iat[r, c]
        except Exception:
            return None
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def headerData(  # type: ignore[override]
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ):
        if role != QtCore.Qt.DisplayRole or self._df is None:
            return None
        if orientation == QtCore.Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return None
        else:
            return str(section + 1)


@persist_plugin_state("burst_browser")
class BurstBrowserWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Burst Browser")

        self._df: pd.DataFrame | None = None
        self._mask: np.ndarray | None = None
        self._col_E: str | None = None
        self._col_S: str | None = None
        self._col_size: str | None = None
        self._have_E: bool = False
        self._have_S: bool = False

        self._model = BurstTableModel(self)

        # Optional experimental-setup information (windows/detectors)
        self.setup_info = None
        self.setup_windows: dict | None = None
        self.setup_detectors: dict | None = None

        self._build_ui()

    # --- UI setup -----------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open folder...", self)
        self.open_button.clicked.connect(self._on_open_folder)
        self.path_label = QtWidgets.QLabel("No file loaded", self)
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.open_button)
        top.addWidget(self.path_label, 1)
        layout.addLayout(top)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        self.table_view = QtWidgets.QTableView(splitter)
        self.table_view.setModel(self._model)
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table_view.setSortingEnabled(True)
        sel_model = self.table_view.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(self._on_table_selection_changed)

        right_widget = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.hist_column_combo = QtWidgets.QComboBox(right_widget)
        self.hist_column_combo.currentIndexChanged.connect(self.update_histogram)
        right_layout.addWidget(self.hist_column_combo)

        # Histogram mode: all gated vs selected bursts
        self.hist_selection_check = QtWidgets.QCheckBox("Use selected bursts", right_widget)
        self.hist_selection_check.setToolTip("If checked, histogram uses only selected rows from the table.")
        self.hist_selection_check.toggled.connect(self.update_histogram)
        right_layout.addWidget(self.hist_selection_check)

        self.hist_plot = pg.PlotWidget(right_widget)
        self.hist_plot.setLabel("left", "Counts")
        right_layout.addWidget(self.hist_plot, 1)

        splitter.addWidget(self.table_view)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Detector selection (for channel-specific views)
        detector_row = QtWidgets.QHBoxLayout()
        detector_row.addWidget(QtWidgets.QLabel("Detector:", self))
        self.detector_combo = QtWidgets.QComboBox(self)
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        detector_row.addWidget(self.detector_combo)
        detector_row.addStretch(1)
        layout.addLayout(detector_row)

        gating_box = QtWidgets.QGroupBox("Gating", self)
        gating_layout = QtWidgets.QGridLayout(gating_box)

        self.e_min_spin = QtWidgets.QDoubleSpinBox(gating_box)
        self.e_min_spin.setRange(0.0, 1.0)
        self.e_min_spin.setSingleStep(0.01)
        self.e_max_spin = QtWidgets.QDoubleSpinBox(gating_box)
        self.e_max_spin.setRange(0.0, 1.0)
        self.e_max_spin.setSingleStep(0.01)

        self.s_min_spin = QtWidgets.QDoubleSpinBox(gating_box)
        self.s_min_spin.setRange(0.0, 1.0)
        self.s_min_spin.setSingleStep(0.01)
        self.s_max_spin = QtWidgets.QDoubleSpinBox(gating_box)
        self.s_max_spin.setRange(0.0, 1.0)
        self.s_max_spin.setSingleStep(0.01)

        self.size_min_spin = QtWidgets.QSpinBox(gating_box)
        self.size_min_spin.setRange(0, 10_000_000)
        self.size_max_spin = QtWidgets.QSpinBox(gating_box)
        self.size_max_spin.setRange(0, 10_000_000)

        gating_layout.addWidget(QtWidgets.QLabel("E min", gating_box), 0, 0)
        gating_layout.addWidget(self.e_min_spin, 0, 1)
        gating_layout.addWidget(QtWidgets.QLabel("E max", gating_box), 0, 2)
        gating_layout.addWidget(self.e_max_spin, 0, 3)

        gating_layout.addWidget(QtWidgets.QLabel("S min", gating_box), 1, 0)
        gating_layout.addWidget(self.s_min_spin, 1, 1)
        gating_layout.addWidget(QtWidgets.QLabel("S max", gating_box), 1, 2)
        gating_layout.addWidget(self.s_max_spin, 1, 3)

        gating_layout.addWidget(QtWidgets.QLabel("Size min", gating_box), 2, 0)
        gating_layout.addWidget(self.size_min_spin, 2, 1)
        gating_layout.addWidget(QtWidgets.QLabel("Size max", gating_box), 2, 2)
        gating_layout.addWidget(self.size_max_spin, 2, 3)

        layout.addWidget(gating_box)

        self.status_label = QtWidgets.QLabel("", self)
        layout.addWidget(self.status_label)

        for w in (
            self.e_min_spin,
            self.e_max_spin,
            self.s_min_spin,
            self.s_max_spin,
            self.size_min_spin,
            self.size_max_spin,
        ):
            w.valueChanged.connect(self._update_gating)

        self._enable_e_controls(False)
        self._enable_s_controls(False)
        self._enable_size_controls(False)

    # --- Helpers ------------------------------------------------------
    def _enable_e_controls(self, enabled: bool) -> None:
        self.e_min_spin.setEnabled(enabled)
        self.e_max_spin.setEnabled(enabled)

    def _enable_s_controls(self, enabled: bool) -> None:
        self.s_min_spin.setEnabled(enabled)
        self.s_max_spin.setEnabled(enabled)

    def _enable_size_controls(self, enabled: bool) -> None:
        self.size_min_spin.setEnabled(enabled)
        self.size_max_spin.setEnabled(enabled)

    def _on_table_selection_changed(self, selected, deselected) -> None:
        """Update histogram when selection changes in selection-only mode."""
        if getattr(self, "hist_selection_check", None) is not None and self.hist_selection_check.isChecked():
            self.update_histogram()

    # --- File / folder loading ----------------------------------------
    def _on_open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder with .bur files")
        if not folder:
            return
        self.load_folder(Path(folder))

    def load_bur(self, path: Path) -> None:
        try:
            df = burstio.read_bur_file(path)
        except Exception as exc:
            logging.error(f"Failed to read BUR file {path}: {exc}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not read BUR file:\n{path}\n\n{exc}")
            return

        self.path_label.setText(str(path))
        self._prepare_dataframe(df)
        self._update_gating()

    def load_folder(self, folder: Path) -> None:
        folder = Path(folder)
        if not folder.exists() or not folder.is_dir():
            QtWidgets.QMessageBox.warning(self, "Folder not found", f"Folder does not exist:\n{folder}")
            return

        bur_files = sorted(folder.glob("**/*.bur"))
        if not bur_files:
            QtWidgets.QMessageBox.information(self, "No files", f"No .bur files found in:\n{folder}")
            return

        dfs: list[pd.DataFrame] = []
        for fn in bur_files:
            try:
                df_part = burstio.read_bur_file(fn)
                df_part["burst_file"] = fn.name
                dfs.append(df_part)
            except Exception as exc:
                logging.warning(f"BurstBrowser: failed to read {fn}: {exc}")

        if not dfs:
            QtWidgets.QMessageBox.information(self, "No data", "No .bur files could be read.")
            return

        df = pd.concat(dfs, ignore_index=True)
        self.path_label.setText(f"{folder} ({len(bur_files)} .bur)")
        self._prepare_dataframe(df)
        self._update_gating()
        self._load_setup_info(folder)

    # --- Data prep ----------------------------------------------------
    def _prepare_dataframe(self, df: pd.DataFrame) -> None:
        df = df.copy()

        if "Number of Photons" in df.columns:
            try:
                n = pd.to_numeric(df["Number of Photons"], errors="coerce")
                df = df[n > 0].reset_index(drop=True)
            except Exception:
                pass

        self._col_E = None
        self._col_S = None
        self._col_size = None
        self._have_E = False
        self._have_S = False

        red_col = None
        green_col = None

        if "E" in df.columns:
            self._col_E = "E"
            self._have_E = True
        elif "Proximity Ratio" in df.columns:
            self._col_E = "Proximity Ratio"
            self._have_E = True
        else:
            photon_cols = [c for c in df.columns if "Number of Photons (" in c]
            red_candidates = [c for c in photon_cols if "red" in c.lower()]
            green_candidates = [c for c in photon_cols if "green" in c.lower()]
            if red_candidates and green_candidates:
                red_col = red_candidates[0]
                green_col = green_candidates[0]
                try:
                    red = pd.to_numeric(df[red_col], errors="coerce")
                    green = pd.to_numeric(df[green_col], errors="coerce")
                    denom = red + green
                    with np.errstate(divide="ignore", invalid="ignore"):
                        e = np.where(denom > 0, red / denom, np.nan)
                    df["E"] = e
                    self._col_E = "E"
                    self._have_E = True
                except Exception:
                    logging.warning("BurstBrowser: failed to compute E from red/green columns")

        if "S" in df.columns:
            self._col_S = "S"
            self._have_S = True
        elif self._have_E and red_col and green_col and "Number of Photons" in df.columns:
            try:
                nd = pd.to_numeric(df[green_col], errors="coerce")
                na = pd.to_numeric(df[red_col], errors="coerce")
                total = pd.to_numeric(df["Number of Photons"], errors="coerce")
                num = nd + na
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = np.where(total > 0, num / total, np.nan)
                df["S"] = s
                self._col_S = "S"
                self._have_S = True
            except Exception:
                logging.warning("BurstBrowser: failed to compute S")

        if "Number of Photons" in df.columns:
            self._col_size = "Number of Photons"
        else:
            cands = [c for c in df.columns if "Number of Photons" in c]
            self._col_size = cands[0] if cands else None

        self._df = df
        self._model.set_dataframe(df)

        # Update detector list now that we know the columns
        self._update_detector_list()

        if self._have_E and self._col_E is not None:
            try:
                col = pd.to_numeric(df[self._col_E], errors="coerce")
                vmin = float(np.nanmin(col)) if np.isfinite(col).any() else 0.0
                vmax = float(np.nanmax(col)) if np.isfinite(col).any() else 1.0
            except Exception:
                vmin, vmax = 0.0, 1.0
            if vmin >= vmax:
                vmin, vmax = 0.0, 1.0
            self.e_min_spin.blockSignals(True)
            self.e_max_spin.blockSignals(True)
            self.e_min_spin.setValue(vmin)
            self.e_max_spin.setValue(vmax)
            self.e_min_spin.blockSignals(False)
            self.e_max_spin.blockSignals(False)
        self._enable_e_controls(self._have_E)

        if self._have_S and self._col_S is not None:
            try:
                col = pd.to_numeric(df[self._col_S], errors="coerce")
                vmin = float(np.nanmin(col)) if np.isfinite(col).any() else 0.0
                vmax = float(np.nanmax(col)) if np.isfinite(col).any() else 1.0
            except Exception:
                vmin, vmax = 0.0, 1.0
            if vmin >= vmax:
                vmin, vmax = 0.0, 1.0
            self.s_min_spin.blockSignals(True)
            self.s_max_spin.blockSignals(True)
            self.s_min_spin.setValue(vmin)
            self.s_max_spin.setValue(vmax)
            self.s_min_spin.blockSignals(False)
            self.s_max_spin.blockSignals(False)
        self._enable_s_controls(self._have_S)

        if self._col_size is not None:
            try:
                col = pd.to_numeric(df[self._col_size], errors="coerce")
                vmin = int(np.nanmin(col)) if np.isfinite(col).any() else 0
                vmax = int(np.nanmax(col)) if np.isfinite(col).any() else 0
            except Exception:
                vmin, vmax = 0, 0
            if vmin < 0:
                vmin = 0
            if vmax < vmin:
                vmax = vmin
            self.size_min_spin.blockSignals(True)
            self.size_max_spin.blockSignals(True)
            self.size_min_spin.setValue(vmin)
            self.size_max_spin.setValue(vmax)
            self.size_min_spin.blockSignals(False)
            self.size_max_spin.blockSignals(False)
        self._enable_size_controls(self._col_size is not None)

        self._populate_hist_columns()

    def _populate_hist_columns(self) -> None:
        self.hist_column_combo.blockSignals(True)
        self.hist_column_combo.clear()
        if self._df is None:
            self.hist_column_combo.blockSignals(False)
            return

        # Always propose global columns first
        preferred: list[str] = []
        for col in ("E", "S", "Number of Photons"):
            if col in self._df.columns and col not in preferred:
                preferred.append(col)

        # If a specific detector is selected, restrict detector-specific
        # columns to that detector (e.g. "Number of Photons (red)",
        # "S prompt red (kHz) | ...", "red Count Rate (KHz)").
        sel_det = None
        try:
            # currentData may be None for "All"
            sel_det = self.detector_combo.currentData() if hasattr(self, "detector_combo") else None
        except Exception:
            sel_det = None

        for c in map(str, self._df.columns):
            if c in preferred:
                continue

            # Detector-specific selection
            if sel_det:
                det = str(sel_det)
                token = f"({det})"
                # Direct detector-tagged columns
                if token in c:
                    preferred.append(c)
                    continue
                # Window columns like "S prompt red (kHz) | ..."
                if f" {det} (" in c or f" {det})" in c:
                    preferred.append(c)
                    continue
                # Count-rate style with detector name somewhere
                if "Count Rate" in c and det.lower() in c.lower():
                    preferred.append(c)
                    continue
            else:
                # No specific detector selected: fall back to old behavior
                if "Count Rate" in c or "Number of Photons (" in c:
                    preferred.append(c)

        if not preferred:
            preferred = list(map(str, self._df.columns))

        for c in preferred:
            self.hist_column_combo.addItem(str(c), str(c))

        self.hist_column_combo.blockSignals(False)
        if self.hist_column_combo.count() > 0:
            self.hist_column_combo.setCurrentIndex(0)

    # --- Gating & histogram -------------------------------------------
    def _update_gating(self) -> None:
        if self._df is None:
            self._mask = None
            self._model.set_mask(np.zeros(0, dtype=bool))
            self._update_status()
            self.update_histogram()
            return

        mask = np.ones(len(self._df), dtype=bool)

        if self._have_E and self._col_E is not None:
            try:
                col = pd.to_numeric(self._df[self._col_E], errors="coerce").to_numpy()
                emin = float(self.e_min_spin.value())
                emax = float(self.e_max_spin.value())
                valid = np.isfinite(col)
                cond = (col >= emin) & (col <= emax)
                mask &= valid & cond
            except Exception:
                pass

        if self._have_S and self._col_S is not None:
            try:
                col = pd.to_numeric(self._df[self._col_S], errors="coerce").to_numpy()
                smin = float(self.s_min_spin.value())
                smax = float(self.s_max_spin.value())
                valid = np.isfinite(col)
                cond = (col >= smin) & (col <= smax)
                mask &= valid & cond
            except Exception:
                pass

        if self._col_size is not None:
            try:
                col = pd.to_numeric(self._df[self._col_size], errors="coerce").to_numpy()
                nmin = int(self.size_min_spin.value())
                nmax = int(self.size_max_spin.value())
                valid = np.isfinite(col)
                cond = (col >= nmin) & (col <= nmax)
                mask &= valid & cond
            except Exception:
                pass

        self._mask = mask
        self._model.set_mask(mask)
        self._update_status()
        self.update_histogram()

    def _update_status(self) -> None:
        if self._df is None or self._mask is None:
            self.status_label.setText("No data loaded")
            return
        total = int(len(self._df))
        selected = int(self._mask.sum())
        self.status_label.setText(f"Bursts: {selected} / {total} selected")

    def _update_detector_list(self) -> None:
        """Populate detector_combo from DataFrame columns and optional setup info."""
        if not hasattr(self, "detector_combo"):
            return
        self.detector_combo.blockSignals(True)
        self.detector_combo.clear()
        self.detector_combo.addItem("All", None)

        dets = set()
        if self._df is not None:
            for c in self._df.columns:
                s = str(c)
                if s.startswith("Number of Photons (") and ")" in s:
                    name = s.split("Number of Photons (", 1)[1].split(")", 1)[0]
                    if name:
                        dets.add(name)

        try:
            if self.setup_detectors:
                for det in self.setup_detectors.keys():
                    dets.add(str(det))
        except Exception:
            pass

        for det in sorted(dets):
            self.detector_combo.addItem(det, det)

        self.detector_combo.blockSignals(False)

    def _on_detector_changed(self, _idx: int) -> None:
        """Refresh histogram-column choices when detector selection changes."""
        self._populate_hist_columns()
        self.update_histogram()

    def _load_setup_info(self, root: Path) -> None:
        """Load experimental setup (windows/detectors) if an Info JSON is present.

        This follows the pattern used in the Burst MLE plugin: it looks for an
        `Info/photon_selection_parameters.json` file either under the selected
        folder or its parent directory and, if present, exposes "windows" and
        "detectors" entries via `setup_windows` and `setup_detectors`.
        """
        self.setup_info = None
        self.setup_windows = None
        self.setup_detectors = None

        try:
            from chisurf.core.settings.file_utils import safe_open_file
            import json
        except Exception:
            return

        candidates = [root / "Info", root.parent / "Info"]
        json_data = None
        used_dir: Path | None = None
        for info_dir in candidates:
            json_file = info_dir / "photon_selection_parameters.json"
            if json_file.exists():
                try:
                    json_data = safe_open_file(
                        json_file,
                        processor=json.load,
                        default_value=None,
                        error_message=f"Could not read setup information from {json_file}",
                    )
                except Exception as exc:
                    logging.warning(f"BurstBrowser: failed to read setup info from {json_file}: {exc}")
                    json_data = None
                used_dir = info_dir
                break

        if not json_data:
            return

        setup = json_data.get("setup_info") or {}
        if not isinstance(setup, dict):
            return

        self.setup_info = setup
        self.setup_windows = setup.get("windows", {}) or {}
        self.setup_detectors = setup.get("detectors", {}) or {}

        logging.info("BurstBrowser: loaded experimental setup from %s", str(used_dir))

        # New detectors may influence detector selection
        self._update_detector_list()

    def update_histogram(self) -> None:
        self.hist_plot.clear()
        if self._df is None or self._mask is None:
            return
        col_name = self.hist_column_combo.currentData()
        if not col_name or col_name not in self._df.columns:
            return

        # Decide whether to use only selected bursts or all gated bursts
        use_selection = getattr(self, "hist_selection_check", None) is not None and self.hist_selection_check.isChecked()
        indexer = None
        if use_selection and self.table_view.selectionModel() is not None:
            selected_rows = self.table_view.selectionModel().selectedRows()
            if not selected_rows:
                return
            rows: list[int] = []
            for idx in selected_rows:
                try:
                    view_row = int(idx.row())
                    if self._model._rows is not None:
                        base_row = int(self._model._rows[view_row])
                    else:
                        base_row = view_row
                except Exception:
                    base_row = None
                if base_row is not None:
                    rows.append(base_row)
            if not rows:
                return
            indexer = rows

        try:
            if indexer is not None:
                series = self._df.loc[indexer, col_name]
            else:
                series = self._df.loc[self._mask, col_name]
            data = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        except Exception:
            return
        if data.size == 0:
            return

        try:
            counts, edges = np.histogram(data, bins=60)
        except Exception:
            return
        if counts.size == 0:
            return
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        bg = pg.BarGraphItem(x=centers, height=counts, width=width, brush="b", pen="k")
        self.hist_plot.addItem(bg)
        self.hist_plot.setLabel("bottom", str(col_name))


__all__ = ["BurstBrowserWidget", "BurstTableModel", "name"]


# ----------------------------
# Default plugin entry style used by ChiSurf
# ----------------------------
if __name__ == "plugin":  # pragma: no cover - used by the plugin host
    widget = BurstBrowserWidget()
    widget.show()
