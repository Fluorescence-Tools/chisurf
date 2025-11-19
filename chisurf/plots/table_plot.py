from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

from qtpy import QtWidgets, QtCore, QtGui

import chisurf.fitting
from chisurf.plots import plotbase


class NoBackgroundProxy(QtCore.QIdentityProxyModel):
    """Proxy model that removes any background color roles.
    This neutralizes background coloring coming from the underlying model (e.g., guidata's DataFrameModel).
    """
    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if role in (QtCore.Qt.BackgroundRole, QtCore.Qt.BackgroundColorRole):
            return None
        return super().data(index, role)

class ReadOnlyColumnProxy(QtCore.QIdentityProxyModel):
    """Proxy model that disables editing for specified column labels.
    Looks up columns by their horizontal header text and removes ItemIsEditable flag.
    Can be stacked with other proxies (e.g., NoBackgroundProxy).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._readonly_headers = set()

    def setReadOnlyHeaders(self, headers):
        self._readonly_headers = set(headers or [])

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        f = super().flags(index)
        try:
            header = self.headerData(index.column(), QtCore.Qt.Horizontal)
            if isinstance(header, str) and header in self._readonly_headers:
                f &= ~QtCore.Qt.ItemIsEditable
        except Exception:
            pass
        return f

try:
    # Used for the model parameter table dialog
    from guidata.widgets.dataframeeditor import DataFrameEditor
except Exception:  # pragma: no cover - optional dependency
    DataFrameEditor = None  # type: ignore


class BooleanToggleDelegate(QtWidgets.QStyledItemDelegate):
    def _is_checked(self, value) -> bool:
        try:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                return bool(int(value))
            if isinstance(value, str):
                v = value.strip().lower()
                return v in ('1', 'true', 't', 'yes', 'y', 'on')
        except Exception:
            pass
        return False

    def _toggle(self, value) -> bool:
        return not self._is_checked(value)

    def _checkbox_rect(self, option: QtWidgets.QStyleOptionViewItem) -> QtCore.QRect:
        rect = QtCore.QRect(option.rect)
        size = 16
        x = rect.x() + (rect.width() - size) // 2
        y = rect.y() + (rect.height() - size) // 2
        return QtCore.QRect(x, y, size, size)

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        checked = self._is_checked(index.data(QtCore.Qt.DisplayRole))
        style = QtWidgets.QApplication.style() if QtWidgets.QApplication.instance() else option.widget.style()
        cb_opt = QtWidgets.QStyleOptionButton()
        cb_opt.state = QtWidgets.QStyle.State_Enabled | (QtWidgets.QStyle.State_On if checked else QtWidgets.QStyle.State_Off)
        cb_opt.rect = self._checkbox_rect(option)
        style.drawControl(QtWidgets.QStyle.CE_CheckBox, cb_opt, painter)

    def createEditor(self, parent, option, index):
        # No inline editor; we toggle directly via editorEvent
        return None

    def editorEvent(self, event: QtCore.QEvent, model: QtCore.QAbstractItemModel, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> bool:
        et = event.type()
        if et in (QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseButtonDblClick):
            new_val = self._toggle(index.data(QtCore.Qt.DisplayRole))
            str_val = 'True' if new_val else 'False'
            return model.setData(index, str_val, QtCore.Qt.EditRole)
        if et == QtCore.QEvent.KeyPress:
            if isinstance(event, QtGui.QKeyEvent) and event.key() in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                new_val = self._toggle(index.data(QtCore.Qt.DisplayRole))
                str_val = 'True' if new_val else 'False'
                return model.setData(index, str_val, QtCore.Qt.EditRole)
        return False

class _FitTableModel(QtCore.QAbstractTableModel):
    """Lightweight table model exposing x, data, model, and residuals.

    Columns:
      0: x (editable)
      1: data (editable)
      2: model (read-only)
      3: weighted residuals (read-only)
    """

    HEADERS = ["x", "data", "model", "w. res."]

    def __init__(self, parent_plot: "FitTablePlot"):
        super().__init__(parent_plot)
        self._plot = parent_plot
        self._x = np.array([], dtype=float)
        self._y = np.array([], dtype=float)
        self._ym = np.array([], dtype=float)
        self._wres = np.array([], dtype=float)

    # ---- Required model API ----
    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else self._x.size

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else 4

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            try:
                return self.HEADERS[section]
            except Exception:
                return None
        return super().headerData(section, orientation, role)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        if row < 0 or row >= self._x.size:
            return None

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            try:
                if col == 0:
                    v = self._x[row]
                elif col == 1:
                    v = self._y[row]
                elif col == 2:
                    v = self._ym[row]
                elif col == 3:
                    v = self._wres[row]
                else:
                    return None
            except Exception:
                return None

            if role == QtCore.Qt.EditRole:
                return repr(float(v))
            try:
                if np.isfinite(v):
                    return f"{float(v):.6g}"
                return "nan"
            except Exception:
                return str(v)

        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        base = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if index.column() in (0, 1):
            base |= QtCore.Qt.ItemIsEditable
        return base

    def setData(self, index: QtCore.QModelIndex, value, role: int = QtCore.Qt.EditRole) -> bool:
        if role != QtCore.Qt.EditRole or not index.isValid():
            return False
        row = index.row()
        col = index.column()
        if row < 0 or row >= self._x.size or col not in (0, 1):
            return False
        try:
            v = float(str(value))
        except Exception:
            return False

        if col == 0:
            self._x[row] = v
        else:
            self._y[row] = v

        # Backpropagate to fit and recompute model/residuals
        self._plot._set_arrays(self._x, self._y)
        # Refresh arrays from updated fit
        self._plot._refresh_arrays_into_model()

        # Emit dataChanged for whole row (all columns) for simplicity
        left = self.index(row, 0)
        right = self.index(row, 3)
        self.dataChanged.emit(left, right, [QtCore.Qt.DisplayRole])
        return True

    # ---- Helpers called by parent plot ----
    def set_arrays(self, x: np.ndarray, y: np.ndarray, ym: np.ndarray, wres: np.ndarray) -> None:
        self.beginResetModel()
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._ym = np.asarray(ym, dtype=float)
        self._wres = np.asarray(wres, dtype=float)
        self.endResetModel()


class FitTablePlot(plotbase.Plot):
    """Data table view for a Fit, implemented with QTableView + model.

    Columns:
      - x (independent variable, editable)
      - data (y, editable)
      - model (y_model, read-only)
      - weighted residuals (wres, read-only)
    """

    name = "Data table"

    def __init__(
        self,
        fit: chisurf.fitting.fit.Fit,
        parent: Optional[QtWidgets.QWidget] = None,
        **kwargs,
    ):
        super().__init__(fit, parent=parent, **kwargs)

        # Top control bar
        controls = QtWidgets.QWidget(self)
        h = QtWidgets.QHBoxLayout(controls)
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(6)

        self.btn_show_model = QtWidgets.QPushButton("Show model", controls)
        self.btn_show_model.setToolTip("Open a table editor for model parameters")
        self.btn_show_model.clicked.connect(self.on_show_model)

        self.btn_copy = QtWidgets.QPushButton("Copy", controls)
        self.btn_copy.setToolTip("Copy Data table (x, data, model, w. res.) to clipboard")
        self.btn_copy.clicked.connect(self.on_copy_table_to_clipboard)

        self.lbl_info = QtWidgets.QLabel("", controls)
        self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        h.addWidget(self.btn_show_model)
        h.addWidget(self.btn_copy)
        h.addStretch(1)
        h.addWidget(self.lbl_info)

        # Main table view + model
        self.table = QtWidgets.QTableView(self)
        self._model = _FitTableModel(self)
        self.table.setModel(self._model)

        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        try:
            hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        except Exception:
            try:
                hh.setResizeMode(QtWidgets.QHeaderView.ResizeToContents)  # Qt4 fallback
            except Exception:
                pass
        hh.setMinimumSectionSize(20)

        self.table.setAlternatingRowColors(False)
        self.table.setWordWrap(False)
        self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

        # Compact font for table only
        try:
            f = self.table.font()
            f.setPointSize(max(7, f.pointSize() - 1))
            f.setStyleStrategy(QtGui.QFont.PreferAntialias)
            self.table.setFont(f)
            self.table.verticalHeader().setDefaultSectionSize(max(16, self.table.fontMetrics().height() + 6))
        except Exception:
            pass

        self.layout.addWidget(controls)
        self.layout.addWidget(self.table)

        # Populate initial arrays
        self._refresh_arrays_into_model()

    # ---- Utilities ----
    def _get_arrays(self):
        """Return arrays aligned to the data length.
        x, data are the full data arrays.
        model and wres are truncated or padded with NaN to match data length.
        """
        fit = self.fit
        data_curve = fit.data
        model_curve = fit.model
        wres_curve = fit.weighted_residuals

        x = np.asarray(data_curve.x, dtype=float)
        y = np.asarray(data_curve.y, dtype=float)
        nd = min(len(x), len(y))
        if nd == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        x = x[:nd].copy()
        y = y[:nd].copy()

        # Model and residuals may have different lengths – align to data length
        ym_raw = np.asarray(model_curve.y, dtype=float)
        wres_raw = np.asarray(wres_curve.y, dtype=float)

        def align(arr: np.ndarray, n: int) -> np.ndarray:
            if arr is None:
                return np.full(n, np.nan, dtype=float)
            m = len(arr)
            if m >= n:
                return arr[:n].astype(float, copy=True)
            out = np.empty(n, dtype=float)
            out[:m] = arr.astype(float, copy=False)
            out[m:] = np.nan
            return out

        ym = align(ym_raw, nd)
        # Embed weighted residuals into the full data length based on fit range
        try:
            xmin, xmax = self.fit.fit_range
        except Exception:
            xmin, xmax = 0, nd - 1
        if nd <= 0:
            return x, y, ym, np.array([])
        # Clip and normalize indices
        xmin = int(np.clip(xmin, 0, nd - 1))
        xmax = int(np.clip(xmax, 0, nd - 1))
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        # Initialize with NaNs and fill inside fit range from residuals
        wres = np.full(nd, np.nan, dtype=float)
        try:
            seg_len = min(wres_raw.size, xmax - xmin + 1, nd - xmin)
            if seg_len > 0:
                wres[xmin:xmin + seg_len] = wres_raw[:seg_len].astype(float, copy=False)
        except Exception:
            # If anything goes wrong, fall back to simple alignment
            wres = align(wres_raw, nd)
        return x, y, ym, wres

    def _set_arrays(self, x: np.ndarray, y: np.ndarray) -> None:
        """Write back x, y to fit.data and trigger recompute."""
        data_curve = self.fit.data
        # Preserve ex/ey lengths or regenerate default if absent
        ex = getattr(data_curve, 'ex', None)
        ey = getattr(data_curve, 'ey', None)
        if ex is not None and len(ex) == len(x):
            pass
        else:
            ex = np.ones_like(x)
        if ey is not None and len(ey) == len(y):
            pass
        else:
            ey = np.ones_like(y)
        data_curve.set_data(x=x, y=y, ex=ex, ey=ey)
        # Recompute model and residuals
        try:
            self.fit.update()
        except Exception:
            # Fallback: attempt to call model.update directly
            try:
                self.fit.model.update()
            except Exception:
                pass

    def _refresh_arrays_into_model(self) -> None:
        x, y, ym, wres = self._get_arrays()
        self._model.set_arrays(x, y, ym, wres)
        n = x.size
        self.lbl_info.setText(f"N={n}  |  chi2r={getattr(self.fit, 'chi2r', float('nan')):.4g}")

    def on_copy_table_to_clipboard(self) -> None:
        """Copy the current Data table (x, data, model, w. res.) to the clipboard.

        Data is exported as tab-separated text with a single header row.
        """
        try:
            model = self._model
            n_rows = model.rowCount()
            n_cols = model.columnCount()
            if n_rows <= 0 or n_cols <= 0:
                return

            # Header
            header_cells = []
            for c in range(n_cols):
                h = model.headerData(c, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
                header_cells.append(str(h) if h is not None else "")
            lines = ["\t".join(header_cells)]

            # Rows
            for r in range(n_rows):
                row_cells = []
                for c in range(n_cols):
                    idx = model.index(r, c)
                    v = model.data(idx, QtCore.Qt.DisplayRole)
                    row_cells.append(str(v) if v is not None else "")
                lines.append("\t".join(row_cells))

            text = "\n".join(lines)
            cb = QtWidgets.QApplication.clipboard()
            cb.setText(text)
        except Exception:
            pass

    def on_show_model(self):
        if DataFrameEditor is None:
            QtWidgets.QMessageBox.warning(self, "DataFrameEditor unavailable", "guidata DataFrameEditor is not installed.")
            return
        # Build a comprehensive parameter table including link info
        model = self.fit.model
        # Prefer all parameters, including nested groups
        try:
            param_dict = model.parameters_all_dict
        except Exception:
            # Fallback to direct parameters list
            param_dict = {p.name: p for p in getattr(model, 'parameters', [])}

        rows = []
        for name, p in param_dict.items():
            try:
                value = float(p.value)
            except Exception:
                value = np.nan
            lb, ub = p.bounds
            fixed = bool(p.fixed)
            bounded = bool(getattr(p, 'bounds_on', False))
            linked = bool(getattr(p, 'is_linked', False))
            # Determine link target name if any
            link_target_name = None
            try:
                link_obj = getattr(p, 'link', None)
                if link_obj is not None:
                    # p.link returns a Parameter or None
                    link_target_name = getattr(link_obj, 'name', None)
            except Exception:
                link_target_name = None
            # Safely convert bounds, allowing None -> NaN
            try:
                lb_val = float(lb) if lb is not None else np.nan
            except Exception:
                lb_val = np.nan
            try:
                ub_val = float(ub) if ub is not None else np.nan
            except Exception:
                ub_val = np.nan
            rows.append({
                'name': name,
                'value': value,
                'lb': lb_val,
                'ub': ub_val,
                'fixed': fixed,
                'bounds_on': bounded,
                'linked': linked,
                'link_target': link_target_name if link_target_name is not None else ''
            })

        df = pd.DataFrame(rows).reset_index(drop=True)
        if df.empty:
            QtWidgets.QMessageBox.information(self, "No parameters", "Model exposes no editable parameters.")
            return

        # Ensure boolean columns are real booleans (no NaN) for proper checkbox behavior
        for _col in ("fixed", "bounds_on", "linked"):
            if _col in df.columns:
                try:
                    df[_col] = df[_col].apply(lambda v: bool(v) if pd.notna(v) else False)
                except Exception:
                    pass

        dlg = DataFrameEditor(self)
        if not dlg.setup_and_check(df, title="Model parameters"):
            return
        # Customize the editor: hide index/row headers and adjust size; install boolean toggle delegates
        try:
            # Hide any row headers (index) on contained table views
            views = dlg.findChildren(QtWidgets.QTableView)
            for v in views:
                try:
                    v.verticalHeader().setVisible(False)
                except Exception:
                    pass
                # Disable alternating row colors and remove background coloring via proxy
                try:
                    v.setAlternatingRowColors(False)
                    orig_model = v.model()
                    # Stack proxies: NoBackgroundProxy -> ReadOnlyColumnProxy
                    if orig_model is not None:
                        nb_source = orig_model
                        if not isinstance(orig_model, NoBackgroundProxy):
                            nb = NoBackgroundProxy(v)
                            nb.setSourceModel(orig_model)
                            nb_source = nb
                        ro = ReadOnlyColumnProxy(v)
                        ro.setSourceModel(nb_source)
                        ro.setReadOnlyHeaders(["name"])  # make 'name' column read-only
                        v.setModel(ro)
                except Exception:
                    pass
                # Also hide a first column named like an index, if present
                try:
                    header0 = v.model().headerData(0, QtCore.Qt.Horizontal)
                    if isinstance(header0, str) and header0.strip().lower() in ("index", "#", ""):
                        v.setColumnHidden(0, True)
                except Exception:
                    pass
                # Install checkbox toggle delegate on boolean columns
                try:
                    model = v.model()
                    if model is not None:
                        ncols = model.columnCount()
                        for ci in range(ncols):
                            header = model.headerData(ci, QtCore.Qt.Horizontal)
                            if isinstance(header, str) and header in ("fixed", "bounds_on", "linked"):
                                v.setItemDelegateForColumn(ci, BooleanToggleDelegate(v))
                except Exception:
                    pass
        except Exception:
            pass
        # Adjust size based on row count, with sane defaults
        try:
            nrows = max(1, len(df))
            height = min(900, 140 + nrows * 28)
            dlg.resize(900, height)
        except Exception:
            try:
                dlg.resize(900, 600)
            except Exception:
                pass
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_df = dlg.get_value()
            # Backpropagate edits to parameters
            pmap = param_dict  # already a name->parameter dict
            
            def _parse_bool(v) -> bool:
                try:
                    if isinstance(v, (bool, np.bool_)):
                        return bool(v)
                    if isinstance(v, (int, np.integer)):
                        return int(v) != 0
                    if isinstance(v, (float, np.floating)):
                        return float(v) != 0.0
                    if isinstance(v, str):
                        s = v.strip().lower()
                        return s in ('1', 'true', 't', 'yes', 'y', 'on')
                except Exception:
                    pass
                return False
            
            for _, row in new_df.iterrows():
                name = row.get('name')
                if name not in pmap:
                    continue
                p = pmap[name]

                # Update value
                try:
                    if pd.notna(row.get('value')):
                        p.value = float(row['value'])
                except Exception:
                    pass

                # Update bounds
                try:
                    if pd.notna(row.get('lb')) and pd.notna(row.get('ub')):
                        p.bounds = (float(row['lb']), float(row['ub']))
                except Exception:
                    pass

                # Update fixed
                try:
                    if 'fixed' in row:
                        p.fixed = _parse_bool(row['fixed'])
                except Exception:
                    pass

                # Update bounds_on
                try:
                    if 'bounds_on' in row:
                        p.bounds_on = _parse_bool(row['bounds_on'])
                except Exception:
                    pass

                # Update linking
                try:
                    want_linked = _parse_bool(row.get('linked'))
                except Exception:
                    want_linked = False
                target_name = str(row.get('link_target')) if row.get('link_target') is not None else ''
                target_name = target_name.strip()
                try:
                    if want_linked and target_name and target_name in pmap and target_name != name:
                        target_param = pmap[target_name]
                        try:
                            p.link = target_param
                        except Exception:
                            # If invalid (e.g., recursive), ignore
                            pass
                    else:
                        # Explicitly unlink if not wanted or invalid
                        try:
                            p.link = None
                        except Exception:
                            pass
                except Exception:
                    pass

            # Recompute model and refresh
            try:
                self.fit.update()
            except Exception:
                try:
                    self.fit.model.update()
                except Exception:
                    pass
            # Update parameter control widgets so changes reflect in the UI
            try:
                self.fit.model.finalize()
            except Exception:
                pass
            self._refresh_arrays_into_model()

    # ---- Plot API ----
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._refresh_arrays_into_model()
