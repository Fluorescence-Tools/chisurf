from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

from qtpy import QtWidgets, QtCore

import chisurf.fitting
from chisurf.plots import plotbase

try:
    # Used for the model parameter table dialog
    from guidata.widgets.dataframeeditor import DataFrameEditor
except Exception:  # pragma: no cover - optional dependency
    DataFrameEditor = None  # type: ignore


class FitTablePlot(plotbase.Plot):
    """
    Plot widget that shows a live-editable table of:
      - x (independent variable)
      - data (y)
      - model (y_model)
      - weighted residuals (wres)

    Editing x or data cells will backpropagate to the Fit's DataCurve and
    trigger a model recomputation and table refresh. A "Show model" button
    opens a DataFrameEditor dialog with model parameters allowing edits that
    are backpropagated to the Fit.
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

        self.lbl_info = QtWidgets.QLabel("", controls)
        self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        h.addWidget(self.btn_show_model)
        h.addStretch(1)
        h.addWidget(self.lbl_info)

        # Main table
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["x", "data", "model", "weighted residuals"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)

        # Wire item changed to backpropagate edits
        self.table.itemChanged.connect(self.on_item_changed)

        self.layout.addWidget(controls)
        self.layout.addWidget(self.table)

        # Populate
        self._block_item_changed = False
        self.update()

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

    def _format_float(self, v: float) -> str:
        try:
            if np.isfinite(v):
                return f"{v:.6g}"
            return "nan"
        except Exception:
            return str(v)

    # ---- GUI population ----
    def _rebuild_table(self):
        x, y, ym, wres = self._get_arrays()
        n = len(x)
        self._block_item_changed = True
        try:
            self.table.clearContents()
            self.table.setRowCount(n)
            for i in range(n):
                # x (editable)
                itx = QtWidgets.QTableWidgetItem(self._format_float(x[i]))
                itx.setFlags(itx.flags() | QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, 0, itx)

                # data (editable)
                ity = QtWidgets.QTableWidgetItem(self._format_float(y[i]))
                ity.setFlags(ity.flags() | QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, 1, ity)

                # model (read-only)
                itm = QtWidgets.QTableWidgetItem(self._format_float(ym[i]))
                itm.setFlags(itm.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, 2, itm)

                # weighted residuals (read-only)
                itw = QtWidgets.QTableWidgetItem(self._format_float(wres[i]))
                itw.setFlags(itw.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, 3, itw)
        finally:
            self._block_item_changed = False

        self.lbl_info.setText(f"N={n}  |  chi2r={getattr(self.fit, 'chi2r', float('nan')):.4g}")

    # ---- Slots ----
    def on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._block_item_changed:
            return
        row = item.row()
        col = item.column()
        # Only propagate edits for x (0) and data (1)
        if col not in (0, 1):
            return
        # Gather full arrays from table to preserve vector integrity
        n = self.table.rowCount()
        x_list: List[float] = []
        y_list: List[float] = []
        for i in range(n):
            try:
                xv = float(self.table.item(i, 0).text())
            except Exception:
                xv = np.nan
            try:
                yv = float(self.table.item(i, 1).text())
            except Exception:
                yv = np.nan
            x_list.append(xv)
            y_list.append(yv)
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        # Basic sanitization: drop NaNs by keeping previous data for them
        x0, y0, _, _ = self._get_arrays()
        if len(x0) == len(x):
            x = np.where(np.isfinite(x), x, x0)
        if len(y0) == len(y):
            y = np.where(np.isfinite(y), y, y0)

        self._set_arrays(x, y)
        self._rebuild_table()

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

        dlg = DataFrameEditor(self)
        if not dlg.setup_and_check(df, title="Model parameters"):
            return
        # Customize the editor: hide index/row headers and adjust size
        try:
            # Hide any row headers (index) on contained table views
            views = dlg.findChildren(QtWidgets.QTableView)
            for v in views:
                try:
                    v.verticalHeader().setVisible(False)
                except Exception:
                    pass
                # Also hide a first column named like an index, if present
                try:
                    header0 = v.model().headerData(0, QtCore.Qt.Horizontal)
                    if isinstance(header0, str) and header0.strip().lower() in ("index", "#", ""):
                        v.setColumnHidden(0, True)
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
                        p.fixed = bool(row['fixed'])
                except Exception:
                    pass

                # Update bounds_on
                try:
                    if 'bounds_on' in row:
                        p.bounds_on = bool(row['bounds_on'])
                except Exception:
                    pass

                # Update linking
                try:
                    want_linked = bool(row.get('linked'))
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
            self._rebuild_table()

    # ---- Plot API ----
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._rebuild_table()
