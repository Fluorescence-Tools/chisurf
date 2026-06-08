from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui

import chisurf as cs
from chisurf import logging
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.gui.plots.table_plot import BooleanToggleDelegate


# ── constants / column enumeration ────────────────────────────────────

COL_FIT = 0
COL_LOCAL_FIT = 1
COL_PARAM = 2
COL_VALUE = 3
COL_FIXED = 4
COL_BOUNDS_LO = 5
COL_BOUNDS_HI = 6
COL_BOUNDS_ON = 7
COL_ERROR = 8
COL_LINKED = 9
COL_DESCRIPTION = 10

HEADERS = [
    "Fit",
    "Local fit",
    "Parameter",
    "Value",
    "Fixed",
    "Bounds lo",
    "Bounds hi",
    "Bounds on",
    "Error",
    "Linked to",
    "Description",
]


# TODO: needs docstring
def _fit_name(fit: Fit) -> str:
    return getattr(fit, "name", f"Fit_{id(fit)}")

# TODO: needs docstring

def _local_fit_name(fit: Fit, idx: Optional[int] = None) -> str:
    if isinstance(fit, FitGroup):
        base = _fit_name(fit)
        return f"{base}[{idx}]" if idx is not None else base
    return ""
# TODO: needs docstring


def _link_summary(p: FittingParameter) -> str:
    if not p.is_linked:
        return ""
    link = getattr(p, "link", None)
    if link is None:
        return ""
    lname = getattr(link, "name", "")
    parent = getattr(link, "parent_fit", None) or getattr(
        getattr(link, "controller", None), "fit", None
    )
    if parent is not None:
        pname = getattr(parent, "name", "")
        return f"{pname}: {lname}"
    return lname


# ── helpers to iterate rows ───────────────────────────────────────────

RowTuple = Tuple[int, Fit, Optional[int], FittingParameter]


def _get_rows(
    fit_list: Optional[List[Fit]] = None,
    skip_global_fit: bool = True,
) -> List[RowTuple]:
    """Return a flat list of (fit_index, fit_or_local_fit, local_idx, param).

    Parameters are sourced from ``model.parameters_all``, which is
    already UID-deduped at the source by ``find_parameters``.
    """
    if fit_list is None:
        fit_list = list(getattr(cs, "fits", []) or [])

    from chisurf.core.models.global_model import GlobalFitModel

    rows: List[RowTuple] = []
    for fi, fit in enumerate(fit_list):
        if skip_global_fit and isinstance(getattr(fit, "model", None), GlobalFitModel):
            continue

        if isinstance(fit, FitGroup) and getattr(fit, "grouped_fits", None):
            for li, local_fit in enumerate(fit.grouped_fits):
                params = getattr(
                    getattr(local_fit, "model", None), "parameters_all", []
                ) or []
                for p in params:
                    rows.append((fi, local_fit, li, p))
        else:
            params = getattr(getattr(fit, "model", None), "parameters_all", []) or []
            for p in params:
                rows.append((fi, fit, None, p))

    return rows


# ── the model ─────────────────────────────────────────────────────────

class ParameterTableModel(QtCore.QAbstractTableModel):
    """Initialize the widget."""
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        # TODO: needs docstring
        super().__init__(parent)
        self._rows: List[RowTuple] = []
        self._fit_list: List[Fit] = []

    def refresh(
        self,
        fit_list: Optional[List[Fit]] = None,
        skip_global_fit: bool = True,
    ):
        self.beginResetModel()
        if fit_list is not None:
            self._fit_list = fit_list
        source = self._fit_list if self._fit_list else None
        self._rows = _get_rows(source, skip_global_fit=skip_global_fit)
        if not self._fit_list:
# TODO: needs docstring
            self._fit_list = [r[1] for r in self._rows]
        self.endResetModel()
# TODO: needs docstring

    # ── row/column  ───────────────────────────────────────────────────
# TODO: needs docstring

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return len(self._rows) if not parent.isValid() else 0

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return len(HEADERS) if not parent.isValid() else 0

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
# TODO: needs docstring
        role: int = QtCore.Qt.DisplayRole,
    ):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return HEADERS[section]
        return None

    # ── data ──────────────────────────────────────────────────────────

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        fit_idx, fit, local_idx, param = self._rows[index.row()]
        col = index.column()

        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return self._display_data(col, fit_idx, fit, local_idx, param)

        if role == QtCore.Qt.ToolTipRole:
            if col == COL_DESCRIPTION:
                d = getattr(param, "description", "") or ""
                return d if d else None
            return self._display_data(col, fit_idx, fit, local_idx, param)

        if role == QtCore.Qt.BackgroundRole:
            return self._bg_color(param)
# TODO: needs docstring

        if role == QtCore.Qt.TextAlignmentRole:
            if col in (COL_VALUE, COL_BOUNDS_LO, COL_BOUNDS_HI, COL_ERROR):
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if col in (COL_FIXED, COL_BOUNDS_ON):
                return int(QtCore.Qt.AlignCenter)

        return None

    def _display_data(self, col, fit_idx, fit, local_idx, param):
        if col == COL_FIT:
            return _fit_name(fit)
        if col == COL_LOCAL_FIT:
            return _local_fit_name(fit, local_idx)
        if col == COL_PARAM:
            return getattr(param, "name", "")
        if col == COL_VALUE:
            v = getattr(param, "value", None)
            return f"{v:.6g}" if v is not None else ""
        if col == COL_FIXED:
            return bool(getattr(param, "fixed", False))
        if col == COL_BOUNDS_LO:
            b = getattr(param, "bounds", None)
            if b is not None and len(b) > 0 and b[0] is not None:
                return f"{b[0]:.6g}"
            return ""
        if col == COL_BOUNDS_HI:
            b = getattr(param, "bounds", None)
            if b is not None and len(b) > 1 and b[1] is not None:
                return f"{b[1]:.6g}"
            return ""
        if col == COL_BOUNDS_ON:
            return bool(getattr(param, "bounds_on", False))
        if col == COL_ERROR:
# TODO: needs docstring
            e = getattr(param, "error_estimate", None)
            if e is not None and np.isfinite(e):
                return f"{e:.4g}"
            return ""
        if col == COL_LINKED:
            return _link_summary(param)
        if col == COL_DESCRIPTION:
            return getattr(param, "description", "") or ""
        return ""

    def _bg_color(self, param: FittingParameter):
        ps = getattr(cs.core.settings, "parameter", {}) or {}
        if getattr(param, "is_output", False):
            c = ps.get("role_color_output", "")
        elif getattr(param, "is_linked", False) and not getattr(
            param, "is_link_master", False
        ):
            c = ps.get("role_color_linked", "")
        else:
            c = ps.get("role_color_input", "")
# TODO: needs docstring
        if c:
            try:
                col = QtGui.QColor(c)
                if col.isValid():
                    return col
            except Exception:
                pass
        return None

    # ── flags / editing ───────────────────────────────────────────────

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        col = index.column()
        param = self._rows[index.row()][3]
        is_follower = getattr(param, "is_linked", False) and not getattr(
            param, "is_link_master", False
        )

        if col in (COL_VALUE, COL_FIXED, COL_BOUNDS_ON):
# TODO: needs docstring
            if is_follower and col == COL_VALUE:
                pass  # not editable
            else:
                f |= QtCore.Qt.ItemIsEditable

        if col in (COL_BOUNDS_LO, COL_BOUNDS_HI):
            bounds_on = getattr(param, "bounds_on", False)
            if bounds_on and not is_follower:
                f |= QtCore.Qt.ItemIsEditable

        return f

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: Any,
        role: int = QtCore.Qt.EditRole,
    ) -> bool:
        if role != QtCore.Qt.EditRole:
            return False
        if not index.isValid():
            return False

        _, fit, local_idx, param = self._rows[index.row()]
        col = index.column()

        if col == COL_VALUE:
            try:
                param.value = float(value)
            except (TypeError, ValueError):
                return False
            self._finalize(fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_FIXED:
            try:
                param.fixed = bool(value) if not isinstance(value, str) else value.lower() in ("true", "1", "yes")
            except Exception:
                return False
            param.fixed = bool(value)
            self._finalize(fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_BOUNDS_ON:
            param.bounds_on = bool(value)
            self._finalize(fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_BOUNDS_LO:
            try:
                old = list(getattr(param, "bounds", (0.0, 0.0)))
                old[0] = float(value)
                param.bounds = tuple(old)
            except Exception:
                return False
            self._finalize(fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_BOUNDS_HI:
# TODO: needs docstring
            try:
                old = list(getattr(param, "bounds", (0.0, 0.0)))
                old[1] = float(value)
                param.bounds = tuple(old)
            except Exception:
                return False
            self._finalize(fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        return False

    @staticmethod
    def _finalize(fit: Fit, local_idx: Optional[int], param: FittingParameter):
        # TODO: needs docstring
        try:
            # TODO: needs docstring
            m = getattr(fit, "model", None)
            if m is not None and hasattr(m, "finalize"):
                m.finalize()
            if hasattr(fit, "update"):
                fit.update()
        except Exception:
            pass

    # ── helpers for the view ──────────────────────────────────────────

    def fit_at_row(self, row: int) -> Fit:
        # TODO: needs docstring
        return self._rows[row][1] if 0 <= row < len(self._rows) else None

    def param_at_row(self, row: int) -> FittingParameter:
        return self._rows[row][3] if 0 <= row < len(self._rows) else None

    def rows_from_params(
        self, params: List[FittingParameter]
    ) -> List[int]:
        uids = {str(getattr(p, "unique_identifier", "")) for p in params}
        return [
            i for i, (_, _, _, p) in enumerate(self._rows)
            if str(getattr(p, "unique_identifier", "")) in uids
        ]

    def get_all_fits(self) -> List[Fit]:
        seen = set()
        result = []
        for _, fit, local_idx, _ in self._rows:
            obj = fit
            if obj is not None:
                uid = str(getattr(obj, "unique_identifier", ""))
                if uid not in seen:
                    seen.add(uid)
                    result.append(obj)
        return result
