from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui

import chisurf as cs
from chisurf import logging
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.parameter import Parameter
from chisurf.gui.plots.table_plot import BooleanToggleDelegate
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client


# ── constants / column enumeration ────────────────────────────────────

COL_ROW = 0
COL_FIT = 1
COL_LOCAL_FIT = 2
COL_PARAM = 3
COL_VALUE = 4
COL_FIXED = 5
COL_BOUNDS_LO = 6
COL_BOUNDS_HI = 7
COL_BOUNDS_ON = 8
COL_ERROR = 9
COL_LINK_ROW = 10
COL_LINK_FITGROUP = COL_LINK_ROW
COL_LINK_SUBFIT = COL_LINK_ROW
COL_LINK_PARAM = COL_LINK_ROW
COL_LINK_FIT = COL_LINK_ROW
COL_LINKED = COL_LINK_ROW

HEADERS = [
    "Row",
    "Fit",
    "Local fit",
    "Parameter",
    "Value",
    "Fixed",
    "Bounds lo",
    "Bounds hi",
    "Bounds on",
    "Error",
    "Link row",
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
    # Fallback: search for the linked parameter in cs.fits to find its owner
    import chisurf as cs
    for fi, fit in enumerate(getattr(cs, "fits", [])):
        from chisurf.core.fitting.fit import FitGroup
        if isinstance(fit, FitGroup):
            for li, gf in enumerate(getattr(fit, "grouped_fits", []) or []):
                pdict = getattr(getattr(gf, "model", None), "parameters_all_dict", {}) or {}
                if lname in pdict and pdict[lname] is link:
                    return f"Fit {fi}[{li}]: {lname}"
        pdict = getattr(getattr(fit, "model", None), "parameters_all_dict", {}) or {}
        if lname in pdict and pdict[lname] is link:
            return f"Fit {fi}: {lname}"
    return lname


def _parse_index_cell(value: Any, *, allow_empty: bool = True) -> Optional[int]:
    """Parse a numeric table index cell.

    Parameters
    ----------
    value : object
        User-entered fit-group or subfit index.
    allow_empty : bool
        Whether an empty cell is accepted as ``None``.

    Returns
    -------
    int or None
        Parsed non-negative integer, or ``None`` for an empty cell.

    Raises
    ------
    ValueError
        If the text cannot be parsed as a non-negative integer.
    """
    text = str(value).strip()
    if not text:
        if allow_empty:
            return None
        raise ValueError("Index is required")
    match = re.fullmatch(r"\d+", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid index: {text}")
    return int(text)


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
        fc = get_fitting_client()
        fit_list = fc.get_fit_objects() if fc is not None else list(getattr(cs, "fits", []) or [])

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
            fc = get_fitting_client()
            self._fit_list = fc.get_fit_objects() if fc is not None else list(getattr(cs, "fits", []) or [])
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
            if col == COL_ROW:
                return str(index.row() + 1)
            return self._display_data(col, fit_idx, fit, local_idx, param)

        if role == QtCore.Qt.ToolTipRole:
            if col == COL_ROW:
                return "Table row number used by the Link row column."
            if col == COL_FIT:
                return _fit_name(fit)
            if col == COL_PARAM:
                d = getattr(param, "description", "") or ""
                return d if d else None
            if col == COL_LINK_ROW:
                return self._link_tooltip(param)
            return self._display_data(col, fit_idx, fit, local_idx, param)

        if role == QtCore.Qt.BackgroundRole:
            return self._bg_color(param)
# TODO: needs docstring

        if role == QtCore.Qt.TextAlignmentRole:
            if col in (COL_VALUE, COL_BOUNDS_LO, COL_BOUNDS_HI, COL_ERROR):
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if col in (COL_ROW, COL_FIXED, COL_BOUNDS_ON, COL_LINK_ROW):
                return int(QtCore.Qt.AlignCenter)

        return None

    def _display_data(self, col, fit_idx, fit, local_idx, param):
        if col == COL_FIT:
            return str(fit_idx)
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
        if col == COL_LINK_ROW:
            target = self._link_target_row(param)
            if target is None:
                return ""
            try:
                return str(self._rows.index(target) + 1)
            except ValueError:
                return self._link_target_label(target)
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

        if col == COL_LINK_ROW:
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

        fit_idx, fit, local_idx, param = self._rows[index.row()]
        col = index.column()
        fc = get_fitting_client()

        if col == COL_VALUE:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return False
            if fc is not None:
                fc.set_parameter_value(
                    parameter_name=str(param.name),
                    value=val,
                    fit_index=fit_idx,
                    local_idx=local_idx,
                )
            self._finalize(fc, fit_idx, fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_FIXED:
            try:
                fixed_val = bool(value) if not isinstance(value, str) else value.lower() in ("true", "1", "yes")
            except Exception:
                return False
            if fc is not None:
                fc.set_parameter_fixed(
                    parameter_name=str(param.name),
                    fixed=fixed_val,
                    fit_index=fit_idx,
                    local_idx=local_idx,
                )
            self._finalize(fc, fit_idx, fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_BOUNDS_ON:
            try:
                bounds_on_val = bool(value) if not isinstance(value, str) else value.lower() in ("true", "1", "yes")
            except Exception:
                return False
            if fc is not None:
                fc.set_parameter_bounds_on(
                    parameter_name=str(param.name),
                    bounds_on=bounds_on_val,
                    fit_index=fit_idx,
                    local_idx=local_idx,
                )
            self._finalize(fc, fit_idx, fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_BOUNDS_LO:
            try:
                old = list(getattr(param, "bounds", (0.0, 0.0)))
                old[0] = float(value)
                new_bounds = tuple(old)
            except Exception:
                return False
            if fc is not None:
                fc.set_parameter_bounds(
                    parameter_name=str(param.name),
                    bounds=new_bounds,
                    fit_index=fit_idx,
                    local_idx=local_idx,
                )
            self._finalize(fc, fit_idx, fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_BOUNDS_HI:
            try:
                old = list(getattr(param, "bounds", (0.0, 0.0)))
                old[1] = float(value)
                new_bounds = tuple(old)
            except Exception:
                return False
            if fc is not None:
                fc.set_parameter_bounds(
                    parameter_name=str(param.name),
                    bounds=new_bounds,
                    fit_index=fit_idx,
                    local_idx=local_idx,
                )
            self._finalize(fc, fit_idx, fit, local_idx, param)
            self.dataChanged.emit(index, index)
            return True

        if col == COL_LINK_ROW:
            return self._set_link_data(index, value, fc)

        return False

    def _set_link_data(self, index: QtCore.QModelIndex, value: Any, fc: Any) -> bool:
        """Apply an edit to the link-row column.

        Parameters
        ----------
        index : QModelIndex
            Edited cell index.
        value : object
            User-entered table value.
        fc : object
            Fitting client used to perform link mutations.

        Returns
        -------
        bool
            ``True`` when the edit was accepted.
        """
        fit_idx, _, local_idx, param = self._rows[index.row()]
        text = str(value).strip()

        if not text:
            return self._unlink_from_table(index, fc)

        try:
            target_row_number = _parse_index_cell(text, allow_empty=False)
        except ValueError as exc:
            logging.log(0, str(exc))
            return False
        if target_row_number is None:
            return False
        target_row_idx = target_row_number - 1
        if target_row_idx < 0 or target_row_idx >= len(self._rows):
            logging.log(0, f"Link row out of range: {target_row_number}")
            return False

        target_fit_idx, _, resolved_target_local_idx, target_param = self._rows[target_row_idx]
        target_param_name = str(getattr(target_param, "name", ""))
        if target_param is param:
            logging.log(0, f"Cannot link {param.name} to itself")
            return False
        try:
            if Parameter.check_recursive_link(target_param, param):
                logging.log(0, f"Cycle detected: cannot link {param.name} -> row {target_row_number}")
                return False
        except Exception:
            pass

        if fc is not None:
            result = fc.link_parameters(
                parameter_name=str(param.name),
                target_parameter_name=target_param_name,
                fit_index=fit_idx,
                target_fit_index=target_fit_idx,
                local_idx=local_idx,
                target_local_idx=resolved_target_local_idx,
            )
            if not result.get("ok", False):
                logging.log(0, f"Link failed: {result.get('error', 'unknown error')}")
                return False
        self._finalize(fc, fit_idx, self._rows[index.row()][1], local_idx, param)
        self.dataChanged.emit(index, index)
        return True

    def _unlink_from_table(self, index: QtCore.QModelIndex, fc: Any) -> bool:
        """Remove a parameter link from a table edit.

        Parameters
        ----------
        index : QModelIndex
            Edited cell index.
        fc : object
            Fitting client used to perform the unlink mutation.

        Returns
        -------
        bool
            ``True`` when unlinking was accepted.
        """
        fit_idx, fit, local_idx, param = self._rows[index.row()]
        if fc is not None:
            result = fc.unlink_parameter(
                parameter_name=str(param.name),
                fit_index=fit_idx,
                local_idx=local_idx,
            )
            if not result.get("ok", False):
                logging.log(0, f"Unlink failed: {result.get('error', 'unknown error')}")
                return False
        self._finalize(fc, fit_idx, fit, local_idx, param)
        self.dataChanged.emit(index, index)
        return True

    def _link_target_row(self, param: FittingParameter) -> Optional[RowTuple]:
        """Find the table row for a parameter's link target.

        Parameters
        ----------
        param : FittingParameter
            Parameter whose link target is being displayed.

        Returns
        -------
        tuple or None
            Target row tuple when the link target is present in the model.
        """
        link = getattr(param, "link", None)
        if link is None:
            return None
        for row in self._rows:
            if row[3] is link:
                return row
        return self._find_external_link_target(link)

    def _link_tooltip(self, param: FittingParameter) -> str:
        """Return a full link dependency description for the tooltip.

        Parameters
        ----------
        param : FittingParameter
            Source parameter for the current row.

        Returns
        -------
        str
            Human-readable dependency target, or entry guidance when unlinked.
        """
        target = self._link_target_row(param)
        if target is None:
            return "Enter the target table row number to link this parameter."
        try:
            row_number = self._rows.index(target) + 1
            return f"Linked to row {row_number}: {self._link_target_label(target)}"
        except ValueError:
            return f"Linked to {self._link_target_label(target)}"

    @staticmethod
    def _link_target_label(row: RowTuple) -> str:
        """Format a link target row with full fit/subfit context.

        Parameters
        ----------
        row : tuple
            Table row tuple for the link target.

        Returns
        -------
        str
            Fit group, subfit, and parameter path.
        """
        fit_idx, fit, local_idx, param = row
        param_name = str(getattr(param, "name", ""))
        fit_name = _fit_name(fit)
        if local_idx is None:
            return f"fit {fit_idx} ({fit_name}), parameter {param_name}"
        return f"fitgroup {fit_idx}, subfit {local_idx} ({fit_name}), parameter {param_name}"

    def _find_external_link_target(self, link: FittingParameter) -> Optional[RowTuple]:
        """Find a linked parameter in the global fit list.

        Parameters
        ----------
        link : FittingParameter
            Link target object.

        Returns
        -------
        tuple or None
            Synthetic row tuple for a target outside the current table rows.
        """
        lname = str(getattr(link, "name", "") or "")
        for fit_idx, fit in enumerate(getattr(cs, "fits", [])):
            if isinstance(fit, FitGroup):
                for local_idx, local_fit in enumerate(getattr(fit, "grouped_fits", []) or []):
                    pdict = getattr(getattr(local_fit, "model", None), "parameters_all_dict", {}) or {}
                    if lname in pdict and pdict[lname] is link:
                        return fit_idx, local_fit, local_idx, link
            pdict = getattr(getattr(fit, "model", None), "parameters_all_dict", {}) or {}
            if lname in pdict and pdict[lname] is link:
                return fit_idx, fit, None, link
        return None

    @staticmethod
    def _finalize(fc, fit_idx, fit, local_idx, param):
        try:
            fc.update_fit(fit_index=fit_idx)
            fc.model_finalize(fit_index=fit_idx)
        except Exception:
            pass

    # ── helpers for the view ──────────────────────────────────────────

    def fit_at_row(self, row: int) -> Fit:
        return self._rows[row][1] if 0 <= row < len(self._rows) else None

    def fit_idx_at_row(self, row: int) -> int:
        return self._rows[row][0] if 0 <= row < len(self._rows) else -1

    def param_at_row(self, row: int) -> FittingParameter:
        return self._rows[row][3] if 0 <= row < len(self._rows) else None

    def link_target_source_row(self, row: int) -> int:
        """Return the source-model row index for a row's link target.

        Parameters
        ----------
        row : int
            Source-model row index.

        Returns
        -------
        int
            Source-model row index for the link target, or ``-1`` when the
            parameter is unlinked or the target is not in the table.
        """
        param = self.param_at_row(row)
        if param is None:
            return -1
        target = self._link_target_row(param)
        if target is None:
            return -1
        try:
            return self._rows.index(target)
        except ValueError:
            return -1

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
