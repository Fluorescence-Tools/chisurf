"""Table widget that renders a list of FittingParameter objects as editable rows.

Provides :class:`ParameterGroupTableModel` (the ``QAbstractTableModel``) and
:class:`ParameterGroupTableWidget` (the ``QWidget`` wrapper with a
``QTableView`` and checkbox delegates).  Designed for
:class:`chisurf.core.dataspec.ParameterGroupTableSection` in the AutoForm
system, but usable standalone::

    model = ParameterGroupTableModel(my_params)
    view = ParameterGroupTableWidget(model=model)
    view.show()

Each row represents one parameter; columns are controlled by the ``columns``
attribute on the section descriptor.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from chisurf import typing
from chisurf.core.fitting.parameter import FittingParameter

# ── column enumeration ──────────────────────────────────────────────────

COL_NAME = 0
COL_VALUE = 1
COL_FIXED = 2
COL_BOUNDS_LO = 3
COL_BOUNDS_HI = 4
COL_BOUNDS_ON = 5
COL_ERROR = 6

#: (id, label, editable, kind)
COLUMN_META = [
    ("name", "Name", False, "str"),
    ("value", "Value", True, "float"),
    ("fixed", "Fixed", True, "bool"),
    ("bounds_lo", "Lo", True, "float"),
    ("bounds_hi", "Hi", True, "float"),
    ("bounds_on", "Bounds", True, "bool"),
    ("error", "Error", False, "float"),
]

COLUMN_IDS = [m[0] for m in COLUMN_META]


# ── boolean checkbox delegate ───────────────────────────────────────────

class _BooleanToggleDelegate(QtWidgets.QStyledItemDelegate):
    """Click-to-toggle checkbox rendered centered in the cell."""

    def _is_checked(self, value) -> bool:
        try:
            if isinstance(value, (bool,)) or value is None:
                return bool(value) if value is not None else False
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "on")
            return bool(int(value))
        except Exception:
            return False

    def _toggle(self, value) -> bool:
        return not self._is_checked(value)

    def _checkbox_rect(self, option: QtWidgets.QStyleOptionViewItem) -> QtCore.QRect:
        rect = option.rect
        size = 16
        x = rect.x() + (rect.width() - size) // 2
        y = rect.y() + (rect.height() - size) // 2
        return QtCore.QRect(x, y, size, size)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        checked = self._is_checked(index.data(QtCore.Qt.DisplayRole))
        style = (
            QtWidgets.QApplication.style()
            if QtWidgets.QApplication.instance()
            else option.widget.style()
        )
        cb_opt = QtWidgets.QStyleOptionButton()
        cb_opt.state = (
            QtWidgets.QStyle.State_Enabled
            | (QtWidgets.QStyle.State_On if checked else QtWidgets.QStyle.State_Off)
        )
        cb_opt.rect = self._checkbox_rect(option)
        style.drawControl(QtWidgets.QStyle.CE_CheckBox, cb_opt, painter)

    def createEditor(self, parent, option, index):
        return None

    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> bool:
        et = event.type()
        if et in (QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseButtonDblClick):
            new_val = self._toggle(index.data(QtCore.Qt.DisplayRole))
            return model.setData(index, str(new_val), QtCore.Qt.EditRole)
        if et == QtCore.QEvent.KeyPress:
            if isinstance(event, QtGui.QKeyEvent) and event.key() in (
                QtCore.Qt.Key_Space,
                QtCore.Qt.Key_Return,
                QtCore.Qt.Key_Enter,
            ):
                new_val = self._toggle(index.data(QtCore.Qt.DisplayRole))
                return model.setData(index, str(new_val), QtCore.Qt.EditRole)
        return False


# ── table model ─────────────────────────────────────────────────────────

class ParameterGroupTableModel(QtCore.QAbstractTableModel):
    """Table model exposing a list of :class:`FittingParameter` objects.

    Each row is one parameter.  Columns are defined by :data:`COLUMN_META`
    and map to the parameter's value, fixed flag, bounds, and error estimate.
    The backing list is *not* copied — edits flow through to the original
    objects immediately.
    """

    def __init__(
        self,
        params: typing.List[FittingParameter],
        parent: typing.Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._params: typing.List[FittingParameter] = list(params)

    # -- row / column count -------------------------------------------------
    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return len(self._params) if not parent.isValid() else 0

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return len(COLUMN_META) if not parent.isValid() else 0

    # -- header data --------------------------------------------------------
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            if 0 <= section < len(COLUMN_META):
                return COLUMN_META[section][1]
        return None

    # -- cell data ----------------------------------------------------------
    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        param = self._params[index.row()]
        col_id, _, editable, kind = COLUMN_META[index.column()]

        if role == QtCore.Qt.DisplayRole:
            return self._display_value(col_id, kind, param)
        if role == QtCore.Qt.EditRole:
            return self._edit_value(col_id, param)
        if role == QtCore.Qt.TextAlignmentRole:
            if kind == "float":
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if kind == "bool":
                return int(QtCore.Qt.AlignCenter)
        if role == QtCore.Qt.ToolTipRole:
            return self._tooltip(col_id, param)
        return None

    @staticmethod
    def _display_value(col_id: str, kind: str, param: FittingParameter) -> str:
        if col_id == "name":
            return str(param.__dict__.get("label_text", param.name))
        if col_id == "value":
            v = param.value
            return f"{v:.6g}" if v is not None else ""
        if col_id == "fixed":
            return str(bool(param.fixed))
        if col_id == "bounds_lo":
            b = param.bounds
            if b is not None and b[0] is not None and param.bounds_on:
                return f"{b[0]:.6g}"
            return ""
        if col_id == "bounds_hi":
            b = param.bounds
            if b is not None and len(b) > 1 and b[1] is not None and param.bounds_on:
                return f"{b[1]:.6g}"
            return ""
        if col_id == "bounds_on":
            return str(bool(param.bounds_on))
        if col_id == "error":
            e = getattr(param, "error_estimate", None)
            if e is not None and _isfinite(e):
                return f"{e:.4g}"
            return ""
        return ""

    @staticmethod
    def _edit_value(col_id: str, param: FittingParameter):
        if col_id == "value":
            return float(param.value)
        if col_id == "fixed":
            return bool(param.fixed)
        if col_id == "bounds_lo":
            b = param.bounds
            return float(b[0]) if b is not None and b[0] is not None else 0.0
        if col_id == "bounds_hi":
            b = param.bounds
            return float(b[1]) if b is not None and len(b) > 1 and b[1] is not None else 0.0
        if col_id == "bounds_on":
            return bool(param.bounds_on)
        return None

    @staticmethod
    def _tooltip(col_id: str, param: FittingParameter) -> typing.Optional[str]:
        if col_id == "name":
            d = getattr(param, "description", "") or ""
            return d if d else None
        if col_id == "value":
            linked = getattr(param, "is_linked", False)
            if linked and not getattr(param, "is_link_master", False):
                link = getattr(param, "link", None)
                lname = getattr(link, "name", "?") if link else "?"
                return f"Linked to {lname} (read-only)"
        return None

    # -- flags / editing ----------------------------------------------------
    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        col_id, _, editable, _ = COLUMN_META[index.column()]
        base = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if not editable:
            return base
        # Linked followers cannot edit value
        param = self._params[index.row()]
        if col_id == "value":
            is_follower = getattr(param, "is_linked", False) and not getattr(
                param, "is_link_master", False
            )
            if is_follower:
                return base
        # Bounds columns only editable when bounds_on is True
        if col_id in ("bounds_lo", "bounds_hi") and not getattr(param, "bounds_on", False):
            return base
        return base | QtCore.Qt.ItemIsEditable

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: typing.Any,
        role: int = QtCore.Qt.EditRole,
    ) -> bool:
        if role != QtCore.Qt.EditRole or not index.isValid():
            return False
        param = self._params[index.row()]
        col_id, _, _, _ = COLUMN_META[index.column()]

        try:
            if col_id == "value":
                is_follower = getattr(param, "is_linked", False) and not getattr(
                    param, "is_link_master", False
                )
                if is_follower:
                    return False
                param.value = float(value)
            elif col_id == "fixed":
                param.fixed = _parse_bool(value)
            elif col_id == "bounds_lo":
                b = list(param.bounds)
                b[0] = float(value)
                param.bounds = tuple(b)
            elif col_id == "bounds_hi":
                b = list(param.bounds)
                b[1] = float(value)
                param.bounds = tuple(b)
            elif col_id == "bounds_on":
                param.bounds_on = _parse_bool(value)
            else:
                return False
        except Exception:
            return False

        self.dataChanged.emit(index, index)
        return True

    # -- helpers ------------------------------------------------------------
    @property
    def parameters(self) -> typing.List[FittingParameter]:
        """Live list of parameters backing the model."""
        return self._params


def _isfinite(v: typing.Any) -> bool:
    try:
        import numpy as np
        return bool(np.isfinite(v))
    except Exception:
        return True


def _parse_bool(value: typing.Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


# ── table widget ────────────────────────────────────────────────────────

class ParameterGroupTableWidget(QtWidgets.QWidget):
    """A ``QTableView`` that edits a list of :class:`FittingParameter` objects.

    Parameters
    ----------
    params : list of FittingParameter
        The parameters to display (one per row).
    section : chisurf.core.dataspec.ParameterGroupTableSection or None
        Section descriptor controlling visible columns and collapsible
        behaviour.  When ``None`` all columns are shown.
    parent : QWidget or None
        Parent widget.
    on_change : callable or None
        Optional callback invoked (with no arguments) after every edit.  In
        the AutoForm context this is wired to trigger a fit recompute.
    """

    def __init__(
        self,
        params: typing.List[FittingParameter],
        section: typing.Any = None,
        parent: typing.Optional[QtWidgets.QWidget] = None,
        on_change: typing.Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent)
        self._section = section
        self._on_change = on_change
        self._params = params

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._model = ParameterGroupTableModel(params)
        self._table = QtWidgets.QTableView()
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(False)
        self._table.setWordWrap(False)
        self._table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.verticalHeader().hide()
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.setShowGrid(True)

        # Compact font
        try:
            f = self._table.font()
            f.setPointSize(max(8, f.pointSize() - 1))
            f.setStyleStrategy(QtGui.QFont.PreferAntialias)
            self._table.setFont(f)
        except Exception:
            pass

        # Column widths
        hh = self._table.horizontalHeader()
        hh.setStretchLastSection(True)
        try:
            hh.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        except Exception:
            try:
                hh.setResizeMode(QtWidgets.QHeaderView.Interactive)
            except Exception:
                pass
        hh.setMinimumSectionSize(40)
        _default_widths = {COL_NAME: 100, COL_VALUE: 80, COL_FIXED: 55,
                           COL_BOUNDS_LO: 65, COL_BOUNDS_HI: 65,
                           COL_BOUNDS_ON: 60, COL_ERROR: 65}
        for col, w in _default_widths.items():
            try:
                self._table.setColumnWidth(col, w)
            except Exception:
                pass

        # Hide columns that are not in the section's whitelist
        self._apply_column_visibility()

        # Boolean toggle delegates on fixed/bounds_on columns
        self._toggle_delegate = _BooleanToggleDelegate(self._table)
        self._table.setItemDelegateForColumn(COL_FIXED, self._toggle_delegate)
        self._table.setItemDelegateForColumn(COL_BOUNDS_ON, self._toggle_delegate)

        # Wire model changes to optional callback
        self._model.dataChanged.connect(self._on_data_changed)

        layout.addWidget(self._table)

    # -- column visibility --------------------------------------------------
    def _apply_column_visibility(self):
        section = self._section
        if section is None:
            return
        cols = getattr(section, "columns", None)
        if not cols:
            return
        visible = set(cols)
        for i, cid in enumerate(COLUMN_IDS):
            self._table.setColumnHidden(i, cid not in visible)

    # -- change dispatch ----------------------------------------------------
    def _on_data_changed(self, *_):
        cb = self._on_change
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    # -- sync ---------------------------------------------------------------
    def sync(self) -> None:
        """Re-read parameter values into the model and repaint."""
        top_left = self._model.index(0, 0)
        bottom_right = self._model.index(
            self._model.rowCount() - 1,
            self._model.columnCount() - 1,
        )
        self._model.dataChanged.emit(top_left, bottom_right)

    # -- accessors ----------------------------------------------------------
    @property
    def table_model(self) -> ParameterGroupTableModel:
        return self._model

    @property
    def table_view(self) -> QtWidgets.QTableView:
        return self._table

    @property
    def parameters(self) -> typing.List[FittingParameter]:
        return self._params
