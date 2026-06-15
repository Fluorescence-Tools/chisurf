from __future__ import annotations

import os
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf.core.data
import chisurf.core.fitting
import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.core.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.core.math.optimization.leastsqbound import OptimizationCancelled

from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client


@chisurf.core.decorators.register
class ModelDataRepresentationSelector(QtWidgets.QTreeWidget):

    @property
    def _fc(self):
        """Try to get the global fitting client; may be None."""
        return get_fitting_client()

    @property
    def selected_fit_index(self) -> int:
        if self.currentIndex().parent().isValid():
            return self.currentIndex().parent().row()
        else:
            return self.currentIndex().row()

    @selected_fit_index.setter
    def selected_fit_index(self, v: int):
        self.setCurrentItem(self.topLevelItem(v))

    @property
    def selected_fit(self):
        fc = self._fc
        if fc is not None:
            fits = fc.list_fits()
            idx = self.selected_fit_index
            if 0 <= idx < len(fits):
                return fits[idx]
        return {}

    @property
    def selected_fits(self) -> typing.List:
        fc = self._fc
        if fc is not None:
            fits = fc.list_fits()
            return [fits[i] for i in self.selected_fit_idx if 0 <= i < len(fits)]
        return []

    @property
    def selected_fit_idx(self) -> typing.List[int]:
        return [r.row() for r in self.selectedIndexes()]

    def selectedIndexes(self) -> typing.List[QtCore.QModelIndex]:
        idx = super().selectedIndexes()[::3]
        return idx

    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.onRemoveFit()

    def onCurveChanged(self):
        fc = self._fc
        if fc is not None:
            fits = fc.list_fits()
            sel = self.selected_fit
            sel_uid = sel.get("uid") if isinstance(sel, dict) else getattr(sel, "unique_identifier", None)
            _mdiarea = getattr(chisurf, "cs", None)
            _mdiarea = getattr(_mdiarea, "mdiarea", None) if _mdiarea is not None else None
            if _mdiarea is not None:
                for fit_window in _mdiarea.subWindowList():
                    fit = getattr(fit_window, 'fit', None)
                    if fit is not None:
                        win_uid = (fit.get("uid") if isinstance(fit, dict)
                                   else str(getattr(fit, "unique_identifier", "")))
                        if win_uid and win_uid == sel_uid:
                            _mdiarea.setActiveSubWindow(fit_window)
                            break
                    elif isinstance(sel, dict) and hasattr(fit_window, 'fit_uid'):
                        if getattr(fit_window, 'fit_uid', None) == sel_uid:
                            _mdiarea.setActiveSubWindow(fit_window)
                            break
        self.change_event()

    def onChangeCurveName(self):
        pass

    def onRemoveFit(self):
        fc = self._fc
        if fc is not None:
            fit_uids = []
            fits = fc.list_fits()
            for si in self.selectedIndexes():
                idx = si.row()
                if 0 <= idx < len(fits):
                    uid = fits[idx].get("uid")
                    if uid:
                        fit_uids.append(uid)
            if fit_uids:
                fc.remove_fits(fit_uids=fit_uids)
        self.update(update_others=True)

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        fc = self._fc
        if fc is not None:
            fits = fc.list_fits()
            for si in self.selectedIndexes():
                idx = si.row()
                if 0 <= idx < len(fits):
                    fit_data = fits[idx]
                    fc.save_fit(
                        filename=fit_data.get("name", "fit_export"),
                        fit_uid=fit_data.get("uid"),
                    )

    def contextMenuEvent(self, event):
        if self.context_menu_enabled:
            menu = QtWidgets.QMenu(self)
            menu.setTitle("Fits")
            menu.addAction("Save").triggered.connect(self.onSaveFit)
            menu.addAction("Close").triggered.connect(self.onRemoveFit)
            menu.addAction("Update").triggered.connect(self.update)
            menu.exec_(event.globalPos())

    def update(self, *args, update_others=True, **kwargs):
        try:
            self.blockSignals(True)
            self.setUpdatesEnabled(False)
            super().update()
            self.clear()

            fc = self._fc
            if fc is not None:
                for nbr, fit_dto in enumerate(fc.list_fits()):
                    widget_name = fit_dto.get("dataset_name", "Unknown")
                    model_name = fit_dto.get("model_name", "Model")
                    item = QtWidgets.QTreeWidgetItem(self, [str(nbr), widget_name, model_name])
                    item.setToolTip(1, fit_dto.get("name", widget_name))
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        finally:
            self.setUpdatesEnabled(True)
            self.blockSignals(False)

    def onItemChanged(self):
        if self.selected_fits:
            fc = self._fc
            if fc is not None:
                ds = self.selected_fits[0]
                idx_new = int(self.currentItem().text(0))
                fits = fc.list_fits()
                uids = [f.get("uid") for f in fits if f.get("uid")]
                ds_uid = ds.get("uid") if isinstance(ds, dict) else getattr(ds, "unique_identifier", None)
                if ds_uid in uids:
                    uids.remove(ds_uid)
                    uids.insert(idx_new, ds_uid)
                    fc.reorder_fits(uids)
            self.update(update_others=True)

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtWidgets.QTreeWidget.show(self)

    def __init__(
            self,
            fit: chisurf.core.fitting.fit.Fit = None,
            experiment=None,
            drag_enabled: bool = False,
            click_close: bool = False,
            change_event: typing.Callable = None,
            curve_types: str = 'experiment',
            get_data_sets: typing.Callable = None,
            parent: QtWidgets.QWidget = None,
            icon: QtGui.QIcon = None,
            context_menu_enabled: bool = True
    ):
        if get_data_sets is None:
            def get_data_sets(**kwargs):
                return chisurf.core.data.get_data(
                    data_set=getattr(chisurf, "imported_datasets", []),
                    **kwargs
                )
            self.get_data_sets = get_data_sets
        else:
            self.get_data_sets = get_data_sets

        if change_event is not None:
            self.change_event = change_event

        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/list-add.png")

        self.curve_type = curve_types
        self.click_close = click_close
        self.fit = fit
        self.experiment = experiment
        self.context_menu_enabled = context_menu_enabled

        super().__init__(parent)
        self.setWindowIcon(icon)
        self.setWordWrap(True)
        self.setAlternatingRowColors(True)

        if drag_enabled:
            self.setAcceptDrops(True)
            self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        # http://python.6.x6.nabble.com/Drag-and-drop-editing-in-QListWidget-or-QListView-td1792540.html
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.drag_item = None
        self.drag_row = None

        self.clicked.connect(self.onCurveChanged)
        self.itemChanged.connect(self.onItemChanged)

        self.setHeaderHidden(False)
        self.setColumnCount(3)
        self.setHeaderLabels(('#', 'Data name', 'Model type'))
        header = self.header()

        # Set resize mode for the first and third columns
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        header.setSectionsClickable(True)


