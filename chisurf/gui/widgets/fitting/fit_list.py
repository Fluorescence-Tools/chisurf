from __future__ import annotations

import os
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf.data
import chisurf.fitting
import chisurf.decorators
import chisurf.gui.decorators
import chisurf.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.math.optimization.leastsqbound import OptimizationCancelled


@chisurf.decorators.register
class ModelDataRepresentationSelector(QtWidgets.QTreeWidget):

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
    def selected_fit(self) -> chisurf.fitting.fit.FitGroup:
        return chisurf.fits[self.selected_fit_index]

    @property
    def selected_fits(self) -> typing.List[chisurf.fitting.fit.FitGroup]:
        return [chisurf.fits[i] for i in self.selected_fit_idx]

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
        for fit_window in chisurf.cs.mdiarea.subWindowList():
            if fit_window.fit == self.selected_fit:
                chisurf.cs.mdiarea.setActiveSubWindow(fit_window)
                break
        self.change_event()

    def onChangeCurveName(self):
        # select current curve and change its name
        pass

    def onRemoveFit(self):
        fit_idxs = [selected_index.row() for selected_index in self.selectedIndexes()]
        for fit_idx in fit_idxs:
            try:
                chisurf.actions.dispatch(
                    name="fit.close",
                    payload={"idx": int(fit_idx)},
                )
            except Exception:
                pass
        self.update(update_others=True)

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        for fit_window in chisurf.cs.mdiarea.subWindowList():
            chisurf.cs.mdiarea.setActiveSubWindow(fit_window)
            chisurf.cs.onSaveFit()

    def contextMenuEvent(self, event):
        if self.context_menu_enabled:
            menu = QtWidgets.QMenu(self)
            menu.setTitle("Fits")
            menu.addAction("Save").triggered.connect(self.onSaveFit)
            menu.addAction("Close").triggered.connect(self.onRemoveFit)
            menu.addAction("Update").triggered.connect(self.update)
            menu.exec_(event.globalPos())

    def update(self, *args, update_others=True, **kwargs):
        # Optimize: avoid triggering expensive fit.update() on every list rebuild
        # and minimize signal/paint churn during population.
        try:
            self.blockSignals(True)
            self.setUpdatesEnabled(False)
            super().update()
            self.clear()

            for nbr, fit in enumerate(chisurf.fits):
                # Only use lightweight data to populate the list; do not call fit.update() here.
                try:
                    widget_name = pathlib.Path(fit.data.name).name
                except Exception:
                    widget_name = getattr(fit.data, 'name', 'Unknown')
                try:
                    model_name = fit.model.__class__.name
                except Exception:
                    model_name = getattr(fit.model.__class__, '__name__', 'Model')
                item = QtWidgets.QTreeWidgetItem(self, [str(nbr), widget_name, model_name])
                item.setToolTip(1, getattr(fit, 'name', widget_name))
                item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        finally:
            self.setUpdatesEnabled(True)
            self.blockSignals(False)

    def onItemChanged(self):
        if self.selected_fits:
            ds = self.selected_fits[0]

            # Find the index of the selected dataset
            index_of_ds = chisurf.fits.index(ds)

            # Remove "c" from its current position
            chisurf.fits.pop(index_of_ds)

            # Insert "c" at position 1
            idx_new = int(self.currentItem().text(0))
            chisurf.fits.insert(idx_new, ds)

            self.update(update_others=True)

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtWidgets.QTreeWidget.show(self)

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
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
                return chisurf.data.get_data(
                    data_set=chisurf.imported_datasets,
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


