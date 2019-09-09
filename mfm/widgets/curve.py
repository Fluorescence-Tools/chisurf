from __future__ import annotations

import os
from typing import List, Callable

from PyQt5 import QtWidgets, QtCore, QtGui

import mfm.fitting
import mfm.widgets


class ExperimentalDataSelector(QtWidgets.QTreeWidget):

    @property
    def curve_name(self) -> str:
        try:
            return self.selected_dataset.name
        except AttributeError:
            return "Noname"

    @property
    def datasets(self) -> List[mfm.experiments.data.ExperimentalData]:
        data_curves = self.get_curves(curve_type=self.curve_type)
        if self.setup is not None:
            return [
                d for d in data_curves if isinstance(d.setup, self.setup)
            ]
        else:
            return data_curves

    @property
    def selected_curve_index(self) -> int:
        if self.currentIndex().parent().isValid():
            return self.currentIndex().parent().row()
        else:
            return self.currentIndex().row()

    @selected_curve_index.setter
    def selected_curve_index(
            self,
            v: int
    ):
        self.setCurrentItem(self.topLevelItem(v))

    @property
    def selected_dataset(
            self
    ) -> mfm.experiments.data.ExperimentalData:
        return self.datasets[self.selected_curve_index]

    @property
    def selected_datasets(
            self
    ) -> List[mfm.experiments.data.ExperimentalData]:
        data_sets_idx = self.selected_dataset_idx
        return [self.datasets[i] for i in data_sets_idx]

    @property
    def selected_dataset_idx(
            self
    ) -> List[int]:
        return [r.row() for r in self.selectedIndexes()]

    def onCurveChanged(self):
        if self.click_close:
            self.hide()
        self.change_event()

    def onChangeCurveName(self):
        # select current curve and change its name
        pass

    def onRemoveDataset(self):
        dataset_idx = [selected_index.row() for selected_index in self.selectedIndexes()]
        mfm.console.execute('mfm.cmd.remove_datasets(%s)' % dataset_idx)
        self.update()

    def onSaveDataset(self):
        filename = mfm.widgets.save_file(file_type="*.csv")
        self.selected_dataset.save(filename)

    def onGroupDatasets(self):
        dg = self.selected_dataset_idx
        mfm.run("mfm.cmd.group_datasets(%s)" % dg)
        self.update()

    def onUnGroupDatasets(self):
        dg = mfm.experiments.data.ExperimentDataGroup(self.selected_datasets)[0]
        dn = list()
        for d in mfm.imported_datasets:
            if d is not dg:
                dn.append(d)
            else:
                dn += dg
        mfm.imported_datasets = dn
        self.update()

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        menu.setTitle("Datasets")
        menu.addAction("Save").triggered.connect(self.onSaveDataset)
        menu.addAction("Remove").triggered.connect(self.onRemoveDataset)
        menu.addAction("Group").triggered.connect(self.onGroupDatasets)
        menu.addAction("Ungroup").triggered.connect(self.onUnGroupDatasets)
        menu.exec_(event.globalPos())

    def update(self, *args, **kwargs):
        QtWidgets.QTreeWidget.update(self, *args, **kwargs)
        try:
            window_title = self.fit.name
            self.setWindowTitle(window_title)
        except AttributeError:
            self.setWindowTitle("")

        self.clear()

        for d in self.datasets:
            # If normal Curve
            fn = d.name
            widget_name = os.path.basename(fn)
            item = QtWidgets.QTreeWidgetItem(self, [widget_name])
            item.setToolTip(0, fn)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

            # If group of curves
            if isinstance(d, mfm.experiments.data.ExperimentDataGroup):
                for di in d:
                    fn = di.name
                    widget_name = os.path.basename(fn)
                    i2 = QtWidgets.QTreeWidgetItem(item, [widget_name])
                    i2.setToolTip(0, fn)
                    i2.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

    def dragMoveEvent(self, event):
        super(ExperimentalDataSelector, self).dragMoveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super(ExperimentalDataSelector, self).dragEnterEvent(event)

    def startDrag(self, supportedActions):
        #self.drag_item = self.currentItem()
        #self.drag_row = self.row(self.drag_item)
        super(ExperimentalDataSelector, self).startDrag(supportedActions)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
            paths.sort()
            for path in paths:
                s = "cs.add_dataset(filename='%s')" % path
                mfm.run(s)
            event.acceptProposedAction()
        else:
            super(ExperimentalDataSelector, self).dropEvent(event)
        self.update()

    def onItemChanged(self):
        if self.selected_datasets:
            ds = self.selected_datasets[0]
            ds.name = str(self.currentItem().text(0))

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtWidgets.QTreeWidget.show(self)

    def __init__(
            self,
            *args,
            fit: mfm.fitting.fit.Fit = None,
            setup=None,
            drag_enabled: bool = False,
            click_close: bool = True,
            change_event: Callable = None,
            curve_types: str = 'experiment',
            **kwargs
    ):
        def get_curves(**kwargs):
            return mfm.experiments.get_data(
                data_set=mfm.imported_datasets,
                **kwargs
            )

        self.get_curves = get_curves

        if change_event is not None:
            self.change_event = change_event
        self.curve_type = curve_types
        self.click_close = click_close
        self.fit = fit
        self.setup = setup

        super(ExperimentalDataSelector, self).__init__()
        self.setWindowIcon(QtGui.QIcon(":/icons/icons/list-add.png"))
        self.setWordWrap(True)
        self.setHeaderHidden(True)
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
