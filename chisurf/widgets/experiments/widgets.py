from __future__ import annotations
import typing
import os

from qtpy import QtWidgets, QtCore, QtGui

import chisurf.fio
import chisurf.widgets.fio
import chisurf.experiments.data
import chisurf.fitting
import chisurf.widgets
import chisurf.decorators
from chisurf.experiments import reader


@chisurf.decorators.register
class ExperimentalDataSelector(
    QtWidgets.QTreeWidget
):
    """

    """

    @property
    def curve_name(
            self
    ) -> str:
        try:
            return self.selected_dataset.name
        except AttributeError:
            return "Noname"

    @property
    def datasets(
            self
    ) -> typing.List[
        chisurf.experiments.data.ExperimentalData
    ]:
        data_curves = self.get_data_sets(
            curve_type=self.curve_type
        )
        if self.data_reader is not None:
            return [
                d for d in data_curves if isinstance(
                    d.data_reader,
                    self.data_reader
                )
            ]
        else:
            return data_curves

    @property
    def selected_curve_index(
            self
    ) -> int:
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
    ) -> chisurf.experiments.data.ExperimentalData:
        return self.datasets[self.selected_curve_index]

    @property
    def selected_datasets(
            self
    ) -> typing.List[
        chisurf.experiments.data.ExperimentalData
    ]:
        data_sets_idx = self.selected_dataset_idx
        return [self.datasets[i] for i in data_sets_idx]

    @property
    def selected_dataset_idx(
            self
    ) -> typing.List[int]:
        return [r.row() for r in self.selectedIndexes()]

    def onCurveChanged(self):
        if self.click_close:
            self.hide()
        self.change_event()

    def onChangeCurveName(self):
        # select current curve and change its name
        pass

    def onRemoveDataset(self):
        dataset_idx = [
            selected_index.row() for selected_index in self.selectedIndexes()
        ]
        chisurf.run(
            'chisurf.macros.remove_datasets(%s)' % dataset_idx
        )
        self.update(update_others=True)

    def onSaveDataset(self):
        filename = chisurf.widgets.save_file(
            file_type="*.*"
        )
        base_name, extension = os.path.splitext(filename)
        if extension.lower() == '.csv':
            self.selected_dataset.save(
                filename=filename,
                file_type='csv'
            )
        else:
            filename = base_name + '.yaml'
            self.selected_dataset.save(
                filename=filename,
                file_type='yaml'
            )

    def onGroupDatasets(self):
        dg = self.selected_dataset_idx
        chisurf.run("chisurf.macros.group_datasets(%s)" % dg)
        self.update()

    def onUnGroupDatasets(self):
        dg = chisurf.experiments.data.ExperimentDataGroup(
            self.selected_datasets
        )[0]
        dn = list()
        for d in chisurf.imported_datasets:
            if d is not dg:
                dn.append(d)
            else:
                dn += dg
        chisurf.imported_datasets = dn
        self.update()

    def contextMenuEvent(self, event):
        if self.context_menu_enabled:
            menu = QtWidgets.QMenu(self)
            menu.setTitle("Datasets")
            menu.addAction("Save").triggered.connect(self.onSaveDataset)
            menu.addAction("Remove").triggered.connect(self.onRemoveDataset)
            menu.addAction("Group").triggered.connect(self.onGroupDatasets)
            menu.addAction("Ungroup").triggered.connect(self.onUnGroupDatasets)
            menu.addAction("Refresh").triggered.connect(self.update)
            menu.exec_(event.globalPos())

    def update(self, *args, update_others=True, **kwargs):
        super().update()
        try:
            window_title = self.fit.name
            self.setWindowTitle(window_title)
        except AttributeError:
            self.setWindowTitle("")
        self.clear()

        for d in self.datasets:
            fn = d.name
            widget_name = os.path.basename(fn)
            item = QtWidgets.QTreeWidgetItem(self, [widget_name])
            item.setToolTip(0, fn)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

            # If group of curves
            if isinstance(d, chisurf.experiments.data.ExperimentDataGroup):
                for di in d:
                    fn = di.name
                    widget_name = os.path.basename(fn)
                    i2 = QtWidgets.QTreeWidgetItem(item, [widget_name])
                    i2.setToolTip(0, fn)
                    i2.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        # update other instances
        if update_others:
            for i in self.get_instances():
                if i is not self:
                    i.update(update_others=False)

    # def dragMoveEvent(
    #         self,
    #         event
    # ):
    #     super().dragMoveEvent(event)
    #
    # def dragEnterEvent(
    #         self,
    #         event
    # ):
    #     if event.mimeData().hasUrls():
    #         event.acceptProposedAction()
    #     else:
    #         super().dragEnterEvent(event)
    #
    # def startDrag(
    #         self,
    #         supportedActions
    # ):
    #     # self.drag_item = self.currentItem()
    #     # self.drag_row = self.row(self.drag_item)
    #     super().startDrag(supportedActions)

    def dropEvent(
            self,
            event: QtGui.QDropEvent
    ):
        if event.mimeData().hasUrls():
            paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
            paths.sort()
            chisurf.run(
                "\n".join(
                    [
                        "chisurf.macros.add_dataset(filename='%s')" % p for p in paths
                    ]
                )
            )
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
        self.update()

    def onItemChanged(self):
        if self.selected_datasets:
            ds = self.selected_datasets[0]
            ds.name = str(self.currentItem().text(0))
            self.update(update_others=True)

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtWidgets.QTreeWidget.show(self)

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            setup=None,
            drag_enabled: bool = False,
            click_close: bool = True,
            change_event: typing.Callable = None,
            curve_types: str = 'experiment',
            get_data_sets: typing.Callable = None,
            parent: QtWidgets.QWidget = None,
            icon: QtGui.QIcon = None,
            context_menu_enabled: bool = True
    ):
        """

        :param fit:
        :param setup:
        :param drag_enabled:
        :param click_close:
        :param change_event:
        :param curve_types:
        :param get_data_sets:
        :param parent:
        :param icon:
        :param context_menu_enabled: if True there is a context menu
        """
        if get_data_sets is None:
            def get_data_sets(**kwargs):
                return chisurf.experiments.get_data(
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
        self.data_reader = setup
        self.context_menu_enabled = context_menu_enabled

        super().__init__(
            parent=parent
        )
        self.setWindowIcon(icon)
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


class FCSController(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    def get_filename(
            self
    ) -> str:
        return chisurf.widgets.get_filename(
                'FCS-CSV files',
                file_type=self.file_type
            )

    def __init__(
            self,
            file_type='Kristine files (*.cor)',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.file_type = file_type
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        self.layout.addWidget(
            chisurf.widgets.fio.CsvWidget()
        )
