"""

"""
from __future__ import annotations
from chisurf import typing

import fnmatch
import numbers
import os
import pathlib

from qtpy import QtGui, QtWidgets
from io import BytesIO

import pyqtgraph as pg
import matplotlib.pyplot as plt

import chisurf.fio
import chisurf.settings
import chisurf.curve
import chisurf.base


def get_widgets_in_layout(
        layout: QtWidgets.QLayout
):
    """Returns a list of all widgets within a layout
    """
    return (layout.itemAt(i) for i in range(layout.count()))


def clear_layout(
        layout: QtWidgets.QLayout
):
    """Clears all widgets within a layout
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clear_layout(child.layout())


def hide_items_in_layout(
        layout: QtWidgets.QLayout
):
    """Hides all items within a Qt-layout
    """
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if type(item) == QtWidgets.QWidgetItem:
            item.widget().hide()


class MyMessageBox(
    QtWidgets.QMessageBox
):

    def __init__(
            self,
            label: str = None,
            info: str = None,
            details: str = None,
            show_fortune: bool = chisurf.settings.cs_settings['fortune']
    ):
        """This Widget can be used to provide an output for warnings
        and exceptions. It can also display fortune cookies.

        :param label:
        :param info:
        :param show_fortune: if True than a fortune cookie is displayed.
        """
        super().__init__()
        self.Icon = 1
        self.setSizeGripEnabled(True)
        self.setIcon(
            QtWidgets.QMessageBox.Information
        )
        if label is not None:
            self.setWindowTitle(label)
        if details is not None:
            self.setDetailedText(details)
        if show_fortune:
            fortune = chisurf.gui.widgets.fortune.get_fortune()
            self.setInformativeText(
                "\n".join(
                    [
                        info,
                        fortune
                    ]
                )
            )
            self.exec_()
            self.setMinimumWidth(450)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
        else:
            self.close()

    def event(self, e) -> bool:
        result = QtWidgets.QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        text_edit = self.findChild(QtWidgets.QTextEdit)
        if text_edit is not None:
            text_edit.setMinimumHeight(0)
            text_edit.setMaximumHeight(16777215)
            text_edit.setMinimumWidth(0)
            text_edit.setMaximumWidth(16777215)
            text_edit.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )

        return result


class FileList(QtWidgets.QListWidget):

    @property
    def filenames(self) -> typing.List[str]:
        fn = list()
        for row in range(self.count()):
            item = self.item(row)
            fn.append(str(item.text()))
        return fn

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                s = str(url.toLocalFile())
                url.setScheme("")
                if s.endswith(self.filename_ending) or \
                        s.endswith(self.filename_ending+'.gz') or \
                        s.endswith(self.filename_ending+'.bz2') or \
                        s.endswith(self.filename_ending+'.zip'):
                    self.addItem(s)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def __init__(
            self,
            accept_drops: bool = True,
            filename_ending: str = "*",
            icon: QtGui.QIcon = None
    ):
        """
        :param accept_drops: if True accepts files that are dropped into the list
        :param kwargs:
        """
        super().__init__()
        self.filename_ending = filename_ending
        self.drag_item = None
        self.drag_row = None

        if accept_drops:
            self.setAcceptDrops(True)
            self.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove
            )

        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/list-add.png")

        self.setWindowIcon(icon)


def get_filename(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: pathlib.Path = None
) -> pathlib.Path:
    """Open a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = chisurf.working_path
    filename_str, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        description,
        str(working_path.absolute()),
        file_type
    )
    filename = pathlib.Path(filename_str)
    chisurf.working_path = pathlib.Path(filename).home()
    return filename


def open_files(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: pathlib.Path = None
):
    """Open a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = chisurf.working_path
    filenames = QtWidgets.QFileDialog.getOpenFileNames(
        None,
        description,
        str(working_path.absolute()),
        file_type
    )[0]
    chisurf.working_path = pathlib.Path(filenames[0]).home()
    return filenames


def save_file(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: pathlib.Path = None
) -> str:
    """Same as open see above a file within a working path. If no path is
    specified the last path is used. After using this function the current
    working path of the running program (ChiSurf) is updated according to the
    folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = chisurf.working_path
    filename = str(
        QtWidgets.QFileDialog.getSaveFileName(
            None,
            description,
            str(working_path.absolute()),
            file_type
        )[0]
    )
    chisurf.working_path = pathlib.Path(filename).home()
    return filename


def get_directory(
        filename_ending: str = None,
        get_files: bool = False,
        directory: pathlib.Path = None
):
    """Opens a new window where you can choose a directory. The current
    working path is updated to this directory.

    It either returns the directory or the files within the directory (if
    get_files is True). The returned files can be filtered for the filename
    ending using the kwarg filename_ending.

    :return: directory str
    """
    fn_ending = filename_ending
    if directory is None:
        directory = chisurf.working_path
    if isinstance(directory, str):
        directory = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                None,
                "Select Directory",
                str(directory.absolute()),
            )
        )
    else:
        directory = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                None,
                "Select Directory"
            )
        )
    chisurf.working_path = pathlib.Path(directory)
    if not get_files:
        return directory
    else:
        filenames = [directory + '/' + s for s in os.listdir(directory)]
        if fn_ending is not None:
            filenames = fnmatch.filter(filenames, fn_ending)
        return directory, filenames


def make_widget_from_yaml(
        variable_dictionary,
        name: str = ''
):
    """
    >>> import numbers
    >>> import pyqtgraph as pg
    >>> import collections
    >>> import yaml
    >>> d = yaml.safe_load(open("./test_session.yaml"))['datasets']
    >>> od =dict(sorted(d.items()))
    >>> w = make_widget_from_yaml(od, 'test')
    >>> w.show()
    :param variable_dictionary: 
    :param name: 
    :return: 
    """
    def make_group(
            d,
            name: str = ''
    ):
        g = QtWidgets.QGroupBox()
        g.setTitle(str(name))
        layout = QtWidgets.QFormLayout()
        g.setLayout(layout)

        for row, key in enumerate(d):
            label = QtWidgets.QLabel(str(key))
            value = d[key]
            if isinstance(value, dict):
                wd = make_group(value, '')
                layout.addRow(str(key), wd)
            else:
                if isinstance(value, bool):
                    wd = QtWidgets.QCheckBox()
                    wd.setChecked(value)
                elif isinstance(value, numbers.Real):
                    wd = pg.SpinBox(value=value)
                else:
                    wd = QtWidgets.QLineEdit()
                    wd.setText(str(value))
                layout.addRow(label, wd)
        return g

    return make_group(variable_dictionary, name)


def tex2svg(
        formula: str,
        fontsize: int = 12,
        dpi: int = 300
):
    """Render TeX formula to SVG.
    Args:
        formula (str): TeX formula.
        fontsize (int, optional): Font size.
        dpi (int, optional): DPI.
    Returns:
        str: SVG render.
    """

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, r'${}$'.format(formula), fontsize=fontsize)

    output = BytesIO()
    fig.savefig(
        output,
        dpi=dpi,
        transparent=True,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.0,
        frameon=False
    )
    plt.close(fig)

    output.seek(0)
    return output.read()


def get_subtree_nodes(tree_widget_item):
    """Returns all QTreeWidgetItems in the subtree rooted at the given node."""
    nodes = []
    nodes.append(tree_widget_item)
    for i in range(tree_widget_item.childCount()):
        nodes.extend(get_subtree_nodes(tree_widget_item.child(i)))
    return nodes


def get_all_items(tree_widget):
    """Returns all QTreeWidgetItems in the given QTreeWidget."""
    all_items = []
    for i in range(tree_widget.topLevelItemCount()):
        top_item = tree_widget.topLevelItem(i)
        all_items.extend(get_subtree_nodes(top_item))
    return all_items


class Controller(
    QtWidgets.QWidget,
    chisurf.base.Base
):
    """
    Used by FittingControllerWidget
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ):
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        d.update(
            {
                'type': 'controller',
                'class': self.__class__.__name__
            }
        )
        return d


class View(
    QtWidgets.QWidget,
    chisurf.base.Base
):
    """
    Used by Plot
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def update(self, *args, **kwargs) -> None:
        super().update()

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ):
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        d.update(
            {
                'type': 'view',
                'class': self.__class__.__name__
            }
        )
        return d
