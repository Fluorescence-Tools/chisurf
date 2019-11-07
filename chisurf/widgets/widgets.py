"""

"""
from __future__ import annotations
from typing import List
from io import BytesIO

import inspect
import fnmatch
import numbers
import os

from qtpy import QtGui, QtWidgets, uic
import qtconsole
import qtconsole.styles
import qtconsole.qtconsoleapp
import qtconsole.inprocess
import IPython.lib

import pyqtgraph as pg
import matplotlib.pyplot as plt

import chisurf.fio
import chisurf.settings
import chisurf.curve
import chisurf.base


class QIPythonWidget(
    qtconsole.qtconsoleapp.RichJupyterWidget
):

    def start_recording(self):
        self._macro = ""
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def run_macro(
            self,
            filename: str = None
    ):
        if filename is None:
            filename = chisurf.widgets.get_filename(
                "Python macros",
                file_type="Python file (*.py)"
            )
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=filename,
                mode='r'
        ) as fp:
            text = fp.read()
            self.execute(text, hidden=False)

    def save_macro(
            self,
            filename: str = None
    ):
        self.stop_recording()
        if filename is None:
            filename = chisurf.widgets.save_file(
                "Python macros",
                file_type="Python file (*.py)"
            )
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            fp.write(self._macro)

    def execute(
            self,
            *args,
            hidden: bool = False,
            **kwargs
    ):
        """

        :param source: the source code that is executed via the command line
        interface.
        :param args:
        :param hidden: if hidden is True the execution is neither recorded in
        the session file nor displayed in the execution history widget
        :param kwargs:
        :return:
        """
        if not hidden:
            try:
                new_text = args[0] + '\n'
                with open(self.session_file, 'a+') as fp:
                    fp.write(new_text)
                if self.recording:
                    self._macro += new_text
                if isinstance(
                        self.history_widget,
                        QtWidgets.QPlainTextEdit
                ):
                    self.history_widget.insertPlainText(new_text)
            except IndexError:
                pass
        super().execute(
            *args,
            **kwargs
        )

    def execute_function(
            self,
            function,
            *args,
            **kwargs
    ):
        """ Gets the function string executes the function on the command line

        :param args:
        :param kwargs:
        :return:
        """
        t = inspect.getsource(function)
        self.execute(t)

    def __init__(
            self,
            history_widget: QtWidgets.QPlainTextEdit = None,
            *args,
            **kwargs
    ):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.history_widget = history_widget

        self.kernel_manager = kernel_manager = qtconsole.inprocess.QtInProcessKernelManager()
        kernel_manager.start_kernel()
        self.kernel_client = kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            IPython.lib.guisupport.get_app_qt4().exit()

        self.exit_requested.connect(stop)
        self.width = kwargs.get(
            'width',
            chisurf.settings.gui['console']['width']
        )
        self._macro = ""
        self.recording = False

        # save nevertheless every inputs into a session file
        self.session_file = chisurf.settings.session_file
        #self.set_default_style(
        #    chisurf.settings.gui['console']['style']
        #)
        self.style_sheet = qtconsole.styles.default_light_style_sheet

    def pushVariables(self, variableDict):
        """ Given a dictionary containing name / value pairs, push those
        variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()

    def printText(self, text):
        """ Prints some plain name to the console """
        self._append_plain_text(text)

    def executeCommand(self, command):
        """ Execute a command in the frame of the console widget """
        self._execute(command, chisurf.settings.cs_settings['show_commands'])


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
            fortune = chisurf.widgets.fortune.get_fortune()
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

    def event(self, e):
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
    def filenames(self) -> List[str]:
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
        working_path: str = None
) -> str:
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
    filename = str(
        QtWidgets.QFileDialog.getOpenFileName(
            None,
            description,
            working_path,
            file_type
        )[0])
    chisurf.working_path = os.path.dirname(filename)
    return filename


def open_files(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: str = None
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
        working_path,
        file_type
    )[0]
    chisurf.working_path = os.path.dirname(filenames[0])
    return filenames


def save_file(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: str = None
):
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
            working_path,
            file_type
        )[0]
    )
    chisurf.working_path = os.path.dirname(filename)
    return filename


def get_directory(
        filename_ending: str = None,
        get_files: bool = False,
        directory: str = None
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
                "Select Directory", directory
            )
        )
    else:
        directory = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                None,
                "Select Directory"
            )
        )
    chisurf.working_path = directory
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


def load_ui(
        target: QtWidgets.QWidget,
        ui_filename: str,
        path: str
):
    filename = os.path.join(
        path,
        ui_filename
    )
    uic.loadUi(
        filename,
        target
    )


def tex2svg(formula, fontsize=12, dpi=300):
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

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def to_dict(
            self
    ):
        return {
            'type': 'controller',
            'class': self.__class__.__name__
        }


class View(
    QtWidgets.QWidget,
    chisurf.base.Base
):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()

    def to_dict(
            self
    ):
        return {
            'type': 'view',
            'class': self.__class__.__name__
        }
