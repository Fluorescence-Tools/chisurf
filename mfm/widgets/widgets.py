from __future__ import annotations
from typing import List

import inspect
import fnmatch
import numbers
import os
import pickle
import random
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from qtconsole.qtconsoleapp import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

import mfm
import mfm.curve

os.environ['PYZMQ_BACKEND'] = 'cython'
DEFAULT_INSTANCE_ARGS = ['qtconsole','--pylab=inline', '--colors=linux']


class QIPythonWidget(RichJupyterWidget):

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
            filename = mfm.widgets.get_filename("Python macros", file_type="Python file (*.py)")
        with open(filename, mode='r') as fp:
            text = fp.read()
            self.execute(text, hidden=False)

    def save_macro(
            self,
            filename: str = None
    ):
        self.stop_recording()
        if filename is None:
            filename = mfm.widgets.save_file("Python macros", file_type="Python file (*.py)")
        with open(filename, mode='w') as fp:
            fp.write(self._macro)

    def execute(
            self,
            *args,
            hidden: bool = False,
            **kwargs
    ):
        if not hidden:
            try:
                new_text = args[0] + '\n'
                with open(self.session_file, 'a+') as fp:
                    fp.write(new_text)
                if self.recording:
                    self._macro += new_text
                if isinstance(self.history_widget, QtWidgets.QPlainTextEdit):
                    self.history_widget.insertPlainText(new_text)
            except IndexError:
                pass
        RichJupyterWidget.execute(self, *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        RichJupyterWidget.__init__(self, *args, **kwargs)
        self.history_widget = kwargs.get('history_widget', None)

        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        self.kernel_client = kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()

        self.exit_requested.connect(stop)
        self.width = kwargs.get(
            'width',
            mfm.settings.cs_settings['gui']['console']['width']
        )
        self._macro = ""
        self.recording = False

        # save nevertheless every inputs into a session file
        filename = datetime.now().strftime('session_%H_%M_%d_%m_%Y.py')
        home = os.path.expanduser("~")
        path = os.path.abspath(os.path.join(home, './.chisurf'))
        if not os.path.isdir(path):
            os.makedirs(path)
        self.session_file = os.path.join(path, filename)
        self.set_default_style(
            mfm.settings.cs_settings['gui']['console']['style']
        )

    def pushVariables(self, variableDict):
        """ Given a dictionary containing name / value pairs, push those variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()

    def printText(self, text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)

    def executeCommand(self, command):
        """ Execute a command in the frame of the console widget """
        self._execute(command, mfm.settings.cs_settings['show_commands'])


def get_widgets_in_layout(layout):
    """Returns a list of all widgets within a layout
    """
    return (layout.itemAt(i) for i in range(layout.count()))


def clear_layout(layout):
    """Clears all widgets within a layout
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clear_layout(child.layout())


def hide_items_in_layout(layout):
    """Hides all items within a Qt-layout
    """
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if type(item) == QtWidgets.QWidgetItem:
            item.widget().hide()


def get_fortune(
        fortunepath: str = './mfm/ui/fortune/',
        min_length: int = 0,
        max_length: int = 100,
        attempts: int = 1000,
        **kwargs
):
    fortune_files = [os.path.splitext(pdat)[0] for pdat in os.listdir(fortunepath) if pdat.endswith(".pdat")]
    attempt = 0
    while True:
        fortune_file = os.path.join(fortunepath, random.choice(fortune_files))
        data = pickle.load(open(fortune_file+".pdat", "rb"))
        (start, length) = random.choice(data)
        print(random.choice(data))
        if length < min_length or (max_length is not None and length > max_length):
            attempt += 1
            if attempt > attempts:
                return ""
            continue
        with open(fortune_file, 'rU') as ffh:
            ffh.seek(start)
            fortunecookie = ffh.read(length)
        return fortunecookie


class MyMessageBox(QtWidgets.QMessageBox):

    def __init__(
            self,
            label: str = None,
            info: str = None
    ):
        super(MyMessageBox, self).__init__()
        self.Icon = 1
        self.setSizeGripEnabled(True)
        self.setIcon(QtWidgets.QMessageBox.Information)
        if label is not None:
            self.setWindowTitle(label)
        if info is not None:
            self.setDetailedText(info)
        if mfm.settings.cs_settings['fortune']:
            fortune = get_fortune(**mfm.settings.cs_settings['fortune'])
            self.setInformativeText(fortune)
            self.exec_()
            self.setMinimumWidth(450)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        else:
            self.close()

    def event(self, e):
        result = QtWidgets.QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        textEdit = self.findChild(QtWidgets.QTextEdit)
        if textEdit != None :
            textEdit.setMinimumHeight(0)
            textEdit.setMaximumHeight(16777215)
            textEdit.setMinimumWidth(0)
            textEdit.setMaximumWidth(16777215)
            textEdit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

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
            super(FileList, self).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super(FileList, self).dragMoveEvent(event)

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
            super(FileList, self).dropEvent(event)

    def __init__(
            self,
            accept_drops: bool = True,
            filename_ending: str = "*"
    ):
        """
        :param accept_drops: if True accepts files that are dropped into the list
        :param kwargs:
        """
        super(FileList, self).__init__()
        self.filename_ending = filename_ending
        self.drag_item = None
        self.drag_row = None

        if accept_drops:
            self.setAcceptDrops(True)
            self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.setWindowIcon(QtGui.QIcon(":/icons/icons/list-add.png"))


def get_filename(
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
        working_path = mfm.working_path
    filename = str(
        QtWidgets.QFileDialog.getOpenFileName(
            None,
            description,
            working_path,
            file_type
        )[0])
    mfm.working_path = os.path.dirname(filename)
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
        working_path = mfm.working_path
    filenames = QtWidgets.QFileDialog.getOpenFileNames(
        None,
        description,
        working_path,
        file_type
    )[0]
    mfm.working_path = os.path.dirname(filenames[0])
    return filenames


def save_file(
        description: str = '',
        file_type: str = 'All files (*.*)',
        working_path: str = None
):
    """Same as open see above a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = mfm.working_path
    filename = str(
        QtWidgets.QFileDialog.getSaveFileName(
            None,
            description,
            working_path,
            file_type
        )[0]
    )
    mfm.working_path = os.path.dirname(filename)
    return filename


def get_directory(**kwargs):
    """Opens a new window where you can choose a directory. The current working path
    is updated to this directory.

    It either returns the directory or the files within the directory (if get_files is True).
    The returned files can be filtered for the filename ending using the kwarg filename_ending.

    :return: directory str
    """
    fn_ending = kwargs.get('filename_ending', None)
    directory = kwargs.get('directory', mfm.working_path)
    get_files = kwargs.get('get_files', False)
    if isinstance(directory, str):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", directory))
    else:
        directory = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
    mfm.working_path = directory
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
    >>> d = yaml.load(open("./test_session.yaml"))['datasets']
    >>> od = collections.OrderedDict(sorted(d.items()))
    >>> w = make_widget_from_yaml(od, 'test')
    >>> w.show()
    :param variable_dictionary: 
    :param name: 
    :return: 
    """
    def make_group(d, name=''):
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


def set_app_style(
        app: QtCore,
        style_sheet_file: str
):
    with open(style_sheet_file, 'r') as fp:
        style_sheet = fp.read()
        app.setStyleSheet(style_sheet)
