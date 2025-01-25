import inspect
import os
import pathlib

import chisurf
from chisurf import typing

from qtpy import QtWidgets, uic


class init_with_ui(object):
    """
    This is a decorator for __init__ methods of QtWidget objects.
    The decorator accepts an ui_filename, calls the super class of
    the object, and initializes the ui file to a target specified
    by the target option. If no target is provided the ui file
    is initialized into the object of the __init__ function.

    """

    def __init__(self, ui_filename: str, path: str = None):
        """

        :param ui_filename: The filename (without path) of the ui file.
        It is assumed that the ui file is in the same path as the file
        of the class that is being initialized.

        """
        self.ui_filename = ui_filename
        self.path = path

    def __call__(self, f: typing.Callable):

        def load_ui(target: QtWidgets.QWidget, ui_filename: str, path: str):
            path = pathlib.Path(path) / ui_filename
            uic.loadUi(path, target)

        def wrapped(cls: QtWidgets.QWidget, *args, **kwargs):
            if self.path is None:
                path = pathlib.Path(inspect.getfile(cls.__class__)).parent
            else:
                path = self.path

            # Call superclass __init__ method if it exists
            try:
                super(cls.__class__, cls).__init__(*args, **kwargs)
            except TypeError:
                super(cls.__class__, cls).__init__()

            super(QtWidgets.QWidget, cls).__init__()
            load_ui(cls, self.ui_filename, path)

            f(cls, *args, **kwargs)

        return wrapped



# reference taken from : http://stackoverflow.com/questions/11872141/drag-a-file-into-qtgui-qlineedit-to-set-url-text
# https://stackoverflow.com/questions/11872141/drag-a-file-into-qtgui-qlineedit-to-set-url-text
class lineEdit_dragFile_injector():
    def __init__(self, lineEdit, auto_inject=True, call=None, target: typing.List[pathlib.Path] = None, sorted: bool = True):
        self.lineEdit = lineEdit
        self.call = call
        self.sorted = sorted
        self.target = target if target is not None else []
        if auto_inject:
            self.inject_dragFile()

    def inject_dragFile(self):
        self.lineEdit.setDragEnabled(True)
        self.lineEdit.dragEnterEvent = self._dragEnterEvent
        self.lineEdit.dragMoveEvent = self._dragMoveEvent
        self.lineEdit.dropEvent = self._dropEvent

    def _dragEnterEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            event.acceptProposedAction()

    def _dragMoveEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            event.acceptProposedAction()

    def _dropEvent(self, event):
        data = event.mimeData()
        if isinstance(self.target, list):
            for url in data.urls():
                if url.scheme() == 'file':
                    filepath = str(url.toLocalFile())
                    filepath = pathlib.Path(filepath).as_posix()
                    chisurf.logging.log(0, f'lineEdit_dragFile_injector::_dropEvent: {filepath}')
                    self.target.append(filepath)

            # Sort the target list of file paths
            if self.sorted:
                self.target.sort()

        if data.urls()[0].scheme() == 'file':
            filepath = str(data.urls()[0].toLocalFile())
            filepath = pathlib.Path(filepath).as_posix()
            self.lineEdit.setText(filepath)

        if callable(self.call):
            self.call()
