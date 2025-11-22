from __future__ import annotations

from chisurf import typing
from qtpy import QtCore
from chisurf.gui import QtWidgets

import qtconsole
import qtconsole.qtconsoleapp
import qtconsole.inprocess
import qtconsole.styles
import qtconsole.manager

import chisurf.fio as io
import chisurf.gui
import chisurf.settings


class QIPythonWidget(
    qtconsole.qtconsoleapp.RichJupyterWidget
    # qtconsole.qtconsoleapp.JupyterWidget
):

    codeRequested = QtCore.Signal(str)

    def start_recording(self):
        self._macro = ""
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def run_macro(self, filename: str = None):
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                "Python macros",
                file_type="Python file (*.py)"
            )
        with io.zipped.open_maybe_zipped(
                filename=filename,
                mode='r'
        ) as fp:
            text = fp.read()
            self.execute(text, hidden=False)

    def save_macro(self, filename: str = None):
        self.stop_recording()
        if filename is None:
            filename = chisurf.gui.widgets.save_file(
                "Python macros",
                file_type="Python file (*.cm.py)"
            )
        with io.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            fp.write(self._macro)

    def do_execute(self, *args, **kwargs):
        super().do_execute(*args, **kwargs)
        # Record changes to history and file log
        new_text = self._history[-1] + '\n'
        with open(self.session_file, 'a+') as fp:
            fp.write(new_text)
        if self.recording:
            self._macro += new_text
        if isinstance(self.history_widget, QtWidgets.QPlainTextEdit):
            self.history_widget.insertPlainText(new_text)

    def __init__(
            self,
            history_widget: QtWidgets.QPlainTextEdit = None,
            recording: bool = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.history_widget = history_widget
        kernel_manager = qtconsole.inprocess.QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_client

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()

        self.exit_requested.connect(stop)
        self.width = kwargs.get('width', chisurf.settings.gui['console_width'])
        self._macro = ""
        self.recording = recording

        # Connect signal used for cross-thread execution of code
        try:
            self.codeRequested.connect(self._execute_from_signal)
        except Exception:
            pass

        # save nevertheless every input into a session file
        self.session_file = chisurf.settings.session_file
        self.set_default_style(chisurf.settings.gui['console_style'])
        self.style_sheet = qtconsole.styles.default_light_style_sheet

    def pushVariables(self, variableDict: typing.Dict[str, object]) -> None:
        """ Given a dictionary containing name / value pairs, push those
        variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()

    def printText(self, text: str):
        """ Prints some plain name to the console """
        self._append_plain_text(text)

    @QtCore.Slot(str)
    def _execute_from_signal(self, code: str) -> None:
        """Internal slot to execute code; runs on this widget's thread."""
        try:
            self.execute(code)
        except Exception:
            pass

    def execute_on_gui_thread(self, code: str = None):
        """Execute code via the IPython console on this widget's thread.

        This allows chisurf.run(...) to be called safely from any thread.
        """
        if code is None:
            return None
        try:
            code_str = str(code)
        except Exception:
            return None

        # If already on this widget's thread (typically the GUI thread), execute directly
        try:
            if QtCore.QThread.currentThread() is self.thread():
                return self.execute(code_str)
        except Exception:
            # If thread affinity check fails, fall back to direct execution
            try:
                return self.execute(code_str)
            except Exception:
                return None

        # Called from a non-GUI thread: emit signal; Qt will deliver it
        # to this widget on its own thread (queued connection).
        try:
            self.codeRequested.emit(code_str)
            return None
        except Exception:
            pass

        # Fallback: execute directly rather than silently dropping the command
        try:
            return self.execute(code_str)
        except Exception:
            return None
