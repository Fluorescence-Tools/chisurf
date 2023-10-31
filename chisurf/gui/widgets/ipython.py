from __future__ import annotations

import inspect

from chisurf import typing
from qtpy import QtWidgets

import qtconsole
import qtconsole.qtconsoleapp
import qtconsole.inprocess
import qtconsole.styles
import qtconsole.manager

import scikit_fluorescence.io
import chisurf.fio
import chisurf.gui
import chisurf.settings


class QIPythonWidget(
    qtconsole.qtconsoleapp.RichJupyterWidget
    # qtconsole.qtconsoleapp.JupyterWidget
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
            filename = chisurf.gui.widgets.get_filename(
                "Python macros",
                file_type="Python file (*.py)"
            )
        with scikit_fluorescence.io.zipped.open_maybe_zipped(
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
            filename = chisurf.gui.widgets.save_file(
                "Python macros",
                file_type="Python file (*.py)"
            )
        with scikit_fluorescence.io.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            fp.write(self._macro)

    def do_execute(self, source, complete, indent):
        if complete:
            self._append_plain_text('\n')
            self._input_buffer_executing = self.input_buffer
            self._executing = True
            self._finalize_input_request()

            # Perform actual execution.
            self._execute(source, False)

        else:
            # Do this inside an edit block so continuation prompts are
            # removed seamlessly via undo/redo.
            cursor = self._get_end_cursor()
            cursor.beginEditBlock()
            try:
                cursor.insertText('\n')
                self._insert_continuation_prompt(cursor, indent)
            finally:
                cursor.endEditBlock()

            # Do not do this inside the edit block. It works as expected
            # when using a QPlainTextEdit control, but does not have an
            # effect when using a QTextEdit. I believe this is a Qt bug.
            self._control.moveCursor(QtGui.QTextCursor.End)

        # Record changes to history and file log
        new_text = source + '\n'
        with open(self.session_file, 'a+') as fp:
            fp.write(new_text)
        if self.recording:
            self._macro += new_text
        if isinstance(self.history_widget, QtWidgets.QPlainTextEdit):
            self.history_widget.insertPlainText(new_text)

    def __init__(
            self,
            history_widget: QtWidgets.QPlainTextEdit = None,
            *args,
            inprocess_kernel: bool = True,
            **kwargs
    ) -> None:
        """

        :param history_widget:
        :param args:
        :param inprocess_kernel: if True the ipython console will be a associated to an inproccess
        kernel. An inprocess kernel can have access to variables and attributes in the UI.

        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        self.history_widget = history_widget
        if inprocess_kernel:
            kernel_manager = qtconsole.inprocess.QtInProcessKernelManager()
            kernel_manager.start_kernel()
            kernel_client = kernel_manager.client()
            kernel_client.start_channels()
        else:
            # In future the inproccess kernel should be replaced with a separated
            # kernel and the UI only inputs commands to the kernel.
            kernel_manager = qtconsole.manager.QtKernelManager(kernel_name="python3")
            kernel_manager.start_kernel()
            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_client

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()

        self.exit_requested.connect(stop)
        self.width = kwargs.get(
            'width',
            chisurf.settings.gui['console_width']
        )
        self._macro = ""
        self.recording = False

        # save nevertheless every inputs into a session file
        self.session_file = chisurf.settings.session_file
        self.set_default_style(
            chisurf.settings.gui['console_style']
        )
        self.style_sheet = qtconsole.styles.default_light_style_sheet

    def pushVariables(
            self,
            variableDict: typing.Dict[str, object]
    ) -> None:
        """ Given a dictionary containing name / value pairs, push those
        variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()

    def printText(
            self,
            text: str
    ):
        """ Prints some plain name to the console """
        self._append_plain_text(text)

