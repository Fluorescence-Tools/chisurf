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
    #qtconsole.qtconsoleapp.JupyterWidget
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

    def execute(
            self,
            *args,
            hidden: bool = None,
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
        if hidden is None:
            hidden = chisurf.settings.cs_settings.get('hidden_execute', False)
        kwargs['hidden'] = hidden
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
            function: typing.Callable,
    ) -> None:
        """ Gets the function string executes the function on the command line

        :param function: A callable function that will executed in the ipython
        environment.

        :return:
        """
        self.execute(
            inspect.getsource(function)
        )

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

