from __future__ import annotations

import pathlib
import pickle
import yaml

import typing

import chisurf.fio as io

from chisurf.gui import QtCore, QtWidgets
import chisurf.fitting
import chisurf.gui.widgets
import chisurf.gui.decorators
from chisurf.gui import uic

from chisurf import plots
from .model import ParameterTransformModel
from chisurf.models import model


class ParameterTransformWidget(model.ModelWidget, ParameterTransformModel):

    plot_classes = [
                    (plots.FitInfo, {}),
                    # (plots.ResidualPlot, {})
    ]

    def create_parameter_widgets(self):
        layout = self.w.gridLayout
        chisurf.gui.widgets.clear_layout(layout)

        n_columns = chisurf.settings.gui['fit_models']['n_columns']
        row = 1

        p_dict = self.parameters_all_dict
        p_keys = list(p_dict.keys())
        p_keys.sort()

        self.set_default_parameter_values()

        for i, pk in enumerate(p_keys):
            p = p_dict[pk]
            pw = chisurf.gui.widgets.fitting.make_fitting_parameter_widget(p, callback=self.finalize)
            column = i % n_columns
            if column == 0:
                row += 1
            layout.addWidget(pw, row, column)

    def set_default_parameter_values(self):
        d = self.codes[self.code_name]['initial']
        param_keys = list(self.parameters_all_dict.keys())

        for k in param_keys:
            initial = d.get(k, None)
            if initial is not None:
                self.parameters_all_dict[k].value = initial['value']
                self.parameters_all_dict[k].bounds = initial['bounds']
                self.parameters_all_dict[k].bounds_on = True

    @property
    def code_name(self):
        current_index = self.w.comboBox.currentIndex()

        if not self.codes:
            return ""

        code_keys = list(self.codes.keys())
        if not code_keys:
            return ""

        if current_index < 0 or current_index >= len(code_keys):
            return code_keys[0] if code_keys else ""

        return code_keys[current_index]

    @code_name.setter
    def code_name(self, v: str):
        idx = self.w.comboBox.findText(v)
        if idx == -1:
            return
        self.w.comboBox.setCurrentIndex(idx)

    @property
    def codes(self) -> typing.Dict:
        return self._codes

    @codes.setter
    def codes(self, v: typing.Dict):
        self._codes = v
        self.w.comboBox.clear()

        if not v:
            return

        code_keys = list(v.keys())
        self.w.comboBox.addItems(code_keys)

    def onCodeChanged(self):
        try:
            code = self.codes[self.code_name]['code']
            self.w.textEdit.setPlainText(code)
            self.onFunctionUpdate()
        except Exception as e:
            # Log the error and show a message to the user
            import logging
            logging.error(f"Error changing code: {str(e)}")
            # Show error message to the user
            from chisurf.gui import QtWidgets
            QtWidgets.QMessageBox.warning(
                self, 
                "Code Error", 
                f"Error loading code: {str(e)}\n\nPlease check the code definition in the YAML file."
            )

    def onFunctionUpdate(self):
        t = self.w.textEdit.toPlainText()
        try:
            self.function = str(t)
            self.create_parameter_widgets()
        except Exception as e:
            # Log the error and show a message to the user
            import logging
            logging.error(f"Error updating function: {str(e)}")
            # Show error message to the user
            from chisurf.gui import QtWidgets
            QtWidgets.QMessageBox.warning(
                self, 
                "Function Error", 
                f"Error in function definition: {str(e)}\n\nPlease correct the function and try again."
            )

    def load_model_file(self, filename: pathlib.Path):
        with io.open_maybe_zipped(filename, 'r') as fp:
            self._code_file = filename
            yaml_content = yaml.safe_load(fp)
            self.codes = yaml_content
            self.w.lineEdit.setText(str(filename.as_posix()))

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            *args,
            code_file: pathlib.Path = None,
            **kwargs
    ):
        super().__init__(fit, *args, **kwargs)

        path = pathlib.Path(__file__).parent.absolute()
        w = uic.loadUi(path / "parameter_transform.ui")

        l = QtWidgets.QVBoxLayout()
        self.setLayout(l)
        l.addWidget(w)
        self.w = w

        self._codes = {}
        if code_file is None:
            code_file = path / 'models.yaml'
        self._code_file = code_file.absolute().as_posix()

        self.load_model_file(code_file)

        self.w.actionFunctionUpdate.triggered.connect(self.onFunctionUpdate)
        self.w.actionCodeChanges.triggered.connect(self.onCodeChanged)

        self.w.checkBox.setChecked(False)
        self.w.comboBox.setCurrentIndex(1)
