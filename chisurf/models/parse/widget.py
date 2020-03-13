"""

"""
from __future__ import annotations

import os
from chisurf import typing

import yaml
from qtpy import QtWidgets, QtCore

import chisurf.decorators
import chisurf.fio
import chisurf.fitting
import chisurf.gui.decorators
import chisurf.models
import chisurf.settings
import chisurf.gui.widgets
from chisurf.models.model import ModelWidget
from chisurf.models.parse import ParseModel


class ParseFormulaWidget(
    QtWidgets.QWidget
):

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="parseWidget.ui"
    )
    def __init__(
            self,
            n_columns: int = None,
            model_file: str = None,
            model_name: str = None,
            model: chisurf.models.Model = None
    ):
        self.model = model
        if n_columns is None:
            n_columns = chisurf.settings.gui['fit_models']['n_columns']
        self.n_columns = n_columns

        self._models = {}
        if model_file is None:
            model_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models.yaml'
            )
        self._model_file = model_file
        self.load_model_file(model_file)

        if model_name is None:
            model_name = list(self._models)[0]

        self.model_name = model_name

        self.actionFormulaChanged.triggered.connect(self.onEquationChanged)
        self.actionModelChanged.triggered.connect(self.onModelChanged)

    def load_model_file(
            self,
            filename: str
    ):
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=filename,
                mode='r'
        ) as fp:
            self._model_file = filename
            self.models = yaml.safe_load(fp)

    @property
    def models(
            self
    ) -> typing.Dict:
        return self._models

    @models.setter
    def models(
            self,
            v: typing.Dict
    ):
        self._models = v
        self.comboBox.clear()
        self.comboBox.addItems(list(v.keys()))

    @property
    def model_name(
            self
    ) -> typing.List[str]:
        """

        :return:
        """
        return list(self.models.keys())[self.comboBox.currentIndex()]

    @model_name.setter
    def model_name(
            self,
            v: str
    ):
        idx = self.comboBox.findText(v)
        self.comboBox.setCurrentIndex(idx)

    @property
    def model_file(self) -> str:
        return self._model_file

    @model_file.setter
    def model_file(
            self,
            v: str
    ):
        self._model_file = v
        self.load_model_file(v)

    def onUpdateFunc(self):
        function_str = str(self.plainTextEdit.toPlainText()).strip()
        chisurf.run("chisurf.macros.parse.change_model('%s')" % function_str)
        try:
            ivs = self.models[self.model_name]['initial']
            for key in ivs.keys():
                self.model.parameter_dict[key].value = ivs[key]
        except (AttributeError, KeyError):
            print("No initial values")

    def onModelChanged(self):
        func = self.models[self.model_name]['equation']
        self.plainTextEdit.setPlainText(
            func
        )
        self.textEdit.setHtml(
            self.models[self.model_name]['description']
        )
        self.onUpdateFunc()
        self.onEquationChanged()

    def onEquationChanged(self):
        self.onUpdateFunc()
        layout = self.gridLayout_2
        chisurf.gui.widgets.clear_layout(layout)
        n_columns = self.n_columns
        row = 1

        for i, k in enumerate(
            sorted(
                self.model.parameters_all_dict.keys()
            )
        ):
            p = self.model.parameters_all_dict[k]
            pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=p
            )
            column = i % n_columns
            if column == 0:
                row += 1
            layout.addWidget(pw, row, column)

    def onLoadModelFile(
            self,
            filename: str = None
    ):
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                'Open models-file',
                'link file (*.yaml)'
            )
        self.load_model_file(
            filename=filename
        )


class ParseModelWidget(
    ParseModel,
    ModelWidget
):

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            *args,
            **kwargs
    ):
        super().__init__(
            fit,
            *args,
            **kwargs
        )
        parse = ParseFormulaWidget(
            model=self
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(
            parse
        )
        self.setLayout(
            layout
        )
        self.parse = parse

    def update_model(self, **kwargs):
        super().update_model(
            **kwargs
        )
