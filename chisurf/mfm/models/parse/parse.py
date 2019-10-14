from __future__ import annotations
from typing import List

import os
import tempfile
from collections import defaultdict, OrderedDict
import sympy
import yaml
from numpy import *
from sympy.printing.latex import latex
from re import Scanner

from qtpy import QtCore, QtWidgets, QtWebEngineWidgets
from qtpy.QtCore import QFile, QFileInfo, QTextStream, QUrl

import mfm
import mfm.decorators
import mfm.widgets
from mfm.models.model import ModelWidget, ModelCurve
from mfm.fitting.parameter import FittingParameter, FittingParameterGroup


class GenerateSymbols(defaultdict):

    def __missing__(self, key):
        return sympy.Symbol(key)


class ParseFormula(FittingParameterGroup):

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit = None,
            model: mfm.models.model.Model = None,
            short: str = '',
            parameters: List[mfm.fitting.parameter.FittingParameter] = None,
            model_file: str = None,
            model_name: str = None,
            **kwargs
    ):
        super().__init__(
            fit=fit,
            model=model,
            short=short,
            parameters=parameters,
            **kwargs
        )

        self._keys = list()
        self._model_file = None
        self._models = dict()
        self._count = 0
        self._func = "x*0"

        if model_file is None:
            model_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models.yaml'
            )
        self.model_file = model_file

        if model_name is None:
            model_name = list(self.models)[0]
        self.model_name = model_name

        self.code = self._func

    @property
    def initial_values(self):
        try:
            ivs = self.models[self.model_name]['initial']
        except AttributeError:
            ivs = OrderedDict([(k, 1.0) for k in self._keys])
        return ivs

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, v):
        self._models = v

    @property
    def model_file(self):
        return self._model_file

    @model_file.setter
    def model_file(self, v):
        self._model_file = v
        self.load_model_file(v)

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, v):
        self._func = v
        self.parse_code()

    def parse_code(self):

        def var_found(
                scanner,
                name: str
        ):
            if name in ['caller', 'e', 'pi']:
                return name
            if name not in self._keys:
                self._keys.append(name)
                ret = 'a[%d]' % self._count
                self._count += 1
            else:
                ret = 'a[%d]' % (self._keys.index(name))
            return ret

        code = self._func
        scanner = Scanner([
            (r"x", lambda y, x: x),
            (r"[a-zA-Z]+\.", lambda y, x: x),
            (r"[a-z]+\(", lambda y, x: x),
            (r"[a-zA-Z_]\w*", var_found),
            (r"\d+\.\d*", lambda y, x: x),
            (r"\d+", lambda y, x: x),
            (r"\+|-|\*|/", lambda y, x: x),
            (r"\s+", None),
            (r"\)+", lambda y, x: x),
            (r"\(+", lambda y, x: x),
            (r",", lambda y, x: x),
        ])
        self._count = 0
        self._keys = list()
        parsed, rubbish = scanner.scan(code)
        parsed = ''.join(parsed)
        if rubbish != '':
            raise Exception('parsed: %s, rubbish %s' % (parsed, rubbish))
        self.code = parsed

        # Define parameters
        self._parameters = list()
        for key in self._keys:
            try:
                iv = self.initial_values[key]
            except KeyError:
                iv = 1.0
            p = FittingParameter(name=key, value=iv)
            self._parameters.append(p)

    def load_model_file(self, filename):
        with mfm.io.zipped.open_maybe_zipped(
                filename=filename,
                mode='r'
        ) as fp:
            self._model_file = filename
            self.models = yaml.safe_load(fp)

    def find_parameters(
            self,
            parameter_type=mfm.parameter.Parameter
    ):
        # do nothing
        pass


class ParseModel(ModelCurve):

    name = "Parse-Model"

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            *args,
            parse: object = None,
            **kwargs
    ):
        super().__init__(
            fit,
            *args,
            **kwargs
        )
        if parse is None:
            parse = ParseFormula()
        self.parse = parse

    def update_model(self, **kwargs):
        a = [p.value for p in self.parse.parameters]
        x = self.fit.data.x
        try:
            y = eval(self.parse.code)
        except:
            y = zeros_like(x) + 1.0
        self._y = y


class ParseFormulaWidget(
    ParseFormula,
    QtWidgets.QWidget
):

    @mfm.decorators.init_with_ui(ui_filename="parseWidget.ui")
    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            model: mfm.models.model.Model,
            short: str = '',
            parameters: List[mfm.fitting.parameter.FittingParameter] = None,
            n_columns: int = None,
            **kwargs
    ):
        if n_columns is None:
            n_columns = mfm.settings.gui['fit_models']['n_columns']
        self.n_columns = n_columns

        self.webview = QtWebEngineWidgets.QWebEngineView()
        self.widget.verticalLayout_4.addWidget(self.webview)

        self.widget.comboBox.currentIndexChanged[int].connect(self.onModelChanged)
        self.widget.toolButton.clicked.connect(self.onUpdateFunc)
        self._func = self.models[self.model_name]['equation']
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(
            layout
        )
        layout.addWidget(self.widget)

    @property
    def func(self):
        return super(ParseFormulaWidget, self.__class__).func

    @func.setter
    def func(self, v):
        super(ParseFormulaWidget, self.__class__).func = v

        self.widget.plainTextEdit.setPlainText(v)
        layout = self.widget.gridLayout_2
        mfm.widgets.clear_layout(layout)
        n_columns = self.n_columns

        pn = list()
        row = 1
        for i, p in enumerate(self._parameters):
            pw = p.make_widget()
            column = i % n_columns
            if column == 0:
                row += 1
            layout.addWidget(pw, row, column)
            pn.append(pw.fitting_parameter)
        self._parameters = pn

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, v):
        ParseFormula.models.fset(self, v)
        self.widget.comboBox.clear()
        self.widget.comboBox.addItems(list(v.keys()))

    @property
    def model_name(self) -> List[str]:
        return list(self.models.keys())[self.widget.comboBox.currentIndex()]

    @model_name.setter
    def model_name(
            self,
            v: str
    ):
        idx = self.widget.comboBox.findText(v)
        self.widget.comboBox.setCurrentIndex(idx)

    def onUpdateEquation(self):
        s = """<html><head>
            <script type="name/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_SVG.js"></script>
            </head><body>
            <link rel="stylesheet" href="http://yui.yahooapis.com/pure/0.6.0/pure-min.css">
            <h3>%s</h3>
            <p><mathjax>
            $$
            """ % self.model_name
        try:
            f = eval(self.func, GenerateSymbols())
            s += latex(f)
        except:
            s += "Error"
        s += "$$</mathjax></p>"
        s += self.models[self.model_name]['description']
        s += "</body></html>"
        file = tempfile.TemporaryFile(
            suffix='.html'
        )
        tempFile = QFile(
            file.name
        )
        tempFile.open(QFile.WriteOnly)
        stream = QTextStream(tempFile)
        stream << s
        tempFile.close()
        fileUrl = QUrl.fromLocalFile(
            QFileInfo(
                tempFile
            ).canonicalFilePath()
        )
        self.webview.load(fileUrl)

    def onUpdateFunc(self):
        function_str = str(self.plainTextEdit.toPlainText())
        mfm.run(
            "\n".join(
                [
                    "cs.current_fit.model.parse.func = '%s'" % function_str,
                    "cs.current_fit.update()"
                ]
            )
        )
        self.onUpdateEquation()

    def onModelChanged(self):
        mfm.run(
            "\n".join(
                [
                    "cs.current_fit.model.parse.model_name = '%s'" %
                    self.model_name,
                    "cs.current_fit.model.parse.func = '%s'" %
                    self.models[self.model_name]['equation'],
                    "cs.current_fit.update()"
                ]
            )
        )
        self.onUpdateEquation()

    def onLoadModelFile(
            self,
            filename: str = None
    ):
        if filename is None:
            filename = mfm.widgets.get_filename(
                'Open models-file',
                'link file (*.yaml)'
            )
        mfm.run(
            "cs.current_fit.model.parse.load_model_file(%s)" % filename
        )


class ParseModelWidget(ParseModel, ModelWidget):

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            **kwargs
    ):
        ModelWidget.__init__(self, fit, **kwargs)
        parse = ParseFormulaWidget(
            fit=fit,
            model=self,
            **kwargs
        )
        ParseModel.__init__(
            self,
            fit=fit,
            parse=parse
        )
        #self.update()

        #self.layout = QtWidgets.QVBoxLayout(self)
        #self.layout.setAlignment(QtCore.Qt.AlignTop)
        #self.layout.addWidget(self.parse)

    def update_model(self, **kwargs):
        ParseModel.update_model(self, **kwargs)

