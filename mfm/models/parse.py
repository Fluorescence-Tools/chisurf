from __future__ import annotations

import tempfile
from collections import defaultdict, OrderedDict
import sympy
import yaml
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import QFile, QFileInfo, QTextStream, QUrl
from numpy import *
from sympy.printing.latex import latex
from re import Scanner

import mfm
from mfm.models.model import ModelWidget, ModelCurve
from mfm.fitting.parameter import FittingParameter, FittingParameterGroup


class GenerateSymbols(defaultdict):
    def __missing__(self, key):
        return sympy.Symbol(key)


class ParseFormula(FittingParameterGroup):

    def __init__(self, **kwargs):
        FittingParameterGroup.__init__(self, **kwargs)

        self._keys = list()
        self._model_file = None
        self._models = dict()
        self._count = 0
        self._func = "x*0"

        self.model_file = kwargs.get('model_file', os.path.join(mfm.package_directory, 'settings/models.yaml'))
        self.model_name = kwargs.get('model_name', self.models.keys()[0])
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

    def var_found(self, scanner, name):
        if name in ['caller', 'e', 'pi']:
            return name
        if name not in self._keys:
            self._keys.append(name)
            ret = 'a[%d]' % self._count
            self._count += 1
        else:
            ret = 'a[%d]' % (self._keys.index(name))
        return ret

    def parse_code(self):

        code = self._func
        scanner = Scanner([
            (r"x", lambda y, x: x),
            (r"[a-zA-Z]+\.", lambda y, x: x),
            (r"[a-z]+\(", lambda y ,x: x),
            (r"[a-zA-Z_]\w*", self.var_found),
            (r"\d+\.\d*", lambda y, x: x),
            (r"\d+", lambda y, x: x),
            (r"\+|-|\*|/", lambda y, x: x),
            (r"\s+", None),
            (r"\)+", lambda y, x: x),
            (r"\(+", lambda y, x: x),
            (r",", lambda y, x: x),
            ])
        self._count = 0
        self._keys = []
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
        with open(filename, 'r') as fp:
            self._model_file = filename
            self.models = yaml.load(fp)

    def find_parameters(self):
        # do nothing
        pass


class ParseModel(ModelCurve):

    name = "Parse-Model"

    def __init__(self, fit, **kwargs):
        ModelCurve.__init__(self, fit, **kwargs)
        self.parse = kwargs.get('parse', ParseFormula())

    def update_model(self, **kwargs):
        a = [p.value for p in self.parse.parameters]
        x = self.fit.data.x
        try:
            y = eval(self.parse.code)
        except:
            y = zeros_like(x) + 1.0
        self._y = y


class ParseFormulaWidget(ParseFormula, QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('mfm/ui/models/parseWidget.ui', self)
        ParseFormula.__init__(self, **kwargs)
        self.n_columns = kwargs.get('n_columns', mfm.settings.cs_settings['gui']['fit_models']['n_columns'])

        #self.webview = QWebView()
        #self.verticalLayout_4.addWidget(self.webview)

        self.comboBox.currentIndexChanged[int].connect(self.onModelChanged)
        self.toolButton.clicked.connect(self.onUpdateFunc)
        self.func = self.models[self.model_name]['equation']

    @property
    def func(self):
        return ParseFormula.func.fget(self)

    @func.setter
    def func(self, v):
        ParseFormula.func.fset(self, v)

        self.plainTextEdit.setPlainText(v)
        layout = self.gridLayout_2
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
        self.comboBox.clear()
        self.comboBox.addItems(list(v.keys()))

    @property
    def model_name(self):
        return list(self.models.keys())[self.comboBox.currentIndex()]

    @model_name.setter
    def model_name(self, v):
        idx = self.comboBox.findText(v)
        self.comboBox.setCurrentIndex(idx)

    def onUpdateEquation(self):
        s = """<html><head>
            <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_SVG.js"></script>
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
        tempFile = QFile(tempfile.mktemp('.html'))
        tempFile.open(QFile.WriteOnly)
        stream = QTextStream(tempFile)
        stream << s
        tempFile.close()
        fileUrl = QUrl.fromLocalFile(QFileInfo(tempFile).canonicalFilePath())
        self.webview.load(fileUrl)

    def onUpdateFunc(self):
        function_str = str(self.plainTextEdit.toPlainText())
        mfm.run("cs.current_fit.models.parse.func = '%s'" % function_str)
        mfm.run("cs.current_fit.update()")
        self.onUpdateEquation()

    def onModelChanged(self):
        mfm.run("cs.current_fit.models.parse.model_name = '%s'" % self.model_name)
        mfm.run("cs.current_fit.models.parse.func = '%s'" % self.models[self.model_name]['equation'])
        mfm.run("cs.current_fit.update()")
        self.onUpdateEquation()

    def onLoadModelFile(self, filename=None):
        if filename is None:
            filename = mfm.widgets.get_filename('Open models-file', 'link file (*.yaml)')
        mfm.run("cs.current_fit.models.parse.load_model_file(%s)" % filename)


class ParseModelWidget(ParseModel, ModelWidget):

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit, **kwargs)
        parse = ParseFormulaWidget(**kwargs)
        ParseModel.__init__(self, fit=fit, parse=parse)
        #self.update()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(self.parse)

    def update_model(self, **kwargs):
        ParseModel.update_model(self, **kwargs)

