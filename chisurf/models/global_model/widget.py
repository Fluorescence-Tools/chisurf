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

from chisurf import plots
from .globalfit import GlobalFitModel
from .parse import ParameterTransformModel
from chisurf.models import model


class GlobalFitModelWidget(GlobalFitModel, model.ModelWidget):

    plot_classes = [
                    (plots.FitInfo, {}),
                    # (plots.ResidualPlot, {})
    ]

    @chisurf.gui.decorators.init_with_ui(ui_filename="globalfit.ui")
    def __init__(self, fit: chisurf.fitting.fit.Fit):
        self.pushButton_3.clicked.connect(self.onSaveTable)
        self.pushButton_4.clicked.connect(self.onLoadTable)
        self.pushButton_5.clicked.connect(self.clear_listed_links)

        self.pushButton_8.clicked.connect(self.setLinks)
        self.addGlobalLink.clicked.connect(self.onAddLink)
        self.comboBox_gfOriginFit.currentIndexChanged[int].connect(self.update_parameter_origin)
        self.comboBox_gfTargetFit.currentIndexChanged[int].connect(self.update_parameter_target)
        self.comboBox_gfTargetParameter.currentIndexChanged[int].connect(self.update_link_text)
        self.comboBox_gfOriginParameter.currentIndexChanged[int].connect(self.update_link_text)
        self.table_GlobalLinks.cellDoubleClicked[int, int].connect(self.onTableGlobalLinksDoubleClicked)
        self.tableWidget.cellDoubleClicked[int, int].connect(self.onRemoveLocalFit)
        self.checkBox_2.stateChanged[int].connect(self.update_parameter_origin)

        self.actionOnClearVariables.triggered.connect(self.onClearVariables)
        self.actionOnAddToLocalFitList.triggered.connect(self.onAddToLocalFitList)
        self.actionOn_clear_local_fits.triggered.connect(self.onClearLocalFits)
        self.actionUpdate_widgets.triggered.connect(self.update_widgets)
        self.actionOnAddGlobalVariable.triggered.connect(self.onAddGlobalVariable)

    @property
    def current_origin_formula(self) -> str:
        return str(self.lineEdit_3.text())

    @property
    def add_all_fits(self) -> bool:
        return bool(self.checkBox.isChecked())

    @property
    def current_global_variable_name(self) -> str:
        return str(self.lineEdit.text())

    @property
    def current_fit_index(self) -> int:
        return self.comboBox.currentIndex()

    @property
    def link_all_of_type(self) -> bool:
        return not self.checkBox_2.isChecked()

    @property
    def clear_on_update(self) -> bool:
        return self.checkBox_3.isChecked()

    @clear_on_update.setter
    def clear_on_update(self, v: bool):
        self.checkBox_3.setChecked(v)

    @property
    def local_fits(self) -> typing.List[chisurf.fitting.fit.Fit]:
        return [
            s for s in chisurf.fits
            if isinstance(s, chisurf.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_idx(self) -> typing.List[int]:
        return [
            i for i, s in enumerate(chisurf.fits)
            if isinstance(s, chisurf.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_names(self) -> typing.List[str]:
        return [f.name for f in self.local_fits]

    @property
    def origin_fit_number(self) -> int:
        return int(self.comboBox_gfOriginFit.currentIndex())  # origin fit fit_index

    @property
    def origin_fit(self) -> chisurf.fitting.fit.Fit:
        ofNbr = self.origin_fit_number
        return self.fits[ofNbr]

    @property
    def origin_parameter(self) -> chisurf.fitting.parameter.FittingParameter:
        return self.origin_fit.model.parameters_all_dict[self.origin_parameter_name]

    @property
    def origin_parameter_name(self) -> str:
        return str(self.comboBox_gfOriginParameter.currentText())

    @property
    def target_fit_number(self) -> int:
        return int(self.comboBox_gfTargetFit.currentIndex())  # target fit fit_index

    @property
    def target_fit(self) -> chisurf.fitting.fit.Fit:
        tfNbr = self.target_fit_number
        return self.fits[tfNbr]

    @property
    def target_parameter_name(self) -> str:
        return str(self.comboBox_gfTargetParameter.currentText())

    @property
    def target_parameter(self) -> chisurf.fitting.parameter.FittingParameter:
        return self.target_fit.model.parameters_all_dict[self.target_parameter_name]

    @property
    def current_link_formula(self):
        return "f[%s]['%s']" % (self.target_fit_number, self.target_parameter_name)

    @property
    def current_origin_link_formula(self):
        if self.link_all_of_type:
            return "f[i]['%s']" % (self.origin_parameter_name)
        else:
            return "f[%s]['%s']" % (self.origin_fit_number, self.origin_parameter_name)

    @property
    def links(self):
        table = self.table_GlobalLinks
        links = []
        for r in range(table.rowCount()):
            # self.tableWidget_2.item(r, 2).data(0).toInt()
            en = bool(table.cellWidget(r, 0).checkState())
            fitA = int(table.item(r, 1).data(0)) - 1
            pA = str(table.item(r, 2).text())
            fB = str(table.item(r, 3).text())
            links.append([en, fitA, pA, fB])
        return links

    @property
    def local_fit_first(self) -> bool:
        return self.checkBoxLocal.isChecked()

    @local_fit_first.setter
    def local_fit_first(self, v: bool):
        if v is True:
            self.checkBoxLocal.setCheckState(2)
        else:
            self.checkBoxLocal.setCheckState(0)

    def update_link_text(self):
        self.lineEdit_2.setText(self.current_link_formula)
        self.lineEdit_3.setText(self.current_origin_link_formula)

    def onRemoveLocalFit(self) -> None:
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        chisurf.run("cs.current_fit.model.remove_local_fit(%s)" % row)

    def onClearLocalFits(self) -> None:
        chisurf.run("cs.current_fit.model.clear_local_fits()")
        self.tableWidget.setRowCount(0)

    def onTableGlobalLinksDoubleClicked(self) -> None:
        row = self.table_GlobalLinks.currentRow()
        self.table_GlobalLinks.removeRow(row)

    def onAddGlobalVariable(self) -> None:
        variable_name = self.current_global_variable_name
        if len(variable_name) > 0 and variable_name not in list(self._global_parameters.keys()):
            chisurf.run(
                "cs.current_fit.model.append_global_parameter(chisurf.parameter.FittingParameterWidget(name='%s'))" %
                self.current_global_variable_name
            )
            layout = self.verticalLayout
            layout.addWidget(self._global_parameters.values()[-1])
        else:
            chisurf.logging.warning("onAddGlobalVariable: No variable name defined.")

    def onClearVariables(self) -> None:
        chisurf.logging.info("onClearVariables")
        self._global_parameters = dict()
        layout = self.verticalLayout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

    def onAddToLocalFitList(self) -> None:
        local_fits = self.local_fits
        local_fits_idx = self.local_fit_idx
        fit_indeces = range(len(local_fits)) if self.add_all_fits else [self.current_fit_index]
        for fitIndex in fit_indeces:
            chisurf.run(
                "cs.current_fit.model.append_fit(chisurf.fits[%s])" % local_fits_idx[fitIndex]
            )

    def append_fit(self, fit: chisurf.fitting.fit):
        if fit not in self.fits:

            table = self.tableWidget
            table.insertRow(table.rowCount())
            rc = table.rowCount() - 1

            tmp = QtWidgets.QTableWidgetItem(fit.name)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 0, tmp)

            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            table.resizeRowsToContents()

            self.update_widgets()

        GlobalFitModel.append_fit(self, fit)

    def onAddLink(self, links: typing.List = None):
        table = self.table_GlobalLinks
        if links is None:
            links = []
            if self.link_all_of_type:
                chisurf.logging.info("Link all of one kind: %s" % self.link_all_of_type)
                for fit_nbr, fit in enumerate(self.fits):
                    fit = self.fits[fit_nbr]
                    pn = [p.name for p in fit.model.parameters_all]
                    if self.origin_parameter_name not in pn:
                        continue
                    origin_parameter = self.fits[fit_nbr].model.parameters_all_dict[self.origin_parameter_name]
                    if origin_parameter is self.target_parameter:
                        continue
                    links.append(
                        [True, fit_nbr, origin_parameter.name,
                         self.current_target_formula]
                    )
            else:
                links.append(
                    [True, self.origin_fit_number, self.origin_parameter_name,
                     self.current_target_formula]
                )
        for link in links:
            en, origin_fit, origin_parameter, formula = link

            rc = table.rowCount()
            table.insertRow(table.rowCount())

            cbe = QtWidgets.QCheckBox(table)
            cbe.setChecked(en)
            table.setCellWidget(rc, 0, cbe)
            table.resizeRowsToContents()
            cbe.setChecked(True)

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setData(0, int(origin_fit + 1))
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 1, tmp)

            tmp = QtWidgets.QTableWidgetItem(origin_parameter)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 2, tmp)

            tmp = QtWidgets.QTableWidgetItem(formula)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 3, tmp)

    def update_parameter_origin(self):
        self.comboBox_gfOriginParameter.clear()
        if len(self.fits) > 0:
            if not self.link_all_of_type:
                pn = [p.name for p in self.fit.model.parameters_all]
                pn.sort()
                self.comboBox_gfOriginParameter.addItems([p for p in pn])
            else:
                names = set([p.name for f in self.fits for p in f.model.parameters_all])
                names = list(names)
                names.sort()
                self.comboBox_gfOriginParameter.addItems(names)

    def update_parameter_target(self):
        self.comboBox_gfTargetParameter.clear()
        if len(self.fits) > 0:
            ftIndex = self.comboBox_gfTargetFit.currentIndex()
            ft = self.fits[ftIndex]
            pn = [p.name for p in self.fit.model.parameters_all]
            pn.sort()
            self.comboBox_gfTargetParameter.addItems([p.name for p in ft.model.parameters_all])

    def update_widgets(self):
        self.comboBox.clear()
        self.comboBox.addItems(self.local_fit_names)

        self.comboBox_gfOriginFit.clear()
        self.comboBox_gfTargetFit.clear()
        usedLocalFitNames = [str(i + 1) for i, f in enumerate(self.fits)]
        self.comboBox_gfOriginFit.addItems(usedLocalFitNames)
        self.comboBox_gfTargetFit.addItems(usedLocalFitNames)

    def onSaveTable(self):
        filename = chisurf.gui.widgets.save_file(
            description='Save link-table',
            file_type='.p'
        )
        pickle.dump(self.links, open(filename, "wb"))

    def onLoadTable(self):
        filename = chisurf.gui.widgets.get_filename(
            description='Open link-table',
            file_type='link file (*.p)'
        )
        with open(filename, "rb") as fp:
            links = pickle.load(fp)
        self.onAddLink(links)

    def clear_listed_links(self):
        self.table_GlobalLinks.setRowCount(0)


class ParameterTransformWidget(ParameterTransformModel, model.ModelWidget):

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
        for k in self.parameters_all_dict.keys():
            initial = d.get(k, None)
            if initial is not None:
                self.parameters_all_dict[k].value = initial['value']
                self.parameters_all_dict[k].bounds = initial['bounds']
                self.parameters_all_dict[k].bounds_on = True

    @property
    def code_name(self):
        return list(self.codes.keys())[self.w.comboBox.currentIndex()]

    @code_name.setter
    def code_name(self, v: str):
        idx = self.w.comboBox.findText(v)
        self.w.comboBox.setCurrentIndex(idx)

    @property
    def codes(self) -> typing.Dict:
        return self._codes

    @codes.setter
    def codes(self, v: typing.Dict):
        self._codes = v
        self.w.comboBox.clear()
        self.w.comboBox.addItems(list(v.keys()))

    def onCodeChanged(self):
        code = self.codes[self.code_name]['code']
        self.w.textEdit.setPlainText(code)
        self.onFunctionUpdate()

    def onFunctionUpdate(self):
        t = self.w.textEdit.toPlainText()
        self.function = str(t)
        self.create_parameter_widgets()

    def load_model_file(self, filename: pathlib.Path):
        with io.open_maybe_zipped(filename, 'r') as fp:
            self._code_file = filename
            self.codes = yaml.safe_load(fp)
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

        w = chisurf.gui.uic.loadUi(path / "parameter_transform.ui")
        l = chisurf.gui.QtWidgets.QVBoxLayout()
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

