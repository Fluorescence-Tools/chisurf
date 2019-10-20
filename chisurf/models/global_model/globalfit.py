from __future__ import annotations
from typing import List, Tuple, Dict

import pickle
import threading
from collections import OrderedDict
from qtpy import QtCore, QtWidgets

import numpy as np

import chisurf.settings as mfm
import chisurf.decorators
from chisurf import plots
from chisurf.curve import Curve
import chisurf.fitting.fit
from chisurf.models import model
from chisurf.fitting.parameter import GlobalFittingParameter


class GlobalFitModel(model.Model, Curve):
    """

    """

    name = "Global fit"

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            **kwargs
    ):
        self.fits = []
        self.fit = fit
        self._global_parameters = OrderedDict()
        self.parameters_calculated = list()
        self._links = list()
        super().__init__(
            fit,
            **kwargs
        )

    @property
    def weighted_residuals(
            self
    ) -> np.array:
        re = list()
        for f in self.fits:
            re.append(f.model.weighted_residuals.flatten())
        return np.concatenate(re)

    @property
    def fit_names(
            self
    ) -> List[str]:
        return [f.name for f in self.fits]

    @property
    def links(
            self
    ) -> List[fitting.parameter.FittingParameter]:
        return self._links

    @links.setter
    def links(
            self,
            v: List[fitting.parameter.FittingParameter]
    ):
        self._links = v if isinstance(v, list) else list()

    @property
    def n_points(self) -> int:
        nbr_points = 0
        for f in self.fits:
            nbr_points += f.model.n_points
        return nbr_points

    @property
    def global_parameters_all(
            self
    ) -> List[fitting.parameter.FittingParameter]:
        return list(self._global_parameters.values())

    @property
    def global_parameters_all_names(
            self
    ) -> List[str]:
        return [p.name for p in self.global_parameters_all]

    @property
    def global_parameters(
            self
    ) -> List[fitting.parameter.FittingParameter]:
        return [p for p in self.global_parameters_all if not p.fixed]

    @property
    def global_parameters_names(
            self
    ) -> List[str]:
        return [p.name for p in self.global_parameters]

    @property
    def global_parameters_bound_all(
            self
    ) -> List[
        Tuple[float, float]
    ]:
        return [pi.bounds for pi in self.global_parameters_all]

    @property
    def global_parameter_linked_all(
            self
    ) -> List[bool]:
        return [p.is_linked for p in self.global_parameters_all]

    @property
    def parameters(
            self
    ) -> List[fitting.parameter.FittingParameter]:
        p = list()
        for f in self.fits:
            p += f.model.parameters
        p += self.global_parameters
        return p

    @property
    def parameter_names(
            self
    ) -> List[str]:
        try:
            re = list()
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += ["%i:%s" % (i + 1, p.name) for p in f.model.parameters]
            re += self.global_parameters_names
            return re
        except AttributeError:
            return list()

    @property
    def parameters_all(
            self
    ) -> List[fitting.parameter.FittingParameter]:
        try:
            re = list()
            for f in self.fits:
                if f.model is not None:
                    re += [p for p in f.model.parameters_all]
            re += self.global_parameters_all
            return re
        except AttributeError:
            return []

    @property
    def global_parameters_values_all(
            self
    ) -> List[float]:
        return [g.value for g in self.global_parameters_all]

    @property
    def global_parameters_fixed_all(
            self
    ) -> List[bool]:
        return [p.fixed for p in self.global_parameters_all]

    @property
    def parameter_names_all(
            self
    ) -> List[str]:
        try:
            re = list()
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += ["%i:%s" % (i + 1, p.name) for p in f.model._parameters]
            re += self.global_parameters_all_names
            return re
        except AttributeError:
            return []

    @property
    def parameter_dict(
            self
    ) -> Dict:
        re = dict()
        for i, f in enumerate(self.fits):
            d = f.model.parameter_dict
            k = [str(i+1)+":"+dk for dk in d.keys()]
            for j, di in enumerate(d.keys()):
                re[k[j]] = d[di]
        return re

    @property
    def data(
            self
    ) -> Tuple[
        np.array,
        np.array,
        np.array
    ]:
        d = list()
        w = list()
        for f in self.fits:
            x, di, wi = f.data[0:-1]
            d.append(di)
            w.append(wi)
        dn = np.hstack(d)
        wn = np.hstack(w)
        xn = np.arange(0, dn.shape[0], 1)
        return xn, dn, wn

    def get_wres(
            self,
            fit: chisurf.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None,
            **kwargs
    ) -> np.array:
        try:
            f = fit
            if xmin is None:
                xmin = f.xmin
            if xmax is None:
                xmax = f.xmax
            x, m = f.model[xmin:xmax]
            x, d, w = f.model.data[xmin:xmax]
            ml = min([len(m), len(d)])
            wr = np.array((d[:ml] - m[:ml]) * w[:ml], dtype=np.float64)
        except:
            wr = np.array([1.0])
        return wr

    def append_fit(
            self,
            fit: chisurf.fitting.fit.Fit
    ) -> None:
        if fit not in self.fits:
            self.fits.append(fit)

    def append_global_parameter(
            self,
            parameter: chisurf.parameter.Parameter
    ) -> None:
        variable_name = parameter.name
        if variable_name not in list(self._global_parameters.keys()):
            self._global_parameters[parameter.name] = parameter

    def setLinks(self):
        self.parameters_calculated = list()
        if self.clear_on_update:
            self.clear_all_links()
        f = [fit.model.parameters_all_dict for fit in self.fits]
        g = self._global_parameters
        for link in self.links:
            en, origin_fit, origin_name, formula = link
            if not en:
                continue
            try:
                origin_parameter = f[origin_fit][origin_name]
                target_parameter = GlobalFittingParameter(f, g, formula)

                origin_parameter.link = target_parameter
                print("f[%s][%s] linked to %s" % (origin_fit, origin_parameter.name, target_parameter.name))
            except IndexError:
                print("not enough fits index out of range")

    def autofitrange(
            self,
            fit: chisurf.fitting.fit.FitGroup
    ):
        self.xmin, self.xmax = None, None
        return self.xmin, self.xmax

    def clear_local_fits(
            self
    ) -> None:
        self.fits = list()

    def remove_local_fit(
            self,
            fit_index: int
    ):
        del self.fits[fit_index]

    def clear_all_links(
            self
    ) -> None:
        for fit in self.fits:
            for p in fit.model.parameters_all:
                p.link = None

    def clear_listed_links(self):
        self.links = list()

    def __str__(self):
        s = "\n"
        s += "Model: Global-fit\n"
        s += "Global-parameters:"
        p0 = list(zip(self.global_parameters_all_names, self.global_parameters_values_all,
                 self.global_parameters_bound_all, self.global_parameters_fixed_all,
                 self.global_parameter_linked_all))
        s += "Parameter \t Value \t Bounds \t Fixed \t Linked\n"
        for p in p0:
            s += "%s \t %.4f \t %s \t %s \t %s \n" % p
        for fit in self.fits:
            s += "\n"
            s += fit.name + "\n"
            s += str(fit.model) + "\n"
        s += "\n"
        return s

    @property
    def _x(
            self
    ) -> np.array:
        x = list()
        for f in self.fits:
            x.append(f.model._x)
        return np.array(x)

    @_x.setter
    def _x(self, v):
        pass

    @property
    def _y(
            self
    ) -> np.array:
        y = list()
        for f in self.fits:
            y.append(f.model._y)
        return np.array(y)

    @_y.setter
    def _y(self, v):
        pass

    def __getitem__(
            self,
            key
    ):
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        x, y = self._x[start:stop:step], self._y[start:stop:step]
        return x, y

    def update(
            self
    ) -> None:
        super().update()
        for f in self.fits:
            f.model.update()

    def update_model(
            self,
            **kwargs
    ) -> None:
        if chisurf.settings.cs_settings['fitting']['parallel_fit']:
            threads = [threading.Thread(target=f.model.update_model) for f in self.fits]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for f in self.fits:
                f.model.update_model()

    def finalize(
            self
    ) -> None:
        super().finalize()
        for fit in self.fits:
            fit.model.finalize()


class GlobalFitModelWidget(GlobalFitModel, model.ModelWidget):

    plot_classes = [#(plots.GlobalFitPlot, {'logy': 'lin',
                    #                       'logx': 'lin'}),
                    (plots.FitInfo, {})
        #,(plots.SurfacePlot, {})
    ]

    @chisurf.decorators.init_with_ui(ui_filename="globalfit_2.ui")
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit
    ):
        self.actionOnAddToLocalFitList.triggered.connect(self.onAddToLocalFitList)
        self.actionOn_clear_local_fits.triggered.connect(self.onClearLocalFits)
        self.actionUpdate_widgets.triggered.connect(self.update_widgets)
        self.actionOnAddGlobalVariable.triggered.connect(self.onAddGlobalVariable)
        self.actionOnClearVariables.triggered.connect(self.onClearVariables)

        self.pushButton_3.clicked.connect(self.onSaveTable)
        self.pushButton_4.clicked.connect(self.onLoadTable)
        self.pushButton_5.clicked.connect(self.clear_listed_links)

        self.pushButton_8.clicked.connect(self.setLinks)
        self.addGlobalLink.clicked.connect(self.onAddLink)
        self.comboBox_gfOriginFit.currentIndexChanged[int].connect(self.update_parameter_origin)
        self.comboBox_gfTargetFit.currentIndexChanged[int].connect(self.update_parameter_target)
        self.comboBox_gfTargetParameter.currentIndexChanged[int].connect(self.update_link_text)
        self.comboBox_gfOriginParameter.currentIndexChanged[int].connect(self.update_link_text)
        self.table_GlobalLinks.cellDoubleClicked [int, int].connect(self.onTableGlobalLinksDoubleClicked)
        self.tableWidget.cellDoubleClicked [int, int].connect(self.onRemoveLocalFit)
        self.checkBox_2.stateChanged [int].connect(self.update_parameter_origin)

    def update_link_text(self):
        self.lineEdit_2.setText(self.current_link_formula)
        self.lineEdit_3.setText(self.current_origin_link_formula)

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
    def clear_on_update(
            self,
            v: bool
    ):
        self.checkBox_3.setChecked(v)

    @property
    def local_fits(self) -> List[chisurf.fitting.fit.Fit]:
        return [
            s for s in chisurf.fits
            if isinstance(s, chisurf.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_idx(self) -> List[int]:
        return [
            i for i, s in enumerate(chisurf.fits)
            if isinstance(s, chisurf.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_names(self) -> List[str]:
        return [f.name for f in self.local_fits]

    def onRemoveLocalFit(self):
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        chisurf.run("cs.current_fit.model.remove_local_fit(%s)" % row)

    def onClearLocalFits(self):
        chisurf.run("cs.current_fit.model.clear_local_fits()")
        self.tableWidget.setRowCount(0)

    def onTableGlobalLinksDoubleClicked(self):
        row = self.table_GlobalLinks.currentRow()
        self.table_GlobalLinks.removeRow(row)

    def onAddGlobalVariable(self):
        variable_name = self.current_global_variable_name
        if len(variable_name) > 0 and variable_name not in list(self._global_parameters.keys()):
            chisurf.run(
                "cs.current_fit.model.append_global_parameter(chisurf.parameter.FittingParameterWidget(name='%s'))" %
                self.current_global_variable_name
            )
            layout = self.verticalLayout
            layout.addWidget(self._global_parameters.values()[-1])
        else:
            print("No variable name defined.")

    def onClearVariables(self):
        print("onClearVariables")
        self._global_parameters = OrderedDict()
        layout = self.verticalLayout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

    def onAddToLocalFitList(self):
        local_fits = self.local_fits
        local_fits_idx = self.local_fit_idx
        fit_indeces = range(len(local_fits)) if self.add_all_fits else [self.current_fit_index]
        for fitIndex in fit_indeces:
            chisurf.run(
                "cs.current_fit.model.append_fit(chisurf.fits[%s])" % local_fits_idx[fitIndex]
            )

    def append_fit(
            self,
            fit: chisurf.fitting.fit
    ):
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

            self.update_widgets(fit_combo=False)

        GlobalFitModel.append_fit(self, fit)

    @property
    def origin_fit_number(self):
        return int(self.comboBox_gfOriginFit.currentIndex())  # origin fit fit_index

    @property
    def origin_fit(self):
        ofNbr = self.origin_fit_number
        return self.fits[ofNbr]

    @property
    def origin_parameter(self):
        return self.origin_fit.model.parameters_all_dict[self.origin_parameter_name]

    @property
    def origin_parameter_name(self):
        return str(self.comboBox_gfOriginParameter.currentText())

    @property
    def target_fit_number(self):
        return int(self.comboBox_gfTargetFit.currentIndex())  # target fit fit_index

    @property
    def target_fit(self):
        tfNbr = self.target_fit_number
        return self.fits[tfNbr]

    @property
    def target_parameter_name(self):
        return str(self.comboBox_gfTargetParameter.currentText())

    @property
    def target_parameter(self):
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

    def onAddLink(
            self,
            links: List = None
    ):
        table = self.table_GlobalLinks
        if links is None:
            links = []
            if self.link_all_of_type:
                print("Link all of one kind: %s" % self.link_all_of_type)
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

    def update_widgets(
            self
    ):
        self.comboBox.clear()
        self.comboBox.addItems(self.local_fit_names)

        self.comboBox_gfOriginFit.clear()
        self.comboBox_gfTargetFit.clear()
        usedLocalFitNames = [str(i + 1) for i, f in enumerate(self.fits)]
        self.comboBox_gfOriginFit.addItems(usedLocalFitNames)
        self.comboBox_gfTargetFit.addItems(usedLocalFitNames)

    def onSaveTable(self):
        filename = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save link-table', '.p'))[0]
        pickle.dump(self.links, open(filename, "wb"))

    def onLoadTable(self):
        filename = chisurf.widgets.get_filename('Open link-table', 'link file (*.p)')
        with open(filename, "rb") as fp:
            links = pickle.load(fp)
        self.onAddLink(links)

    def clear_listed_links(self):
        self.table_GlobalLinks.setRowCount(0)

    @property
    def local_fit_first(self):
        return self.checkBoxLocal.isChecked()

    @local_fit_first.setter
    def local_fit_first(self, v):
        if v is True:
            self.checkBoxLocal.setCheckState(2)
        else:
            self.checkBoxLocal.setCheckState(0)

