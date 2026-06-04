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
from chisurf.models.parameter_transform import ParameterTransformWidget
from chisurf.models import model


class GlobalFitModelWidget(GlobalFitModel, model.ModelWidget):

    plot_classes = [
                    (plots.FitInfo, {}),
                    # (plots.ResidualPlot, {})
    ]

    @chisurf.gui.decorators.init_with_ui(ui_filename="globalfit.ui")
    def __init__(self, fit: chisurf.fitting.fit.Fit):
        """Initialize the global-fit widget.

        Connects UI signals and subscribes to model-level fit-append
        notifications.

        Parameters
        ----------
        fit : chisurf.fitting.fit.Fit
            The parent fit (FitGroup).
        """
        self.pushButton_3.clicked.connect(self.onSaveTable)
        self.pushButton_4.clicked.connect(self.onLoadTable)
        self.pushButton_5.clicked.connect(self.clear_listed_links)

        self.pushButton_8.clicked.connect(self.setLinks)
        self.addGlobalLink.clicked.connect(self.onAddLink)
        # Ensure the "Used fits -> add" tool button actually appends the selected/local fits
        # to the global fit. Without this connection, clicking the button has no effect.
        self.toolButton_7.clicked.connect(self.onAddToLocalFitList)
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

        self.groupBox_2.toggled.connect(self.widget_3.setVisible)
        self.widget_3.setVisible(self.groupBox_2.isChecked())

        # Subscribe UI to model-level fit-append notifications so the table updates
        # regardless of whether fits are appended via GUI or macros/actions.
        try:
            self.on_fit_appended(self._on_fit_appended_ui)
        except Exception:
            # If the base does not provide the subscription API, ignore silently.
            pass

    @property
    def add_all_fits(self) -> bool:
        """If True, add all available local fits at once."""
        return bool(self.checkBox.isChecked())

    @property
    def current_global_variable_name(self) -> str:
        """Name entered in the global-variable text field."""
        return str(self.lineEdit.text())

    @property
    def current_fit_index(self) -> int:
        """Currently selected local-fit index in the combo box."""
        return self.comboBox.currentIndex()

    @property
    def link_all_of_type(self) -> bool:
        """If True, link all parameters sharing the same name."""
        return not self.checkBox_2.isChecked()

    @property
    def clear_on_update(self) -> bool:
        """Whether links are cleared before re-evaluation."""
        return self.checkBox_3.isChecked()

    @clear_on_update.setter
    def clear_on_update(self, v: bool):
        """Set whether links are cleared before re-evaluation.

        Parameters
        ----------
        v : bool
            New state.
        """
        self.checkBox_3.setChecked(v)

    @property
    def local_fits(self) -> typing.List[chisurf.fitting.fit.Fit]:
        """Fits available to add to the global model (not already included)."""
        return [
            s for s in chisurf.fits
            if isinstance(s, chisurf.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_idx(self) -> typing.List[int]:
        """Indices of available local fits in the global fit list."""
        return [
            i for i, s in enumerate(chisurf.fits)
            if isinstance(s, chisurf.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_names(self) -> typing.List[str]:
        """Names of available local fits."""
        return [f.name for f in self.local_fits]

    @property
    def origin_fit_number(self) -> int:
        """Index of the origin fit in the global model."""
        return int(self.comboBox_gfOriginFit.currentIndex())

    @property
    def origin_fit(self) -> chisurf.fitting.fit.Fit:
        """The origin fit for a link."""
        ofNbr = self.origin_fit_number
        return self.fits[ofNbr]

    @property
    def origin_parameter(self) -> chisurf.fitting.parameter.FittingParameter:
        """The origin parameter for a link."""
        return self.origin_fit.model.parameters_all_dict[self.origin_parameter_name]

    @property
    def origin_parameter_name(self) -> str:
        """Name of the selected origin parameter."""
        return str(self.comboBox_gfOriginParameter.currentText())

    @property
    def target_fit_number(self) -> int:
        """Index of the target fit for a link."""
        return int(self.comboBox_gfTargetFit.currentIndex())

    @property
    def target_fit(self) -> chisurf.fitting.fit.Fit:
        """The target fit for a link."""
        tfNbr = self.target_fit_number
        return self.fits[tfNbr]

    @property
    def target_parameter_name(self) -> str:
        """Name of the selected target parameter."""
        return str(self.comboBox_gfTargetParameter.currentText())

    @property
    def target_parameter(self) -> chisurf.fitting.parameter.FittingParameter:
        """The target parameter for a link."""
        return self.target_fit.model.parameters_all_dict[self.target_parameter_name]

    @property
    def current_link_formula(self):
        """Default link formula string for the current selection."""
        return f"f[{self.target_fit_number}]['{self.target_parameter_name}']"

    @property
    def current_target_formula(self) -> str:
        """Target formula string."""
        if self.checkBox_4.isChecked():
            return str(self.lineEdit_2.text())
        return self.current_link_formula

    @property
    def current_origin_link_formula(self):
        """Origin formula string for a link."""
        if self.link_all_of_type:
            return f"f[i]['{self.origin_parameter_name}']"
        else:
            return f"f[{self.origin_fit_number}]['{self.origin_parameter_name}']"

    @property
    def links(self):
        """Link definitions read from the global-links table."""
        table = self.table_GlobalLinks
        links = []
        for r in range(table.rowCount()):
            en = bool(table.cellWidget(r, 0).checkState())
            fitA = int(table.item(r, 1).data(0)) - 1
            pA = str(table.item(r, 2).text())
            fB = str(table.item(r, 3).text())
            links.append([en, fitA, pA, fB])
        return links

    @property
    def local_fit_first(self) -> bool:
        """If True, the local-fit list is displayed first."""
        return self.checkBoxLocal.isChecked()

    @local_fit_first.setter
    def local_fit_first(self, v: bool):
        """Set whether the local-fit list is displayed first.

        Parameters
        ----------
        v : bool
            New state.
        """
        if v is True:
            self.checkBoxLocal.setCheckState(2)
        else:
            self.checkBoxLocal.setCheckState(0)

    def update_link_text(self):
        """Update the formula text field unless custom editing is enabled."""
        if not self.checkBox_4.isChecked():
            self.lineEdit_2.setText(self.current_link_formula)

    def onRemoveLocalFit(self) -> None:
        """Qt slot: remove the selected local fit from the table."""
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        chisurf.actions.dispatch(
            name="model.remove_local_fit",
            payload={"row": int(row)},
        )

    def onClearLocalFits(self) -> None:
        """Qt slot: clear all local fits from the table."""
        chisurf.actions.dispatch(
            name="model.clear_local_fits",
            payload={},
        )
        self.tableWidget.setRowCount(0)

    def onTableGlobalLinksDoubleClicked(self) -> None:
        """Qt slot: remove a link row on double-click."""
        row = self.table_GlobalLinks.currentRow()
        self.table_GlobalLinks.removeRow(row)

    def onAddGlobalVariable(self) -> None:
        """Qt slot: add a new global variable from the text field."""
        variable_name = self.current_global_variable_name
        if len(variable_name) > 0 and variable_name not in list(self._global_parameters.keys()):
            chisurf.actions.dispatch(
                name="model.append_global_parameter",
                payload={"parameter_name": str(self.current_global_variable_name)},
            )
            layout = self.verticalLayout
            layout.addWidget(self._global_parameters.values()[-1])
        else:
            chisurf.logging.warning("onAddGlobalVariable: No variable name defined.")

    def onClearVariables(self) -> None:
        """Qt slot: remove all global parameters."""
        chisurf.logging.info("onClearVariables")
        self._global_parameters = dict()
        layout = self.verticalLayout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

    def onAddToLocalFitList(self) -> None:
        """Qt slot: add selected (or all) local fits to the global model."""
        print("onAddToLocalFitList")
        chisurf.logging.info("onAddToLocalFitList")
        local_fits = self.local_fits
        local_fits_idx = self.local_fit_idx
        fit_indeces = range(len(local_fits)) if self.add_all_fits else [self.current_fit_index]
        print("fit_indeces:", fit_indeces)
        for fitIndex in fit_indeces:
            print(f"onAddToLocalFitList:fitIndex:{fitIndex}")
            chisurf.actions.dispatch(
                name="model.append_fit",
                payload={"fit_index": int(local_fits_idx[fitIndex])},
            )

    def append_fit(self, fit: chisurf.fitting.fit):
        """Append a fit to the global model.

        Defer UI updates to the model-level callback to avoid double insertion.

        Parameters
        ----------
        fit : chisurf.fitting.fit.Fit
            The fit instance to append.
        """
        GlobalFitModel.append_fit(self, fit)

    # --- UI reaction to model notifications ---
    def _on_fit_appended_ui(self, fit: chisurf.fitting.fit) -> None:
        """Model callback: update the table when a fit is appended.

        Parameters
        ----------
        fit : chisurf.fitting.fit.Fit
            The newly appended fit.
        """
        try:
            table = self.tableWidget
            existing_names = set()
            for r in range(table.rowCount()):
                item = table.item(r, 0)
                if item is not None:
                    existing_names.add(str(item.text()))
            if str(fit.name) in existing_names:
                self.update_widgets()
                return

            table.insertRow(table.rowCount())
            rc = table.rowCount() - 1

            tmp = QtWidgets.QTableWidgetItem(fit.name)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 0, tmp)

            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            table.resizeRowsToContents()

            self.update_widgets()
        except Exception:
            pass

    def onAddLink(self, links: typing.List = None):
        """Qt slot: add a link row (or multiple rows) to the global-links table.

        Parameters
        ----------
        links : list, optional
            Pre-defined link definitions. If None, a new link is created
            from the current UI selection.
        """
        table = self.table_GlobalLinks
        if not isinstance(links, list):
            links = None
        if links is None:
            links = []
            if self.link_all_of_type:
                chisurf.logging.info(f"Link all of one kind: {self.link_all_of_type}")
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
        """Refresh the origin-parameter combo box based on the selected origin fit."""
        self.comboBox_gfOriginParameter.clear()
        if len(self.fits) > 0:
            if not self.link_all_of_type:
                origin_index = self.comboBox_gfOriginFit.currentIndex()
                pn = []
                if 0 <= origin_index < len(self.fits):
                    fit = self.fits[origin_index]
                    pn = [p.name for p in fit.model.parameters_all]
                pn.sort()
                self.comboBox_gfOriginParameter.addItems(pn)
            else:
                names = set([p.name for f in self.fits for p in f.model.parameters_all])
                names = list(names)
                names.sort()
                self.comboBox_gfOriginParameter.addItems(names)

        self.update_link_text()

    def update_parameter_target(self):
        """Refresh the target-parameter combo box based on the selected target fit."""
        self.comboBox_gfTargetParameter.clear()
        if len(self.fits) > 0:
            ftIndex = self.comboBox_gfTargetFit.currentIndex()
            ft = self.fits[ftIndex]
            pn = [p.name for p in self.fit.model.parameters_all]
            pn.sort()
            self.comboBox_gfTargetParameter.addItems([p.name for p in ft.model.parameters_all])

    def update_widgets(self):
        """Refresh all combo boxes and parameter lists from model state."""
        self.comboBox.clear()
        self.comboBox.addItems(self.local_fit_names)

        self.comboBox_gfOriginFit.clear()
        self.comboBox_gfTargetFit.clear()
        usedLocalFitNames = [str(i + 1) for i, f in enumerate(self.fits)]
        self.comboBox_gfOriginFit.addItems(usedLocalFitNames)
        self.comboBox_gfTargetFit.addItems(usedLocalFitNames)

    def onSaveTable(self):
        """Qt slot: save the link table to a pickle file."""
        filename = chisurf.gui.widgets.save_file(
            description='Save link-table',
            file_type='.p'
        )
        pickle.dump(self.links, open(filename, "wb"))

    def onLoadTable(self):
        """Qt slot: load a link table from a pickle file."""
        filename = chisurf.gui.widgets.get_filename(
            description='Open link-table',
            file_type='link file (*.p)'
        )
        with open(filename, "rb") as fp:
            links = pickle.load(fp)
        self.onAddLink(links)

    def clear_listed_links(self):
        """Qt slot: clear all rows from the global-links table."""
        self.table_GlobalLinks.setRowCount(0)

