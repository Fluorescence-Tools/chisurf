from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets
import chisurf.core.fitting
import chisurf.gui.widgets
import chisurf.gui.decorators
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

from chisurf.gui import plots
from chisurf.core.models.global_model.globalfit import GlobalFitModel
from chisurf.gui.widgets.models import model_widget as model


class GlobalFitModelWidget(GlobalFitModel, model.ModelWidget):

    plot_classes = [
                    (plots.FitInfo, {}),
                    # (plots.ResidualPlot, {})
    ]

    @chisurf.gui.decorators.init_with_ui(ui_filename="globalfit.ui")
    def __init__(self, fit: chisurf.core.fitting.fit.Fit):
        """Initialize the global-fit widget.

        Connects UI signals and subscribes to model-level fit-append
        notifications.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            The parent fit (FitGroup).
        """
        self.toolButton_7.clicked.connect(self.onAddToLocalFitList)
        self.tableWidget.cellDoubleClicked[int, int].connect(self.onRemoveLocalFit)

        self.actionOnClearVariables.triggered.connect(self.onClearVariables)
        self.actionOnAddToLocalFitList.triggered.connect(self.onAddToLocalFitList)
        self.actionOn_clear_local_fits.triggered.connect(self.onClearLocalFits)
        self.actionUpdate_widgets.triggered.connect(self.update_widgets)
        self.actionOnAddGlobalVariable.triggered.connect(self.onAddGlobalVariable)

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
    def local_fits(self) -> typing.List[chisurf.core.fitting.fit.Fit]:
        """Fits available to add to the global model (not already included)."""
        fit_objects = get_fitting_client().get_fit_objects()
        return [
            s for s in fit_objects
            if isinstance(s, chisurf.core.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_idx(self) -> typing.List[int]:
        """Indices of available local fits in the global fit list."""
        fit_objects = get_fitting_client().get_fit_objects()
        return [
            i for i, s in enumerate(fit_objects)
            if isinstance(s, chisurf.core.fitting.fit.Fit) and s.model is not self
        ]

    @property
    def local_fit_names(self) -> typing.List[str]:
        """Names of available local fits."""
        return [f.name for f in self.local_fits]

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

    def onRemoveLocalFit(self) -> None:
        """Qt slot: remove the selected local fit from the table."""
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        chisurf.core.actions.dispatch(
            name="model.remove_local_fit",
            payload={"row": int(row)},
        )

    def onClearLocalFits(self) -> None:
        """Qt slot: clear all local fits from the table."""
        chisurf.core.actions.dispatch(
            name="model.clear_local_fits",
            payload={},
        )
        self.tableWidget.setRowCount(0)

    def onAddGlobalVariable(self) -> None:
        """Qt slot: add a new global variable from the text field."""
        variable_name = self.current_global_variable_name
        if len(variable_name) > 0 and variable_name not in list(self._global_parameters.keys()):
            chisurf.core.actions.dispatch(
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
            chisurf.core.actions.dispatch(
                name="model.append_fit",
                payload={"fit_index": int(local_fits_idx[fitIndex])},
            )

    def append_fit(self, fit: chisurf.core.fitting.fit):
        """Append a fit to the global model.

        Defer UI updates to the model-level callback to avoid double insertion.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            The fit instance to append.
        """
        GlobalFitModel.append_fit(self, fit)

    # --- UI reaction to model notifications ---
    def _on_fit_appended_ui(self, fit: chisurf.core.fitting.fit) -> None:
        """Model callback: update the table when a fit is appended.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
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

    def update_widgets(self):
        """Refresh all combo boxes and parameter lists from model state."""
        self.comboBox.clear()
        self.comboBox.addItems(self.local_fit_names)
