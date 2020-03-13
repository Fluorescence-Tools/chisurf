from __future__ import annotations
from chisurf import typing

import sys

from qtpy import QtWidgets
from guiqwt.builder import make
from guiqwt.plot import CurveDialog

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.models.tcspc
import chisurf.models.tcspc.widgets
from chisurf.fluorescence.fret.fret_line import FRETLineGenerator


class FRETLineGeneratorWidget(
    QtWidgets.QWidget
):

    name = "FRET-Line Generator"

    models = [
        (
            chisurf.models.tcspc.widgets.GaussianModelWidget,
            {'hide_corrections': True,
             'hide_fit': True,
             'hide_generic': True,
             'hide_convolve': True,
             'hide_rotation': True,
             'hide_error': True,
             'hide_donor': True
             }
        ),
        (
            chisurf.models.tcspc.widgets.FRETrateModelWidget,
            {'hide_corrections': True,
             'hide_fit': True,
             'hide_generic': True,
             'hide_convolve': True,
             'hide_rotation': True,
             'hide_error': True,
             'hide_donor': True,
             'enable_mix_model_donor': True
             }
        ),
    ]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(
            self,
            model_index: int
    ):
        self.comboBox.setCurrentIndex(model_index)
        model, parameter = self.models[model_index]
        self.fret_line_generator.model = model

    @property
    def current_model_index(self) -> int:
        return int(self.comboBox.currentIndex())

    @property
    def n_points(self) -> int:
        return int(self.spinBox.value())

    @property
    def parameter_range(self) -> typing.Tuple[float, float]:
        return float(self.doubleSpinBox.value()), float(self.doubleSpinBox_2.value())

    @chisurf.gui.decorators.init_with_ui(ui_filename="fret_line.ui")
    def __init__(
            self,
            verbose: bool = chisurf.verbose,
            *args,
            **kwargs
    ):
        win = CurveDialog(edit=False, toolbar=True)

        # Make Plot
        plot = win.get_plot()
        self.verticalLayout_5.addWidget(plot)
        self.fret_line_plot = plot

        self.fret_line_generator = FRETLineGenerator(*args, **kwargs)

        self.verbose = verbose
        self.model_names = [str(model[0].name) for model in self.models]
        self.comboBox.addItems(self.model_names)
        self.model = self.current_model_index

        self.onModelChanged()
        self.update_parameter()

        self.actionModel_changed.triggered.connect(self.onModelChanged)
        self.actionParameter_changed.triggered.connect(self.onParameterChanged)
        self.actionUpdate_Parameter.triggered.connect(self.update_parameter)
        self.actionCalculate_FRET_Line.triggered.connect(self.onCalculate)
        self.actionClear_plot.triggered.connect(self.onClearPlot)

    def onClearPlot(self):
        print("onClearPlot")
        self.fret_line_plot.del_all_items()

    def onCalculate(self):
        print("onCalculate")
        self.fret_line_generator.model.find_parameters()
        self.fret_line_generator.update(
            parameter_name=self.parameter_name,
            parameter_range=self.parameter_range
        )
        fret_line = make.curve(
            self.fret_line_generator.fluorescence_averaged_lifetimes,
            self.fret_line_generator.fret_efficiencies,
            color="r",
            linewidth=2
        )
        self.fret_line_plot.add_item(fret_line)
        self.fret_line_plot.do_autoscale()
        self.lineEdit.setText(self.fret_line_generator.transfer_efficency_string)
        self.lineEdit_2.setText(self.fret_line_generator.fdfa_string)
        self.lineEdit_3.setText("%s" % list(self.fret_line_generator.polynom_coefficients))

    def onParameterChanged(self):
        print("onParameterChanged")
        self.parameter_name = str(self.comboBox_2.currentText())

    def update_parameter(self):
        print("update_parameter")
        self.comboBox_2.clear()
        print(self.fret_line_generator.model.parameter_names)
        self.comboBox_2.addItems(self.fret_line_generator.model.parameter_names)

    def onModelChanged(self):
        print("onModelChanged")
        self.fret_line_generator.model.close()
        self.model = self.current_model_index
        self.fret_line_generator.model.find_parameters()
        self.verticalLayout.addWidget(self.fret_line_generator.model)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FRETLineGeneratorWidget()
    win.show()
    sys.exit(app.exec_())
