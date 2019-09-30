import os

from qtpy import QtWidgets, uic

import mfm
import mfm.widgets
from mfm.experiments import reader
import mfm.io.widgets


class CsvTCSPCWidget(
    QtWidgets.QWidget,
):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "csvTCSPCWidget.ui"
            ),
            self
        )
        self.actionDtChanged.triggered.connect(self.onParametersChanged)
        self.actionRebinChanged.triggered.connect(self.onParametersChanged)
        self.actionRepratechange.triggered.connect(self.onParametersChanged)
        self.actionPolarizationChange.triggered.connect(self.onParametersChanged)
        self.actionGfactorChanged.triggered.connect(self.onParametersChanged)
        self.actionIsjordiChanged.triggered.connect(self.onParametersChanged)
        self.actionMatrixColumnsChanged.triggered.connect(self.onParametersChanged)

    def onParametersChanged(self):
        is_jordi = bool(self.checkBox_3.isChecked())
        try:
            matrix_columns = list(
                map(
                    int,
                    str(
                        self.lineEdit.text()
                    ).strip().split(' ')
                )
            )
        except ValueError:
            matrix_columns = []
        gfactor = float(self.doubleSpinBox_3.value())
        pol = 'vm'
        if self.radioButton_4.isChecked():
            pol = 'vv/vh'
        if self.radioButton_3.isChecked():
            pol = 'vv'
        elif self.radioButton_2.isChecked():
            pol = 'vh'
        elif self.radioButton.isChecked():
            pol = 'vm'
        rep_rate = self.doubleSpinBox.value()

        rebin_y = int(self.comboBox.currentText())
        rebin_x = int(self.comboBox_2.currentText())
        rebin = int(self.comboBox.currentText())
        dt = float(
            self.doubleSpinBox_2.value()
        ) * rebin if self.checkBox_2.isChecked() else 1.0 * rebin

        mfm.run(
            "\n".join(
                [
                    "cs.current_setup.is_jordi = %s" % is_jordi,
                    "cs.current_setup.use_header = %s" % (not is_jordi),
                    "cs.current_setup.matrix_columns = %s" % matrix_columns,
                    "cs.current_setup.g_factor = %f" % gfactor,
                    "cs.current_setup.polarization = '%s'" % pol,
                    "cs.current_setup.rep_rate = %s" % rep_rate,
                    "cs.current_setup.rebin = (%s, %s)" % (rebin_x, rebin_y),
                    "cs.current_setup.dt = %s" % dt
                ]
            )
        )


class TCSPCReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    def get_filename(
            self
    ) -> str:
        return mfm.widgets.get_filename(
            description='CSV-TCSPC file',
            file_type='All files (*.*)',
            working_path=None
        )

    @property
    def filename(
            self
    ) -> str:
        return self.get_filename()

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        self.layout.addWidget(
            mfm.io.widgets.CsvWidget()
        )
        self.layout.addWidget(
            CsvTCSPCWidget()
        )

