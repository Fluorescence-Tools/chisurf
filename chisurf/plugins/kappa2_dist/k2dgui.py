from __future__ import annotations
import typing

import sys
import numpy as np
from qtpy import QtWidgets, uic
import pyqtgraph as pg
import pyqtgraph.dockarea
import os


from .k2dfun import kappasq_all_delta, kappasq_all, kappasq_dwt, s2delta


class Kappa2Dist(QtWidgets.QWidget):

    name = "Kappa2Dist"

    def __init__(self, kappa2: float = 0.667, *args, **kwargs):
        # initialize UI
        super().__init__(*args, **kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "kappa2_dist.ui"
            ),
            self
        )
        self.k2 = np.array([], dtype=np.double)
        self.kappa2 = kappa2

        # pyqtgraph
        area = pyqtgraph.dockarea.DockArea()
        self.verticalLayout.addWidget(area)
        d1 = pyqtgraph.dockarea.Dock(
            "res",
            size=(500, 80),
            hideTitle=True
        )
        p1 = pg.PlotWidget()
        d1.addWidget(p1)
        area.addDock(d1, 'top')
        self.kappa2_plot = p1.getPlotItem()
        self.kappa2_curve = self.kappa2_plot.plot(
            x=[0.0],
            y=[0.0],
            name='kappa2'
        )
        # Add save button
        self.saveButton = QtWidgets.QPushButton("Save")
        self.saveButton.setToolTip("Save computed kappa2 distribution to file")
        self.gridLayout.addWidget(self.saveButton, 0, 4, 1, 2)
        self.saveButton.setEnabled(False)  # Disable until data is computed

        # Connections
        self.pushButton.clicked.connect(self.onUpdateHist)
        self.doubleSpinBox_4.valueChanged.connect(self.onUpdateRapp)
        self.saveButton.clicked.connect(self.onSaveDistribution)
        self.hide()

    def onUpdateHist(self) -> None:
        if self.model == "cone":
            if self.rAD_known:
                x, k2hist, self.k2 = kappasq_all_delta(
                    delta=self.delta,
                    sD2=self.SD2,
                    sA2=self.SA2,
                    step=self.step,
                    n_bins=self.n_bins
                )
                self.lineEdit.setText(
                    "%.2f" % (self.delta * 180 / np.pi)
                )
            else:
                x, k2hist, self.k2 = kappasq_all(
                    self.SD2,
                    self.SA2
                )
        elif self.model == "diffusion":
            x, k2hist, self.k2 = kappasq_dwt(
                sD2=self.SD2,
                sA2=self.SA2,
                fret_efficiency=self.fret_efficiency,
                n_bins=self.n_bins
            )
        self.k2scale = x
        self.k2hist = k2hist
        self.kappa2_curve.setData(x=x[1:], y=k2hist)

        # Enable save button now that we have data
        self.saveButton.setEnabled(True)

        self.onUpdateRapp()

    def onUpdateRapp(self) -> None:
        k2 = self.k2_true
        k2scale = self.k2scale[1:]
        k2hist = self.k2hist
        # Rapp
        Rapp_scale = (1. / k2 * k2scale)**(1.0/6.0)
        Rapp_mean = np.dot(k2hist, Rapp_scale)/sum(k2hist)
        self.Rapp_mean = Rapp_mean
        Rapp_sd = np.sqrt(np.dot(k2hist, (Rapp_scale-Rapp_mean)**2) / sum(k2hist))
        self.RappSD = Rapp_sd
        # kappa2
        self.k2_mean = np.dot(k2hist, k2scale)/sum(k2hist)
        self.k2_sd = np.sqrt(np.dot(k2hist, (k2scale-self.k2_mean)**2) / sum(k2hist))

    @property
    def model(self) -> str:
        if self.radioButton_2.isChecked():
            return "cone"
        elif self.radioButton.isChecked():
            return "diffusion"

    @property
    def rAD_known(self) -> bool:
        return self.checkBox.isChecked()

    @property
    def k2_mean(self) -> float:
        return float(self.doubleSpinBox_10.value())

    @k2_mean.setter
    def k2_mean(self, v: float):
        self.doubleSpinBox_10.setValue(v)

    @property
    def k2_sd(self) -> float:
        return float(self.doubleSpinBox_9.value())

    @k2_sd.setter
    def k2_sd(self, v: float):
        self.doubleSpinBox_9.setValue(v)

    @property
    def min_max(self) -> typing.Tuple[float, float]:
        k2 = self.k2.flatten()
        return min(k2), max(k2)

    @property
    def n_bins(self) -> int:
        return int(self.spinBox.value())

    @property
    def r_0(self) -> float:
        return float(self.doubleSpinBox_2.value())

    @property
    def r_Dinf(self) -> float:
        return float(self.doubleSpinBox.value())

    @property
    def r_ADinf(self) -> float:
        return float(self.doubleSpinBox_7.value())

    @property
    def r_Ainf(self):
        return float(self.doubleSpinBox_5.value())

    @property
    def step(self):
        return float(self.doubleSpinBox_3.value())

    @property
    def fret_efficiency(self):
        return float(self.doubleSpinBox_11.value())

    @property
    def SD2(self) -> float:
        return -np.sqrt(self.r_Dinf/self.r_0)

    @property
    def SA2(self) -> float:
        return np.sqrt(self.r_Ainf/self.r_0)

    @property
    def delta(self) -> float:
        s2_delta, delta = s2delta(
            s2_donor=self.SD2,
            s2_acceptor=self.SA2,
            r_inf_AD=self.r_ADinf,
            r_0=self.r_0
        )
        return delta

    @property
    def k2_true(self) -> float:
        return float(self.doubleSpinBox_4.value())

    @property
    def Rapp_mean(self) -> float:
        return float(self.doubleSpinBox_6.value())

    @Rapp_mean.setter
    def Rapp_mean(self, v: float):
        self.doubleSpinBox_6.setValue(v)

    @property
    def RappSD(self) -> float:
        return float(self.doubleSpinBox_8.value())

    @RappSD.setter
    def RappSD(self, v: float):
        self.doubleSpinBox_8.setValue(v)

    def onSaveDistribution(self) -> None:
        """
        Save the computed kappa2 distribution to a file.
        Opens a file dialog for the user to select the save location.
        Saves the k2 scale and histogram values as a CSV file.
        """
        try:
            # Get file name from user
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Kappa2 Distribution",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )

            if file_path:
                # If user didn't add .csv extension, add it
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'

                # Prepare data for saving
                k2scale = self.k2scale[1:]  # Skip the first bin edge
                k2hist = self.k2hist

                # Create header with metadata
                header = [
                    f"# Kappa2 Distribution",
                    f"# Model: {self.model}",
                    f"# SD2: {self.SD2:.6f}",
                    f"# SA2: {self.SA2:.6f}",
                    f"# Mean kappa2: {self.k2_mean:.6f}",
                    f"# SD kappa2: {self.k2_sd:.6f}",
                    f"# Assumed kappa2: {self.k2_true:.6f}",
                    f"# Mean Rapp: {self.Rapp_mean:.6f}",
                    f"# SD Rapp: {self.RappSD:.6f}",
                    f"#",
                    f"# kappa2,probability"
                ]

                # Write to file
                with open(file_path, 'w') as f:
                    # Write header
                    f.write('\n'.join(header) + '\n')

                    # Write data
                    for x, y in zip(k2scale, k2hist):
                        f.write(f"{x:.6f},{y:.6f}\n")

                QtWidgets.QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Kappa2 distribution saved to:\n{file_path}"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"An error occurred while saving the file:\n{str(e)}"
            )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Kappa2Dist()
    win.show()
    sys.exit(app.exec_())