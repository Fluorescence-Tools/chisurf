import sys
from chisurf.tools.broken.potential_energy import PotentialEnergyWidget
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PotentialEnergyWidget()
    win.show()
    sys.exit(app.exec_())
