import sys
from chisurf.gui.tools.structure.potential_energy import PotentialEnergyWidget
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PotentialEnergyWidget()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
