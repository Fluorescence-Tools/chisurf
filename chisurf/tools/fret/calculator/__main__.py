import sys
from chisurf.tools.fret.fret_calculator import FRETCalculator
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FRETCalculator()
    win.show()
    sys.exit(app.exec_())
