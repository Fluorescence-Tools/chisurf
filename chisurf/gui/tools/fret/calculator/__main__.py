import sys
from chisurf.gui.tools.fret.calculator import FRETCalculator
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FRETCalculator()
    win.show()
    sys.exit(app.exec_())
