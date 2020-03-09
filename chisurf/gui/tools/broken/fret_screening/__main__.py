import sys
from chisurf.gui.tools.broken.fret_screening import FPSScreenTrajectory
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FPSScreenTrajectory()
    win.show()
    sys.exit(app.exec_())
