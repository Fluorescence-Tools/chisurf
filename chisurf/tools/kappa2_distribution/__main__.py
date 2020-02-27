import sys
from chisurf.tools.kappa2_distribution.kappa2dist import Kappa2Dist
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Kappa2Dist()
    win.show()
    sys.exit(app.exec_())
