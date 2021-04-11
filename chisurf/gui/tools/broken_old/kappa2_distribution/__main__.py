import sys
from chisurf.gui.tools.kappa2_distribution.kappa2dist import Kappa2Dist
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Kappa2Dist()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

