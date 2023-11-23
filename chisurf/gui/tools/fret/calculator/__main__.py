import sys
from chisurf.gui.tools.fret.calculator import FRETCalculator
from chisurf.gui import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FRETCalculator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
