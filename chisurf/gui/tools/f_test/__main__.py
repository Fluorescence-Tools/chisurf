import sys
from chisurf.gui.tools.f_test.f_calculator import FTestWidget
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FTestWidget()
    win.show()
    sys.exit(app.exec_())
