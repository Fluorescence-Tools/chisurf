import sys
from chisurf.gui import QtWidgets

from chisurf.plugins.core.f_test.f_calculator import FTestWidget


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FTestWidget()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
