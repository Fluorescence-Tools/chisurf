import sys
from .widget import RotateTranslateTrajectoryWidget
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RotateTranslateTrajectoryWidget()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
