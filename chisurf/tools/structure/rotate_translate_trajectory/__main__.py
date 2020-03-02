import sys
from chisurf.tools.structure.rotate_translate_trajectory import RotateTranslateTrajectoryWidget
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = RotateTranslateTrajectoryWidget()
    win.show()
    sys.exit(app.exec_())
