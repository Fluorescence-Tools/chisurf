import sys
from chisurf.gui.tools.structure import AlignTrajectoryWidget
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AlignTrajectoryWidget()
    win.show()
    app.exec_()

