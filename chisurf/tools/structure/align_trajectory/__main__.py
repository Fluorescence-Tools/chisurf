import sys
from chisurf.tools.structure.align_trajectory import AlignTrajectoryWidget
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AlignTrajectoryWidget()
    win.show()
    app.exec_()

