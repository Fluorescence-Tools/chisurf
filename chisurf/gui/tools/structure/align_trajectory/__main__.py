import sys
from chisurf.gui.tools.structure.align_trajectory import AlignTrajectoryWidget
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    win = AlignTrajectoryWidget()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
