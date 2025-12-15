import sys
from .widget import AlignTrajectoryWidget
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    win = AlignTrajectoryWidget()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
