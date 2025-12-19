import sys

from qtpy.QtWidgets import QApplication

from .widget import PotentialEnergyWidget


def main():
    app = QApplication(sys.argv)
    win = PotentialEnergyWidget()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
