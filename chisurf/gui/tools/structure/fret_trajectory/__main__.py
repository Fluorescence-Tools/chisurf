import sys
from . import gui
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = gui.Structure2Transfer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
