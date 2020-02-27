import sys
from . import gui
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = gui.Structure2Transfer()
    win.show()
    sys.exit(app.exec_())
