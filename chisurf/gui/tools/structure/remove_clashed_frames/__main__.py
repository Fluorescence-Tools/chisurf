import sys
from chisurf.gui.tools.structure import RemoveClashedFrames
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = RemoveClashedFrames()
    win.show()
    sys.exit(app.exec_())
