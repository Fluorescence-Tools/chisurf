import sys
from chisurf.gui.tools.broken.fret_lines import FRETLineGeneratorWidget
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FRETLineGeneratorWidget()
    win.show()
    sys.exit(app.exec_())
