import sys
from chisurf.tools.broken.fret_lines.fret_lines import FRETLineGeneratorWidget
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FRETLineGeneratorWidget()
    win.show()
    sys.exit(app.exec_())
