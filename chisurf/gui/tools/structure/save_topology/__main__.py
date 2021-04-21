import sys
from chisurf.gui.tools.structure.save_topology import SaveTopology
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SaveTopology()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
