import sys
from chisurf.tools.structure.save_topology import SaveTopology
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = SaveTopology()
    win.show()
    sys.exit(app.exec_())
