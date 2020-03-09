import sys
from . import label_structure
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = label_structure.LabelStructure()
    win.show()
    sys.exit(app.exec_())
