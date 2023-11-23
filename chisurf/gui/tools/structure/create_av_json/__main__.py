import sys
from . import label_structure
from chisurf.gui import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = label_structure.LabelStructure()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()