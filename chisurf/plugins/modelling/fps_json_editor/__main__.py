import sys

from chisurf.gui import QtWidgets
from chisurf.plugins.modelling.fps_json_editor.label_structure import LabelStructure


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = LabelStructure()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
