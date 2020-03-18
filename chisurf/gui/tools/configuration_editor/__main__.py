import sys
from . import configuration_editor
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = configuration_editor.ParameterEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
