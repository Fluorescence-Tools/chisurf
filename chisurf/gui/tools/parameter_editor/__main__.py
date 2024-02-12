import sys
from . import parameter_editor
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = parameter_editor.ParameterEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
