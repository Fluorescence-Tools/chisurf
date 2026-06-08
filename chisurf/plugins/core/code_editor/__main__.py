import sys

from chisurf.gui import QtWidgets
from chisurf.plugins.core.code_editor import CodeEditor


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CodeEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
