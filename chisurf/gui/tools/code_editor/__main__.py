import sys
from . import qsci_editor
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    win = qsci_editor.CodeEditor()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
