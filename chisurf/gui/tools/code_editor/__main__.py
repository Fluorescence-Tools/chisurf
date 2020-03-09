import sys
from . import qsci_editor
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = qsci_editor.CodeEditor()
    win.show()
    app.exec_()

