import sys

from chisurf.gui import QtWidgets
from chisurf.plugins.core.code_editor import CodeEditorWindow


def main():
    """Run the standalone code editor application."""
    app = QtWidgets.QApplication(sys.argv)
    win = CodeEditorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
