"""Code Editor plugin launcher."""

from chisurf.plugins.core.code_editor.window import CodeEditorWindow

if __name__ == "plugin":
    window = CodeEditorWindow()
    window.show()


if __name__ == "__main__":
    import sys

    from qtpy import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    window = CodeEditorWindow()
    window.show()
    sys.exit(app.exec_())
