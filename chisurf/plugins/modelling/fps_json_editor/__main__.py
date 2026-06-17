import sys

from chisurf.gui import QtWidgets
from chisurf.plugins.modelling.fps_json_editor.gui.tool import FpsJsonEditorTool


def main():
    """Launch the FPS JSON Editor window."""
    app = QtWidgets.QApplication(sys.argv)
    win = FpsJsonEditorTool()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
