"""Plugin Check plugin launcher."""

from chisurf.plugins.core.plugin_check.gui.tool import PluginCheckTool

if __name__ == "plugin":
    window = PluginCheckTool()
    window.show()


if __name__ == "__main__":
    import sys

    from qtpy import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    window = PluginCheckTool()
    window.show()
    sys.exit(app.exec_())
