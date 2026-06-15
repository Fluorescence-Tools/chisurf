"""Compatibility shim for legacy ribbon launcher."""

from chisurf.plugins.core.globalview.gui.tool import GraphWizard  # noqa: F401

if __name__ == "plugin":
    graph_wiz = GraphWizard()
    graph_wiz.show()

if __name__ == "__main__":
    import sys
    from qtpy import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    graph_wiz = GraphWizard()
    graph_wiz.setWindowTitle("🕸️ ChiSurf Parameter Network")
    graph_wiz.show()
    sys.exit(app.exec_())
