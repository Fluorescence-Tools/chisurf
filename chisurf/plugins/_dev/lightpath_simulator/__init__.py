import os

# Plugin metadata
name = "Dev:Spectroscopy:Light Path Simulator"

def plugin():
    from qtpy import QtWidgets, QtGui
    from .simulator_widget import LightPathSimulatorWidget
    from chisurf.plugins._dev.fluorophore_db import get_db
    
    # Use the shared database from the fluorophore_db directory
    db = get_db()
    
    widget = LightPathSimulatorWidget(db)
    widget.show()
    return widget

def get_icon():
    """Lazy load icon to avoid Qt dependency on import."""
    from qtpy import QtGui
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    if os.path.exists(icon_path):
        return QtGui.QIcon(icon_path)
    return QtGui.QIcon()

if __name__ == "__main__" or __name__ == "plugin":
    from qtpy import QtWidgets
    # For testing and chi-surf plugin loading
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])
    w = plugin()
    if __name__ == "__main__":
        app.exec_()
