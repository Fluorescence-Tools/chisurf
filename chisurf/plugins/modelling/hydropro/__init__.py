import sys
from qtpy import QtWidgets
from .hydrogui import HydroGui

# Plugin display name
name = "Structure:Computation:HydroPro"

if __name__ == "plugin":
    window = HydroGui()
    window.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = HydroGui()
    win.show()
    sys.exit(app.exec_())
