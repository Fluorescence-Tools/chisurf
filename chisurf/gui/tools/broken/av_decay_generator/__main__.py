import sys
from . import dye_diffusion
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = dye_diffusion.TransientDecayGenerator()
    win.show()
    sys.exit(app.exec_())
