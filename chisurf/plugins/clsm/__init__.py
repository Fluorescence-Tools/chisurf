import clsmview.gui

name = "Imaging:CLSM-Draw"

import sys

import chisurf
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

from PyQt5.QtWidgets import *

log = chisurf.logging.info


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clsm = clsmview.gui.CLSMPixelSelect()
    clsm.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    clsm = clsmview.gui.CLSMPixelSelect()
    clsm.show()
