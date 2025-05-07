name = "Tools:QuEst (Quenching estimator)"

import sys

import chisurf
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

from PyQt5.QtWidgets import *

log = chisurf.logging.info


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ndx = TransientDecayGenerator()
    ndx.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    ndx = TransientDecayGenerator()
    ndx.show()
