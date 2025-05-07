name = "Tools:ndXplorer"

import sys

import chisurf
import ndxplorer

from PyQt5.QtWidgets import *

log = chisurf.logging.info


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ndx = ndxplorer.NDXplorer()
    ndx.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    ndx = ndxplorer.NDXplorer()
    ndx.show()
