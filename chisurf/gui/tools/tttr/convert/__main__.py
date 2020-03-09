import sys
from chisurf.gui.tools.tttr import TTTRConvert
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TTTRConvert()
    gui.show()
    app.exec_()

