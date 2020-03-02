import sys
from chisurf.tools.broken.correlate import CorrelatorWidget
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CorrelatorWidget()
    gui.show()
    app.exec_()

