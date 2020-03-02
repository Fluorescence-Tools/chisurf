import sys
from chisurf.tools.tttr.decay import HistogramTTTR
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = HistogramTTTR()
    gui.show()
    app.exec_()

