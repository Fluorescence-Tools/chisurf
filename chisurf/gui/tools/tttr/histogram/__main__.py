import sys
from chisurf.gui.tools.tttr.histogram import HistogramTTTR
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    gui = HistogramTTTR()
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
