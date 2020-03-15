import sys
from chisurf.gui.tools.tttr.correlate import CorrelateTTTR
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    gui = CorrelateTTTR()
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
