import sys
from chisurf.gui.tools.tttr.convert import TTTRConvert
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    gui = TTTRConvert()
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
