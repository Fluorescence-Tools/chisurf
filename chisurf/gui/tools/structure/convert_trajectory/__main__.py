import sys
from chisurf.gui.tools.structure.convert_trajectory import MDConverter
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    gui = MDConverter()
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
