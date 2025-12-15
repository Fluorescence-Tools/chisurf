import sys
from .widget import MDConverter
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    gui = MDConverter()
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
