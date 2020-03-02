import sys
from chisurf.tools.structure.convert_trajectory import MDConverter
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MDConverter()
    win.show()
    app.exec_()

