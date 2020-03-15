import sys
from chisurf.gui.tools.structure.remove_clashed_frames import RemoveClashedFrames
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RemoveClashedFrames()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
