import sys
from chisurf.gui.tools.structure.join_trajectories import JoinTrajectoriesWidget
from qtpy import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = JoinTrajectoriesWidget()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
