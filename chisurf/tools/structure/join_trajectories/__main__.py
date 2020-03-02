import sys
from chisurf.tools.structure.join_trajectories import JoinTrajectoriesWidget
from qtpy import QtWidgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = JoinTrajectoriesWidget()
    win.show()
    sys.exit(app.exec_())
