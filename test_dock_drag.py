import sys
from qtpy import QtWidgets, QtCore, QtGui

class EventFilter(QtCore.QObject):
    def __init__(self, mdi, dock):
        super().__init__()
        self.mdi = mdi
        self.dock = dock

    def eventFilter(self, obj, event):
        if obj == self.dock:
             # print(f"Event for dock: {event.type()}")
             pass

        # 3 = MouseButtonRelease, 175 = NonClientAreaMouseButtonRelease
        if event.type() in (3, 175):
            if isinstance(obj, QtWidgets.QDockWidget) and obj.isFloating():
                pos = QtGui.QCursor.pos()
                mdi_rect = self.mdi.rect()
                top_left = self.mdi.mapToGlobal(mdi_rect.topLeft())
                bottom_right = self.mdi.mapToGlobal(mdi_rect.bottomRight())
                global_rect = QtCore.QRect(top_left, bottom_right)
                if global_rect.contains(pos):
                    print("DROPPED ON MDI!")
                    w = obj.widget()
                    if w:
                        title = obj.windowTitle()
                        w.setProperty("_original_dock_name", obj.objectName())
                        obj.setParent(None)
                        obj.close()
                        sub = self.mdi.addSubWindow(w)
                        sub.setWindowTitle(title)
                        w.show()
                        sub.show()
                        print(f"Widget moved to MDI. Dock widget() is now: {obj.widget()}")
                    return True
        
        # 17 = Show, 12 = ShowWindowRequest
        if event.type() in (17, 12):
            if isinstance(obj, QtWidgets.QDockWidget):
                print(f"Dock {obj.objectName()} SHOW/SHOW_REQ event. widget is: {obj.widget()}")
                if obj.widget() is None:
                    for sub in self.mdi.subWindowList():
                        w = sub.widget()
                        if w and w.property("_original_dock_name") == obj.objectName():
                            print(f"RE-DOCKING: Restoring {obj.objectName()} from MDI")
                            sub.setWidget(None)
                            sub.close()
                            obj.setWidget(w)
                            w.show()
                            return True
        
        return super().eventFilter(obj, event)

app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QMainWindow()
mdi = QtWidgets.QMdiArea()
win.setCentralWidget(mdi)
win.resize(800, 600)

dock = QtWidgets.QDockWidget("Test Dock")
dock.setObjectName("myDock")
w = QtWidgets.QLabel("I am a dock widget content")
dock.setWidget(w)
win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

ef = EventFilter(mdi, dock)
app.installEventFilter(ef)

win.show()

def simulate():
    print("\n--- Phase 1: Drag to MDI ---")
    dock.setFloating(True)
    center = mdi.mapToGlobal(mdi.rect().center())
    dock.move(center)
    evt = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, QtCore.QPoint(0,0), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    app.postEvent(dock, evt)
    
    # After 1 second, simulate "reenabling" the dock
    QtCore.QTimer.singleShot(1500, simulate_redock)

def simulate_redock():
    print("\n--- Phase 2: Re-enable Dock ---")
    dock.show()

QtCore.QTimer.singleShot(1000, simulate)
QtCore.QTimer.singleShot(4000, lambda: sys.exit(0))
sys.exit(app.exec_())
