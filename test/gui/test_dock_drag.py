import sys
import pytest
from qtpy import QtWidgets, QtCore, QtGui

class EventFilter(QtCore.QObject):
    def __init__(self, mdi, dock):
        super().__init__()
        self.mdi = mdi
        self.dock = dock

    def eventFilter(self, obj, event):
        # 3 = MouseButtonRelease, 175 = NonClientAreaMouseButtonRelease
        if event.type() in (3, 175):
            if isinstance(obj, QtWidgets.QDockWidget) and obj.isFloating():
                pos = QtGui.QCursor.pos()
                mdi_rect = self.mdi.rect()
                top_left = self.mdi.mapToGlobal(mdi_rect.topLeft())
                bottom_right = self.mdi.mapToGlobal(mdi_rect.bottomRight())
                global_rect = QtCore.QRect(top_left, bottom_right)
                if global_rect.contains(pos):
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
                    return True
        
        # 17 = Show, 12 = ShowWindowRequest
        if event.type() in (17, 12):
            if isinstance(obj, QtWidgets.QDockWidget):
                if obj.widget() is None:
                    for sub in self.mdi.subWindowList():
                        w = sub.widget()
                        if w and w.property("_original_dock_name") == obj.objectName():
                            sub.setWidget(None)
                            sub.close()
                            obj.setWidget(w)
                            w.show()
                            return True
        
        return super().eventFilter(obj, event)

@pytest.fixture
def dock_mdi_setup(qtbot):
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    mdi = QtWidgets.QMdiArea()
    win.setCentralWidget(mdi)
    win.resize(800, 600)

    dock = QtWidgets.QDockWidget("Test Dock")
    dock.setObjectName("myDock")
    w = QtWidgets.QLabel("I am a dock widget content")
    dock.setWidget(w)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    ef = EventFilter(mdi, dock)
    QtWidgets.QApplication.instance().installEventFilter(ef)
    
    win.show()
    return win, mdi, dock

def test_dock_drag_to_mdi(dock_mdi_setup, qtbot):
    win, mdi, dock = dock_mdi_setup
    
    # Simulate Drag to MDI
    dock.setFloating(True)
    center = mdi.mapToGlobal(mdi.rect().center())
    dock.move(center)
    
    # Position cursor at the center of MDI to satisfy EventFilter logic
    QtGui.QCursor.setPos(center)
    
    # Post MouseButtonRelease event
    evt = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, QtCore.QPoint(0,0), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.postEvent(dock, evt)
    
    # Process events to allow Filter to run
    qtbot.wait(500)
    
    # Assert widget moved to MDI
    assert len(mdi.subWindowList()) == 1
    assert dock.widget() is None

def test_redock_from_mdi(dock_mdi_setup, qtbot):
    win, mdi, dock = dock_mdi_setup
    
    # First move to MDI
    dock.setFloating(True)
    center = mdi.mapToGlobal(mdi.rect().center())
    dock.move(center)
    QtGui.QCursor.setPos(center)
    evt = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, QtCore.QPoint(0,0), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.postEvent(dock, evt)
    qtbot.wait(500)
    
    assert len(mdi.subWindowList()) == 1
    
    # Now simulate showing the empty dock to trigger re-docking
    dock.show()
    qtbot.wait(500)
    
    # Assert widget moved back to Dock
    assert len(mdi.subWindowList()) == 0
    assert dock.widget() is not None
    assert dock.widget().text() == "I am a dock widget content"
