from qtpy import QtCore, QtGui, QtWidgets


class DockDropOverlay(QtWidgets.QWidget):
    """A translucent overlay that highlights the active drop zone within the DockArea.

    This widget is a child of the DockArea and is drawn on top of other widgets.
    It does not capture mouse events.
    """

    def __init__(self, parent: QtWidgets.QWidget):
        """Initialize the dock drop overlay.

        Parameters
        ----------
        parent : QWidget
            The parent DockArea widget.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.hide()

    def set_highlight(self, rect: QtCore.QRect) -> None:
        """Set the geometry of the overlay relative to the parent DockArea.

        Parameters
        ----------
        rect : QRect
            The bounding rectangle relative to the parent to highlight.
        """
        if rect.isNull():
            self.hide()
            return
        self.setGeometry(rect)
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draw the premium styled translucent overlay.

        Parameters
        ----------
        event : QPaintEvent
            The paint event.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        color = QtGui.QColor(0, 150, 255, 60)
        border_color = QtGui.QColor(0, 150, 255, 180)

        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect().adjusted(2, 2, -2, -2)), 6, 6)

        painter.fillPath(path, color)
        painter.setPen(QtGui.QPen(border_color, 2))
        painter.drawPath(path)
