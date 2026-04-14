"""
Screenshot

A minimal plugin that captures a screenshot of the ChiSurf main window and copies it to the clipboard.

Behavior:
- It captures the current main window and copies the image to the clipboard.
- It displays a temporary message box confirming the action.
"""

# Display name used by the Plugins menu (category: name)
name = "Main:Tools:Screenshot"

import chisurf
from chisurf.gui import QtWidgets


def _find_main_window():
    """Return the main window instance if available, else None."""
    try:
        # Import here to avoid heavy imports on module load
        from chisurf.gui.main import Main as _Main
    except Exception:
        _Main = None

    app = QtWidgets.QApplication.instance()
    if app is None:
        return None

    # Prefer an explicit instance of our Main window
    if _Main is not None:
        for w in app.topLevelWidgets():
            if isinstance(w, _Main):
                return w

    # Fallback to the active window if it's a QMainWindow
    w = app.activeWindow()
    if isinstance(w, QtWidgets.QMainWindow):
        return w

    # As a last resort, return the first visible main window-like widget
    for w in app.topLevelWidgets():
        if isinstance(w, QtWidgets.QMainWindow) and w.isVisible():
            return w

    return None


def _run_screenshot():
    """Grab main window screenshot, copy to clipboard, and exit."""

    win = _find_main_window()
    if win is None:
        try:
            chisurf.logging.error("Screenshot plugin: Main window not found.")
        except Exception:
            pass
        return

    # Ensure the UI has processed pending paints before grabbing
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()

    # Grab pixmap of the window and copy to clipboard immediately
    try:
        pixmap = win.grab()  # QPixmap of the widget
        if not pixmap.isNull():
            cb = QtWidgets.QApplication.clipboard()
            if cb is not None:
                cb.setPixmap(pixmap)

                # Feedback in status bar
                try:
                    if hasattr(win, 'statusBar') and win.statusBar() is not None:
                        win.statusBar().showMessage("Screenshot copied to clipboard", 3000)
                except Exception:
                    pass

                # Show a temporary "Toast" message overlay
                try:
                    from chisurf.gui import QtCore
                    # Create a styled label
                    toast = QtWidgets.QLabel("Screenshot copied to clipboard", win)
                    toast.setStyleSheet("""
                        QLabel {
                            background-color: #222222;
                            color: #ffffff;
                            padding: 15px;
                            border-radius: 8px;
                            font-size: 14px;
                            border: 1px solid #444444;
                        }
                    """)
                    toast.setAlignment(QtCore.Qt.AlignCenter)
                    # Make it a frameless, non-interactive overlay
                    toast.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.ToolTip | QtCore.Qt.WindowStaysOnTopHint)
                    toast.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
                    toast.adjustSize()

                    # Center it relative to the main window
                    try:
                        main_rect = win.geometry()
                        toast_rect = toast.geometry()
                        x = main_rect.x() + (main_rect.width() - toast_rect.width()) // 2
                        y = main_rect.y() + (main_rect.height() - toast_rect.height()) // 2
                        toast.move(x, y)
                    except Exception:
                        pass

                    toast.show()

                    # Auto-destruct after 1.5 seconds
                    # We store the reference on the window to prevent premature garbage collection
                    if not hasattr(win, '_screenshot_toasts'):
                        win._screenshot_toasts = []
                    win._screenshot_toasts.append(toast)

                    def _cleanup():
                        try:
                            toast.hide()
                            toast.deleteLater()
                            if toast in win._screenshot_toasts:
                                win._screenshot_toasts.remove(toast)
                        except Exception:
                            pass

                    QtCore.QTimer.singleShot(500, _cleanup)

                except Exception:
                    pass

                try:
                    chisurf.logging.info("Screenshot copied to clipboard.")
                except Exception:
                    pass
        else:
            try:
                chisurf.logging.error("Screenshot plugin: Grabbed pixmap is null.")
            except Exception:
                pass
    except Exception as e:
        try:
            chisurf.logging.error(f"Screenshot plugin failed: {e}")
        except Exception:
            pass


# Execute when run by the plugin loader
if __name__ == "plugin":
    _run_screenshot()
