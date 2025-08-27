"""
Screenshot

A minimal plugin that captures a screenshot of the ChiSurf main window.

Behavior:
- When launched, it prompts the user where to save the screenshot.
- It captures the current main window and saves the image.
- It closes immediately after saving (or if the user cancels).
"""

# Display name used by the Plugins menu (category: name)
name = "Tools:Screenshot"

import pathlib

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


ess_image_filters = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp);;All Files (*.*)"


def _run_screenshot():
    """Prompt for a filename (pre-filled with current date), grab main window screenshot, save, and exit."""
    try:
        from chisurf.gui.widgets import general as _general
    except Exception:
        _general = None

    win = _find_main_window()
    if win is None:
        try:
            chisurf.logging.error("Screenshot plugin: Main window not found.")
        except Exception:
            pass
        return

    # Build default filename based on current local date/time
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        ts = "screenshot"

    # Determine working directory
    _wp = getattr(chisurf, 'working_path', None)
    if not _wp:
        _wp = pathlib.Path.home()

    # Construct the initial path with suggested filename
    initial_path = pathlib.Path(_wp) / f"{ts}.png"

    # Ask for filename using QFileDialog directly so we can pass a pre-filled path
    filename, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
        None,
        caption="Save screenshot",
        dir=str(initial_path),
        filter=ess_image_filters
    )

    # Update working_path if the user chose a path
    if filename:
        try:
            chisurf.working_path = pathlib.Path(filename).parent
        except Exception:
            pass

    if not filename:
        # User cancelled, just exit
        return

    path = pathlib.Path(filename)

    # If user didn't provide an extension, infer from selected filter, default to .png
    if path.suffix == "":
        suffix = ".png"
        try:
            sel = (selected_filter or "").lower()
            if "*.jpg" in sel or "*.jpeg" in sel:
                suffix = ".jpg"
            elif "*.bmp" in sel:
                suffix = ".bmp"
        except Exception:
            pass
        path = path.with_suffix(suffix)

    # Ensure the UI has processed pending paints before grabbing
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()

    try:
        pixmap = win.grab()  # QPixmap of the widget
        ok = pixmap.save(str(path))
        if ok:
            try:
                chisurf.logging.info(f"Screenshot saved to {path}")
            except Exception:
                pass
        else:
            try:
                chisurf.logging.error(f"Failed to save screenshot to {path}")
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
