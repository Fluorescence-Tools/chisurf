import sys
from PyQt5.QtWidgets import QApplication


def main():
    """Entry point to launch the ChiSurf Updater GUI standalone."""
    # Reuse existing QApplication if present (e.g., when launched from another Qt app)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Import here to avoid side effects at module import time
    from . import UpdaterWidget

    widget = UpdaterWidget()
    widget.show()

    # Start the Qt event loop
    # Handle both PyQt5 exec_ and newer exec naming
    if hasattr(app, 'exec_'):
        sys.exit(app.exec_())
    else:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
