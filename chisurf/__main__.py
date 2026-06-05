from __future__ import annotations

import sys
import traceback
import pathlib

try:
    from qtpy import QtWidgets, uic
except Exception:
    QtWidgets = None
    uic = None


# Define a simplified dialog class for error handling
class SimpleErrorDialog(QtWidgets.QDialog if QtWidgets is not None else object):
    def __init__(self, exception_text, parent=None):
        if QtWidgets is None or uic is None:
            raise RuntimeError("Qt is not available for SimpleErrorDialog")

        super().__init__(parent)

        uic.loadUi(
            pathlib.Path(__file__).parent / "gui" / "widgets" / "simple_error_dialog.ui",
            self,
        )

        # Set the exception text
        self.exception_text_box.setText(exception_text)

        # Connect signals
        self.clear_button.clicked.connect(self.clear_settings)
        self.cancel_button.clicked.connect(self.reject)

    def clear_settings(self):
        if QtWidgets is None:
            raise RuntimeError("Qt is not available for SimpleErrorDialog")
        try:
            # Import settings functions directly
            from chisurf.core.settings import clear_settings_folder, clear_logging_files

            clear_settings_folder()
            clear_logging_files()
            QtWidgets.QMessageBox.information(
                self,
                "Settings Cleared",
                "User settings have been cleared. Please restart ChiSurf.",
            )
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to clear settings: {str(e)}",
            )


def main():
    try:
        # Enable faulthandler to dump C-level traceback on segfault before
        # the OS kills the process.  The output file is opened at startup to
        # avoid allocation inside the signal handler.
        import faulthandler
        from datetime import datetime
        # faulthandler.enable() already registers SIGSEGV, SIGABRT, SIGBUS,
        # SIGFPE, SIGILL — do NOT call faulthandler.register(signal.SIGSEGV)
        # on top of it (that raises RuntimeError).
        _crash_log = pathlib.Path.home() / ".chisurf" / f"crash_{datetime.now():%Y%m%d_%H%M%S}.log"
        _crash_log.parent.mkdir(parents=True, exist_ok=True)
        _fh = _crash_log.open("w")
        faulthandler.enable(file=_fh, all_threads=True)

        # Import Qt and settings modules inside the try block to catch import errors
        from chisurf.core.settings import clear_settings_folder, clear_logging_files
        from chisurf.gui import get_app

        # Start the application
        app = get_app()
        try:
            exit_code = app.exec()
        finally:
            _fh.flush()
            _fh.close()

        # Hard-exit the process after the Qt event loop finishes. This avoids
        # running full Python interpreter finalization (Py_Finalize), which can
        # trigger shutdown-time crashes in C extensions (e.g. Qt bindings) when
        # complex object graphs are torn down.
        import os
        os._exit(exit_code)

    except Exception as e:
        # Always print to stderr for CLI users and logging
        exception_text = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"ChiSurf encountered an error during startup:\n{exception_text}", file=sys.stderr)

        # Handle the exception
        try:
            if QtWidgets is None:
                raise RuntimeError("Qt is not available")

            # Create a basic QApplication if one doesn't exist yet
            if not QtWidgets.QApplication.instance():
                app = QtWidgets.QApplication(sys.argv)

            # Format the exception traceback
            exception_text = f"{str(e)}\n\n{traceback.format_exc()}"

            # Show the dialog
            dialog = SimpleErrorDialog(exception_text)
            result = dialog.exec()

        except Exception as inner_e:
            # If PyQt5 fails, fall back to console output
            print(f"ChiSurf failed to start: {e}")
            print(f"Additionally, failed to show error dialog: {inner_e}")
            print(traceback.format_exc())
            print("\nTo clear settings manually, delete the folder: ~/.chisurf")

        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()
