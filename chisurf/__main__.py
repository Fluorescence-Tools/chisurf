from __future__ import annotations

import sys
import traceback


def main():
    try:
        # Import Qt and settings modules inside the try block to catch import errors
        from PyQt5 import QtWidgets, QtCore
        from chisurf.settings import clear_settings_folder, clear_logging_files
        from chisurf.gui import get_app

        # Define the dialog class inside the try block to ensure Qt is available
        class ClearSettingsDialog(QtWidgets.QDialog):
            def __init__(self, exception_text, parent=None):
                super().__init__(parent)

                # Load the UI file
                import pathlib
                from PyQt5 import uic
                uic.loadUi(pathlib.Path(__file__).parent / "gui" / "clear_settings_dialog.ui", self)

                # Set the exception text
                self.exception_text_box.setText(exception_text)

                # Connect signals
                self.clear_button.clicked.connect(self.clear_settings)
                self.cancel_button.clicked.connect(self.reject)

            def clear_settings(self):
                try:
                    clear_settings_folder()
                    clear_logging_files()
                    QtWidgets.QMessageBox.information(
                        self, 
                        "Settings Cleared", 
                        "User settings have been cleared. Please restart ChiSurf."
                    )
                    self.accept()
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, 
                        "Error", 
                        f"Failed to clear settings: {str(e)}"
                    )

        # Start the application
        app = get_app()
        sys.exit(app.exec_())

    except Exception as e:
        # Handle the exception
        try:
            # Try to use PyQt5 for the error dialog
            from PyQt5 import QtWidgets

            # Create a basic QApplication if one doesn't exist yet
            if not QtWidgets.QApplication.instance():
                app = QtWidgets.QApplication(sys.argv)

            # Format the exception traceback
            exception_text = f"{str(e)}\n\n{traceback.format_exc()}"

            # Define a simplified dialog class for error handling
            class SimpleErrorDialog(QtWidgets.QDialog):
                def __init__(self, exception_text, parent=None):
                    super().__init__(parent)

                    # Load the UI file
                    import pathlib
                    from PyQt5 import uic
                    uic.loadUi(pathlib.Path(__file__).parent / "gui" / "simple_error_dialog.ui", self)

                    # Set the exception text
                    self.exception_text_box.setText(exception_text)

                    # Connect signals
                    self.clear_button.clicked.connect(self.clear_settings)
                    self.cancel_button.clicked.connect(self.reject)

                def clear_settings(self):
                    try:
                        # Import settings functions directly
                        from chisurf.settings import clear_settings_folder, clear_logging_files
                        clear_settings_folder()
                        clear_logging_files()
                        QtWidgets.QMessageBox.information(
                            self, 
                            "Settings Cleared", 
                            "User settings have been cleared. Please restart ChiSurf."
                        )
                        self.accept()
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(
                            self, 
                            "Error", 
                            f"Failed to clear settings: {str(e)}"
                        )

            # Show the dialog
            dialog = SimpleErrorDialog(exception_text)
            result = dialog.exec_()

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
