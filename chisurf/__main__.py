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
                self.setWindowTitle("ChiSurf Error")
                self.setMinimumWidth(500)

                layout = QtWidgets.QVBoxLayout()

                # Error message
                error_label = QtWidgets.QLabel("ChiSurf failed to start due to an error:")
                layout.addWidget(error_label)

                # Exception details in a text box
                exception_text_box = QtWidgets.QTextEdit()
                exception_text_box.setReadOnly(True)
                exception_text_box.setText(exception_text)
                exception_text_box.setMinimumHeight(200)
                layout.addWidget(exception_text_box)

                # Question about clearing settings
                question_label = QtWidgets.QLabel("Would you like to clear user settings? This might resolve the issue.")
                layout.addWidget(question_label)

                # Buttons
                button_box = QtWidgets.QHBoxLayout()

                clear_button = QtWidgets.QPushButton("Clear Settings")
                clear_button.clicked.connect(self.clear_settings)

                cancel_button = QtWidgets.QPushButton("Cancel")
                cancel_button.clicked.connect(self.reject)

                button_box.addWidget(clear_button)
                button_box.addWidget(cancel_button)

                layout.addLayout(button_box)
                self.setLayout(layout)

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
                    self.setWindowTitle("ChiSurf Error")
                    self.setMinimumWidth(500)

                    layout = QtWidgets.QVBoxLayout()

                    # Error message
                    error_label = QtWidgets.QLabel("ChiSurf failed to start due to an error:")
                    layout.addWidget(error_label)

                    # Exception details in a text box
                    exception_text_box = QtWidgets.QTextEdit()
                    exception_text_box.setReadOnly(True)
                    exception_text_box.setText(exception_text)
                    exception_text_box.setMinimumHeight(200)
                    layout.addWidget(exception_text_box)

                    # Question about clearing settings
                    question_label = QtWidgets.QLabel("Would you like to clear user settings? This might resolve the issue.")
                    layout.addWidget(question_label)

                    # Buttons
                    button_box = QtWidgets.QHBoxLayout()

                    clear_button = QtWidgets.QPushButton("Clear Settings")
                    clear_button.clicked.connect(self.clear_settings)

                    cancel_button = QtWidgets.QPushButton("Cancel")
                    cancel_button.clicked.connect(self.reject)

                    button_box.addWidget(clear_button)
                    button_box.addWidget(cancel_button)

                    layout.addLayout(button_box)
                    self.setLayout(layout)

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
