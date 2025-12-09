import sys
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from qtpy.QtWidgets import QApplication
from chisurf.plugins.updater import UpdaterWidget

# Create the application
app = QApplication(sys.argv)

# Create the updater widget
widget = UpdaterWidget()

# Show the widget
widget.show()

# Print test instructions
print("Test Instructions:")
print("1. Enter a local folder path (e.g., Q:\\chisurf\\conda) in the URL input field")
print("2. Click 'Check for Updates'")
print("3. Verify that the available versions combobox is populated with versions from the local folder")
print("4. Enter a remote URL (e.g., https://www.peulen.xyz/downloads/chisurf/conda/) in the URL input field")
print("5. Click 'Check for Updates'")
print("6. Verify that the available versions combobox is populated with versions from the remote URL")
print("7. Use the Browse button to select a local folder")
print("8. Verify that the URL input field is updated with the selected folder path")
print("9. Click 'Check for Updates'")
print("10. Verify that the available versions combobox is populated with versions from the selected folder")

# Run the application
sys.exit(app.exec())