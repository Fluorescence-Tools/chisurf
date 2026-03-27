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

# Print test information
print("Test Instructions:")
print("1. Verify that the 'Remote URL' radio button is selected by default")
print("2. Verify that the URL input field shows a remote URL (e.g., https://www.peulen.xyz/downloads/chisurf/conda/)")
print("3. Switch to 'Local Folder' and verify that the URL input field updates to show a local path (if available)")
print("4. Enter a new local path and switch back to 'Remote URL'")
print("5. Verify that the URL input field updates to show the remote URL again")
print("6. Enter a new remote URL and switch back to 'Local Folder'")
print("7. Verify that the URL input field updates to show the local path again")
print("8. Click 'Apply' and verify that the URL is saved in settings")

# Run the application
sys.exit(app.exec())