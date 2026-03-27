import sys
import pathlib
import os

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

print("Test Instructions:")
print("1. Select 'Remote URL' and enter a valid URL (e.g., https://www.peulen.xyz/downloads/chisurf/conda/)")
print("2. Click 'Apply'")
print("3. Click 'Check for Updates'")
print("4. Verify that the available versions combobox is populated with versions from the remote source")
print("5. Select 'Local Folder' and enter a valid local path (e.g., Q:\\chisurf\\conda)")
print("6. Click 'Apply'")
print("7. Click 'Check for Updates'")
print("8. Verify that the available versions combobox is populated with versions from the local source")

# Run the application
sys.exit(app.exec())