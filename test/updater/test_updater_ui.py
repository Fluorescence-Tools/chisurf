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

# Run the application
sys.exit(app.exec())