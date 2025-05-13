"""
Spectra Viewer

This plugin provides a PyQt5 GUI application to view and compare absorption and emission spectra
of various fluorophores. It includes functionality to display dye structures/images and can
integrate with the download_atto script to download data for ATTO dyes.
"""

import os
import sys
import json
import numpy as np
import subprocess
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import chisurf
from chisurf.fio.ascii import Csv, load_xy
from chisurf.math.datatools import overlapping_region, align_x_spacing, minmax

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Spectra Viewer"

# Load the plugin icon
icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
if os.path.exists(icon_path):
    icon = QtGui.QIcon(icon_path)
else:
    # If icon doesn't exist, try to create it
    try:
        import icon
        icon = QtGui.QIcon(icon_path)
    except:
        icon = None

class SpectraViewerWidget(QtWidgets.QMainWindow):
    """Main widget for the Spectra Viewer plugin."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectra Viewer")
        self.resize(1000, 800)

        # Initialize data structures
        self.plugin_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.spectra_dir = self.plugin_dir / "spectra"
        self.fluorophores_dir = self.spectra_dir / "fluorophores"
        self.filters_dir = self.spectra_dir / "filters"

        # Create directories if they don't exist
        self.fluorophores_dir.mkdir(exist_ok=True)
        self.spectra_dir.mkdir(exist_ok=True)
        self.filters_dir.mkdir(exist_ok=True)

        # Ensure the download directory exists
        download_dir = self.plugin_dir / "download"
        download_dir.mkdir(exist_ok=True)

        # Initialize data containers
        self.current_item = None
        self.current_item_data = None
        self.absorption_spectrum = None
        self.emission_spectrum = None
        self.item_image = None
        self.item_metadata = None

        # Setup menubar
        self.setup_menubar()

        # Setup UI
        self.setup_ui()

        # Populate comboboxes with available items
        self.populate_item_list()

    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        main_layout.addWidget(left_panel)

        # Right panel for plots and images
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel)

        # Spectra selection group
        spectra_group = QtWidgets.QGroupBox("Spectra Selection")
        spectra_layout = QtWidgets.QVBoxLayout(spectra_group)
        left_layout.addWidget(spectra_group)

        # Spectra type selection
        spectra_type_layout = QtWidgets.QHBoxLayout()
        spectra_layout.addLayout(spectra_type_layout)

        spectra_type_layout.addWidget(QtWidgets.QLabel("Type:"))
        self.item_type_combo = QtWidgets.QComboBox()

        # Dynamically populate item types based on available directories
        self.populate_item_types()

        self.item_type_combo.currentIndexChanged.connect(self.on_item_type_changed)
        spectra_type_layout.addWidget(self.item_type_combo)

        # Item selection
        item_select_layout = QtWidgets.QHBoxLayout()
        spectra_layout.addLayout(item_select_layout)

        item_select_layout.addWidget(QtWidgets.QLabel("Item:"))
        self.item_combo = QtWidgets.QComboBox()
        self.item_combo.currentIndexChanged.connect(self.on_item_changed)
        item_select_layout.addWidget(self.item_combo)

        # Item information group
        info_group = QtWidgets.QGroupBox("Item Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        left_layout.addWidget(info_group)

        # Item description
        info_layout.addWidget(QtWidgets.QLabel("Description:"))
        self.item_description = QtWidgets.QTextEdit()
        self.item_description.setReadOnly(True)
        self.item_description.setMaximumHeight(100)
        info_layout.addWidget(self.item_description)

        # Optical properties
        info_layout.addWidget(QtWidgets.QLabel("Optical Properties:"))
        self.optical_properties = QtWidgets.QTableWidget()
        self.optical_properties.setColumnCount(2)
        self.optical_properties.setHorizontalHeaderLabels(["Property", "Value"])
        self.optical_properties.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.optical_properties.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        info_layout.addWidget(self.optical_properties)

        # Display options group
        display_group = QtWidgets.QGroupBox("Display Options")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        left_layout.addWidget(display_group)

        # Checkboxes for what to display
        self.show_abs = QtWidgets.QCheckBox("Show Absorption Spectrum")
        self.show_abs.setChecked(True)
        self.show_abs.stateChanged.connect(self.update_plots)
        display_layout.addWidget(self.show_abs)

        self.show_em = QtWidgets.QCheckBox("Show Emission Spectrum")
        self.show_em.setChecked(True)
        self.show_em.stateChanged.connect(self.update_plots)
        display_layout.addWidget(self.show_em)

        self.normalize_spectra = QtWidgets.QCheckBox("Normalize Spectra")
        self.normalize_spectra.setChecked(True)
        self.normalize_spectra.stateChanged.connect(self.update_plots)
        display_layout.addWidget(self.normalize_spectra)

        # Add stretch to push everything to the top
        left_layout.addStretch()

        # Create plot widget for spectra
        self.spectra_plot = pg.PlotWidget()
        self.spectra_plot.setBackground('w')
        self.spectra_plot.setLabel('left', 'Intensity', units='a.u.')
        self.spectra_plot.setLabel('bottom', 'Wavelength', units='nm')
        # Disable SI prefix for wavelength axis to show 'nm' instead of 'knm'
        self.spectra_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.spectra_plot.addLegend()
        right_layout.addWidget(self.spectra_plot)

        # Create a label for the structure images section
        structure_label = QtWidgets.QLabel("Images")
        structure_label.setAlignment(QtCore.Qt.AlignCenter)
        structure_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(structure_label)

        # Create image view for structure
        self.structure_scroll = QtWidgets.QScrollArea()
        self.structure_scroll.setWidgetResizable(True)
        self.structure_content = QtWidgets.QWidget()
        self.structure_layout = QtWidgets.QHBoxLayout(self.structure_content)
        self.structure_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.structure_layout.setSpacing(10)  # Add some spacing between images
        self.structure_layout.setContentsMargins(10, 10, 10, 10)  # Add margins around the images
        self.structure_scroll.setWidget(self.structure_content)
        self.structure_scroll.setMaximumHeight(250)  # Limit the height of the scroll area
        right_layout.addWidget(self.structure_scroll)


    def populate_download_combo(self):
        """Populate the download combobox with available download scripts."""
        # Clear the combobox first
        self.download_combo.clear()

        # Get available download scripts
        download_scripts = self.get_available_download_scripts()

        # Add each script to the combobox
        for script_name in download_scripts.keys():
            self.download_combo.addItem(script_name)

        # If no scripts were found, add a placeholder
        if self.download_combo.count() == 0:
            self.download_combo.addItem("No download scripts found")

    def download_selected_data(self):
        """Download data from the selected source."""
        # Get the selected script name
        script_name = self.download_combo.currentText()

        # Skip if no valid script is selected
        if script_name == "No download scripts found":
            QtWidgets.QMessageBox.warning(self, "Warning", "No download scripts available.")
            return

        # Find the corresponding action in the menu and trigger it
        for action in self.menuBar().findChildren(QtWidgets.QAction):
            if action.text() == f"Download {script_name}":
                action.trigger()
                return

        # Fallback if the action is not found
        QtWidgets.QMessageBox.warning(self, "Warning", f"Download script for {script_name} not found.")

    def populate_item_list(self):
        """Populate the item combobox with available items based on the selected type."""
        # Clear the combobox first
        self.item_combo.clear()

        # Get the selected type display name
        item_type_display = self.item_type_combo.currentText()

        # Skip if no item type is selected or if it's the placeholder
        if not item_type_display or item_type_display == "No item types found":
            self.item_combo.addItem("No items found")
            return

        # Convert display name to directory name (e.g., "Atto Dyes" -> "atto_dyes")
        item_type_dir = item_type_display.lower().replace(' ', '_')

        # Look for items in the corresponding directory
        type_dir = self.fluorophores_dir / item_type_dir
        if type_dir.exists():
            # Get all subdirectories (each is an item)
            items = [d.name for d in type_dir.iterdir() if d.is_dir()]
            items.sort()
            self.item_combo.addItems(items)

        # If no items were found, add a placeholder
        if self.item_combo.count() == 0:
            self.item_combo.addItem(f"No {item_type_display} found - use Download from menu")

    def populate_item_types(self):
        """Dynamically populate the item type combo box based on available directories."""
        # Clear the combo box
        self.item_type_combo.clear()

        # Check if the fluorophores directory exists
        if not self.fluorophores_dir.exists():
            self.fluorophores_dir.mkdir(exist_ok=True)
            self.item_type_combo.addItem("No item types found")
            return

        # Get all subdirectories in the fluorophores directory
        item_types = []
        for d in self.fluorophores_dir.iterdir():
            if d.is_dir():
                # Convert directory name to a display name (e.g., "atto_dyes" -> "Atto Dyes")
                display_name = d.name.replace('_', ' ').title()
                item_types.append((display_name, d.name))

        # Sort item types alphabetically
        item_types.sort()

        # Add item types to the combo box
        if item_types:
            for display_name, _ in item_types:
                self.item_type_combo.addItem(display_name)
        else:
            self.item_type_combo.addItem("No item types found")

    def on_item_type_changed(self):
        """Handle changes to the item type selection."""
        self.populate_item_list()

    def on_item_changed(self):
        """Handle changes to the item selection."""
        item_name = self.item_combo.currentText()

        # Skip if the item is a placeholder
        if item_name.startswith("No "):
            return

        # Get the item directory
        item_type_display = self.item_type_combo.currentText()

        # Convert display name to directory name (e.g., "Atto Dyes" -> "atto_dyes")
        item_type_dir = item_type_display.lower().replace(' ', '_')

        # Get the item directory
        item_dir = self.fluorophores_dir / item_type_dir / item_name

        # Load the item metadata
        metadata_file = item_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.item_metadata = json.load(f)

                # Update the description
                self.item_description.setText(self.item_metadata.get("description", ""))

                # Update the optical properties table
                optical_props = self.item_metadata.get("optical_properties", {})
                self.optical_properties.setRowCount(len(optical_props))
                for i, (prop, value) in enumerate(optical_props.items()):
                    self.optical_properties.setItem(i, 0, QtWidgets.QTableWidgetItem(prop))
                    self.optical_properties.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))

                # Load the absorption spectrum
                abs_file = self.item_metadata.get("absorption_spectrum", "")
                if abs_file:
                    abs_path = item_dir / "spectra" / abs_file
                    if abs_path.exists():
                        try:
                            x, y = load_xy(str(abs_path), delimiter="\t", skiprows=2)
                            self.absorption_spectrum = (x, y)
                        except Exception as e:
                            print(f"Error loading absorption spectrum: {e}")
                            self.absorption_spectrum = None
                else:
                    self.absorption_spectrum = None

                # Load the emission spectrum
                em_file = self.item_metadata.get("emission_spectrum", "")
                if em_file:
                    em_path = item_dir / "spectra" / em_file
                    if em_path.exists():
                        try:
                            x, y = load_xy(str(em_path), delimiter="\t", skiprows=2)
                            self.emission_spectrum = (x, y)
                        except Exception as e:
                            print(f"Error loading emission spectrum: {e}")
                            self.emission_spectrum = None
                else:
                    self.emission_spectrum = None

                # Load the structure images
                self.clear_structure_images()
                for img_file in self.item_metadata.get("structure_images", []):
                    img_path = item_dir / img_file
                    if img_path.exists():
                        self.add_structure_image(str(img_path))

                # Update the plots
                self.update_plots()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load item metadata: {str(e)}")
        else:
            QtWidgets.QMessageBox.warning(self, "Missing Metadata", 
                                         f"No metadata found for {item_name}. The item may not be properly installed.")


    def clear_structure_images(self):
        """Clear all structure images from the structure tab."""
        # Remove all widgets from the structure layout
        while self.structure_layout.count():
            item = self.structure_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_structure_image(self, image_path):
        """Add a structure image to the structure tab."""
        try:
            # Create a label for the image
            label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(image_path)

            # Scale the pixmap to a reasonable size while maintaining aspect ratio
            max_height = 200  # Maximum height for the images
            scaled_pixmap = pixmap.scaledToHeight(max_height, QtCore.Qt.SmoothTransformation)

            label.setPixmap(scaled_pixmap)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

            # Add the label to the structure layout
            self.structure_layout.addWidget(label)
        except Exception as e:
            print(f"Error adding structure image: {e}")

    def setup_menubar(self):
        """Set up the menubar with options."""
        # Create menubar
        menubar = self.menuBar()

        # Create Tools menu
        tools_menu = menubar.addMenu('Tools')

        # Create Download submenu
        download_menu = tools_menu.addMenu('Download Data')

        # Scan the download folder for available scripts
        download_scripts = self.get_available_download_scripts()

        # Add an action for each download script
        for script_name, script_path in download_scripts.items():
            download_action = QtWidgets.QAction(f'Download {script_name}', self)
            download_action.setData(script_path)
            download_action.triggered.connect(self.run_download_script)
            download_menu.addAction(download_action)

    def get_available_download_scripts(self):
        """Scan the download folder for available Python scripts."""
        download_scripts = {}
        download_dir = self.plugin_dir / "download"

        if download_dir.exists() and download_dir.is_dir():
            for file in download_dir.glob("*.py"):
                if file.is_file() and not file.name.startswith("__"):
                    # Use the filename without extension as the script name
                    script_name = file.stem.replace("_", " ").title()
                    download_scripts[script_name] = str(file)

        return download_scripts

    def run_download_script(self):
        """Run the selected download script and display the output in a popup window."""
        action = self.sender()
        if not action:
            return

        script_path = action.data()
        script_name = action.text().replace("Download ", "")

        try:
            # Use a directory named after the script for all downloads
            # This creates a consistent naming convention for all data types
            output_dir = self.fluorophores_dir / script_name.lower().replace(" ", "_")

            # Create the output directory if it doesn't exist
            output_dir.mkdir(exist_ok=True)

            # Create a dialog to show the output
            output_dialog = QtWidgets.QDialog(self)
            output_dialog.setWindowTitle(f"Download {script_name} Output")
            output_dialog.resize(800, 600)

            # Create a text edit to display the output
            output_text = QtWidgets.QTextEdit()
            output_text.setReadOnly(True)
            output_text.setFont(QtGui.QFont("Courier New", 10))

            # Create a layout for the dialog
            layout = QtWidgets.QVBoxLayout(output_dialog)
            layout.addWidget(output_text)

            # Add a close button
            close_button = QtWidgets.QPushButton("Close")
            close_button.clicked.connect(output_dialog.accept)
            layout.addWidget(close_button)

            # Show the dialog
            output_dialog.show()

            # Update the text edit to show that the script is running
            output_text.append(f"Running {os.path.basename(script_path)} script...\n")
            QtWidgets.QApplication.processEvents()

            # Run the script with the output directory as an argument
            process = subprocess.Popen(
                [sys.executable, script_path, "--output", str(output_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.plugin_dir)
            )

            # Read and display the output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_text.append(output.strip())
                    QtWidgets.QApplication.processEvents()

            # Get the return code
            return_code = process.poll()

            # Display any errors
            if return_code != 0:
                error_output = process.stderr.read()
                output_text.append(f"\nError (return code {return_code}):\n{error_output}")
            else:
                output_text.append("\nScript completed successfully.")

            # Refresh the item list
            self.populate_item_list()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run {os.path.basename(script_path)}: {str(e)}")


    def update_plots(self):
        """Update the plots with current data."""
        self.spectra_plot.clear()

        # Plot absorption spectrum if available and enabled
        if self.absorption_spectrum is not None and self.show_abs.isChecked():
            x, y = self.absorption_spectrum

            # Normalize if requested
            if self.normalize_spectra.isChecked():
                y = y / np.max(y)

            self.spectra_plot.plot(x, y, pen=pg.mkPen('b', width=2), name='Absorption')

        # Plot emission spectrum if available and enabled
        if self.emission_spectrum is not None and self.show_em.isChecked():
            x, y = self.emission_spectrum

            # Normalize if requested
            if self.normalize_spectra.isChecked():
                y = y / np.max(y)

            self.spectra_plot.plot(x, y, pen=pg.mkPen('r', width=2), name='Emission')

        # Set fixed x-axis range from 250 nm to 1000 nm for all dyes
        self.spectra_plot.setXRange(250, 1000)

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the SpectraViewerWidget class
    window = SpectraViewerWidget()
    # Show the window
    window.show()
