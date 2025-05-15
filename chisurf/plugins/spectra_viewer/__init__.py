"""
Spectra Viewer

This plugin provides a comprehensive tool for visualizing, comparing, and analyzing 
absorption and emission spectra of fluorophores. It allows users to:
1. Load and display multiple absorption and emission spectra simultaneously
2. Normalize spectra for easier comparison
3. Calculate Förster radius (R₀) for FRET pairs with customizable parameters
4. View and edit fluorophore metadata including optical properties
5. Display molecular structures and chemical diagrams of fluorophores
6. Add custom dyes with their spectra and properties
7. Download spectral data for commercial fluorophores

The plugin features an intuitive interface with interactive plots and supports 
side-by-side comparison of multiple fluorophores. This makes it particularly 
useful for:
- Selecting optimal FRET pairs for experiments
- Designing multiplexed fluorescence experiments
- Comparing spectral properties across different fluorophore families
- Educational purposes to understand spectral characteristics and FRET theory

The Förster radius calculator implements the complete overlap integral calculation
with all relevant parameters (orientation factor, quantum yield, refractive index),
providing accurate R₀ values for FRET experimental design.
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

# Import the database module
from .database import SpectraDatabase

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
        if os.path.exists(icon_path):
            icon = QtGui.QIcon(icon_path)
        else:
            # Create a default icon if the icon file doesn't exist
            icon = QtGui.QIcon()
    except Exception as e:
        print(f"Error loading icon for spectra_viewer: {e}")
        # Create a default icon if there's an error
        icon = QtGui.QIcon()

class SpectraViewerWidget(QtWidgets.QMainWindow):
    """Main widget for the Spectra Viewer plugin."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectra Viewer")
        self.resize(1000, 800)

        # Initialize data structures
        self.plugin_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Initialize database
        self.db = SpectraDatabase()

        # Create database tables if they don't exist
        with self.db:
            self.db.create_tables()

        # Define paths for backward compatibility (but don't create them)
        self.spectra_dir = self.plugin_dir / "spectra"
        self.fluorophores_dir = self.spectra_dir / "fluorophores"
        self.filters_dir = self.spectra_dir / "filters"

        # Ensure only the download directory exists (needed for scripts)
        download_dir = self.plugin_dir / "download"
        download_dir.mkdir(exist_ok=True, parents=True)

        # Initialize data containers
        self.current_item = None
        self.current_item_data = None
        self.current_item_id = None
        self.current_type_id = None
        self.absorption_spectrum = None
        self.emission_spectrum = None
        self.item_image = None
        self.item_metadata = None
        self.image_paths = []  # List to store image paths for the current item

        # Dictionary to store multiple spectra
        # Format: {spectrum_id: {'type': 'absorption'|'emission', 'name': name, 'data': (x, y), 'metadata': metadata}}
        self.loaded_spectra = {}

        # Selected spectra for display and calculations
        self.selected_spectra = []

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

        # Item information group (hidden by default)
        self.info_group = QtWidgets.QGroupBox("Item Information")
        info_layout = QtWidgets.QVBoxLayout(self.info_group)
        left_layout.addWidget(self.info_group)
        self.info_group.setVisible(False)  # Hide by default

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

        # Edit metadata button
        self.edit_metadata_button = QtWidgets.QPushButton("Edit Metadata")
        self.edit_metadata_button.clicked.connect(self.edit_metadata)
        info_layout.addWidget(self.edit_metadata_button)

        # Loaded spectra group
        loaded_spectra_group = QtWidgets.QGroupBox("Loaded Spectra")
        loaded_spectra_layout = QtWidgets.QVBoxLayout(loaded_spectra_group)
        left_layout.addWidget(loaded_spectra_group)

        # List widget for loaded spectra
        self.spectra_list = QtWidgets.QListWidget()
        self.spectra_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.spectra_list.itemSelectionChanged.connect(self.on_spectra_selection_changed)
        loaded_spectra_layout.addWidget(self.spectra_list)

        # Buttons for managing spectra
        spectra_buttons_layout = QtWidgets.QHBoxLayout()
        loaded_spectra_layout.addLayout(spectra_buttons_layout)

        self.add_spectrum_button = QtWidgets.QPushButton("Add Current")
        self.add_spectrum_button.clicked.connect(self.add_current_spectrum)
        spectra_buttons_layout.addWidget(self.add_spectrum_button)

        self.remove_spectrum_button = QtWidgets.QPushButton("Remove Selected")
        self.remove_spectrum_button.clicked.connect(self.remove_selected_spectra)
        spectra_buttons_layout.addWidget(self.remove_spectrum_button)

        # Display options group
        display_group = QtWidgets.QGroupBox("Display Options")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        left_layout.addWidget(display_group)

        # Display options

        self.normalize_spectra = QtWidgets.QCheckBox("Normalize Spectra")
        self.normalize_spectra.setChecked(True)
        self.normalize_spectra.stateChanged.connect(self.update_plots)
        display_layout.addWidget(self.normalize_spectra)

        # Checkbox to show/hide item information
        self.show_info_checkbox = QtWidgets.QCheckBox("Show Item Information")
        self.show_info_checkbox.setChecked(False)
        self.show_info_checkbox.stateChanged.connect(self.toggle_info_group)
        display_layout.addWidget(self.show_info_checkbox)

        # Förster radius calculation group
        forster_group = QtWidgets.QGroupBox("Förster Radius Calculation")
        forster_layout = QtWidgets.QVBoxLayout(forster_group)
        left_layout.addWidget(forster_group)
        forster_group.setEnabled(False)

        # Donor and acceptor selection
        donor_layout = QtWidgets.QHBoxLayout()
        forster_layout.addLayout(donor_layout)
        donor_layout.addWidget(QtWidgets.QLabel("Donor:"))
        self.donor_combo = QtWidgets.QComboBox()
        donor_layout.addWidget(self.donor_combo)

        acceptor_layout = QtWidgets.QHBoxLayout()
        forster_layout.addLayout(acceptor_layout)
        acceptor_layout.addWidget(QtWidgets.QLabel("Acceptor:"))
        self.acceptor_combo = QtWidgets.QComboBox()
        acceptor_layout.addWidget(self.acceptor_combo)

        # Parameters for Förster radius calculation
        params_layout = QtWidgets.QGridLayout()
        forster_layout.addLayout(params_layout)

        params_layout.addWidget(QtWidgets.QLabel("Orientation Factor (κ²):"), 0, 0)
        self.kappa_squared = QtWidgets.QDoubleSpinBox()
        self.kappa_squared.setRange(0.01, 4.0)
        self.kappa_squared.setValue(0.667)
        self.kappa_squared.setSingleStep(0.01)
        params_layout.addWidget(self.kappa_squared, 0, 1)

        params_layout.addWidget(QtWidgets.QLabel("Donor QY:"), 1, 0)
        self.donor_qy = QtWidgets.QDoubleSpinBox()
        self.donor_qy.setRange(0.01, 1.0)
        self.donor_qy.setValue(0.5)
        self.donor_qy.setSingleStep(0.01)
        params_layout.addWidget(self.donor_qy, 1, 1)

        params_layout.addWidget(QtWidgets.QLabel("Acceptor Ext. Coef. (M⁻¹cm⁻¹):"), 2, 0)
        self.acceptor_ext_coef = QtWidgets.QDoubleSpinBox()
        self.acceptor_ext_coef.setRange(1000, 300000)
        self.acceptor_ext_coef.setValue(100000)
        self.acceptor_ext_coef.setSingleStep(1000)
        self.acceptor_ext_coef.setDecimals(0)
        params_layout.addWidget(self.acceptor_ext_coef, 2, 1)

        params_layout.addWidget(QtWidgets.QLabel("Refractive Index:"), 3, 0)
        self.refractive_index = QtWidgets.QDoubleSpinBox()
        self.refractive_index.setRange(1.0, 2.0)
        self.refractive_index.setValue(1.33)
        self.refractive_index.setSingleStep(0.01)
        params_layout.addWidget(self.refractive_index, 3, 1)

        # Calculate button and result display
        calc_layout = QtWidgets.QHBoxLayout()
        forster_layout.addLayout(calc_layout)

        self.calculate_button = QtWidgets.QPushButton("Calculate R₀")
        self.calculate_button.clicked.connect(self.calculate_forster_radius)
        calc_layout.addWidget(self.calculate_button)

        self.forster_result = QtWidgets.QLineEdit()
        self.forster_result.setReadOnly(True)
        calc_layout.addWidget(self.forster_result)


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

        # Create a button to show images in a separate window
        self.show_images_button = QtWidgets.QPushButton("Show Images")
        self.show_images_button.clicked.connect(self.show_images_in_window)
        self.show_images_button.setEnabled(False)  # Disabled by default until images are available
        right_layout.addWidget(self.show_images_button)

        # Create image view for structure (hidden by default)
        self.structure_scroll = QtWidgets.QScrollArea()
        self.structure_scroll.setWidgetResizable(True)
        self.structure_content = QtWidgets.QWidget()
        self.structure_layout = QtWidgets.QHBoxLayout(self.structure_content)
        self.structure_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.structure_layout.setSpacing(10)  # Add some spacing between images
        self.structure_layout.setContentsMargins(10, 10, 10, 10)  # Add margins around the images
        self.structure_scroll.setWidget(self.structure_content)
        self.structure_scroll.setMaximumHeight(250)  # Limit the height of the scroll area
        # Images are hidden by default and will be shown in a separate window


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

        # Get the selected item type
        item_type_index = self.item_type_combo.currentIndex()
        if item_type_index < 0:
            self.item_combo.addItem("No items found")
            return

        item_type_display = self.item_type_combo.currentText()
        item_type_id = self.item_type_combo.itemData(item_type_index)

        # Skip if no item type is selected or if it's a placeholder
        if not item_type_display or item_type_display == "No item types found" or item_type_id == -1:
            self.item_combo.addItem("No items found")
            return

        # Get items from the database
        with self.db:
            items = self.db.get_items_by_type(item_type_id)

        # Add items to the combo box
        if items:
            for item_id, item_name, _ in items:
                # Store the item_id as user data
                self.item_combo.addItem(item_name, item_id)
        else:
            # Check if we have any data in the old format
            old_format_items = []

            # Convert display name to directory name (e.g., "Atto Dyes" -> "atto_dyes")
            item_type_dir = item_type_display.lower().replace(' ', '_')

            # Get the directory for this item type
            type_dir = self.fluorophores_dir / item_type_dir

            # Check if the directory exists
            if type_dir.exists():
                # Get all subdirectories (each is an item)
                old_format_items = [d.name for d in type_dir.iterdir() if d.is_dir()]
                old_format_items.sort()

            if old_format_items:
                # We have old format data but no database entries
                # Suggest running the migration script
                QtWidgets.QMessageBox.information(
                    self, 
                    "Database Migration Required",
                    f"Found {item_type_display} data in the old file format. Please run the migration script to convert it to the new database format."
                )

                # Add the old format items to the combo box
                self.item_combo.addItems(old_format_items)
            else:
                self.item_combo.addItem(f"No {item_type_display} found - use Download from menu")

    def populate_item_types(self):
        """Dynamically populate the item type combo box based on available item types in the database."""
        # Clear the combo box
        self.item_type_combo.clear()

        # Get available item types from the database
        with self.db:
            item_types = self.db.get_item_types()

        # Add to combo box
        if item_types:
            for type_id, name, display_name in item_types:
                # Store the type_id as user data
                self.item_type_combo.addItem(display_name, type_id)
        else:
            # Check if we have any data in the old format
            old_format_types = []

            # Check if the fluorophores directory exists
            if self.fluorophores_dir.exists():
                for d in self.fluorophores_dir.iterdir():
                    if d.is_dir():
                        # Convert directory name to a display name (e.g., "atto_dyes" -> "Atto Dyes")
                        display_name = d.name.replace('_', ' ').title()
                        old_format_types.append((display_name, d.name))

            if old_format_types:
                # We have old format data but no database entries
                # Suggest running the migration script
                QtWidgets.QMessageBox.information(
                    self, 
                    "Database Migration Required",
                    "Found spectra data in the old file format. Please run the migration script to convert it to the new database format."
                )

                # Sort item types alphabetically
                old_format_types.sort()

                for display_name, name in old_format_types:
                    self.item_type_combo.addItem(display_name, name)
            else:
                self.item_type_combo.addItem("No item types found", -1)

    def on_item_type_changed(self):
        """Handle changes to the item type selection."""
        self.populate_item_list()

    def on_item_changed(self):
        """Handle changes to the item selection."""
        item_index = self.item_combo.currentIndex()
        if item_index < 0:
            return

        item_name = self.item_combo.currentText()

        # Skip if the item is a placeholder
        if item_name.startswith("No "):
            return

        # Get the item ID from the combo box
        item_id = self.item_combo.itemData(item_index)

        # If item_id is None, this might be old format data
        if item_id is None:
            # Show a message about migration
            QtWidgets.QMessageBox.information(
                self, 
                "Database Migration Required",
                f"Found item {item_name} in the old file format. Please run the migration script to convert it to the new database format."
            )

            # Try to load from the old format
            self.load_item_from_files(item_name)
            return

        # Store the current item
        self.current_item = item_name
        self.current_item_id = item_id

        # Get the item type
        type_index = self.item_type_combo.currentIndex()
        self.current_type_id = self.item_type_combo.itemData(type_index)

        # Load the item data from the database
        with self.db:
            # Get the item description
            item_data = self.db.get_item_by_name_and_type(item_name, self.current_type_id)
            if item_data:
                _, _, description = item_data
                self.item_description.setText(description)

                # Get optical properties
                optical_props = self.db.get_optical_properties(item_id)
                self.optical_properties.setRowCount(len(optical_props))
                for i, (prop, value) in enumerate(optical_props.items()):
                    self.optical_properties.setItem(i, 0, QtWidgets.QTableWidgetItem(prop))
                    self.optical_properties.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))

                # Create a metadata dictionary for compatibility with existing code
                self.item_metadata = {
                    "name": item_name,
                    "description": description,
                    "optical_properties": optical_props
                }

                # Get absorption spectrum
                abs_data = self.db.get_spectrum(item_id, "absorption")
                if abs_data:
                    self.absorption_spectrum = abs_data
                else:
                    self.absorption_spectrum = None

                # Get emission spectrum
                em_data = self.db.get_spectrum(item_id, "emission")
                if em_data:
                    self.emission_spectrum = em_data
                else:
                    self.emission_spectrum = None

                # Get images
                self.clear_structure_images()
                images = self.db.get_images(item_id)
                for image_name, image_data, image_format in images:
                    self.add_structure_image_from_data(image_name, image_data, image_format)

                # Update the plots
                self.update_plots()

                # Highlight missing fields for Förster radius calculation
                self.highlight_missing_fields()

                # Show the item information section if the checkbox is checked
                self.info_group.setVisible(self.show_info_checkbox.isChecked())
            else:
                QtWidgets.QMessageBox.warning(self, "Missing Data", 
                                             f"No data found for {item_name} in the database.")

    def load_item_from_files(self, item_name):
        """Load item data from files (old format)."""
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

                # Highlight missing fields for Förster radius calculation
                self.highlight_missing_fields()

                # Show the item information section if the checkbox is checked
                self.info_group.setVisible(self.show_info_checkbox.isChecked())

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

        # Clear the image data list
        self.image_data_list = []

        # Disable the show images button
        self.show_images_button.setEnabled(False)

    def add_structure_image(self, image_path):
        """Add a structure image to the structure tab from a file path."""
        try:
            # Read the image file
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Get the image name and format
            image_name = os.path.basename(image_path)
            image_format = os.path.splitext(image_name)[1].lstrip('.')

            # Add the image using the data
            self.add_structure_image_from_data(image_name, image_data, image_format)
        except Exception as e:
            print(f"Error adding structure image from path: {e}")

    def add_structure_image_from_data(self, image_name, image_data, image_format):
        """Add a structure image to the structure tab from image data."""
        try:
            # Create a label for the image
            label = QtWidgets.QLabel()

            # Create a QPixmap from the image data
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(image_data)

            # Scale the pixmap to a reasonable size while maintaining aspect ratio
            max_height = 200  # Maximum height for the images
            scaled_pixmap = pixmap.scaledToHeight(max_height, QtCore.Qt.SmoothTransformation)

            label.setPixmap(scaled_pixmap)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

            # Add the label to the structure layout
            self.structure_layout.addWidget(label)

            # Store the image data for later use
            if not hasattr(self, 'image_data_list'):
                self.image_data_list = []
            self.image_data_list.append((image_name, image_data, image_format))

            # Enable the show images button
            self.show_images_button.setEnabled(True)
        except Exception as e:
            print(f"Error adding structure image from data: {e}")

    def show_images_in_window(self):
        """Show images in a separate window."""
        # Check if we have any image data
        if not hasattr(self, 'image_data_list') or not self.image_data_list:
            QtWidgets.QMessageBox.information(self, "No Images", "No images loaded. Please load some items with images first.")
            return

        # Create a dialog to show the images
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Structure Images")
        dialog.resize(800, 600)

        # Create a scroll area for the images
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        # Add each image to the layout
        for image_name, image_data, image_format in self.image_data_list:
            try:
                # Add the image name as a header
                header = QtWidgets.QLabel(image_name)
                header.setAlignment(QtCore.Qt.AlignCenter)
                header.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
                layout.addWidget(header)

                # Create a label for the image
                label = QtWidgets.QLabel()
                pixmap = QtGui.QPixmap()
                pixmap.loadFromData(image_data)

                # Scale the pixmap to a reasonable size while maintaining aspect ratio
                max_width = 700  # Maximum width for the images
                scaled_pixmap = pixmap.scaledToWidth(max_width, QtCore.Qt.SmoothTransformation)

                label.setPixmap(scaled_pixmap)
                label.setAlignment(QtCore.Qt.AlignCenter)

                # Add the label to the layout
                layout.addWidget(label)

                # Add some spacing
                layout.addSpacing(20)
            except Exception as e:
                print(f"Error adding image to dialog: {e}")

        # If no images were successfully added
        if layout.count() == 0:
            layout.addWidget(QtWidgets.QLabel("No images could be displayed."))

        scroll.setWidget(content)

        # Create a layout for the dialog
        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.addWidget(scroll)

        # Add a close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        dialog_layout.addWidget(close_button)

        # Show the dialog
        dialog.exec_()

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

        # Create Database submenu
        database_menu = tools_menu.addMenu('Database')

        # Add "Add Custom Dye" action
        add_dye_action = QtWidgets.QAction('Add Custom Dye', self)
        add_dye_action.triggered.connect(self.add_custom_dye)
        tools_menu.addAction(add_dye_action)

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

            # Run the script (no need to create directories since we're using the database)
            process = subprocess.Popen(
                [sys.executable, script_path],
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

            # Reload the database and refresh the UI
            with self.db:
                # Refresh the item types and item list
                self.populate_item_types()
                self.populate_item_list()

                # Update the download combo
                self.populate_download_combo()

                # Show a success message
                QtWidgets.QMessageBox.information(
                    self, 
                    "Download Complete",
                    f"{script_name} data has been downloaded and stored in the database."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run {os.path.basename(script_path)}: {str(e)}")



    def on_spectra_selection_changed(self):
        """Handle changes to the spectra selection in the list widget."""
        self.selected_spectra = []
        for item in self.spectra_list.selectedItems():
            spectrum_id = item.data(QtCore.Qt.UserRole)
            self.selected_spectra.append(spectrum_id)

        # Update donor and acceptor combo boxes
        self.update_donor_acceptor_combos()

        # Update the plots
        self.update_plots()

    def update_donor_acceptor_combos(self):
        """Update the donor and acceptor combo boxes with available spectra."""
        # Save current selections
        donor_idx = self.donor_combo.currentIndex()
        acceptor_idx = self.acceptor_combo.currentIndex()

        # Clear and repopulate
        self.donor_combo.clear()
        self.acceptor_combo.clear()

        for spectrum_id, spectrum in self.loaded_spectra.items():
            name = spectrum['name']
            spectrum_type = spectrum['type']
            display_name = f"{name} ({spectrum_type})"

            # Add to donor combo (typically emission spectra)
            if spectrum_type == 'emission':
                self.donor_combo.addItem(display_name, spectrum_id)

            # Add to acceptor combo (typically absorption spectra)
            if spectrum_type == 'absorption':
                self.acceptor_combo.addItem(display_name, spectrum_id)

        # Restore selections if possible
        if donor_idx >= 0 and donor_idx < self.donor_combo.count():
            self.donor_combo.setCurrentIndex(donor_idx)
        if acceptor_idx >= 0 and acceptor_idx < self.acceptor_combo.count():
            self.acceptor_combo.setCurrentIndex(acceptor_idx)

    def add_current_spectrum(self):
        """Add the current spectrum to the list of loaded spectra."""
        item_name = self.item_combo.currentText()

        # Skip if the item is a placeholder
        if item_name.startswith("No "):
            return

        # Add absorption spectrum if available
        if self.absorption_spectrum is not None:
            spectrum_id = f"{item_name}_abs_{len(self.loaded_spectra)}"
            self.loaded_spectra[spectrum_id] = {
                'type': 'absorption',
                'name': item_name,
                'data': self.absorption_spectrum,
                'metadata': self.item_metadata
            }

            # Add to list widget
            item = QtWidgets.QListWidgetItem(f"{item_name} (absorption)")
            item.setData(QtCore.Qt.UserRole, spectrum_id)
            self.spectra_list.addItem(item)

        # Add emission spectrum if available
        if self.emission_spectrum is not None:
            spectrum_id = f"{item_name}_em_{len(self.loaded_spectra)}"
            self.loaded_spectra[spectrum_id] = {
                'type': 'emission',
                'name': item_name,
                'data': self.emission_spectrum,
                'metadata': self.item_metadata
            }

            # Add to list widget
            item = QtWidgets.QListWidgetItem(f"{item_name} (emission)")
            item.setData(QtCore.Qt.UserRole, spectrum_id)
            self.spectra_list.addItem(item)

        # Update donor and acceptor combo boxes
        self.update_donor_acceptor_combos()

    def remove_selected_spectra(self):
        """Remove selected spectra from the list."""
        for item in self.spectra_list.selectedItems():
            spectrum_id = item.data(QtCore.Qt.UserRole)
            if spectrum_id in self.loaded_spectra:
                del self.loaded_spectra[spectrum_id]

            # Remove from list widget
            self.spectra_list.takeItem(self.spectra_list.row(item))

        # Update selected spectra list
        self.selected_spectra = [s for s in self.selected_spectra if s in self.loaded_spectra]

        # Update donor and acceptor combo boxes
        self.update_donor_acceptor_combos()

        # Update the plots
        self.update_plots()

    def calculate_spectral_overlap(self, donor_id, acceptor_id):
        """Calculate the spectral overlap integral between donor emission and acceptor absorption."""
        if donor_id not in self.loaded_spectra or acceptor_id not in self.loaded_spectra:
            return None

        donor_spectrum = self.loaded_spectra[donor_id]
        acceptor_spectrum = self.loaded_spectra[acceptor_id]

        # Check if the spectra are of the correct type
        if donor_spectrum['type'] != 'emission' or acceptor_spectrum['type'] != 'absorption':
            return None

        # Get the data
        donor_x, donor_y = donor_spectrum['data']
        acceptor_x, acceptor_y = acceptor_spectrum['data']

        # Find overlapping region
        x_min = max(min(donor_x), min(acceptor_x))
        x_max = min(max(donor_x), max(acceptor_x))

        # Filter data to overlapping region
        donor_mask = (donor_x >= x_min) & (donor_x <= x_max)
        acceptor_mask = (acceptor_x >= x_min) & (acceptor_x <= x_max)

        donor_x_overlap = donor_x[donor_mask]
        donor_y_overlap = donor_y[donor_mask]

        acceptor_x_overlap = acceptor_x[acceptor_mask]
        acceptor_y_overlap = acceptor_y[acceptor_mask]

        # Interpolate acceptor data to match donor wavelengths
        acceptor_y_interp = np.interp(donor_x_overlap, acceptor_x_overlap, acceptor_y_overlap)

        # Normalize donor emission
        donor_y_norm = donor_y_overlap / np.trapz(donor_y_overlap, donor_x_overlap)

        # Calculate spectral overlap integral: J = ∫ fD(λ) * εA(λ) * λ^4 * dλ
        # Note: wavelength should be in cm, extinction coefficient in M^-1 cm^-1
        # For simplicity, we'll assume the wavelength is in nm and convert to cm
        # and that the extinction coefficient is already in M^-1 cm^-1
        wavelength_cm = donor_x_overlap / 1e7  # Convert nm to cm

        # Calculate λ^4 * fD(λ) * εA(λ)
        integrand = donor_y_norm * acceptor_y_interp * wavelength_cm**4

        # Calculate the integral
        overlap_integral = np.trapz(integrand, wavelength_cm)

        return overlap_integral

    def calculate_forster_radius(self):
        """Calculate the Förster radius from selected donor and acceptor spectra."""
        # Get selected donor and acceptor
        donor_idx = self.donor_combo.currentIndex()
        acceptor_idx = self.acceptor_combo.currentIndex()

        if donor_idx < 0 or acceptor_idx < 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select both donor and acceptor spectra.")
            return

        donor_id = self.donor_combo.itemData(donor_idx)
        acceptor_id = self.acceptor_combo.itemData(acceptor_idx)

        # Calculate spectral overlap
        overlap_integral = self.calculate_spectral_overlap(donor_id, acceptor_id)

        if overlap_integral is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Failed to calculate spectral overlap. Make sure donor is emission and acceptor is absorption.")
            return

        # Get parameters
        kappa_squared = self.kappa_squared.value()
        donor_qy = self.donor_qy.value()
        refractive_index = self.refractive_index.value()

        # Get acceptor extinction coefficient (max value)
        acceptor_spectrum = self.loaded_spectra[acceptor_id]
        acceptor_x, acceptor_y = acceptor_spectrum['data']

        # Try to get parameters from metadata if available
        donor_metadata = self.loaded_spectra[donor_id]['metadata']
        acceptor_metadata = self.loaded_spectra[acceptor_id]['metadata']

        if donor_metadata:
            optical_props = donor_metadata.get("optical_properties", {})
            if "Quantum Yield" in optical_props:
                try:
                    qy = float(optical_props["Quantum Yield"])
                    self.donor_qy.setValue(qy)
                    donor_qy = qy
                except (ValueError, TypeError):
                    pass

        # Try to get extinction coefficient from metadata
        if acceptor_metadata:
            optical_props = acceptor_metadata.get("optical_properties", {})
            if "Extinction Coefficient" in optical_props:
                try:
                    max_extinction_coef = float(optical_props["Extinction Coefficient"])
                    self.acceptor_ext_coef.setValue(max_extinction_coef)
                except (ValueError, TypeError):
                    pass

        # Use the value from the UI field
        max_extinction_coef = self.acceptor_ext_coef.value()

        # Calculate Förster radius
        # R₀⁶ = (9000 * ln(10) * κ² * Φᴅ * J) / (128 * π⁵ * n⁴ * Nₐ)
        # where:
        # κ² is the orientation factor
        # Φᴅ is the quantum yield of the donor
        # J is the spectral overlap integral
        # n is the refractive index of the medium
        # Nₐ is Avogadro's number (6.022 × 10²³)

        # Constants
        avogadro = 6.022e23
        ln10 = np.log(10)
        pi = np.pi

        # Calculate R₀ in Å
        numerator = 9000 * ln10 * kappa_squared * donor_qy * overlap_integral
        denominator = 128 * pi**5 * refractive_index**4 * avogadro

        r0_sixth = numerator / denominator
        r0 = r0_sixth**(1/6)  # in cm
        r0_angstrom = r0 * 1e8  # Convert to Å

        # Display result
        self.forster_result.setText(f"{r0_angstrom:.2f} Å")

        # Show detailed calculation in a message box
        details = (
            f"Calculation Details:\n"
            f"Donor: {self.donor_combo.currentText()}\n"
            f"Acceptor: {self.acceptor_combo.currentText()}\n"
            f"Orientation Factor (κ²): {kappa_squared}\n"
            f"Donor Quantum Yield: {donor_qy}\n"
            f"Acceptor Extinction Coefficient (max): {max_extinction_coef:.2e} M⁻¹cm⁻¹\n"
            f"Refractive Index: {refractive_index}\n"
            f"Spectral Overlap Integral (J): {overlap_integral:.3e} M⁻¹cm³\n"
            f"Förster Radius (R₀): {r0_angstrom:.2f} Å"
        )

        QtWidgets.QMessageBox.information(self, "Förster Radius Calculation", details)

    def highlight_missing_fields(self):
        """Highlight missing fields for Förster radius calculation."""
        if not self.item_metadata:
            return

        # Check for quantum yield and extinction coefficient
        optical_props = self.item_metadata.get("optical_properties", {})
        has_qy = "Quantum Yield" in optical_props
        has_ext_coef = "Extinction Coefficient" in optical_props

        # Update the donor QY field background color
        if has_qy:
            try:
                qy = float(optical_props["Quantum Yield"])
                self.donor_qy.setValue(qy)
                self.donor_qy.setStyleSheet("")  # Reset style
            except (ValueError, TypeError):
                self.donor_qy.setStyleSheet("background-color: #FFEEEE;")  # Light red
        else:
            self.donor_qy.setStyleSheet("background-color: #FFEEEE;")  # Light red

        # Update the acceptor extinction coefficient field background color
        if has_ext_coef:
            try:
                ext_coef = float(optical_props["Extinction Coefficient"])
                self.acceptor_ext_coef.setValue(ext_coef)
                self.acceptor_ext_coef.setStyleSheet("")  # Reset style
            except (ValueError, TypeError):
                self.acceptor_ext_coef.setStyleSheet("background-color: #FFEEEE;")  # Light red
        else:
            self.acceptor_ext_coef.setStyleSheet("background-color: #FFEEEE;")  # Light red

    def toggle_info_group(self):
        """Toggle the visibility of the item information group."""
        self.info_group.setVisible(self.show_info_checkbox.isChecked())

    def add_custom_dye(self):
        """Open a dialog to add a custom dye to the database."""
        # Create a dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Custom Dye")
        dialog.resize(600, 700)

        # Create a layout for the dialog
        layout = QtWidgets.QVBoxLayout(dialog)

        # Create a form layout for the dye information
        form_layout = QtWidgets.QFormLayout()
        layout.addLayout(form_layout)

        # Dye name
        dye_name_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Dye Name:", dye_name_edit)

        # Dye type (category)
        dye_type_combo = QtWidgets.QComboBox()

        # Get existing dye types
        existing_types = []
        if self.fluorophores_dir.exists():
            for d in self.fluorophores_dir.iterdir():
                if d.is_dir():
                    display_name = d.name.replace('_', ' ').title()
                    existing_types.append((display_name, d.name))

        # Add existing types to combo box
        for display_name, dir_name in existing_types:
            dye_type_combo.addItem(display_name, dir_name)

        # Add option for new type
        dye_type_combo.addItem("New Type...", "new_type")

        # Create a line edit for new type (initially hidden)
        new_type_edit = QtWidgets.QLineEdit()
        new_type_edit.setPlaceholderText("Enter new dye type")
        new_type_edit.setVisible(False)

        # Function to toggle new type edit visibility
        def on_type_changed():
            if dye_type_combo.currentData() == "new_type":
                new_type_edit.setVisible(True)
            else:
                new_type_edit.setVisible(False)

        dye_type_combo.currentIndexChanged.connect(on_type_changed)

        form_layout.addRow("Dye Type:", dye_type_combo)
        form_layout.addRow("", new_type_edit)

        # Description
        description_edit = QtWidgets.QTextEdit()
        description_edit.setMaximumHeight(100)
        form_layout.addRow("Description:", description_edit)

        # Optical properties
        layout.addWidget(QtWidgets.QLabel("Optical Properties:"))

        # Create a grid layout for optical properties
        optical_props_layout = QtWidgets.QGridLayout()
        layout.addLayout(optical_props_layout)

        # Add common optical properties
        optical_props_layout.addWidget(QtWidgets.QLabel("Absorption Wavelength (nm):"), 0, 0)
        abs_wavelength_edit = QtWidgets.QLineEdit()
        optical_props_layout.addWidget(abs_wavelength_edit, 0, 1)

        optical_props_layout.addWidget(QtWidgets.QLabel("Extinction Coefficient:"), 1, 0)
        ext_coef_edit = QtWidgets.QLineEdit()
        optical_props_layout.addWidget(ext_coef_edit, 1, 1)

        optical_props_layout.addWidget(QtWidgets.QLabel("Emission Wavelength (nm):"), 2, 0)
        em_wavelength_edit = QtWidgets.QLineEdit()
        optical_props_layout.addWidget(em_wavelength_edit, 2, 1)

        optical_props_layout.addWidget(QtWidgets.QLabel("Quantum Yield (%):"), 3, 0)
        qy_edit = QtWidgets.QLineEdit()
        optical_props_layout.addWidget(qy_edit, 3, 1)

        optical_props_layout.addWidget(QtWidgets.QLabel("Fluorescence Lifetime (ns):"), 4, 0)
        lifetime_edit = QtWidgets.QLineEdit()
        optical_props_layout.addWidget(lifetime_edit, 4, 1)

        # Spectra files
        layout.addWidget(QtWidgets.QLabel("Spectra Files:"))

        # Absorption spectrum
        abs_spectrum_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(abs_spectrum_layout)

        abs_spectrum_label = QtWidgets.QLabel("Absorption Spectrum:")
        abs_spectrum_layout.addWidget(abs_spectrum_label)

        abs_spectrum_path = QtWidgets.QLineEdit()
        abs_spectrum_path.setReadOnly(True)
        abs_spectrum_layout.addWidget(abs_spectrum_path)

        abs_spectrum_button = QtWidgets.QPushButton("Browse...")
        abs_spectrum_layout.addWidget(abs_spectrum_button)

        # Emission spectrum
        em_spectrum_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(em_spectrum_layout)

        em_spectrum_label = QtWidgets.QLabel("Emission Spectrum:")
        em_spectrum_layout.addWidget(em_spectrum_label)

        em_spectrum_path = QtWidgets.QLineEdit()
        em_spectrum_path.setReadOnly(True)
        em_spectrum_layout.addWidget(em_spectrum_path)

        em_spectrum_button = QtWidgets.QPushButton("Browse...")
        em_spectrum_layout.addWidget(em_spectrum_button)

        # Structure images
        layout.addWidget(QtWidgets.QLabel("Structure Images:"))

        # List widget to display selected images
        images_list = QtWidgets.QListWidget()
        images_list.setMaximumHeight(100)
        layout.addWidget(images_list)

        # Buttons for adding/removing images
        images_button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(images_button_layout)

        add_image_button = QtWidgets.QPushButton("Add Image...")
        images_button_layout.addWidget(add_image_button)

        remove_image_button = QtWidgets.QPushButton("Remove Selected")
        images_button_layout.addWidget(remove_image_button)

        # Variables to store selected files
        abs_spectrum_file = None
        em_spectrum_file = None
        image_files = []

        # Function to browse for absorption spectrum
        def browse_abs_spectrum():
            nonlocal abs_spectrum_file
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "Select Absorption Spectrum", "", "Text Files (*.txt);;All Files (*.*)"
            )
            if file_path:
                abs_spectrum_file = file_path
                abs_spectrum_path.setText(file_path)

        abs_spectrum_button.clicked.connect(browse_abs_spectrum)

        # Function to browse for emission spectrum
        def browse_em_spectrum():
            nonlocal em_spectrum_file
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "Select Emission Spectrum", "", "Text Files (*.txt);;All Files (*.*)"
            )
            if file_path:
                em_spectrum_file = file_path
                em_spectrum_path.setText(file_path)

        em_spectrum_button.clicked.connect(browse_em_spectrum)

        # Function to add an image
        def add_image():
            file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                dialog, "Select Structure Images", "", "Image Files (*.png *.jpg *.jpeg);;All Files (*.*)"
            )
            for file_path in file_paths:
                if file_path and file_path not in image_files:
                    image_files.append(file_path)
                    item = QtWidgets.QListWidgetItem(os.path.basename(file_path))
                    item.setData(QtCore.Qt.UserRole, file_path)
                    images_list.addItem(item)

        add_image_button.clicked.connect(add_image)

        # Function to remove selected images
        def remove_image():
            for item in images_list.selectedItems():
                file_path = item.data(QtCore.Qt.UserRole)
                if file_path in image_files:
                    image_files.remove(file_path)
                images_list.takeItem(images_list.row(item))

        remove_image_button.clicked.connect(remove_image)

        # Add buttons for OK and Cancel
        button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(button_layout)

        ok_button = QtWidgets.QPushButton("OK")
        button_layout.addWidget(ok_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(cancel_button)

        # Connect buttons
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)

        # Show the dialog
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Get the dye name
            dye_name = dye_name_edit.text().strip()
            if not dye_name:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a dye name.")
                return

            # Get the dye type
            if dye_type_combo.currentData() == "new_type":
                dye_type_display = new_type_edit.text().strip()
                if not dye_type_display:
                    QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a dye type.")
                    return
                dye_type_dir = dye_type_display.lower().replace(' ', '_')
            else:
                dye_type_display = dye_type_combo.currentText()
                dye_type_dir = dye_type_combo.currentData()

            # Check if spectra files are provided
            if not abs_spectrum_file and not em_spectrum_file:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please provide at least one spectrum file.")
                return

            try:
                # Store data in the database
                with self.db:
                    # Add or get the item type
                    type_id = self.db.add_item_type(dye_type_dir, dye_type_display)

                    # Add the item
                    description = description_edit.toPlainText()
                    item_id = self.db.add_item(dye_name, type_id, description)

                    # Add optical properties
                    optical_properties = {
                        "λabs": abs_wavelength_edit.text(),
                        "εmax": ext_coef_edit.text(),
                        "λfl": em_wavelength_edit.text(),
                        "ηfl": qy_edit.text(),
                        "τfl": lifetime_edit.text()
                    }

                    for prop_name, prop_value in optical_properties.items():
                        if prop_value:  # Only add non-empty properties
                            self.db.add_optical_property(item_id, prop_name, prop_value)

                    # Add absorption spectrum if provided
                    if abs_spectrum_file:
                        try:
                            # Load the spectrum data
                            from chisurf.fio.ascii import load_xy
                            x, y = load_xy(abs_spectrum_file, delimiter="\t", skiprows=2)
                            # Add to database
                            self.db.add_spectrum(item_id, "absorption", np.array(x), np.array(y))
                        except Exception as e:
                            print(f"Error loading absorption spectrum: {e}")

                    # Add emission spectrum if provided
                    if em_spectrum_file:
                        try:
                            # Load the spectrum data
                            from chisurf.fio.ascii import load_xy
                            x, y = load_xy(em_spectrum_file, delimiter="\t", skiprows=2)
                            # Add to database
                            self.db.add_spectrum(item_id, "emission", np.array(x), np.array(y))
                        except Exception as e:
                            print(f"Error loading emission spectrum: {e}")

                    # Add images if provided
                    for image_file in image_files:
                        try:
                            image_filename = os.path.basename(image_file)
                            # The add_image method reads the file and stores it as a BLOB
                            self.db.add_image(item_id, image_filename)
                        except Exception as e:
                            print(f"Error adding image: {e}")

                # Refresh the UI
                self.populate_item_types()
                self.populate_item_list()

                QtWidgets.QMessageBox.information(self, "Success", f"Custom dye '{dye_name}' added successfully to the database.")

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to add custom dye: {str(e)}")

    def edit_metadata(self):
        """Open a dialog to edit metadata for the selected spectrum."""
        # Get selected items
        selected_items = self.spectra_list.selectedItems()

        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a spectrum to edit its metadata.")
            return

        # Use the first selected item
        item = selected_items[0]
        spectrum_id = item.data(QtCore.Qt.UserRole)

        if spectrum_id not in self.loaded_spectra:
            return

        spectrum = self.loaded_spectra[spectrum_id]
        metadata = spectrum['metadata']

        if not metadata:
            metadata = {"description": "", "optical_properties": {}}

        # Create a dialog for editing
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Edit Metadata - {spectrum['name']}")
        dialog.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Description
        layout.addWidget(QtWidgets.QLabel("Description:"))
        description_edit = QtWidgets.QTextEdit()
        description_edit.setText(metadata.get("description", ""))
        layout.addWidget(description_edit)

        # Optical properties
        layout.addWidget(QtWidgets.QLabel("Optical Properties:"))

        properties_table = QtWidgets.QTableWidget()
        properties_table.setColumnCount(2)
        properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        properties_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        optical_props = metadata.get("optical_properties", {})
        properties_table.setRowCount(len(optical_props) + 5)  # Add extra rows for new properties

        for i, (prop, value) in enumerate(optical_props.items()):
            properties_table.setItem(i, 0, QtWidgets.QTableWidgetItem(prop))
            properties_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))

        layout.addWidget(properties_table)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton("Save")
        cancel_button = QtWidgets.QPushButton("Cancel")

        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        # Connect buttons
        save_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Update metadata
            new_metadata = {
                "description": description_edit.toPlainText(),
                "optical_properties": {}
            }

            # Get optical properties from table
            for i in range(properties_table.rowCount()):
                prop_item = properties_table.item(i, 0)
                value_item = properties_table.item(i, 1)

                if prop_item and value_item and prop_item.text().strip():
                    prop = prop_item.text().strip()
                    value = value_item.text().strip()

                    # Try to convert to number if possible
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass

                    new_metadata["optical_properties"][prop] = value

            # Update the spectrum metadata
            spectrum['metadata'] = new_metadata
            self.loaded_spectra[spectrum_id] = spectrum

            # If this is the current item, update the display
            if spectrum_id == self.current_item:
                self.item_description.setText(new_metadata.get("description", ""))

                optical_props = new_metadata.get("optical_properties", {})
                self.optical_properties.setRowCount(len(optical_props))
                for i, (prop, value) in enumerate(optical_props.items()):
                    self.optical_properties.setItem(i, 0, QtWidgets.QTableWidgetItem(prop))
                    self.optical_properties.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))

    def update_plots(self):
        """Update the plots with current data."""
        self.spectra_plot.clear()

        # Always plot all loaded spectra
        if self.loaded_spectra:
            for spectrum_id, spectrum in self.loaded_spectra.items():
                spectrum_type = spectrum['type']
                name = spectrum['name']
                x, y = spectrum['data']

                # Normalize if requested
                if self.normalize_spectra.isChecked():
                    y = y / np.max(y)

                # Choose color based on selection status and type
                if spectrum_id in self.selected_spectra:
                    # Selected spectra in color
                    if spectrum_type == 'absorption':
                        color = pg.mkPen('b', width=2)
                    else:  # emission
                        color = pg.mkPen('r', width=2)
                    self.spectra_plot.plot(x, y, pen=color, name=f"{name} ({spectrum_type})")
                else:
                    # Non-selected spectra in gray with transparency
                    color = pg.mkPen(QtGui.QColor(128, 128, 128, 80), width=1)  # gray with alpha=0.7
                    self.spectra_plot.plot(x, y, pen=color)
        else:
            # Fall back to displaying current spectra if no spectra are loaded
            # Plot absorption spectrum if available
            if self.absorption_spectrum is not None:
                x, y = self.absorption_spectrum

                # Normalize if requested
                if self.normalize_spectra.isChecked():
                    y = y / np.max(y)

                self.spectra_plot.plot(x, y, pen=pg.mkPen('b', width=2), name='Absorption')

            # Plot emission spectrum if available
            if self.emission_spectrum is not None:
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
