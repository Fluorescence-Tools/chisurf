from __future__ import annotations

import pathlib
import yaml
import re
import os

from qtpy import QtCore, QtGui, QtWidgets

import chisurf
import chisurf.fio as io
from chisurf import logging
import chisurf.settings
from chisurf.settings import cs_settings
from chisurf.gui.widgets.settings_editor import SettingsEditor, SettingsTreeModel, SettingsItemDelegate


class FilePathItemDelegate(SettingsItemDelegate):
    """A delegate for editing file paths with a file browser button."""

    def createEditor(self, parent, option, index):
        """Create a widget with a file browser button for file paths."""
        if not index.isValid() or index.column() != 1:
            return super().createEditor(parent, option, index)

        # Get the setting name
        setting_name = index.sibling(index.row(), 0).data(QtCore.Qt.DisplayRole)

        # Check if this is a file path setting
        if setting_name.lower() == "filename":
            # Create a widget to hold the line edit and button
            editor = QtWidgets.QWidget(parent)
            layout = QtWidgets.QHBoxLayout(editor)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # Create line edit for the file path
            line_edit = QtWidgets.QLineEdit(editor)
            line_edit.setText(index.data(QtCore.Qt.DisplayRole))

            # Create browse button
            browse_button = QtWidgets.QPushButton("...", editor)
            browse_button.setMaximumWidth(30)

            # Add widgets to layout
            layout.addWidget(line_edit)
            layout.addWidget(browse_button)

            # Connect browse button to file dialog
            browse_button.clicked.connect(lambda: self._browse_file(line_edit))

            # Store the line edit for later access
            editor.setProperty("line_edit", line_edit)

            return editor
        else:
            return super().createEditor(parent, option, index)

    def _browse_file(self, line_edit):
        """Open a file dialog and set the selected file path in the line edit."""
        current_path = line_edit.text()
        start_dir = os.path.dirname(current_path) if current_path else ""

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            line_edit.window(),
            "Select Data File",
            start_dir,
            "Data Files (*.dat *.csv);;All Files (*.*)"
        )

        if file_path:
            # Convert to relative path if possible
            try:
                # Get the directory of the YAML file being edited
                yaml_dir = os.path.dirname(line_edit.window().filename)
                if yaml_dir and os.path.commonpath([yaml_dir, file_path]) == yaml_dir:
                    # Make path relative to the YAML file
                    rel_path = os.path.relpath(file_path, yaml_dir)
                    # Use ./ prefix for clarity
                    if not rel_path.startswith('.'):
                        rel_path = f"./{rel_path}"
                    file_path = rel_path
            except (ValueError, AttributeError):
                # If there's an error, use the absolute path
                pass

            line_edit.setText(file_path)

    def setModelData(self, editor, model, index):
        """Set the model data from the editor."""
        if not index.isValid() or index.column() != 1:
            return super().setModelData(editor, model, index)

        # Get the setting name
        setting_name = index.sibling(index.row(), 0).data(QtCore.Qt.DisplayRole)

        # Check if this is a file path setting
        if setting_name.lower() == "filename":
            # Get the line edit from the editor
            line_edit = editor.property("line_edit")
            if line_edit:
                # Set the model data from the line edit
                model.setData(index, line_edit.text())
        else:
            super().setModelData(editor, model, index)


class UCFRETExperimentEditor(SettingsEditor):
    """A specialized editor for UCFRET experiment YAML files."""

    def __init__(
        self,
        *args,
        filename: str = None,
        **kwargs
    ):
        # Initialize parent
        super().__init__(*args, filename=filename, **kwargs)

        # Set window title
        self.setWindowTitle("UCFRET Experiment Editor")

    def setup_ui(self):
        """Set up the user interface."""
        # Call parent setup_ui
        super().setup_ui()

        # Replace the delegate with our custom version
        self.delegate = FilePathItemDelegate()
        self.tree_view.setItemDelegate(self.delegate)

        # Add a button to create a new experiment file
        self.new_button = QtWidgets.QPushButton("New")
        self.new_button.clicked.connect(self.create_new_experiment)

        # Insert the new button before the reload button
        button_layout = self.layout().itemAt(2).layout()
        button_layout.insertWidget(1, self.new_button)

        # Add a help button
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        button_layout.insertWidget(2, self.help_button)

    @staticmethod
    def create_experiment_from_template(parent=None):
        """
        Create a new experiment file from a template.

        This static method handles the template selection and file creation process.
        It can be called without creating an editor instance first.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget for dialogs, by default None

        Returns
        -------
        str or None
            The path to the created file, or None if the operation was cancelled or failed
        """
        # Create a dialog to select a template
        template_dialog = QtWidgets.QDialog(parent)
        template_dialog.setWindowTitle("Select Template")
        template_dialog.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout(template_dialog)

        # Add a label
        label = QtWidgets.QLabel("Select a template for the new experiment file:")
        layout.addWidget(label)

        # Add a list widget with templates
        template_list = QtWidgets.QListWidget()
        template_list.addItem("Empty Template")
        template_list.addItem("T4 Lysozyme E5pAcF S44C (5-44)")
        template_list.setCurrentRow(0)  # Select the first item by default
        layout.addWidget(template_list)

        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(template_dialog.accept)
        button_box.rejected.connect(template_dialog.reject)
        layout.addWidget(button_box)

        # Show the dialog
        if template_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        # Get the selected template
        selected_template = template_list.currentRow()

        # Ask for a filename
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent,
            "Create New Experiment File",
            str(chisurf.working_path),  # Use the current working path
            "YAML Files (*.yml *.yaml)"
        )

        if not filename:
            return None

        # Update working path
        chisurf.working_path = pathlib.Path(filename).parent

        # Create a new experiment file with the selected template
        try:
            # Default minimal experiment structure
            default_settings = {
                "Sample": {
                    "Name": "Sample Name",
                    "Measurement ID": "DA_Measurement",
                    "Reference": {
                        "Donor": "D0_Measurement"
                    }
                },
                "Measurement datasets": {
                    "DA_Measurement": {
                        "Setup settings": "Donor",
                        "Instrument response function": {
                            "Measurement ID": "IRF_DA"
                        },
                        "Description": "Description of the DA measurement",
                        "Data": {
                            "Filename": "./sample_DA.dat"
                        }
                    },
                    "D0_Measurement": {
                        "Setup settings": "Donor",
                        "Instrument response function": {
                            "Measurement ID": "IRF_D0"
                        },
                        "Description": "Description of the D0 measurement",
                        "Data": {
                            "Filename": "./sample_D0.dat"
                        }
                    },
                    "IRF_DA": {
                        "Setup settings": "IRF",
                        "Description": "Description of the DA IRF measurement",
                        "Data": {
                            "Filename": "./IRF_DA.dat"
                        }
                    },
                    "IRF_D0": {
                        "Setup settings": "IRF",
                        "Description": "Description of the D0 IRF measurement",
                        "Data": {
                            "Filename": "./IRF_D0.dat"
                        }
                    }
                }
            }

            settings = None

            if selected_template == 1:  # T4 Lysozyme E5pAcF S44C (5-44)
                # Try to find the 5-44.yml file
                example_paths = [
                    # First check if there's a template file in the plugin directory
                    pathlib.Path(chisurf.__file__).parent / "plugins" / "ucfret" / "templates" / "5-44.yml",
                    # Then check the module example directories
                    pathlib.Path(chisurf.__file__).parent / ".." / ".." / ".." / "modules" / "ucfret" / "example" / "5-44" / "5-44.yml",
                    pathlib.Path(chisurf.__file__).parent / ".." / ".." / "modules" / "ucfret" / "example" / "5-44" / "5-44.yml"
                ]

                for path in example_paths:
                    try:
                        path = path.resolve()
                        if path.exists():
                            with open(path, 'r') as f:
                                settings = yaml.safe_load(f)
                            break
                    except Exception as e:
                        logging.log(1, f"Error loading template from {path}: {e}")

                if settings is None:
                    # If we couldn't find the 5-44.yml file, show an error and use the default template
                    QtWidgets.QMessageBox.warning(
                        parent,
                        "Template Not Found",
                        "Could not find the 5-44.yml template file. Using default template instead."
                    )

            # If we're using the default template or couldn't find the 5-44.yml file
            if settings is None:
                # Try to load the default template from the YAML file
                default_template_path = pathlib.Path(chisurf.__file__).parent / "plugins" / "ucfret" / "default_experiment_template.yml"

                try:
                    if default_template_path.exists():
                        with open(default_template_path, 'r') as f:
                            settings = yaml.safe_load(f)
                    else:
                        # If the template file doesn't exist, show an error and use the minimal structure
                        QtWidgets.QMessageBox.warning(
                            parent,
                            "Template Not Found",
                            "Could not find the default template file. Using minimal default settings."
                        )
                        settings = default_settings
                except Exception as e:
                    logging.log(1, f"Error loading default template: {e}")
                    settings = default_settings

            # Write the settings to the new file
            with open(filename, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)

            # Show help message
            QtWidgets.QMessageBox.information(
                parent,
                "New Experiment File Created",
                "A new experiment file has been created with default values.\n\n"
                "You should now:\n"
                "1. Update the Sample information\n"
                "2. Set the correct paths to your data files\n"
                "3. Update the Setup information as needed\n\n"
                "Click the Help button for more information when the editor opens."
            )

            return filename

        except Exception as e:
            logging.log(1, f"Error creating new experiment file: {e}")
            QtWidgets.QMessageBox.critical(
                parent,
                "Error",
                f"Failed to create new experiment file: {e}"
            )
            return None

    def create_new_experiment(self):
        """
        Create a new experiment file.

        This method is called when the "New" button is clicked in the editor.
        It uses the create_experiment_from_template static method to create
        a new experiment file and then loads it into the editor.
        """
        filename = self.create_experiment_from_template(self)
        if filename:
            self.load_file(filename)

    def show_help(self):
        """Show help information about the experiment file format."""
        help_text = """
<h3>UCFRET Experiment File Format</h3>

<p>This editor allows you to create and edit UCFRET experiment files, which define the structure of your FRET experiment and link to the data files.</p>

<h4>Key Sections:</h4>

<ul>
<li><b>Sample</b> - Basic information about the sample and measurement</li>
<li><b>Measurement datasets</b> - Information about each measurement, including:
  <ul>
    <li>Donor-Acceptor (DA) measurement</li>
    <li>Donor-only (D0) measurement</li>
    <li>Instrument Response Functions (IRFs) for both channels</li>
  </ul>
</li>
<li><b>Setup parts</b> - Information about the components of your experimental setup</li>
<li><b>Setup</b> - Description of the complete experimental setup</li>
<li><b>Setup settings</b> - Configuration settings for the setup</li>
</ul>

<h4>Required Data Files:</h4>

<p>You need to provide paths to the following data files:</p>
<ul>
<li>Donor-Acceptor (DA) fluorescence decay data</li>
<li>Donor-only (D0) fluorescence decay data</li>
<li>Instrument Response Function (IRF) for DA channel</li>
<li>Instrument Response Function (IRF) for D0 channel</li>
</ul>

<p>These files should be in a two-column format where:
<ul>
<li>The first column represents time (in nanoseconds)</li>
<li>The second column represents photon counts</li>
</ul>
</p>

<p>Click the "..." button next to a filename field to browse for a data file.</p>
"""

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("UCFRET Experiment Editor Help")
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setText(help_text)
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.exec_()
