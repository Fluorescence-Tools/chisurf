"""
FCS Power Series Wizard

This module implements a wizard for creating and analyzing FCS power series data.
"""

import sys
import os.path
import pathlib
import typing
import numpy as np

from chisurf.gui import QtWidgets, QtGui, QtCore

import chisurf.gui
import chisurf.gui.widgets
import chisurf.gui.decorators
import chisurf.gui.tools
import chisurf.plugins
import chisurf.gui.tools.parameter_editor

import chisurf.data
import chisurf.experiments
import chisurf.curve
import chisurf.fitting
import chisurf.macros
import chisurf.settings

import pyqtgraph as pg


class FCSPowerSeriesWizard(QtWidgets.QWizard):
    """
    A wizard for creating and analyzing FCS power series data.

    This wizard guides the user through the process of:
    1. Loading FCS curves via drag and drop
    2. Selecting an equation to describe the FCS curves
    3. Selecting which parameters are linked
    4. Creating a set of FCS fits for the loaded datasets
    """

    data: typing.Dict[str, chisurf.curve.Curve] = {}

    def __init__(self, *args, **kwargs):
        """Initialize the FCS Power Series Wizard."""
        self.ui_file = "wizard.ui"
        self.ui_path = pathlib.Path(__file__).parent

        # Initialize UI
        super().__init__(*args, **kwargs)
        self.setWindowTitle("FCS Power Series Wizard")

        # Create pages
        self.create_pages()

        # Initialize parameter linking table
        self.update_parameter_linking()

        # Connect signals
        self.button(QtWidgets.QWizard.FinishButton).clicked.connect(self.onFinish)
        self.model_selector.currentIndexChanged.connect(self.update_parameter_linking)

        # Set window properties
        self.resize(800, 600)

    def create_pages(self):
        """Create the wizard pages."""
        # Page 1: Select Data
        self.page_select_data = QtWidgets.QWizardPage()
        self.page_select_data.setTitle("Select FCS Data")
        self.page_select_data.setSubTitle("Drag and drop FCS data files or select them using the file browser.")

        # Add isComplete method to validate that at least one dataset is selected
        def isComplete():
            datasets = self.data_selector.get_datasets()
            return len(datasets) > 0

        self.page_select_data.isComplete = isComplete

        layout_select_data = QtWidgets.QVBoxLayout(self.page_select_data)

        # Create data selector widget
        self.data_selector = chisurf.gui.widgets.experiments.widgets.ExperimentalDataSelector(
            parent=self.page_select_data,
            drag_enabled=True,
            click_close=False,
            context_menu_enabled=True
        )
        layout_select_data.addWidget(self.data_selector)

        # Add refresh button
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(lambda: self.data_selector.update())
        layout_select_data.addWidget(refresh_button)

        # Connect data selector signals
        self.data_selector.model().rowsInserted.connect(lambda: self.page_select_data.completeChanged.emit())
        self.data_selector.model().rowsRemoved.connect(lambda: self.page_select_data.completeChanged.emit())

        # Add page to wizard
        self.addPage(self.page_select_data)

        # Page 2: Select Model
        self.page_select_model = QtWidgets.QWizardPage()
        self.page_select_model.setTitle("Select FCS Model")
        self.page_select_model.setSubTitle("Select an equation to describe the FCS curves.")

        layout_select_model = QtWidgets.QVBoxLayout(self.page_select_model)

        # Create model selector widget
        model_label = QtWidgets.QLabel("Model:")
        layout_select_model.addWidget(model_label)

        self.model_selector = QtWidgets.QComboBox(self.page_select_model)
        layout_select_model.addWidget(self.model_selector)

        # Add description label
        description_label = QtWidgets.QLabel("Description:")
        layout_select_model.addWidget(description_label)

        self.model_description = QtWidgets.QTextEdit()
        self.model_description.setReadOnly(True)
        layout_select_model.addWidget(self.model_description)

        # Load available FCS models
        self.load_fcs_models()

        # Add page to wizard
        self.addPage(self.page_select_model)

        # Page 3: Link Parameters
        self.page_link_params = QtWidgets.QWizardPage()
        self.page_link_params.setTitle("Link Parameters")
        self.page_link_params.setSubTitle("Select which parameters should be linked between fits.")

        layout_link_params = QtWidgets.QVBoxLayout(self.page_link_params)

        # Create parameter linking widget
        self.param_linking = QtWidgets.QTableWidget(self.page_link_params)
        self.param_linking.setColumnCount(2)
        self.param_linking.setHorizontalHeaderLabels(["Parameter", "Link"])
        self.param_linking.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.param_linking.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.param_linking.setAlternatingRowColors(True)
        layout_link_params.addWidget(self.param_linking)

        # Add instructions
        instructions = QtWidgets.QLabel("Check the parameters that should be linked between fits. "
                                       "The first dataset will be used as the master fit.")
        layout_link_params.addWidget(instructions)

        # Add page to wizard
        self.addPage(self.page_link_params)

    def load_fcs_models(self):
        """Load available FCS models from the models.yaml file."""
        # Get the path to the FCS models file
        models_file = pathlib.Path(chisurf.models.fcs.fcs.__file__).parent / 'models.yaml'

        # Load the models
        with open(models_file, 'r') as f:
            import yaml
            models = yaml.safe_load(f)

        # Add models to the combo box
        for model_name in models.keys():
            self.model_selector.addItem(model_name)

    def update_parameter_linking(self):
        """Update the parameter linking table based on the selected model."""
        # Get the selected model
        model_name = self.model_selector.currentText()

        # Clear the table
        self.param_linking.setRowCount(0)

        # Get the model parameters
        models_file = pathlib.Path(chisurf.models.fcs.fcs.__file__).parent / 'models.yaml'
        with open(models_file, 'r') as f:
            import yaml
            models = yaml.safe_load(f)

        # Get parameters for the selected model
        if model_name in models:
            # Update model description
            description = models[model_name].get('description', 'No description available')
            equation = models[model_name].get('equation', '')
            self.model_description.setText(f"Description: {description}\n\nEquation: {equation}")

            params = models[model_name]['initial'].keys()

            # Add parameters to the table
            for i, param in enumerate(params):
                self.param_linking.insertRow(i)
                self.param_linking.setItem(i, 0, QtWidgets.QTableWidgetItem(param))

                # Add checkbox for linking
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(True)  # Default to linked
                self.param_linking.setCellWidget(i, 1, checkbox)

    def onFinish(self):
        """Create the FCS fits when the wizard is finished."""
        # Get selected datasets
        datasets = self.data_selector.get_datasets()
        if not datasets:
            chisurf.gui.widgets.MyMessageBox(
                "No fits created!",
                info="No datasets selected.",
                show_fortune=True
            )
            return

        # Get selected model
        model_name = self.model_selector.currentText()

        # Create fits for each dataset
        for dataset in datasets:
            chisurf.imported_datasets.append(dataset)

        # Create a fit for the first dataset (master fit)
        chisurf.macros.core_fit.add_fit(
            model_name=model_name,
            dataset_indices=[0]
        )
        master_fit = chisurf.fits[-1]

        # Create fits for the remaining datasets and link parameters
        for i in range(1, len(datasets)):
            chisurf.macros.core_fit.add_fit(
                model_name=model_name,
                dataset_indices=[i]
            )
            current_fit = chisurf.fits[-1]

            # Link parameters based on user selection
            for row in range(self.param_linking.rowCount()):
                param_name = self.param_linking.item(row, 0).text()
                checkbox = self.param_linking.cellWidget(row, 1)

                if checkbox.isChecked():
                    # Link parameter to master fit
                    current_fit.model.parameters_all_dict[param_name].link = master_fit.model.parameters_all_dict[param_name]

        # Create a global fit
        chisurf.macros.core_fit.add_fit(model_name='Global fit', dataset_indices=[0])
        global_fit = chisurf.fits[-1]

        # Add all fits to the global fit
        for i in range(len(datasets)):
            global_fit.model.append_fit(chisurf.fits[i])

        # Update the fits
        for fit in chisurf.fits:
            fit.update()


if __name__ == "plugin":
    wizard = FCSPowerSeriesWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = FCSPowerSeriesWizard()
    wizard.show()
    sys.exit(app.exec_())
