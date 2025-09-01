"""
Model Manager for ChiSurf

This plugin allows you to manage all models and experiments in ChiSurf. You can:
- View all available models and experiments
- Enable or disable models and experiments
- View model and experiment descriptions

The model manager provides a convenient interface for configuring which models
and experiments are available in ChiSurf.
"""

import sys
import pathlib
import importlib
import yaml
from typing import Optional, Dict, List, Type

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QCheckBox,
    QMessageBox, QGroupBox, QScrollArea, QSplitter, QTextEdit, QLineEdit,
    QTabWidget
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

import chisurf
import chisurf.models
import chisurf.experiments
import chisurf.settings

# Define the plugin name - this will appear in the Plugins menu
name = "Setup:Model Manager"


class ModelManagerWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Manager")
        self.resize(800, 500)

        # Get settings
        self.settings = chisurf.settings.cs_settings.get('plugins', {})
        self.disabled_models = self.settings.get('disabled_models', [])
        self.disabled_experiments = self.settings.get('disabled_experiments', [])
        self.hide_disabled = self.settings.get('hide_disabled_models', False)  # Default to not hiding

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Display settings file location
        settings_file_group = QGroupBox("Settings File Location")
        settings_file_layout = QVBoxLayout(settings_file_group)
        self.settings_file_edit = QLineEdit()
        self.settings_file_edit.setReadOnly(True)
        self.settings_file_edit.setMaximumHeight(50)

        # Determine which settings file is being used
        if chisurf.settings.cs_settings.get('use_source_folder', True):
            settings_file = pathlib.Path(chisurf.settings.__file__).parent / 'settings_chisurf.yaml'
        else:
            settings_file = chisurf.settings.chisurf_settings_file

        self.settings_file_edit.setText(str(settings_file))
        settings_file_layout.addWidget(self.settings_file_edit)
        main_layout.addWidget(settings_file_group)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create models tab
        self.models_tab = QWidget()
        self.tab_widget.addTab(self.models_tab, "Models")
        self.setup_models_tab()

        # Create experiments tab
        self.experiments_tab = QWidget()
        self.tab_widget.addTab(self.experiments_tab, "Experiments")
        self.setup_experiments_tab()

        # Create buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)

        refresh_button = QPushButton("Refresh Lists")
        refresh_button.clicked.connect(self.load_all)
        button_layout.addWidget(refresh_button)

        main_layout.addLayout(button_layout)

        # Load models and experiments
        self.load_all()

    def setup_models_tab(self):
        """Set up the models tab with a splitter for list and details."""
        layout = QVBoxLayout(self.models_tab)

        # Create splitter for list and details
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Create list widget for models
        list_group = QGroupBox("Available Models")
        list_layout = QVBoxLayout(list_group)
        self.model_list = QListWidget()
        self.model_list.setMinimumWidth(300)
        self.model_list.currentItemChanged.connect(self.on_model_selected)
        list_layout.addWidget(self.model_list)
        splitter.addWidget(list_group)

        # Create details widget
        details_group = QGroupBox("Model Details")
        details_layout = QVBoxLayout(details_group)

        # Model name and status
        name_layout = QHBoxLayout()
        self.model_name_label = QLabel("Select a model")
        name_layout.addWidget(self.model_name_label)
        name_layout.addStretch()
        details_layout.addLayout(name_layout)

        # Model status
        status_layout = QHBoxLayout()
        self.model_disabled_checkbox = QCheckBox("Disable model")
        self.model_disabled_checkbox.stateChanged.connect(self.on_model_disabled_changed)
        status_layout.addWidget(self.model_disabled_checkbox)
        status_layout.addStretch()
        details_layout.addLayout(status_layout)

        # Model experiment
        exp_layout = QHBoxLayout()
        exp_label = QLabel("Experiment:")
        exp_layout.addWidget(exp_label)
        self.model_experiment_label = QLabel("Not available")
        exp_layout.addWidget(self.model_experiment_label)
        exp_layout.addStretch()
        details_layout.addLayout(exp_layout)

        # Model module path
        module_layout = QHBoxLayout()
        module_label = QLabel("Module Path:")
        module_layout.addWidget(module_label)
        self.model_module_edit = QTextEdit()
        self.model_module_edit.setReadOnly(True)
        self.model_module_edit.setMaximumHeight(50)
        module_layout.addWidget(self.model_module_edit)
        details_layout.addLayout(module_layout)

        # Model description
        self.model_description_edit = QTextEdit()
        self.model_description_edit.setReadOnly(True)
        details_layout.addWidget(self.model_description_edit)

        splitter.addWidget(details_group)

    def setup_experiments_tab(self):
        """Set up the experiments tab with a splitter for list and details."""
        layout = QVBoxLayout(self.experiments_tab)

        # Create splitter for list and details
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Create list widget for experiments
        list_group = QGroupBox("Available Experiments")
        list_layout = QVBoxLayout(list_group)
        self.experiment_list = QListWidget()
        self.experiment_list.setMinimumWidth(300)
        self.experiment_list.currentItemChanged.connect(self.on_experiment_selected)
        list_layout.addWidget(self.experiment_list)
        splitter.addWidget(list_group)

        # Create details widget
        details_group = QGroupBox("Experiment Details")
        details_layout = QVBoxLayout(details_group)

        # Experiment name and status
        name_layout = QHBoxLayout()
        self.experiment_name_label = QLabel("Select an experiment")
        name_layout.addWidget(self.experiment_name_label)
        name_layout.addStretch()
        details_layout.addLayout(name_layout)

        # Experiment status
        status_layout = QHBoxLayout()
        self.experiment_disabled_checkbox = QCheckBox("Disable experiment")
        self.experiment_disabled_checkbox.stateChanged.connect(self.on_experiment_disabled_changed)
        status_layout.addWidget(self.experiment_disabled_checkbox)
        status_layout.addStretch()
        details_layout.addLayout(status_layout)

        # Experiment module path
        module_layout = QHBoxLayout()
        module_label = QLabel("Module Path:")
        module_layout.addWidget(module_label)
        details_layout.addLayout(module_layout)

        # Experiment models
        models_group = QGroupBox("Associated Models")
        models_layout = QVBoxLayout(models_group)
        self.experiment_models_list = QListWidget()
        models_layout.addWidget(self.experiment_models_list)
        details_layout.addWidget(models_group)

        splitter.addWidget(details_group)

    def load_all(self):
        """Load all models and experiments."""
        self.load_models()
        self.load_experiments()

    def load_models(self):
        """Load all available models and display them in the list."""
        self.model_list.clear()
        self.models = {}

        # Get all experiments
        experiments = chisurf.experiments.types

        # Collect all models from all experiments
        for exp_name, experiment in experiments.items():
            for model_class in experiment.model_classes:
                model_name = model_class.name

                # Format display name with experiment:model format
                display_name = f"{experiment.name} - {model_name}"

                # Check if this model is marked as disabled
                is_disabled = model_name in self.disabled_models

                # Create list item
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, model_name)  # Keep original model name as data

                # Mark models based on status
                if is_disabled:
                    item.setForeground(Qt.gray)
                    item.setText(f"{display_name} [DISABLED]")

                # Add to list widget
                self.model_list.addItem(item)

                # Store model metadata
                doc = model_class.__doc__
                if doc is None:
                    doc = "No description available."

                # Get the full module path
                module_path = f"{model_class.__module__}.{model_class.__name__}"

                self.models[model_name] = {
                    'name': model_name,
                    'display_name': display_name,
                    'class': model_class,
                    'experiment': exp_name,
                    'is_disabled': is_disabled,
                    'doc': doc,
                    'module_path': module_path
                }

    def load_experiments(self):
        """Load all available experiments and display them in the list."""
        self.experiment_list.clear()
        self.experiments = {}

        # Get all experiments
        experiments = chisurf.experiments.types

        for _, experiment in experiments.items():
            # Check if this experiment is marked as disabled
            exp_name = experiment.name
            is_disabled = exp_name in self.disabled_experiments

            # Create list item
            item = QListWidgetItem(exp_name)
            item.setData(Qt.UserRole, exp_name)

            # Mark experiments based on status
            if is_disabled:
                item.setForeground(Qt.gray)
                item.setText(f"{exp_name} [DISABLED]")

            # Add to list widget
            self.experiment_list.addItem(item)

            # Store experiment metadata
            doc = experiment.__doc__
            if doc is None:
                doc = "No description available."

            self.experiments[exp_name] = {
                'name': exp_name,
                'experiment': experiment,
                'is_disabled': is_disabled,
                'doc': doc,
                'models': [model.name for model in experiment.model_classes]
            }

    def on_model_selected(self, current, previous):
        """Handle model selection in the list."""
        if current is None:
            self.model_name_label.setText("Select a model")
            self.model_experiment_label.setText("Not available")
            self.model_disabled_checkbox.setChecked(False)
            self.model_module_edit.clear()
            self.model_description_edit.clear()
            return

        model_name = current.data(Qt.UserRole)
        model_info = self.models[model_name]

        self.model_name_label.setText(model_info['name'])
        self.model_disabled_checkbox.setChecked(model_info['is_disabled'])
        self.model_experiment_label.setText(model_info['experiment'])
        self.model_module_edit.setText(model_info['module_path'])
        self.model_description_edit.setText(model_info['doc'])

    def on_experiment_selected(self, current, previous):
        """Handle experiment selection in the list."""
        if current is None:
            self.experiment_name_label.setText("Select an experiment")
            self.experiment_disabled_checkbox.setChecked(False)
            self.experiment_models_list.clear()
            return

        exp_name = current.data(Qt.UserRole)
        exp_info = self.experiments[exp_name]

        self.experiment_name_label.setText(exp_info['name'])
        self.experiment_disabled_checkbox.setChecked(exp_info['is_disabled'])

        # Update models list
        self.experiment_models_list.clear()
        for model_name in exp_info['models']:
            # Get the model info if available
            if model_name in self.models:
                model_info = self.models[model_name]
                display_name = model_info.get('display_name', model_name)
            else:
                display_name = model_name

            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, model_name)

            # Check if model is disabled
            if model_name in self.disabled_models:
                item.setForeground(Qt.gray)
                item.setText(f"{display_name} [DISABLED]")
            self.experiment_models_list.addItem(item)

    def on_model_disabled_changed(self, state):
        """Handle model disabled checkbox state change."""
        current = self.model_list.currentItem()
        if current is None:
            return

        model_name = current.data(Qt.UserRole)
        model_info = self.models[model_name]
        display_name = model_info.get('display_name', model_name)

        if state == Qt.Checked:
            if model_name not in self.disabled_models:
                self.disabled_models.append(model_name)
            model_info['is_disabled'] = True
        else:
            if model_name in self.disabled_models:
                self.disabled_models.remove(model_name)
            model_info['is_disabled'] = False

        # Update the list item
        if model_info['is_disabled']:
            current.setForeground(Qt.gray)
            current.setText(f"{display_name} [DISABLED]")
        else:
            current.setForeground(Qt.black)
            current.setText(display_name)

        # Update the experiment models list if this model is in the current experiment
        exp_item = self.experiment_list.currentItem()
        if exp_item is not None:
            exp_name = exp_item.data(Qt.UserRole)
            exp_info = self.experiments[exp_name]
            if model_name in exp_info['models']:
                self.on_experiment_selected(exp_item, None)

    def on_experiment_disabled_changed(self, state):
        """Handle experiment disabled checkbox state change."""
        current = self.experiment_list.currentItem()
        if current is None:
            return

        exp_name = current.data(Qt.UserRole)
        exp_info = self.experiments[exp_name]

        if state == Qt.Checked:
            if exp_name not in self.disabled_experiments:
                self.disabled_experiments.append(exp_name)
            exp_info['is_disabled'] = True
        else:
            if exp_name in self.disabled_experiments:
                self.disabled_experiments.remove(exp_name)
            exp_info['is_disabled'] = False

        # Update the list item
        if exp_info['is_disabled']:
            current.setForeground(Qt.gray)
            current.setText(f"{exp_name} [DISABLED]")
        else:
            current.setForeground(Qt.black)
            current.setText(exp_name)


    def save_settings(self):
        """Save settings to the settings file."""
        # Update settings
        self.settings['disabled_models'] = self.disabled_models
        self.settings['disabled_experiments'] = self.disabled_experiments
        self.settings['hide_disabled_models'] = self.hide_disabled

        # Update settings in chisurf
        chisurf.settings.cs_settings['plugins'] = self.settings

        # Determine which settings file to use
        if chisurf.settings.cs_settings.get('use_source_folder', True):
            settings_file = pathlib.Path(chisurf.settings.__file__).parent / 'settings_chisurf.yaml'
        else:
            settings_file = chisurf.settings.chisurf_settings_file

        # Save settings to file
        try:
            with open(settings_file, 'w') as f:
                yaml.dump(chisurf.settings.cs_settings, f, default_flow_style=False)
            QMessageBox.information(
                self,
                "Settings Saved",
                f"Model and experiment settings have been "
                f"saved successfully to:\n{settings_file}\n\nA restart "
                f"of the software is required for the changes to take effect."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings to {settings_file}: {e}")


# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the ModelManagerWidget class
    window = ModelManagerWidget()
    # Show the window
    window.show()
