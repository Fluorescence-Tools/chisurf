from __future__ import annotations

import os

import pathlib
import webbrowser

import chisurf.gui
import chisurf.macros.core_fit
from chisurf import typing

import numpy as np
from chisurf.gui import QtWidgets, QtGui, QtCore, uic

import chisurf
import chisurf.decorators
import chisurf.base
import chisurf.fio
import chisurf.experiments
import chisurf.macros

import chisurf.gui.tools
import chisurf.gui.tools.settings_editor
import chisurf.gui.widgets
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments.modelling

import chisurf.models
import chisurf.plugins
import chisurf.fitting
import chisurf.gui.resources


class Main(QtWidgets.QMainWindow):
    """

    Attributes
    ----------
    current_dataset : chisurf.base.Data
        The dataset that is currently selected in the ChiSurf GUI. This
        dataset corresponds to the analysis window selected by the user in
        the UI.
    current_model_class : chisurf.model.Model
        The model used in the analysis (fit) of the currently selected analysis
        windows.
    fit_idx : int
        The index of the currently selected fit in the fit list chisurf.fits
        The current fit index corresponds to the currently selected fit window
        in the list of all fits of the fit.
    current_experiment_idx : int
        The index of the experiment type currently selected in the UI out of
        the list all supported experiments. This corresponds to the index of
        the UI combo box used to select the experiment.
    current_experiment : chisurf.experiments.Experiment
        The experiment currently selected in the GUI.
    current_setup_idx : int
        The index of the setup currently selected in the GUI.
    current_setup_name : str
        The name of the setup currently selected in the GUI.
    current_setup : chisurf.experiments.reader.ExperimentReader
        The current experiment setup / experiment reader selecetd in the GUI
    experiment_names : list
        A list containing the names of the experiments.

    """

    _current_dataset: chisurf.base.Data = None
    experiment_names: typing.List[str] = list()

    @property
    def current_dataset(self) -> chisurf.base.Data:
        return self._current_dataset

    @current_dataset.setter
    def current_dataset(self, dataset_index: int):
        self.dataset_selector.selected_curve_index = dataset_index

    @property
    def current_model_class(self):
        return self._current_model_class

    @property
    def fit_idx(self) -> int:
        return self._fit_idx

    @property
    def current_experiment_idx(self) -> int:
        return self._current_experiment_idx

    @current_experiment_idx.setter
    def current_experiment_idx(self, v: int):
        self.set_current_experiment_idx(v)

    @property
    def current_experiment(self) -> chisurf.experiments.Experiment:
        return chisurf.experiment[self.comboBox_experimentSelect.currentText()]

    @current_experiment.setter
    def current_experiment(self, name: str) -> None:
        combo = self.comboBox_experimentSelect
        # find the row in the combo whose text matches your experiment name
        idx = combo.findText(name)
        if idx == -1:
            raise ValueError(f"Experiment “{name}” not found in comboBox")
        # if it’s different from the current index, update both the combo and your internal idx
        if combo.currentIndex() != idx:
            combo.setCurrentIndex(idx)
            self._current_experiment_idx = idx

    @property
    def current_setup_idx(self) -> int:
        return self._current_setup_idx

    @current_setup_idx.setter
    def current_setup_idx(self, v: int):
        self.set_current_experiment_idx(v)

    @property
    def current_setup_name(self):
        return self.current_setup.name

    @property
    def current_setup(self) -> chisurf.experiments.reader.ExperimentReader:
        current_setup = self.current_experiment.readers[
            self.current_setup_idx
        ]
        return current_setup

    @current_setup.setter
    def current_setup(self, name: str) -> None:
        i = self.current_setup_idx
        j = i
        for j, s in enumerate(
                self.current_experiment.readers
        ):
            if s.name == name:
                break
        if j != i:
            self.current_setup_idx = j

    @property
    def current_experiment_reader(self):
        if isinstance(
            self.current_setup,
            chisurf.experiments.reader.ExperimentReader
        ):
            return self.current_setup
        elif isinstance(
                self.current_setup,
                chisurf.experiments.reader.ExperimentReaderController
        ):
            return self.current_setup.experiment_reader

    @property
    def current_model_name(self) -> str:
        return self.current_model_class.name

    @property
    def current_fit(self) -> chisurf.fitting.fit.FitGroup:
        return self._current_fit

    @current_fit.setter
    def current_fit(self, v: chisurf.fitting.fit.FitGroup) -> None:
        self._current_fit = v

    def set_current_experiment_idx(self, v):
        self.comboBox_experimentSelect.setCurrentIndex(v)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if chisurf.settings.gui['confirm_close_program']:
            reply = chisurf.gui.widgets.general.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to quit?",
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def subWindowActivated(self):
        sub_window = self.mdiarea.currentSubWindow()
        if sub_window is not None:
            for fit_idx, f in enumerate(chisurf.fits):
                if f == sub_window.fit:
                    if self.current_fit is not chisurf.fits[fit_idx]:
                        chisurf.run(f"cs.current_fit = chisurf.fits[{fit_idx}]")
                        self._fit_idx = fit_idx
                        break

            self.current_fit_widget = sub_window.fit_widget

            window_title = chisurf.__name__ + "(" + chisurf.__version__ + "): " + self.current_fit.name
            self.setWindowTitle(window_title)

            chisurf.gui.widgets.hide_items_in_layout(self.modelLayout)
            chisurf.gui.widgets.hide_items_in_layout(self.plotOptionsLayout)
            self.current_fit.model.show()
            self.current_fit_widget.show()
            sub_window.current_plot_controller.show()

    def onRunMacro(
            self,
            filename: pathlib.Path = None,
            executor: str = 'console',
            globals=None, locals=None
    ):
        chisurf.logging.info(f"Running macro: {filename}")
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                "Python macros",
                file_type="Python file (*.py)"
            )
        if executor == 'console':
            filename_str = filename.as_posix()
            chisurf.run(f"chisurf.console.run_macro(filename='{filename_str}')")
        elif executor == 'exec':
            if globals is None:
                # Create a globals dictionary with essential modules and variables
                globals = {
                    "__name__": "__main__",
                    "chisurf": chisurf,
                    "np": np,
                    "os": os,
                    "QtCore": QtCore,
                    "QtGui": QtGui,
                    "cs": self  # Add the main window as 'cs'
                }
            globals.update({"__file__": filename})

            # Get the directory of the macro file
            import sys
            import importlib
            macro_dir = str(pathlib.Path(filename).parent)

            # Temporarily add the macro directory to sys.path for relative imports
            original_sys_path = sys.path.copy()
            if macro_dir not in sys.path:
                sys.path.insert(0, macro_dir)

            try:
                # Determine if this is part of a package
                if str(filename).find('\\plugins\\') > -1:
                    parts = str(filename).split('\\plugins\\')
                    if len(parts) > 1:
                        plugin_path = parts[1].split('\\')
                        if len(plugin_path) > 0:
                            package_name = plugin_path[0]

                            # Check if this is a user plugin (in home directory) or a built-in plugin
                            user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
                            is_user_plugin = str(filename).startswith(str(user_plugin_root))

                            if is_user_plugin:
                                # For user plugins, we don't set a package name as they're not part of the chisurf package
                                globals.update({"__package__": None})
                                chisurf.logging.info(f"Running user plugin: {package_name}")

                                # Check if the user plugin has a name defined in its __init__.py
                                user_plugin_path = user_plugin_root / package_name / "__init__.py"
                                if user_plugin_path.exists():
                                    try:
                                        # Read the source
                                        source = user_plugin_path.read_text(encoding="utf-8")

                                        # Parse into an AST
                                        tree = ast.parse(source, filename=str(user_plugin_path))

                                        # Look for a name assignment
                                        for node in ast.walk(tree):
                                            if isinstance(node, ast.Assign):
                                                for target in node.targets:
                                                    if isinstance(target, ast.Name) and target.id == 'name':
                                                        if isinstance(node.value, ast.Str):
                                                            plugin_name = node.value.s
                                                            chisurf.logging.info(f"User plugin name: {plugin_name}")
                                                        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                                            plugin_name = node.value.value
                                                            chisurf.logging.info(f"User plugin name: {plugin_name}")
                                    except Exception as e:
                                        chisurf.logging.warning(f"Error extracting name from {user_plugin_path}: {e}")
                            else:
                                # For built-in plugins, set the package name as before
                                globals.update({"__package__": f"chisurf.plugins.{package_name}"})

                                # Reload all modules related to this plugin to ensure full recompilation
                                plugin_module_prefix = f"chisurf.plugins.{package_name}"
                                for module_name in list(sys.modules.keys()):
                                    if module_name.startswith(plugin_module_prefix):
                                        try:
                                            chisurf.logging.info(f"Reloading module: {module_name}")
                                            importlib.reload(sys.modules[module_name])
                                        except Exception as e:
                                            chisurf.logging.warning(f"Failed to reload module {module_name}: {e}")

                try:
                    # Check if the file exists
                    if not pathlib.Path(filename).exists():
                        # Try to find the file in the user plugins directory
                        user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
                        if '\\plugins\\' in str(filename):
                            parts = str(filename).split('\\plugins\\')
                            if len(parts) > 1:
                                plugin_path = parts[1]
                                user_plugin_path = user_plugin_root / plugin_path
                                if user_plugin_path.exists():
                                    filename = user_plugin_path
                                    chisurf.logging.info(f"Found file in user plugins directory: {filename}")
                                else:
                                    chisurf.logging.error(f"File not found: {filename}")
                                    chisurf.logging.error(f"Also checked user plugin path: {user_plugin_path}")
                                    raise FileNotFoundError(f"File not found: {filename}")
                        else:
                            chisurf.logging.error(f"File not found: {filename}")
                            raise FileNotFoundError(f"File not found: {filename}")

                    with open(filename, 'rb') as file:
                        exec(compile(file.read(), filename, 'exec'), globals, locals)
                except Exception as e:
                    chisurf.logging.error(f"Error executing macro: {e}")
                    raise
            finally:
                # Restore the original sys.path
                sys.path = original_sys_path

    def onTileWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.tileSubWindows()

    def onTabWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.TabbedView)
        self.mdiarea.setTabsClosable(True)
        self.mdiarea.setTabsMovable(True)

    def onCascadeWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.cascadeSubWindows()

    def onCurrentDatasetChanged(self):
        self._current_dataset = self.dataset_selector.selected_dataset
        self.comboBox_Model.clear()
        ds = self.current_dataset
        if chisurf.imported_datasets:
            # Get all model names from the experiment
            all_model_names = ds.experiment.get_model_names()

            # Get the list of disabled models from settings
            disabled_models = chisurf.settings.cs_settings.get('plugins', {}).get('disabled_models', [])

            # Filter out disabled models
            model_names = [name for name in all_model_names if name not in disabled_models]

            # Add only enabled models to the combobox
            self.comboBox_Model.addItems(model_names)

    def onCurrentModelChanged(self):
        model_idx = self.comboBox_Model.currentIndex()
        if model_idx >= 0:  # Make sure a valid model is selected
            # Get the selected model name from the combobox
            selected_model_name = self.comboBox_Model.currentText()

            # Find the corresponding model class in the experiment's model classes
            for model_class in self.current_dataset.experiment.model_classes:
                if model_class.name == selected_model_name:
                    self._current_model_class = model_class
                    break

    def onAddFit(self, *args, data_idx: typing.List[int] = None):
        if data_idx is None:
            data_idx = [r.row() for r in self.dataset_selector.selectedIndexes()]
        chisurf.run(f"chisurf.macros.add_fit(model_name='{self.current_model_name}', dataset_indices={data_idx})")

    def onExperimentChanged(self):
        experiment_name = self.comboBox_experimentSelect.currentText()
        chisurf.run(f"cs.current_experiment = '{experiment_name}'")

        # Add setups for selected experiment
        self.comboBox_setupSelect.blockSignals(True)
        self.comboBox_setupSelect.clear()
        self.comboBox_setupSelect.addItems(
            self.current_experiment.reader_names
        )
        self.comboBox_setupSelect.blockSignals(False)
        self._current_experiment_idx = self.comboBox_experimentSelect.currentIndex()
        self.onSetupChanged()

    def onLoadFitResults(self, **kwargs):
        filename = chisurf.gui.widgets.get_filename(
            file_type="*.json",
            description="Load results into fit-models",
            **kwargs
        )
        chisurf.run(f"chisurf.macros.load_fit_result({self.fit_idx}, {filename})")

    def onSaveProject(self, event: QtCore.QEvent = None):
        """
        Save the current state of the application as a project.

        This method prompts the user for a directory and project name, then calls
        the save_project function to save the project.
        """
        # Get directory to save project
        path, _ = chisurf.gui.widgets.get_directory()
        if not path:
            return

        # Get project name
        project_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Project",
            "Project name:",
            QtWidgets.QLineEdit.Normal,
            "chisurf_project"
        )
        if not ok or not project_name:
            return

        # Save project
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.core_fit.save_project(target_path=r"{path.as_posix()}", project_name="{project_name}")')

    def onLoadProject(self, event: QtCore.QEvent = None):
        """
        Load a project from a project folder.

        This method prompts the user for a project folder, then calls
        the load_project function to load the project.
        """
        # Get project directory
        path, _ = chisurf.gui.widgets.get_directory(
            caption="Select Project Folder"
        )
        if not path:
            return

        # Check if this is a valid project folder
        project_file = path / "project.yaml"
        if not project_file.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Project",
                f"The selected folder does not contain a valid project file (project.yaml)."
            )
            return

        # Load project
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.core_fit.load_project(project_path=r"{path.as_posix()}")')

    def set_current_setup_idx(self, v: int):
        self.comboBox_setupSelect.setCurrentIndex(v)
        self._current_setup_idx = v

    def onSetupChanged(self):
        chisurf.gui.widgets.hide_items_in_layout(
            self.layout_experiment_reader
        )
        chisurf.run(f"cs.current_setup = '{self.current_setup_name}'")
        try:
            self.layout_experiment_reader.addWidget(
                self.current_setup
            )
            self.current_setup.show()
        except TypeError:
            self.layout_experiment_reader.addWidget(
                self.current_setup.controller
            )
            self.current_setup.controller.show()
        self._current_setup_idx = self.comboBox_setupSelect.currentIndex()

    def onCloseAllFits(self):
        for sub_window in chisurf.gui.fit_windows:
            sub_window.widget().close_confirm = False
            sub_window.close()

        chisurf.fits.clear()
        chisurf.gui.fit_windows.clear()

        # Clear the analysis dock layouts
        chisurf.gui.widgets.clear_layout(self.modelLayout)
        chisurf.gui.widgets.clear_layout(self.plotOptionsLayout)

    def onAddDataset(self):
        filename = self.current_setup.controller.get_filename()
        if isinstance(filename, list):
            l = [r"{}".format(pathlib.Path(f).as_posix()) for f in filename]
            s = '|'.join(l)
        elif isinstance(filename, pathlib.Path):
            s = r"{}".format(filename.as_posix())
        else:
            s = r"{}".format(filename)
        s = s.replace("\\", "/")
        chisurf.run(f'chisurf.macros.add_dataset(filename=r"{s}")')

    def onSaveFits(self, event: QtCore.QEvent = None):
        path, _ = chisurf.gui.widgets.get_directory()
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.save_fits(target_path=r"{path.as_posix()}")')

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        path, _ = chisurf.gui.widgets.get_directory(**kwargs)
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.save_fit(target_path=r"{path.as_posix()}")')

    def onOpenHelp(self):
        webbrowser.open_new(chisurf.info.help_url)

    def onOpenUpdate(self):
        webbrowser.open_new(chisurf.info.help_url)

    def onClearLocalSettings(self):
        """Reset local settings and show a confirmation popup."""
        # Clear the settings folder
        chisurf.settings.clear_settings_folder()

        # Show a confirmation popup
        chisurf.gui.widgets.general.MyMessageBox(
            label="Settings Reset",
            info="Local settings have been reset successfully.",
            show_fortune=False
        )

    def onClearUserStyles(self):
        """Clear user style files (QSS) and show a confirmation popup."""
        # Get the path to the user styles folder
        user_styles_path = chisurf.settings.get_path('settings') / 'styles'

        # Check if the folder exists
        if user_styles_path.exists() and user_styles_path.is_dir():
            # Delete all QSS files in the folder
            for file in user_styles_path.glob('*.qss'):
                try:
                    file.unlink()
                except Exception as e:
                    chisurf.logging.warning(f"Could not delete style file {file}: {e}")

            # Show a confirmation popup
            chisurf.gui.widgets.general.MyMessageBox(
                label="Styles Reset",
                info="User style files have been cleared successfully. Restart the application to apply default styles.",
                show_fortune=False
            )
        else:
            # Show a message if the folder doesn't exist
            chisurf.gui.widgets.general.MyMessageBox(
                label="Styles Reset",
                info="No user style files found.",
                show_fortune=False
            )

    def onClearUserPlugins(self):
        """Clear user plugin folder and show a confirmation popup."""
        # Clear the user plugins folder
        chisurf.settings.clear_user_plugins_folder()

        # Show a confirmation popup
        chisurf.gui.widgets.general.MyMessageBox(
            label="User Plugins Reset",
            info="User plugins folder has been cleared successfully. Restart the application to apply changes.",
            show_fortune=False
        )

    def load_toolbar_plugins(self):
        """Load plugins into the toolbar based on toolbar_plugins setting."""
        import pathlib
        import pkgutil
        import ast

        # Get the list of toolbar plugins from settings
        toolbar_plugins = chisurf.settings.cs_settings.get('plugins', {}).get('toolbar_plugins', [])

        if not toolbar_plugins:
            return

        # Create a toolbar for plugins if it doesn't exist
        if not hasattr(self, 'plugins_toolbar'):
            self.plugins_toolbar = self.addToolBar("Plugins")
            self.plugins_toolbar.setObjectName("pluginsToolBar")
            # Set icon size to match standard toolbar (16x16)
            self.plugins_toolbar.setIconSize(QtCore.QSize(16, 16))

        # Determine built-in plugin directory
        plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent

        # Determine user plugin directory
        user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'

        # Helper function to read module docstring
        def read_module_docstring(package_path):
            init_py = package_path / "__init__.py"
            if not init_py.exists():
                return None

            # Read the source
            source = init_py.read_text(encoding="utf-8")

            # Parse into an AST and extract the docstring
            tree = ast.parse(source, filename=str(init_py))
            return ast.get_docstring(tree)

        # Helper function to get plugin name from module without importing
        def get_plugin_name(plugin_dir, module_name, check_user_dir=True):
            """Extract plugin name without importing the module."""
            # Default value
            name = module_name

            # Path to the __init__.py file
            init_py = plugin_dir / module_name / "__init__.py"

            # Check if the file exists in the built-in directory
            if not init_py.exists() and check_user_dir:
                # Try to find it in the user plugins directory
                user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
                user_init_py = user_plugin_root / module_name / "__init__.py"
                if user_init_py.exists():
                    init_py = user_init_py
                else:
                    return name
            elif not init_py.exists():
                return name

            try:
                # Read the source
                source = init_py.read_text(encoding="utf-8")

                # Parse into an AST
                tree = ast.parse(source, filename=str(init_py))

                # Look for a name assignment
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == 'name':
                                if isinstance(node.value, ast.Str):
                                    name = node.value.s
                                elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    name = node.value.value

                return name
            except Exception as e:
                chisurf.logging.warning(f"Error extracting name from {init_py}: {e}")
                return name

        # Load each toolbar plugin
        for plugin_name in toolbar_plugins:
            try:
                # Find the module name for this plugin
                module_name = None
                for module_info in chisurf.plugins.__path__:
                    for _, name, _ in pkgutil.iter_modules([module_info]):
                        # Get the plugin name without importing
                        extracted_name = get_plugin_name(plugin_root, name)
                        # Check if the extracted name matches the plugin name
                        # For user plugins, we need to check both with and without category prefix
                        if extracted_name == plugin_name:
                            module_name = name
                            break
                        # Also check if the clean name (without category) matches
                        # This is for backward compatibility with plugins that don't use category prefix
                        clean_extracted = extracted_name.split(':')[-1]
                        clean_plugin = plugin_name.split(':')[-1]
                        if clean_extracted == clean_plugin:
                            module_name = name
                            break
                    if module_name:
                        break

                if not module_name:
                    # Try to find the module in the user plugins directory
                    user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
                    for name in os.listdir(user_plugin_root) if os.path.isdir(user_plugin_root) else []:
                        if os.path.isdir(user_plugin_root / name):
                            # Get the plugin name from the user directory
                            extracted_name = get_plugin_name(user_plugin_root, name, check_user_dir=False)
                            # Check if the extracted name matches the plugin name
                            if extracted_name == plugin_name:
                                module_name = name
                                break
                            # Also check if the clean name (without category) matches
                            clean_extracted = extracted_name.split(':')[-1]
                            clean_plugin = plugin_name.split(':')[-1]
                            if clean_extracted == clean_plugin:
                                module_name = name
                                break

                if not module_name:
                    chisurf.logging.warning(f"Could not find module for plugin: {plugin_name}")
                    continue

                # Build the module path for later use with onRunMacro
                module_path = f"chisurf.plugins.{module_name}"

                # Check if this is a user plugin
                user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
                user_plugin_dir = user_plugin_root / module_name
                is_user_plugin = user_plugin_dir.exists()

                # If it's a user plugin, get the name from the user plugin directory
                if is_user_plugin:
                    user_plugin_name = get_plugin_name(user_plugin_root, module_name, check_user_dir=False)
                    if ":" in user_plugin_name:
                        # Use the user plugin name if it has a category prefix
                        plugin_name = user_plugin_name
                        chisurf.logging.info(f"Using user plugin name: {plugin_name}")

                # Get the clean plugin name (without sorting prefix)
                clean_name = plugin_name.split(':')[-1]

                # Create an action for the plugin with empty text (icon only)
                action = QtWidgets.QAction("", self)

                # Set icon if available
                # Check both built-in and user plugin directories for icons
                icon_path = plugin_root / module_name / 'icon.png'
                user_icon_path = user_plugin_root / module_name / 'icon.png'

                if icon_path.exists():
                    action.setIcon(QtGui.QIcon(str(icon_path)))
                elif user_icon_path.exists():
                    action.setIcon(QtGui.QIcon(str(user_icon_path)))

                # Get plugin description from docstring
                # Check both built-in and user plugin directories for docstrings
                plugin_path = plugin_root / module_name
                user_plugin_path = user_plugin_root / module_name

                description = read_module_docstring(plugin_path)
                if description is None:
                    # Try user plugin path
                    description = read_module_docstring(user_plugin_path)
                    if description is None:
                        description = "No description available."

                # Set tooltip to show plugin name followed by description
                action.setToolTip(f"{clean_name}: {description}")

                # Connect the action to a function that will load and show the plugin
                action.triggered.connect(lambda checked=False, m=module_path: self.load_and_show_plugin(m))

                # Add the action to the toolbar
                self.plugins_toolbar.addAction(action)

                # Log the plugin name for debugging
                chisurf.logging.info(f"Added plugin to toolbar: {plugin_name} (module: {module_name})")

            except Exception as e:
                chisurf.logging.error(f"Error loading toolbar plugin {plugin_name}: {e}")

    def load_and_show_plugin(self, module_path):
        """Load and show a plugin from its module path."""
        try:
            import pathlib
            from functools import partial

            # Extract the module name from the module path
            module_name = module_path.split('.')[-1]

            # Determine the built-in plugin directory
            plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent
            plugin_dir = plugin_root / module_name

            # Determine the user plugin directory
            user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
            user_plugin_dir = user_plugin_root / module_name

            # Check if the plugin exists in the built-in directory
            if plugin_dir.exists():
                # Use the built-in plugin
                plugin_dir_to_use = plugin_dir
                is_user_plugin = False
            # Check if the plugin exists in the user directory
            elif user_plugin_dir.exists():
                # Use the user plugin
                plugin_dir_to_use = user_plugin_dir
                is_user_plugin = True
            else:
                chisurf.logging.warning(f"Plugin directory not found in either built-in or user locations: {module_name}")
                return

            # Determine which file to run: wizard.py if it exists, else __init__.py
            wizard_path = plugin_dir_to_use / "wizard.py"
            init_path = plugin_dir_to_use / "__init__.py"

            # Check if wizard.py exists
            if wizard_path.exists():
                # Run the wizard.py with specific parameters
                adr = "https://github.com/fluorescence-tools/chisurf"  # Default value
                p = partial(
                    self.onRunMacro, wizard_path,
                    executor='exec',
                    globals={'__name__': 'plugin', 'adr': adr}
                )
                p()
            # If no wizard.py, run the plugin's __init__.py using onRunMacro
            elif init_path.exists():
                # Run the __init__.py with specific parameters
                self.onRunMacro(
                    init_path,
                    executor='exec',
                    globals={'__name__': 'plugin'}
                )
            else:
                chisurf.logging.warning(f"No wizard.py or __init__.py found for plugin: {module_path}")

        except Exception as e:
            chisurf.logging.error(f"Error loading plugin {module_path}: {e}")


    def init_console(self):
        self.verticalLayout_4.addWidget(chisurf.console)
        chisurf.console.pushVariables({'cs': self})
        chisurf.console.pushVariables({'chisurf': chisurf})
        chisurf.console.pushVariables({'np': np})
        chisurf.console.pushVariables({'os': os})
        chisurf.console.pushVariables({'QtCore': QtCore})
        chisurf.console.pushVariables({'QtGui': QtGui})
        chisurf.console.set_default_style('linux')
        chisurf.run = chisurf.console.execute
        chisurf.run(str(chisurf.settings.gui['console_init']))

    def _setup_experiment(self, exp_type, config):
        """
        Set up an experiment based on its configuration.

        Args:
            exp_type (str): The experiment type key in chisurf.experiments.types
            config (dict): Configuration for the experiment with readers and models
        """
        # Get the experiment instance
        experiment = chisurf.experiments.types[exp_type]

        # Set up readers if defined
        if 'readers' in config:
            readers_list = []
            for reader_config in config['readers']:
                reader_class = self._resolve_class(reader_config['reader_class'])
                controller_class = self._resolve_class(reader_config.get('controller_class'))

                # Create reader instance with parameters
                reader_params = reader_config.get('reader_params', {})
                reader_params['experiment'] = experiment
                reader = reader_class(**reader_params)

                # Create controller instance if specified
                controller = None
                if controller_class:
                    controller_params = reader_config.get('controller_params', {})
                    # Add settings from cs_settings if specified
                    if 'settings_key' in reader_config:
                        settings_key = reader_config['settings_key']
                        if settings_key in chisurf.settings.cs_settings:
                            controller_params.update(chisurf.settings.cs_settings[settings_key])
                    controller = controller_class(**controller_params)

                readers_list.append((reader, controller))

            experiment.add_readers(readers_list)

        # Set up models if defined
        if 'models' in config:
            model_classes = [self._resolve_class(model_class) for model_class in config['models']]
            experiment.add_model_classes(models=model_classes)

        # Register the experiment
        chisurf.experiment[experiment.name] = experiment

        return experiment

    def _resolve_class(self, class_path):
        """
        Resolve a class from its string path.

        Args:
            class_path (str): The full path to the class

        Returns:
            class: The resolved class
        """
        if not class_path:
            return None

        if not isinstance(class_path, str):
            return class_path

        parts = class_path.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]

        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def init_setups(self):
        """
        Initialize experiment setups based on configuration from YAML file.
        """
        import yaml
        import pathlib
        import shutil
        import chisurf.experiments

        # Define paths for experiment configuration file
        source_config_file = pathlib.Path(chisurf.settings.get_path('chisurf')) / "settings" / "experiment_configs.yaml"
        user_config_file = pathlib.Path(chisurf.settings.get_path('settings')) / "experiment_configs.yaml"

        # Ensure the user config file exists
        if not user_config_file.exists():
            # If user config doesn't exist but source does, copy it
            if source_config_file.exists():
                shutil.copyfile(source_config_file, user_config_file)
            else:
                # If neither exists, we'll use default configurations later
                chisurf.logging.warning(f"Experiment configuration file not found: {source_config_file}")
                experiment_configs = {}

        # Load experiment configurations from YAML file if it exists
        if user_config_file.exists():
            try:
                with open(user_config_file, 'r') as f:
                    experiment_configs = yaml.safe_load(f)
            except Exception as e:
                chisurf.logging.error(f"Error loading experiment configurations: {e}")
                experiment_configs = {}

        # Set up each standard experiment based on its configuration
        if experiment_configs:
            for exp_type, config in experiment_configs.items():
                # Skip the global experiment and experiment_types, they're handled separately
                if exp_type == 'global' or exp_type == 'experiment_types':
                    continue
                self._setup_experiment(exp_type, config)
        else:
            # Fallback to default setup if no configurations are available
            chisurf.logging.warning("Using default experiment configurations")
            # Set up each experiment type with minimal configuration
            for exp_type, experiment in chisurf.experiments.types.items():
                chisurf.experiment[experiment.name] = experiment

        # Set up global dataset using configuration from YAML
        global_config = experiment_configs.get('global', {})
        global_fit = chisurf.experiments.experiment.Experiment(
            name=global_config.get('name', 'Global'),
            hidden=global_config.get('hidden', True)
        )

        # Set up global reader
        if 'readers' in global_config and global_config['readers']:
            reader_config = global_config['readers'][0]
            reader_class = self._resolve_class(reader_config['reader_class'])
            reader_params = reader_config.get('reader_params', {})
            reader_params['experiment'] = global_fit
            global_setup = reader_class(**reader_params)
            global_fit.add_reader(global_setup)
        else:
            # Fallback to default if not configured
            global_setup = chisurf.experiments.globalfit.GlobalFitSetup(
                name='Global-Fit',
                experiment=global_fit
            )
            global_fit.add_reader(global_setup)

        # Set up global models
        if 'models' in global_config:
            model_classes = [self._resolve_class(model_class) for model_class in global_config['models']]
            global_fit.add_model_classes(models=model_classes)

        chisurf.experiment[global_fit.name] = global_fit

        chisurf.macros.add_dataset(global_setup, name="Global Dataset")

        # Update UI
        # Filter out hidden experiments
        self.experiment_names = [
            b.name for b in list(chisurf.experiment.values()) 
            if not b.hidden
        ]
        self.comboBox_experimentSelect.addItems(
            self.experiment_names
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(pathlib.Path(__file__).parent / "gui.ui", self)

        self.current_fit_widget = None
        self._current_fit = None
        self._current_model_class = None
        self._current_experiment_idx = 0
        self._fit_idx = 0
        self._current_setup_idx = 0

        self.experiment_names = list()
        self.dataset_selector = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=False,
            curve_types='all',
            change_event=self.onCurrentDatasetChanged,
            drag_enabled=True,
            experiment=None
        )

        # widget listing the existing fits
        self.fit_selector = chisurf.gui.widgets.fitting.ModelDataRepresentationSelector(parent=self)

        self.about = uic.loadUi(pathlib.Path(__file__).parent / "about.ui")

        # Setup status bar with progress bar and message
        self.status = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status)

        # Create a QWidget to hold the progress bar and message
        status_widget = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_widget)

        # Set spacing and margins to zero
        status_layout.setSpacing(0)  # Set spacing between widgets to zero
        status_layout.setContentsMargins(0, 0, 0, 0)  # Set margins to zero

        # Create a progress bar
        self.progress_bar = QtWidgets.QProgressBar(self.status)
        self.progress_bar.setFixedWidth(150)  # Set a fixed width for the progress bar
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_bar.setFixedHeight(15)  # Adjust the height as needed

        # Create a label for the status message
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setMaximumHeight(20)  # Set a maximum height for the status message

        # Add the progress bar and status message to the status layout
        status_layout.addWidget(self.status_label)
        status_layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        status_layout.addWidget(self.progress_bar)

        # Add the status widget to the status bar, aligning to the left
        self.status.addWidget(status_widget, 1)  # 1 gives the widget some stretch

    def update(self):
        super().update()
        self.fit_selector.update()
        self.dataset_selector.update()

    def arrange_widgets(self):
        # self.setCentralWidget(self.mdiarea)

        ##########################################################
        #      Help and About widgets                            #
        ##########################################################
        self.about.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.about.hide()

        ##########################################################
        #      IPython console                                   #
        #      Push variables to console and add it to           #
        #      user interface                                    #
        ##########################################################
        self.dockWidgetScriptEdit.setVisible(chisurf.settings.gui['show_macro_edit'])
        self.dockWidget_console.setVisible(chisurf.settings.gui['show_console'])
        # Set the height of the console dock widget
        if 'console_height' in chisurf.settings.gui:
            from qtpy.QtCore import Qt
            self.resizeDocks([self.dockWidget_console], [chisurf.settings.gui['console_height']], Qt.Vertical)
        self.init_console()

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        self.tabifyDockWidget(self.dockWidgetReadData, self.dockWidgetDatasets)
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetAnalysis)
        self.tabifyDockWidget(self.dockWidgetAnalysis, self.dockWidgetPlot)
        self.tabifyDockWidget(self.dockWidgetPlot, self.dockWidgetScriptEdit)
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetHistory)
        self.editor = chisurf.gui.tools.code_editor.CodeEditor()

        self.verticalLayout_10.addWidget(self.editor)

        # Add data selector widget
        self.verticalLayout_8.addWidget(self.dataset_selector)

        # Add fit selector widget
        self.verticalLayout_5.addWidget(self.fit_selector)

        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidgetReadData.raise_()

    def define_actions(self):
        ##########################################################
        # GUI ACTIONS
        ##########################################################
        self.actionTile_windows.triggered.connect(self.onTileWindows)
        self.actionTab_windows.triggered.connect(self.onTabWindows)
        self.actionCascade.triggered.connect(self.onCascadeWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)
        self.actionAbout.triggered.connect(self.about.show)
        self.actionHelp_2.triggered.connect(self.onOpenHelp)
        self.actionUpdate.triggered.connect(self.onOpenUpdate)

        ##########################################################
        #      Record and run recorded macros                    #
        ##########################################################
        self.actionRecord.triggered.connect(chisurf.console.start_recording)
        self.actionStop.triggered.connect(chisurf.console.save_macro)
        self.actionRun.triggered.connect(
            lambda: self.onRunMacro(filename=None, executor='console')
        )

        ##########################################################
        #    Connect changes in User-interface to actions like:  #
        #    Loading dataset, changing setups, models, etc.      #
        ##########################################################
        self.actionSetupChanged.triggered.connect(self.onSetupChanged)
        self.actionExperimentChanged.triggered.connect(self.onExperimentChanged)
        self.actionChange_current_dataset.triggered.connect(self.onCurrentDatasetChanged)
        self.comboBox_Model.currentIndexChanged.connect(self.onCurrentModelChanged)
        self.comboBox_experimentSelect.currentIndexChanged.connect(self.onExperimentChanged)
        self.comboBox_setupSelect.currentIndexChanged.connect(self.onSetupChanged)
        self.actionAdd_fit.triggered.connect(self.onAddFit)
        self.actionSaveAllFits.triggered.connect(self.onSaveFits)
        self.actionSaveCurrentFit.triggered.connect(self.onSaveFit)
        self.actionClose_Fit.triggered.connect(chisurf.macros.core_fit.close_fit)
        self.actionClose_all_fits.triggered.connect(self.onCloseAllFits)
        self.actionLoad_Data.triggered.connect(self.onAddDataset)
        self.actionLoad_result_in_current_fit.triggered.connect(self.onLoadFitResults)

        # Add actions for saving and loading projects (disabled)
        self.actionSaveProject = QtWidgets.QAction("Save Project...", self)
        self.actionSaveProject.setShortcut("Ctrl+Shift+S")
        self.actionSaveProject.triggered.connect(self.onSaveProject)
        self.actionSaveProject.setEnabled(False)
        self.menuFile.addAction(self.actionSaveProject)

        self.actionLoadProject = QtWidgets.QAction("Load Project...", self)
        self.actionLoadProject.setShortcut("Ctrl+Shift+O")
        self.actionLoadProject.triggered.connect(self.onLoadProject)
        self.actionLoadProject.setEnabled(False)
        self.menuFile.addAction(self.actionLoadProject)

    def load_tools(self):
        import chisurf
        import chisurf.gui
        import chisurf.gui.tools

        ##########################################################
        #      Fluorescence widgets                              #
        #      (Commented widgets don't work at the moment       #
        ##########################################################


        self.f_test = chisurf.gui.tools.f_test.FTestWidget()
        self.actionF_Test.triggered.connect(self.f_test.show)

        ##########################################################
        #      Load toolbar plugins                              #
        ##########################################################
        self.load_toolbar_plugins()

        ##########################################################
        #      Settings                                          #
        ##########################################################
        # Configuration editor
        self.configuration = chisurf.gui.tools.settings_editor.SettingsEditor(
            filename=chisurf.settings.chisurf_settings_file
        )
        self.actionSettings.triggered.connect(self.configuration.show)
        # Reset local settings, i.e., the settings file in the user folder
        self.actionClear_local_settings.triggered.connect(self.onClearLocalSettings)
        # Clear logging files, i.e., the log files in the user folder
        self.actionClear_logging_files.triggered.connect(chisurf.settings.clear_logging_files)
        # Clear user styles, i.e., the QSS files in the user folder
        self.actionClear_user_styles = QtWidgets.QAction("Clear user styles", self)
        self.actionClear_user_styles.triggered.connect(self.onClearUserStyles)
        self.menuSettings.addAction(self.actionClear_user_styles)

        # Clear user plugins, i.e., the plugins in the user folder
        self.actionClear_user_plugins = QtWidgets.QAction("Clear user plugins", self)
        self.actionClear_user_plugins.triggered.connect(self.onClearUserPlugins)
        self.menuSettings.addAction(self.actionClear_user_plugins)


        ##########################################################
        #      Initialize                                        #
        ##########################################################
        self.onExperimentChanged()
