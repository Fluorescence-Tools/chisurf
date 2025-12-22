from __future__ import annotations

import os
import ast
import json
import pathlib
import traceback

import chisurf.gui
import chisurf.macros.core_fit
from chisurf import typing

import numpy as np
from chisurf.gui import QtWidgets, QtGui, QtCore, uic
from chisurf.gui.widgets import system_info_watermark as _system_info_watermark
from chisurf.gui.gui_tweaks import apply_platform_window_tweaks, apply_dock_tab_colors


class TruncatingStatusBar(QtWidgets.QStatusBar):
    """QStatusBar that truncates overly long messages by keeping the start and end
    and inserting ellipsis in the middle.

    This ensures the status bar stays readable even for very long messages.
    """
    def __init__(self, *args, max_message_length: int = 160, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_message_length = max(7, int(max_message_length))  # minimum to allow x...y

    def setMaxMessageLength(self, n: int):
        try:
            self._max_message_length = max(7, int(n))
        except Exception:
            pass

    def _format_message(self, message: str) -> str:
        try:
            s = str(message)
        except Exception:
            return message
        max_len = self._max_message_length
        if not s or len(s) <= max_len:
            return s
        # Compute how many chars to keep from start and end, reserving 3 for '...'
        keep_total = max_len - 3
        start_keep = keep_total // 2
        end_keep = keep_total - start_keep
        return f"{s[:start_keep]}...{s[-end_keep:]}"

    # Override showMessage to apply truncation automatically
    def showMessage(self, message: str, timeout: int = 0):  # type: ignore[override]
        truncated = self._format_message(message)
        super().showMessage(truncated, timeout)

import chisurf
import chisurf.decorators
import chisurf.base
import chisurf.fio
import chisurf.experiments
import chisurf.macros

import chisurf.gui.widgets.settings_editor
import chisurf.gui.widgets
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments.modelling
from chisurf.gui.widgets.general import LogListWidget

import chisurf.models
import chisurf.plugins
import chisurf.fitting
import chisurf.gui.resources
import chisurf.plugins.misc.code_editor
import chisurf.gui.project_helpers as project_helpers
import chisurf.gui.fit_helpers as fit_helpers
import chisurf.gui.misc_helpers as misc_helpers


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
    current_experiment : chisurf.experiments.core.Experiment
        The experiment currently selected in the GUI.
    current_setup_idx : int
        The index of the setup currently selected in the GUI.
    current_setup_name : str
        The name of the setup currently selected in the GUI.
    current_setup : chisurf.experiments.core.reader.ExperimentReader
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
    def current_experiment(self) -> chisurf.experiments.core.Experiment:
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
            # Call onExperimentChanged to update the GUI
            self.onExperimentChanged()

    @property
    def current_setup_idx(self) -> int:
        return self._current_setup_idx

    @current_setup_idx.setter
    def current_setup_idx(self, v: int):
        self.set_current_setup_idx(v)

    @property
    def current_setup_name(self):
        return self.current_setup.name

    @property
    def current_setup(self) -> chisurf.experiments.core.reader.ExperimentReader:
        readers = self.current_experiment.readers
        if not readers:
            raise IndexError("No experiment readers defined for the current experiment")
        idx = getattr(self, "_current_setup_idx", 0)
        try:
            combo_idx = self.comboBox_setupSelect.currentIndex()
        except Exception:
            combo_idx = idx
        if idx < 0 or idx >= len(readers):
            if 0 <= combo_idx < len(readers):
                idx = combo_idx
            else:
                idx = 0
        self._current_setup_idx = idx
        current_setup = readers[idx]
        return current_setup

    @current_setup.setter
    def current_setup(self, name: str) -> None:
        i = self.current_setup_idx
        j = i
        setup_found = False
        for j, s in enumerate(
                self.current_experiment.readers
        ):
            if s.name == name:
                setup_found = True
                break
        if not setup_found:
            # Display a popup message if the setup name doesn't exist
            chisurf.gui.widgets.general.MyMessageBox(
                label="Setup Not Found",
                info=f"Setup '{name}' does not exist in the current experiment.",
                show_fortune=False
            )
            return
        if j != i:
            self.current_setup_idx = j
            # Call onSetupChanged to update the GUI
            self.onSetupChanged()
            
    @property
    def filter_hide_enabled(self) -> bool:
        """
        Property to check if the hide filter checkbox is checked.
        
        Returns:
            bool: True if non-matching log entries should be hidden, False otherwise
        """
        return self.checkBox_filter_hide.isChecked()
        
    @filter_hide_enabled.setter
    def filter_hide_enabled(self, value: bool) -> None:
        """
        Property to set the state of the hide filter checkbox.
        
        Args:
            value (bool): True to hide non-matching log entries, False to gray them out
        """
        self.checkBox_filter_hide.setChecked(value)

    @property
    def current_experiment_reader(self):
        if isinstance(
            self.current_setup,
            chisurf.experiments.core.reader.ExperimentReader
        ):
            return self.current_setup
        elif isinstance(
                self.current_setup,
                chisurf.experiments.core.reader.ExperimentReaderController
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
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        # Controlled spin-down: close all fits before the main window exits.
        try:
            self.onCloseAllFits()
        except Exception:
            pass

        event.accept()

    def subWindowActivated(self):
        sub_window = self.mdiarea.currentSubWindow()
        if sub_window is not None:
            # Clear existing widgets from layouts
            chisurf.gui.widgets.hide_items_in_layout(self.modelLayout)
            header_layout = getattr(self, "analysisHeaderLayout", None)
            if header_layout is not None:
                chisurf.gui.widgets.hide_items_in_layout(header_layout)
            chisurf.gui.widgets.hide_items_in_layout(self.plotOptionsLayout)

            # Handle fit windows first
            if hasattr(sub_window, 'fit') and sub_window.fit is not None:
                for fit_idx, f in enumerate(chisurf.fits):
                    if f == sub_window.fit:
                        if self.current_fit is not chisurf.fits[fit_idx]:
                            chisurf.run(f"cs.current_fit = chisurf.fits[{fit_idx}]")
                            self._fit_idx = fit_idx
                            break

                self.current_fit_widget = sub_window.fit_widget

                window_title = chisurf.__name__ + "(" + chisurf.__version__ + "): " + self.current_fit.name
                self.setWindowTitle(window_title)

                self.current_fit.model.show()
                self.current_fit_widget.show()
                sub_window.current_plot_controller.show()
            # Handle plugin windows with plot controllers (like sm_acquisition)
            elif hasattr(sub_window, 'current_plot_controller') and sub_window.current_plot_controller is not None:
                # Add and show the plugin's plot controller
                self.plotOptionsLayout.addWidget(sub_window.current_plot_controller)
                sub_window.current_plot_controller.show()
                # Update window title for plugin windows
                window_title = chisurf.__name__ + "(" + chisurf.__version__ + "): " + sub_window.windowTitle()
                self.setWindowTitle(window_title)

    def onRunMacro(
            self,
            filename: pathlib.Path = None,
            executor: str = 'console',
            globals=None, locals=None
    ):
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                "Python macros",
                file_type="Python file (*.py)"
            )
        chisurf.logging.info(f"Running script: {filename}")
        if executor == 'console':
            filename_str = filename.as_posix()
            chisurf.console.run_macro(filename=filename.as_posix())
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
                        # plugin_path looks like [pkg, subpkg, ..., filename]
                        # Use all path components except the filename to build the
                        # full nested package name, falling back to the first
                        # component for legacy single-level plugins.
                        if len(plugin_path) > 1:
                            module_parts = plugin_path[:-1]
                        elif len(plugin_path) == 1:
                            module_parts = plugin_path
                        else:
                            module_parts = []

                        if module_parts:
                            package_name = '.'.join(module_parts)

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
                                # For built-in plugins, set the full nested package name
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
        # If multiple datasets are selected, schedule per-dataset fit
        # creation on the Qt event loop. This mirrors clicking "Add Fit"
        # repeatedly while keeping each add_fit call isolated, which has
        # proven stable.
        if not data_idx:
            return

        indices = list(data_idx)
        model_name = self.current_model_name

        def _create_next_fit():
            if not indices:
                return
            idx = indices.pop(0)
            try:
                chisurf.macros.core_fit.add_fit(
                    dataset_indices=[idx],
                    model_name=model_name,
                )
            except Exception as e:
                # Surface errors instead of silently swallowing them so that
                # model/widget construction problems (e.g. for new Gaussian
                # PDA models) can be diagnosed.
                msg = f"Add fit failed for dataset index {idx} with model '{model_name}': {e}"
                try:
                    chisurf.logging.error(msg)
                    chisurf.logging.error(traceback.format_exc())
                except Exception:
                    pass
                try:
                    # Show a short status-bar message to the user
                    self.status.showMessage(msg, 10000)
                except Exception:
                    pass
            if indices:
                QtCore.QTimer.singleShot(0, _create_next_fit)

        _create_next_fit()

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

    def onLoadFit(self, **kwargs):
        filename = chisurf.gui.widgets.get_filename(
            file_type="*.fit.json",
            description="Load fit (fit.json)",
            **kwargs
        )
        if not filename:
            return
        chisurf.run(f"chisurf.macros.core_fit.load_fit_project(r\"{filename}\")")

    def _recent_projects_file(self) -> pathlib.Path:
        try:
            return chisurf.settings.get_path('settings') / 'recent_projects.json'
        except Exception:
            return pathlib.Path.home() / '.chisurf' / 'recent_projects.json'

    def _load_recent_projects(self) -> list[str]:
        fp = self._recent_projects_file()
        try:
            if not fp.is_file():
                return []
        except Exception:
            return []
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            return []
        if isinstance(data, dict):
            items = data.get('projects', [])
        else:
            items = data
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for it in items:
            try:
                s = str(it)
            except Exception:
                continue
            if s:
                out.append(s)
        return out

    def _store_recent_projects(self, projects: list[str]) -> None:
        fp = self._recent_projects_file()
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        payload = {'projects': list(projects or [])}
        try:
            fp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        except Exception:
            try:
                chisurf.logging.exception(f"Failed to write recent projects file: {fp}")
            except Exception:
                pass

    def _set_recent_projects(self, projects: list[str]) -> None:
        try:
            self._recent_projects = list(projects or [])
        except Exception:
            self._recent_projects = []

    def add_recent_project(self, project_path) -> None:
        try:
            p = os.path.abspath(os.path.normpath(str(project_path)))
        except Exception:
            return
        if not p:
            return

        try:
            current = list(getattr(self, '_recent_projects', []) or [])
        except Exception:
            current = []

        def _key(s: str) -> str:
            try:
                s2 = os.path.normpath(str(s))
            except Exception:
                s2 = str(s)
            return s2.lower() if os.name == 'nt' else s2

        seen = set()
        new_list: list[str] = []
        for item in [p] + current:
            if not item:
                continue
            k = _key(item)
            if k in seen:
                continue
            seen.add(k)
            new_list.append(item)

        max_n = 10
        new_list = new_list[:max_n]
        self._set_recent_projects(new_list)
        self._store_recent_projects(new_list)
        self._refresh_recent_projects_menu()

    def _clear_recent_projects(self) -> None:
        self._set_recent_projects([])
        self._store_recent_projects([])
        self._refresh_recent_projects_menu()

    def _open_recent_project(self, project_dir: str) -> None:
        try:
            path = pathlib.Path(project_dir)
        except Exception:
            return

        try:
            project_file = path / 'project.json'
            if not project_file.exists():
                QtWidgets.QMessageBox.warning(
                    self,
                    'Invalid Project',
                    'The selected folder does not contain a valid project file (project.json).'
                )
                try:
                    current = list(getattr(self, '_recent_projects', []) or [])
                    current = [p for p in current if os.path.normpath(p) != os.path.normpath(project_dir)]
                    self._set_recent_projects(current)
                    self._store_recent_projects(current)
                    self._refresh_recent_projects_menu()
                except Exception:
                    pass
                return
        except Exception:
            return

        try:
            chisurf.working_path = path
        except Exception:
            pass

        try:
            chisurf.macros.core_fit.load_project(project_path=path.as_posix())
        except Exception:
            try:
                chisurf.logging.exception(f"Failed to load recent project: {path}")
            except Exception:
                pass
            return

        try:
            self._current_project_dir = path
        except Exception:
            pass

        self.add_recent_project(path)

    def _refresh_recent_projects_menu(self) -> None:
        menu = getattr(self, '_menu_recent_projects', None)
        if menu is None:
            return
        try:
            menu.clear()
        except Exception:
            return

        try:
            projects = list(getattr(self, '_recent_projects', []) or [])
        except Exception:
            projects = []

        if not projects:
            try:
                a = QtWidgets.QAction('No recent projects', self)
                a.setEnabled(False)
                menu.addAction(a)
            except Exception:
                pass
        else:
            for i, p in enumerate(projects):
                try:
                    label = f"&{i + 1} {p}"
                    a = QtWidgets.QAction(label, self)
                    a.triggered.connect(lambda _checked=False, pp=p: self._open_recent_project(pp))
                    menu.addAction(a)
                except Exception:
                    continue

        try:
            menu.addSeparator()
        except Exception:
            pass
        try:
            clear_action = QtWidgets.QAction('Clear Recent Projects', self)
            clear_action.triggered.connect(self._clear_recent_projects)
            menu.addAction(clear_action)
        except Exception:
            pass

    def _init_recent_projects_menu(self) -> None:
        if getattr(self, '_menu_recent_projects', None) is not None:
            self._refresh_recent_projects_menu()
            return

        try:
            recent_menu = QtWidgets.QMenu('Recent Projects', self)
            self._menu_recent_projects = recent_menu
        except Exception:
            return

        try:
            self.menuProject.insertMenu(self.actionClose_Project, recent_menu)
        except Exception:
            try:
                self.menuProject.addMenu(recent_menu)
            except Exception:
                pass

        try:
            self._set_recent_projects(self._load_recent_projects())
        except Exception:
            self._set_recent_projects([])

        self._refresh_recent_projects_menu()

    def onSaveProject(self, event: QtCore.QEvent = None):
        project_helpers.save_project(self, event=event)

    def onSaveProjectAs(self, event: QtCore.QEvent = None):
        project_helpers.save_project_as(self, event=event)

    def onLoadProject(self, event: QtCore.QEvent = None):
        """
        Load a project from a project folder.

        This method prompts the user for a project folder, then calls
        the load_project function to load the project.
        """
        # Inform user about experimental status
        chisurf.gui.widgets.general.MyMessageBox(
            label="Project Load",
            info="Loading a saved project. This feature is experimental.",
            show_fortune=False
        )

        # Get project directory
        path, _ = chisurf.gui.widgets.get_directory(
            caption="Select Project Folder"
        )
        if not path:
            return

        # Check if this is a valid project folder (JSON-based project format)
        project_file = path / "project.json"
        if not project_file.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Project",
                f"The selected folder does not contain a valid project file (project.json)."
            )
            return

        # Load project
        chisurf.working_path = path
        try:
            chisurf.macros.core_fit.load_project(project_path=path.as_posix())
        except Exception:
            try:
                chisurf.logging.exception(f"Project load failed: {path}")
            except Exception:
                pass
            return

        try:
            self._current_project_dir = path
        except Exception:
            pass

        try:
            self.add_recent_project(path)
        except Exception:
            pass

    def reinitialize(self):
        log = getattr(chisurf, 'logging', None)

        def _log_exception(step: str) -> None:
            try:
                exc_fn = getattr(log, 'exception', None)
                if callable(exc_fn):
                    exc_fn(f"reinitialize: {step} failed")
            except Exception:
                pass

        def _run(step: str, fn) -> None:
            try:
                fn()
            except Exception:
                _log_exception(step)

        def _close_subwindows() -> None:
            for sw in list(self.mdiarea.subWindowList()):
                try:
                    sw.close()
                except Exception:
                    try:
                        title = sw.windowTitle()
                    except Exception:
                        title = None
                    step = f"close subwindow {title}" if title else "close subwindow"
                    _log_exception(step)

        _run('onCloseAllFits', self.onCloseAllFits)
        _run('close subwindows', _close_subwindows)
        _run('clear imported_datasets', chisurf.imported_datasets.clear)
        _run('dataset_selector.update', self.dataset_selector.update)
        _run('fit_selector.update', self.fit_selector.update)

        def _reset_state():
            self._current_dataset = None
            self._current_fit = None
            self._fit_idx = 0

        _run('reset current state', _reset_state)
        _run('comboBox_Model.clear', self.comboBox_Model.clear)

    def onCloseProject(self, event: QtCore.QEvent = None):
        try:
            self.reinitialize()
        except Exception:
            pass

        try:
            self._current_project_dir = None
        except Exception:
            pass

    def set_current_setup_idx(self, v: int):
        try:
            count = self.comboBox_setupSelect.count()
        except Exception:
            count = 0
        if count > 0:
            if v < 0:
                v = 0
            elif v >= count:
                v = count - 1
        else:
            v = 0
        try:
            self.comboBox_setupSelect.setCurrentIndex(v)
        except Exception:
            pass
        self._current_setup_idx = v

    def onSetupChanged(self):
        chisurf.gui.widgets.hide_items_in_layout(
            self.layout_experiment_reader
        )
        readers = self.current_experiment.readers
        if not readers:
            self._current_setup_idx = 0
            return
        try:
            widget = self.current_setup
            self.layout_experiment_reader.addWidget(widget)
            widget.show()
            # Update UI elements if the widget has an updateUI method
            if hasattr(widget, 'updateUI') and callable(widget.updateUI):
                widget.updateUI()
        except TypeError:
            widget = self.current_setup.controller
            self.layout_experiment_reader.addWidget(widget)
            widget.show()
            # Update UI elements if the controller has an updateUI method
            if hasattr(widget, 'updateUI') and callable(widget.updateUI):
                widget.updateUI()
        self._current_setup_idx = self.comboBox_setupSelect.currentIndex()

        # If the reader/controller implements a hook for context help, pass
        # a callable it can use to open the help window at the right place.
        try:
            if hasattr(widget, 'set_help_callback') and callable(widget.set_help_callback):
                widget.set_help_callback(self.open_context_help_for_reader)
        except Exception:
            pass

    def onCloseAllFits(self):
        fit_helpers.close_all_fits(self)

    def onAddDataset(self):
        misc_helpers.add_dataset(self)

    def onSaveFits(self, event: QtCore.QEvent = None):
        fit_helpers.save_fits(self, event=event)

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        fit_helpers.save_fit(self, event=event, **kwargs)

    def onOpenHelp(self):
        """Open the help plugin."""
        misc_helpers.open_help(self)

    def open_context_help_for_reader(self, topic: str | None = None) -> None:
        """Open the help plugin, optionally with a filter for a given topic.

        Parameters
        ----------
        topic : str or None
            Optional free-text topic, e.g. an experiment or reader name.
            When provided, the help browser's filter box is pre-filled so the
            relevant documentation entries are highlighted.
        """

        misc_helpers.open_help(self, topic=topic)

    def onOpenUpdate(self):
        """Open the updater plugin."""
        misc_helpers.open_updater(self)

    def onOpenAbout(self):
        """Open the about plugin."""
        misc_helpers.open_about(self)

    def onClearLocalSettings(self):
        """Reset local settings and show a confirmation popup."""
        misc_helpers.clear_local_settings(self)

    def onClearUserStyles(self):
        """Clear user style files (QSS) and show a confirmation popup."""
        misc_helpers.clear_user_styles(self)

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

    def onDockWidgetPlotVisibilityChanged(self, visible):
        """Update the Plot Controller when the dockWidgetPlot becomes visible.

        Args:
            visible (bool): Whether the dock widget is visible
        """
        if visible and self.mdiarea.currentSubWindow() is not None:
            # Get the current subwindow
            sub_window = self.mdiarea.currentSubWindow()
            # Update the current plot controller
            if hasattr(sub_window, 'current_plot_controller') and hasattr(sub_window.current_plot_controller, 'update'):
                sub_window.current_plot_controller.update()

    def load_toolbar_plugins(self):
        """Load plugins into the toolbar based on toolbar_plugins setting."""
        import pathlib
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

        # Build an index of available plugins using chisurf.plugins.iter_plugins
        try:
            plugin_infos = list(chisurf.plugins.iter_plugins())
        except Exception:
            plugin_infos = []

        def _find_plugin_info(target_name: str):
            clean_target = target_name.split(':')[-1].strip() if ':' in target_name else target_name
            for info in plugin_infos:
                pname = info.get('plugin_name') or info.get('module_name') or ''
                if not pname:
                    continue
                if pname == target_name:
                    return info
                clean = pname.split(':')[-1].strip() if ':' in pname else pname
                if clean == clean_target:
                    return info
            return None

        # Load each toolbar plugin
        for plugin_name in toolbar_plugins:
            try:
                info = _find_plugin_info(plugin_name)
                if info is None:
                    chisurf.logging.warning(f"Could not find module for plugin: {plugin_name}")
                    continue

                module_path = info.get('module_path')
                module_name = info.get('module_name') or ''
                package_dir = pathlib.Path(info.get('package_dir'))

                # Get the clean plugin name (without sorting prefix)
                clean_name = plugin_name.split(':')[-1]

                # Create an action for the plugin with empty text (icon only)
                action = QtWidgets.QAction("", self)

                # Set icon if available
                icon_path = package_dir / 'icon.png'
                if icon_path.exists():
                    action.setIcon(QtGui.QIcon(str(icon_path)))
                else:
                    action.setText(clean_name)

                # Get plugin description from metadata or docstring
                description = info.get('description')
                if not description:
                    description = read_module_docstring(package_dir) or "No description available."

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

    def _run_plugin_from_dir(self, plugin_dir_to_use):
        """Helper to run a plugin given its directory."""
        try:
            import pathlib
            from functools import partial

            plugin_dir_to_use = pathlib.Path(plugin_dir_to_use)
            wizard_path = plugin_dir_to_use / "wizard.py"
            init_path = plugin_dir_to_use / "__init__.py"

            # Check if wizard.py exists
            if wizard_path.exists():
                adr = "https://github.com/fluorescence-tools/chisurf"  # Default value
                p = partial(
                    self.onRunMacro, wizard_path,
                    executor='exec',
                    globals={'__name__': 'plugin', 'adr': adr}
                )
                p()
            elif init_path.exists():
                # If no wizard.py, run the plugin's __init__.py using onRunMacro
                self.onRunMacro(
                    init_path,
                    executor='exec',
                    globals={'__name__': 'plugin'}
                )
            else:
                chisurf.logging.warning(f"No wizard.py or __init__.py found for plugin directory: {plugin_dir_to_use}")
        except Exception as e:
            chisurf.logging.error(f"Error running plugin from {plugin_dir_to_use}: {e}")

    def load_and_show_plugin(self, module_path):
        """Load and show a plugin from its module path."""
        try:
            import pathlib

            plugin_dir_to_use = None

            # First try to resolve the plugin via chisurf.plugins.iter_plugins
            try:
                for info in chisurf.plugins.iter_plugins():
                    if info.get('module_path') == module_path:
                        plugin_dir_to_use = pathlib.Path(info.get('package_dir'))
                        break
            except Exception:
                plugin_dir_to_use = None

            # Fallback to legacy behavior using flat module names
            if plugin_dir_to_use is None:
                module_name = module_path.split('.')[-1]
                plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent
                plugin_dir = plugin_root / module_name
                user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
                user_plugin_dir = user_plugin_root / module_name

                if plugin_dir.exists():
                    plugin_dir_to_use = plugin_dir
                elif user_plugin_dir.exists():
                    plugin_dir_to_use = user_plugin_dir
                else:
                    chisurf.logging.warning(f"Plugin directory not found in either built-in or user locations: {module_path}")
                    return

            self._run_plugin_from_dir(plugin_dir_to_use)

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
        chisurf.run = chisurf.console.execute_on_gui_thread
        try:
            chisurf.log = chisurf.console.log_on_gui_thread
        except Exception:
            pass
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
                    # Couple the controller to its reader so that
                    # ExperimentReaderController.experiment_reader is set and
                    # controller UIs can manipulate reader attributes
                    # (e.g. FCS noise/weighting mode).
                    controller = controller_class(
                        experiment_reader=reader,
                        **controller_params
                    )

                readers_list.append((reader, controller))

            experiment.add_readers(readers_list)

        # Set up models if defined
        if 'models' in config:
            model_classes = [self._resolve_class(model_class) for model_class in config['models']]
            experiment.add_model_classes(models=model_classes)

        try:
            chisurf.models.load_user_models()
            user_model_classes = list(chisurf.models.iter_user_models_for_experiment(exp_type, experiment.name))
        except Exception:
            user_model_classes = []
        if user_model_classes:
            experiment.add_model_classes(models=user_model_classes)

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
        import copy
        import yaml
        import pathlib
        import shutil
        import chisurf.experiments

        def _load_yaml_config(path: pathlib.Path) -> dict:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}

        def _deep_merge_dicts(base: dict, override: dict) -> dict:
            result = copy.deepcopy(base) if base else {}
            for key, value in (override or {}).items():
                if (
                    isinstance(value, dict)
                    and isinstance(result.get(key), dict)
                ):
                    result[key] = _deep_merge_dicts(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            return result

        def _summarize_experiment_config_diff(default_cfg: dict, user_cfg: dict, max_lines: int = 10) -> str:
            default_cfg = default_cfg or {}
            user_cfg = user_cfg or {}
            lines: list[str] = []

            try:
                default_keys = set(default_cfg.keys())
                user_keys = set(user_cfg.keys())
            except Exception:
                default_keys = set()
                user_keys = set()

            added_sections = sorted(user_keys - default_keys)
            removed_sections = sorted(default_keys - user_keys)
            common_sections = default_keys & user_keys

            if added_sections:
                lines.append("  - Added sections: " + ", ".join(added_sections))
            if removed_sections:
                lines.append("  - Removed sections: " + ", ".join(removed_sections))

            try:
                default_types = default_cfg.get("experiment_types") or {}
                user_types = user_cfg.get("experiment_types") or {}
                if isinstance(default_types, dict) and isinstance(user_types, dict):
                    def_type_keys = set(default_types.keys())
                    user_type_keys = set(user_types.keys())
                    added_types = sorted(user_type_keys - def_type_keys)
                    removed_types = sorted(def_type_keys - user_type_keys)
                    changed_types = []
                    for key in sorted(def_type_keys & user_type_keys):
                        d_val = default_types.get(key) or {}
                        u_val = user_types.get(key) or {}
                        if not isinstance(d_val, dict) or not isinstance(u_val, dict):
                            if d_val != u_val:
                                changed_types.append(key)
                            continue
                        name_changed = d_val.get("name", key) != u_val.get("name", key)
                        hidden_changed = bool(d_val.get("hidden", False)) != bool(u_val.get("hidden", False))
                        if name_changed or hidden_changed:
                            changed_types.append(key)
                    if added_types:
                        lines.append("  - Added experiment types: " + ", ".join(added_types))
                    if removed_types:
                        lines.append("  - Removed experiment types: " + ", ".join(removed_types))
                    if changed_types:
                        lines.append("  - Modified experiment types: " + ", ".join(changed_types))
            except Exception:
                pass

            changed_sections = []
            for key in sorted(common_sections):
                if key in ("experiment_types", "global"):
                    continue
                try:
                    if default_cfg.get(key) != user_cfg.get(key):
                        changed_sections.append(key)
                except Exception:
                    continue
            if changed_sections:
                lines.append("  - Modified experiment sections: " + ", ".join(changed_sections))

            if not lines:
                return ""

            if len(lines) > max_lines:
                extra = len(lines) - max_lines
                lines = lines[:max_lines]
                lines.append(f"  ... and {extra} more change(s).")

            return "\n".join(lines)

        # Define paths for experiment configuration file
        source_config_file = pathlib.Path(chisurf.settings.get_path('chisurf')) / "settings" / "experiment_configs.yaml"
        user_config_file = pathlib.Path(chisurf.settings.get_path('settings')) / "experiment_configs.yaml"

        check_updates = True
        try:
            check_updates = bool(chisurf.settings.cs_settings.get('check_experiment_config_updates_on_startup', True))
        except Exception:
            check_updates = True

        if check_updates and source_config_file.exists() and user_config_file.exists():
            try:
                files_differ = source_config_file.read_bytes() != user_config_file.read_bytes()
            except Exception:
                files_differ = False

            if files_differ:
                app_running = False
                try:
                    app_running = QtWidgets.QApplication.instance() is not None
                except Exception:
                    app_running = False

                if app_running:
                    diff_summary = ""
                    try:
                        default_for_diff = _load_yaml_config(source_config_file)
                        user_for_diff = _load_yaml_config(user_config_file)
                        if isinstance(default_for_diff, dict) or isinstance(user_for_diff, dict):
                            diff_summary = _summarize_experiment_config_diff(
                                default_for_diff or {},
                                user_for_diff or {}
                            )
                    except Exception:
                        diff_summary = ""

                    msg = QtWidgets.QMessageBox(self)
                    msg.setWindowTitle("Experiment configuration update available")
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("The experiment configuration file in your settings folder differs from the latest shipped version.")
                    base_info = (
                        "Do you want to update your experiment configuration to the new default?\n\n"
                        "This will overwrite your current user experiment configuration file."
                    )
                    if diff_summary:
                        msg.setInformativeText(
                            base_info + "\n\nChanges detected compared to your current configuration:\n" +
                            diff_summary
                        )
                    else:
                        msg.setInformativeText(base_info)
                    yes_button = msg.addButton("Update", QtWidgets.QMessageBox.YesRole)
                    msg.addButton("Skip", QtWidgets.QMessageBox.NoRole)
                    try:
                        checkbox = QtWidgets.QCheckBox("Don't check experiment configuration updates on startup")
                        msg.setCheckBox(checkbox)
                    except Exception:
                        checkbox = None

                    msg.exec_()

                    try:
                        if checkbox is not None and checkbox.isChecked():
                            from chisurf.settings.settings_utils import set_check_experiment_config_updates_on_startup as _set_exp_flag
                            _set_exp_flag(False)
                            try:
                                chisurf.settings.cs_settings['check_experiment_config_updates_on_startup'] = False
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        if msg.clickedButton() is yes_button:
                            shutil.copyfile(source_config_file, user_config_file)
                    except Exception:
                        pass

        # Ensure the user config file exists
        if not user_config_file.exists():
            # If user config doesn't exist but source does, copy it
            if source_config_file.exists():
                shutil.copyfile(source_config_file, user_config_file)
            else:
                # If neither exists, we'll use default configurations later
                chisurf.logging.warning(f"Experiment configuration file not found: {source_config_file}")
                experiment_configs = {}

        # Load packaged defaults and user overrides (if any), then merge them so
        # newly shipped experiments automatically appear unless the user
        # explicitly overrides them.
        default_configs = _load_yaml_config(source_config_file)
        user_configs = _load_yaml_config(user_config_file) if user_config_file.exists() else {}
        if default_configs and user_configs:
            experiment_configs = _deep_merge_dicts(default_configs, user_configs)
        elif default_configs:
            experiment_configs = default_configs
        else:
            experiment_configs = user_configs

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
        global_fit = chisurf.experiments.core.Experiment(
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
        try:
            self.analysisHeaderWidget = QtWidgets.QWidget(self.dockWidgetAnalysis)
            self.analysisHeaderLayout = QtWidgets.QVBoxLayout(self.analysisHeaderWidget)
            self.analysisHeaderLayout.setContentsMargins(0, 0, 0, 0)
            self.analysisHeaderLayout.setSpacing(0)
            try:
                self.verticalLayout_3.insertWidget(0, self.analysisHeaderWidget)
            except Exception:
                try:
                    layout = self.dockWidgetAnalysis.layout()
                except Exception:
                    layout = None
                if layout is not None:
                    layout.insertWidget(0, self.analysisHeaderWidget)
        except Exception:
            self.analysisHeaderWidget = None
            self.analysisHeaderLayout = None
        try:
            self.toolButton_reader_help.clicked.connect(self._on_reader_help_clicked)
        except Exception:
            pass

        # Apply small platform-specific tweaks to the outer window chrome.
        # On Windows we request a dark titlebar via the DWM API so the
        # window frame matches the dark theme without giving up native
        # resizing/snap behavior. This is a no-op on other platforms.
        try:
            self._apply_platform_window_tweaks()
        except Exception:
            pass

        # Enable drop on the 'Drop files here' label and connect event filter
        try:
            self.label_filedrop.setAcceptDrops(True)
            self.label_filedrop.installEventFilter(self)
            # Helpful tooltip
            if hasattr(self.label_filedrop, 'setToolTip'):
                self.label_filedrop.setToolTip("Drop files here to open with the current setup")
        except Exception:
            pass

        # Replace the standard QListWidget with our custom LogListWidget
        # First, save any existing items
        existing_items = []
        if hasattr(self, 'plainTextEditLog'):
            for i in range(self.plainTextEditLog.count()):
                existing_items.append(self.plainTextEditLog.item(i).text())
        
        # Get the parent widget of plainTextEditLog
        parent_widget = self.plainTextEditLog.parent()
        # Get the layout containing plainTextEditLog
        layout = parent_widget.layout()
        # Find the index of plainTextEditLog in the layout
        for i in range(layout.count()):
            if layout.itemAt(i).widget() == self.plainTextEditLog:
                layout_index = i
                break
        
        # Remove the old widget
        self.plainTextEditLog.setParent(None)
        
        # Create and add the new widget
        self.plainTextEditLog = LogListWidget(parent_widget)
        self.plainTextEditLog.setObjectName("plainTextEditLog")
        layout.insertWidget(layout_index, self.plainTextEditLog)
        
        # Restore any existing items
        for item_text in existing_items:
            self.plainTextEditLog.addItem(item_text)

        self.current_fit_widget = None
        self._current_fit = None
        self._current_model_class = None
        self._current_experiment_idx = 0
        self._fit_idx = 0
        self._current_setup_idx = 0
        self._system_info_watermark = None
        self._current_project_dir = None

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

        # Setup status bar with progress bar and message
        self.status = TruncatingStatusBar(self)
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

        # Warm up heavy imports after the UI is ready to make first-time actions more responsive
        try:
            # Run shortly after the event loop starts so the window can appear first
            QtCore.QTimer.singleShot(100, self.warmup_imports)
        except Exception:
            # If QTimer is not available for some reason, just ignore
            pass

        try:
            self._init_system_info_watermark()
            QtCore.QTimer.singleShot(0, self._update_system_info_watermark_geometry)
        except Exception:
            pass

    def update(self):
        super().update()
        self.fit_selector.update()
        self.dataset_selector.update()

    def _init_system_info_watermark(self) -> None:
        try:
            parent = getattr(self, "mdiarea", None)
        except Exception:
            parent = None
        label = getattr(self, "_system_info_watermark", None)
        try:
            label = _system_info_watermark.ensure_watermark(parent, label)
        except Exception:
            return
        self._system_info_watermark = label

    def _update_system_info_watermark_geometry(self) -> None:
        label = getattr(self, "_system_info_watermark", None)
        try:
            _system_info_watermark.update_geometry(label)
        except Exception:
            pass

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        try:
            self._update_system_info_watermark_geometry()
        except Exception:
            pass

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        try:
            self._update_system_info_watermark_geometry()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # Handle drag-and-drop onto the 'Drop files here' label
        try:
            label = getattr(self, 'label_filedrop', None)
        except Exception:
            label = None
        if obj is not None and label is not None and obj is label:
            t = event.type()
            if t == QtCore.QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif t == QtCore.QEvent.DragMove:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif t == QtCore.QEvent.Drop:
                if event.mimeData().hasUrls():
                    try:
                        paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
                        paths = [p for p in paths if p]
                        if paths:
                            paths.sort()
                            # Same behavior as dropping on the dataset selector
                            command = "\n".join([f"chisurf.macros.add_dataset(filename=r'{p}')" for p in paths])
                            try:
                                chisurf.run(command)
                            except Exception:
                                # Fallback: call directly without chisurf.run
                                for p in paths:
                                    try:
                                        chisurf.macros.add_dataset(filename=rf"{p}")
                                    except Exception:
                                        pass
                            event.acceptProposedAction()
                            try:
                                self.status.showMessage(f"Added {len(paths)} file(s)", 3000)
                            except Exception:
                                pass
                    finally:
                        pass
                    return True
        return super().eventFilter(obj, event)

    def warmup_imports(self):
        """Preload heavy modules to improve first-use responsiveness.
        This shifts import cost to just after startup.
        """
        try:
            import importlib
            # Core visualization libs typically used when adding a fit
            import pyqtgraph as _pg  # noqa: F401
            from matplotlib import colors as _mcolors  # noqa: F401
            # Numeric/scientific routines used during fitting
            import scipy.linalg as _sl  # noqa: F401
            import scipy.stats as _sstats  # noqa: F401
            # Ensure fitting widgets are fully imported
            import chisurf.gui.widgets.fitting as _fitwidgets  # noqa: F401
            # Touch a commonly used class to trigger any uic loads
            _ = getattr(_fitwidgets, 'FittingControllerWidget', None)
            # Optionally warm up model registry that may be consulted
            _ = importlib.import_module('chisurf.models.global_model.globalfit')
        except Exception as e:
            try:
                chisurf.logging.debug(f"warmup_imports encountered: {e}")
            except Exception:
                pass

    def _apply_platform_window_tweaks(self) -> None:
        """Apply small, non-invasive tweaks to the main window frame.

        Currently this enables a dark titlebar on supported Windows
        versions using the DWM "immersive dark mode" attribute, so the
        outer chrome looks less like a stock bright Windows app while
        retaining native move/resize/snap behavior.
        """
        apply_platform_window_tweaks(self)

    def _on_reader_help_clicked(self) -> None:
        try:
            exp_name = self.comboBox_experimentSelect.currentText().strip()
        except Exception:
            exp_name = ""
        try:
            setup_name = self.comboBox_setupSelect.currentText().strip()
        except Exception:
            setup_name = ""

        topic = None

        # Resolve context help topic from configurable reader rules in settings.
        try:
            help_cfg = getattr(chisurf.settings, "help", {}) or {}
            rules = help_cfg.get("reader_rules", []) or []

            current_setup = getattr(chisurf.cs, "current_setup", None)
            reader_name = None
            is_jordi = False
            try:
                if current_setup is not None:
                    reader_name = getattr(current_setup, "experiment_reader", None)
                    is_jordi = bool(getattr(current_setup, "is_jordi", False))
            except Exception:
                reader_name = None
                is_jordi = False

            reader_name_lc = reader_name.strip() if isinstance(reader_name, str) else None

            for rule in rules:
                try:
                    rule_exp = str(rule.get("experiment", "")).strip()
                except Exception:
                    rule_exp = ""
                if not rule_exp or rule_exp != exp_name:
                    continue

                match = True

                rule_reader = rule.get("reader")
                if rule_reader is not None:
                    try:
                        rule_reader_str = str(rule_reader).strip()
                    except Exception:
                        rule_reader_str = ""
                    if not reader_name_lc or reader_name_lc.lower() != rule_reader_str.lower():
                        match = False

                if not match:
                    continue

                rule_setup = rule.get("setup")
                if rule_setup is not None:
                    try:
                        rule_setup_str = str(rule_setup).strip()
                    except Exception:
                        rule_setup_str = ""
                    if setup_name != rule_setup_str:
                        match = False

                if not match:
                    continue

                try:
                    doc = str(rule.get("doc", "")).strip()
                except Exception:
                    doc = ""
                if not doc:
                    continue

                anchor = None
                if is_jordi and "jordi_anchor" in rule:
                    try:
                        anchor = str(rule.get("jordi_anchor", "")).strip() or None
                    except Exception:
                        anchor = None
                if not anchor:
                    try:
                        anchor = str(rule.get("anchor", "")).strip() or None
                    except Exception:
                        anchor = None

                topic = f"{doc}#{anchor}" if anchor else doc
                break
        except Exception:
            topic = None

        # Fallback for all other experiments/readers: keep the original
        # behavior of using experiment + setup names as a text filter.
        if topic is None:
            parts = [p for p in (exp_name, setup_name) if p]
            raw_topic = " ".join(parts) if parts else ""
            topic = raw_topic.replace("/", " ") or None

        try:
            self.open_context_help_for_reader(topic)
        except Exception:
            # Fallback: plain help window
            try:
                self.onOpenHelp()
            except Exception:
                pass

    def update_setup_ui(self):
        """Update the UI to reflect changes in current_setup properties."""
        # Call onSetupChanged to update the UI
        self.onSetupChanged()

    def arrange_widgets(self):
        # self.setCentralWidget(self.mdiarea)

        ##########################################################
        #      Help and About widgets                            #
        ##########################################################

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
        self.editor = chisurf.plugins.misc.code_editor.CodeEditor()

        self.verticalLayout_10.addWidget(self.editor)

        # Add data selector widget
        self.verticalLayout_8.addWidget(self.dataset_selector)

        # Add fit selector widget
        self.verticalLayout_5.addWidget(self.fit_selector)

        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidgetReadData.raise_()

        apply_dock_tab_colors(self)

    def filter_log_content(self):
        """
        Filter log content based on filter text and hide checkbox state.
        If checkBox_filter_hide is checked, hide non-matching lines.
        If unchecked, highlight matching lines and gray out non-matching lines.
        """
        filter_text = self.lineEdit_LogFilter.text().strip().lower()
        hide_non_matching = self.checkBox_filter_hide.isChecked()
        
        # Initialize _original_log_items if it doesn't exist
        if not hasattr(self, '_original_log_items'):
            self._original_log_items = []
            # Store all current items
            for i in range(self.plainTextEditLog.count()):
                self._original_log_items.append(self.plainTextEditLog.item(i).text())
        
        # If there's no filter text, show all content normally
        if not filter_text:
            # Restore the original content with normal formatting
            self.plainTextEditLog.clear()
            for item_text in self._original_log_items:
                item = QtWidgets.QListWidgetItem(item_text)
                self.plainTextEditLog.addItem(item)
            return
        
        # Clear the current content
        self.plainTextEditLog.clear()
        
        # Add items back to the log with appropriate formatting
        if self._original_log_items:
            for item_text in self._original_log_items:
                # Check if this item contains the filter text
                if filter_text in item_text.lower():
                    # Always add matching items
                    item = QtWidgets.QListWidgetItem(item_text)
                    # Highlight matching items
                    item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))  # Black text
                    item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 0, 50)))  # Light yellow background
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    self.plainTextEditLog.addItem(item)
                elif not hide_non_matching:
                    # Only add non-matching items if hide_non_matching is False
                    item = QtWidgets.QListWidgetItem(item_text)
                    # Gray out non-matching items
                    item.setForeground(QtGui.QBrush(QtGui.QColor(150, 150, 150)))  # Gray text
                    self.plainTextEditLog.addItem(item)
            
            # Show a message if no items match the filter
            if self.plainTextEditLog.count() == 0:
                item = QtWidgets.QListWidgetItem("No matching log entries found.")
                self.plainTextEditLog.addItem(item)
        else:
            item = QtWidgets.QListWidgetItem("No log entries found.")
            self.plainTextEditLog.addItem(item)
            
    def update_log_filter(self):
        """
        Update the log filter when new log entries are added.
        This method should be called after new log entries are added to plainTextEditLog.
        """
        # Get the latest item added to the list
        if self.plainTextEditLog.count() > 0:
            latest_item = self.plainTextEditLog.item(self.plainTextEditLog.count() - 1).text()
            
            # Add the new item to our original items list
            if hasattr(self, '_original_log_items'):
                self._original_log_items.append(latest_item)
        
        # Apply highlighting/graying out if there's a filter text
        # This will be called regardless of filter text to ensure proper formatting
        self.filter_log_content()

    def define_actions(self):
        ##########################################################
        # GUI ACTIONS
        ##########################################################
        # Connect log filter and hide checkbox
        self.lineEdit_LogFilter.textChanged.connect(self.filter_log_content)
        self.checkBox_filter_hide.stateChanged.connect(self.filter_log_content)
        
        self.actionTile_windows.triggered.connect(self.onTileWindows)
        self.actionTab_windows.triggered.connect(self.onTabWindows)
        self.actionCascade.triggered.connect(self.onCascadeWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)
        self.dockWidgetPlot.visibilityChanged.connect(self.onDockWidgetPlotVisibilityChanged)

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
        self.actionLoad_Fit.triggered.connect(self.onLoadFit)

        # Use actions from .ui file for saving and loading projects
        # Now enabled by default and backed by the JSON-based project macros.
        self.actionSave_Project.triggered.connect(self.onSaveProject)
        self.actionSave_Project.setEnabled(True)

        try:
            action = QtWidgets.QAction("Save Project As...", self)
            action.setShortcut("Ctrl+Shift+S")
            action.triggered.connect(self.onSaveProjectAs)
            self.actionSave_Project_As = action
            try:
                self.menuProject.insertAction(self.actionClose_Project, action)
            except Exception:
                self.menuProject.addAction(action)
        except Exception:
            pass

        self.actionOpen_Project.triggered.connect(self.onLoadProject)
        self.actionOpen_Project.setEnabled(True)
        self.actionClose_Project.triggered.connect(self.onCloseProject)
        self.actionClose_Project.setEnabled(True)

        try:
            self._init_recent_projects_menu()
        except Exception:
            pass

    def onOpenFretRdaAxisSettings(self):
        """Open a dialog for global FRET R_DA axis settings."""
        try:
            from chisurf.models.pda.widgets import FretRdaAxisSettingsWidget
        except Exception as e:
            try:
                chisurf.logging.error(f"Could not load FretRdaAxisSettingsWidget: {e}")
            except Exception:
                pass
            try:
                QtWidgets.QMessageBox.warning(
                    self,
                    "FRET RDA axis settings",
                    (
                        "The RDA axis settings widget could not be loaded.\n"
                        "Please check that chisurf.models.pda.widgets is available."
                    ),
                )
            except Exception:
                pass
            return

        dlg = getattr(self, "_fret_rda_axis_dialog", None)
        if dlg is None or not isinstance(dlg, QtWidgets.QDialog):
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("FRET RDA axis settings")
            layout = QtWidgets.QVBoxLayout(dlg)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)
            widget = FretRdaAxisSettingsWidget(dlg)
            layout.addWidget(widget)
            self._fret_rda_axis_dialog = dlg

        dlg.show()
        try:
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            pass

    def load_tools(self):
        import chisurf


        ##########################################################
        #      Load toolbar plugins                              #
        ##########################################################
        self.load_toolbar_plugins()

        ##########################################################
        #      Settings                                          #
        ##########################################################
        # Configuration editor
        self.configuration = chisurf.gui.widgets.settings_editor.SettingsEditor(
            filename=chisurf.settings.chisurf_settings_file,
            window_title="ChiSurf Settings"
        )
        self.actionSettings.triggered.connect(self.configuration.show)
        # Global FRET R_DA axis settings dialog
        self.actionFretRdaAxisSettings = QtWidgets.QAction("FRET RDA axis ...", self)
        self.actionFretRdaAxisSettings.triggered.connect(self.onOpenFretRdaAxisSettings)
        try:
            # Place just before the "Clear local settings" entry
            self.menuSettings.insertAction(self.actionClear_local_settings, self.actionFretRdaAxisSettings)
        except Exception:
            # Fallback: append to the Settings menu
            self.menuSettings.addAction(self.actionFretRdaAxisSettings)
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
