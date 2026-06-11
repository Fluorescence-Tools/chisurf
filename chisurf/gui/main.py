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
from chisurf import logging
from chisurf.gui import QtWidgets, QtGui, QtCore, uic
from chisurf.gui.gui_tweaks import apply_dock_tab_colors
from chisurf.gui import misc_helpers, project_helpers, fit_helpers


import chisurf as cs
import chisurf.core.decorators
import chisurf.core.base
import chisurf.core.fio
import chisurf.core.experiments
import chisurf.macros
import chisurf.core.settings
from chisurf.core.actions import record_action

import chisurf.gui.widgets.settings_editor
import chisurf.gui.widgets
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.history_browser
import chisurf.gui.widgets.experiments.modelling

# Heavy imports moved to deferred/local usage or warmup_imports
# import cs.core.models
# import cs.plugins
# import cs.core.fitting
import chisurf.gui.resources
import chisurf.plugins.core.code_editor


class _MdiDropEventFilter(QtCore.QObject):
    def __init__(self, mdiarea):
        super().__init__(mdiarea)
        self.mdiarea = mdiarea

    def eventFilter(self, obj, event):
        event_types = (3, 175)
        if hasattr(QtCore.QEvent, 'NonClientAreaMouseButtonRelease'):
            event_types = event_types + (QtCore.QEvent.NonClientAreaMouseButtonRelease,)
            
        if event.type() in event_types:
            if isinstance(obj, QtWidgets.QDockWidget) and obj.isFloating():
                pos = QtGui.QCursor.pos()
                try:
                    mdi_rect = self.mdiarea.rect()
                    top_left = self.mdiarea.mapToGlobal(mdi_rect.topLeft())
                    bottom_right = self.mdiarea.mapToGlobal(mdi_rect.bottomRight())
                    global_rect = QtCore.QRect(top_left, bottom_right)
                except Exception:
                    global_rect = None
                
                if global_rect is not None and global_rect.contains(pos):
                    widget = obj.widget()
                    if widget is not None:
                        title = obj.windowTitle()
                        size = obj.size()
                        
                        # Tag widget with original dock name for re-docking
                        widget.setProperty("_original_dock_name", obj.objectName())
                        
                        # Unparent carefully to prevent deletion on dock close
                        widget.setParent(None)
                        obj.close()
                        
                        try:
                            cs.logging.info(f"Converting dock '{title}' to MDI subwindow")
                        except Exception:
                            pass
                        
                        sub = self.mdiarea.addSubWindow(widget)
                        sub.setWindowTitle(title)
                        sub.resize(size)
                        
                        # Explicitly show both the inner widget and the subwindow wrapper
                        widget.show()
                        sub.show()
                        return True
        
        # Handle re-docking when the dock is re-enabled/shown
        if event.type() == 17: # QEvent.Show
            if isinstance(obj, QtWidgets.QDockWidget) and obj.widget() is None:
                dock_name = obj.objectName()
                if dock_name:
                    for sub in self.mdiarea.subWindowList():
                        w = sub.widget()
                        if w and w.property("_original_dock_name") == dock_name:
                            try:
                                cs.logging.info(f"Restoring dock '{dock_name}' from MDI")
                            except Exception:
                                pass
                            # Move back to dock
                            sub.setWidget(None)
                            sub.close()
                            obj.setWidget(w)
                            w.show()
                            return True

        return super().eventFilter(obj, event)


from chisurf.gui.main_helper import (
    ProjectMixin,
    SetupMixin,
    HistoryMixin,
    StateMixin,
    DevMixin,
)

class Main(
    QtWidgets.QMainWindow,
    ProjectMixin,
    SetupMixin,
    HistoryMixin,
    StateMixin,
    DevMixin,
):
    """

    Attributes
    ----------
    current_dataset : cs.core.base.Data
        The dataset that is currently selected in the ChiSurf GUI. This
        dataset corresponds to the analysis window selected by the user in
        the UI.
    current_model_class : cs.model.Model
        The model used in the analysis (fit) of the currently selected analysis
        windows.
    fit_idx : int
        The index of the currently selected fit in the fit list cs.fits
        The current fit index corresponds to the currently selected fit window
        in the list of all fits of the fit.
    current_experiment_idx : int
        The index of the experiment type currently selected in the UI out of
        the list all supported experiments. This corresponds to the index of
        the UI combo box used to select the experiment.
    current_experiment : cs.core.experiments.core.Experiment
        The experiment currently selected in the GUI.
    current_setup_idx : int
        The index of the setup currently selected in the GUI.
    current_setup_name : str
        The name of the setup currently selected in the GUI.
    current_setup : cs.core.experiments.core.reader.ExperimentReader
        The current experiment setup / experiment reader selecetd in the GUI
    experiment_names : list
        A list containing the names of the experiments.

    """

    _current_dataset: cs.core.base.Data = None
    experiment_names: typing.List[str] = list()

    @property
    def current_dataset(self) -> cs.core.base.Data:
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
    def current_experiment(self) -> cs.core.experiments.core.Experiment | None:
        name = self.comboBox_experimentSelect.currentText()
        if not name:
            return None
        return cs.experiment.get(name)

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
            self._refresh_experiment_ui()

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
    def current_setup(self) -> cs.core.experiments.core.reader.ExperimentReader:
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
            cs.gui.widgets.general.MyMessageBox(
                label="Setup Not Found",
                info=f"Setup '{name}' does not exist in the current experiment.",
                show_fortune=False
            )
            return
        if j != i:
            self.current_setup_idx = j
            self._refresh_setup_ui()
            
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
            cs.core.experiments.core.reader.ExperimentReader
        ):
            return self.current_setup
        elif isinstance(
                self.current_setup,
                cs.core.experiments.core.reader.ExperimentReaderController
        ):
            return self.current_setup.experiment_reader

    @property
    def current_model_name(self) -> str:
        return self.current_model_class.name

    @property
    def current_fit(self) -> cs.core.fitting.fit.FitGroup:
        return self._current_fit

    @current_fit.setter
    def current_fit(self, v: cs.core.fitting.fit.FitGroup) -> None:
        self._current_fit = v

    def set_current_experiment_idx(self, v):
        self.comboBox_experimentSelect.setCurrentIndex(v)

    _READ_DATA_DOCK_WIDTH = 390

    def _save_window_state(self):
        """Persist dock layout and window geometry via QSettings."""
        settings = QtCore.QSettings("ChiSurf", "MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("state", self.saveState())

    def _restore_window_state(self):
        """Restore dock layout and window geometry from QSettings."""
        settings = QtCore.QSettings("ChiSurf", "MainWindow")
        geo = settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = settings.value("state")
        if state is not None:
            self.restoreState(state)

    def _apply_read_data_dock_width(self) -> None:
        """Apply the startup width for the left read-data dock."""
        dock = getattr(self, "dockWidgetReadData", None)
        if dock is None:
            return
        try:
            self.resizeDocks([dock], [self._READ_DATA_DOCK_WIDTH], QtCore.Qt.Horizontal)
        except Exception:
            dock.resize(self._READ_DATA_DOCK_WIDTH, dock.height())

    def closeEvent(self, event: QtGui.QCloseEvent):
        # Always save window state regardless of confirmation
        try:
            self._save_window_state()
        except Exception:
            pass
        if cs.core.settings.gui['confirm_close_program']:
            reply = cs.gui.widgets.general.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to quit?",
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        # Save setup defaults before closing
        try:
            self._save_setup_defaults()
        except Exception:
            pass
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
            cs.gui.widgets.hide_items_in_layout(self.modelLayout)
            header_layout = getattr(self, "analysisHeaderLayout", None)
            if header_layout is not None:
                cs.gui.widgets.hide_items_in_layout(header_layout)
            cs.gui.widgets.hide_items_in_layout(self.plotOptionsLayout)

            # Handle fit windows first
            if hasattr(sub_window, 'fit') and sub_window.fit is not None:
                for fit_idx, f in enumerate(cs.fits):
                    if f == sub_window.fit:
                        if self.current_fit is not cs.fits[fit_idx]:
                            cs.run(f"cs.current_fit = cs.fits[{fit_idx}]")
                            self._fit_idx = fit_idx
                            break

                self.current_fit_widget = sub_window.fit_widget

                window_title = cs.__name__ + "(" + cs.__version__ + "): " + self.current_fit.name
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
                window_title = cs.__name__ + "(" + cs.__version__ + "): " + sub_window.windowTitle()
                self.setWindowTitle(window_title)

    def onRunMacro(
            self,
            filename: pathlib.Path = None,
            executor: str = 'console',
            globals=None, locals=None
    ):
        misc_helpers.run_macro(filename=filename, executor=executor, globals=globals, locals=locals, main_window=self)

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
        if cs.imported_datasets:
            # Get all model names from the experiment
            all_model_names = ds.experiment.get_model_names()

            # Get the list of disabled models from settings
            disabled_models = cs.core.settings.cs_settings.get('plugins', {}).get('disabled_models', [])

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
        fit_helpers.add_fits_for_datasets(
            window=self,
            data_idx=data_idx,
            model_name=self.current_model_name,
        )


    def onLoadFit(self, **kwargs):
        filename = cs.gui.widgets.get_filename(
            file_type="*.fit.json",
            description="Load fit (fit.json)",
            **kwargs
        )
        if not filename:
            return
        cs.core.actions.dispatch(
            name="fit.load",
            payload={"filename": str(filename)},
        )


    def onCloseAllFits(self):
        cs.core.actions.dispatch(
            name="fit.close_all",
            payload={},
        )

        # Clear the analysis dock layouts
        cs.gui.widgets.clear_layout(self.modelLayout)
        header_layout = getattr(self, "analysisHeaderLayout", None)
        if header_layout is not None:
            cs.gui.widgets.clear_layout(header_layout)
        cs.gui.widgets.clear_layout(self.plotOptionsLayout)

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
        cs.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": s, "experiment_reader": None},
        )

    def onSaveFits(self, event: QtCore.QEvent = None):
        path, _ = cs.gui.widgets.get_directory()
        if not path:
            return
        cs.working_path = path
        cs.core.actions.dispatch(
            name="fit.save_all",
            payload={"target_path": path.as_posix()},
        )

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        # Prefer default directory from the current fit's data filename, if available
        try:
            default_dir = None
            fit = getattr(self, 'current_fit', None)
            data_obj = getattr(fit, 'data', None) if fit is not None else None
            filename = getattr(data_obj, 'filename', None) if data_obj is not None else None
            if isinstance(filename, str):
                fn = filename.strip()
                if fn and fn.lower() != 'none':
                    p = pathlib.Path(fn)
                    # Use parent folder only for absolute paths
                    if p.is_absolute():
                        default_dir = p.parent
            # Only set the directory if the caller did not specify one
            if ('directory' not in kwargs or kwargs.get('directory') is None) and default_dir is not None:
                kwargs['directory'] = default_dir
        except Exception as e:
            cs.logging.warning(f"onSaveFit: could not infer data folder from fit.data.filename: {e}")

        path, _ = cs.gui.widgets.get_directory(**kwargs)
        if not path:
            return
        # Keep behavior: user chooses where to save; update working path accordingly
        cs.working_path = path
        cs.core.actions.dispatch(
            name="fit.save",
            payload={"target_path": path.as_posix()},
        )

    def onOpenHelp(self):
        """Open the help plugin."""
        try:
            self.open_context_help_for_reader(None)
        except Exception as e:
            cs.gui.widgets.general.MyMessageBox(
                label="Help Plugin Error",
                info=f"Error loading help plugin: {str(e)}",
                show_fortune=False
            )

    def open_context_help_for_reader(self, topic: str | None = None) -> None:
        """Open the help plugin, optionally with a filter for a given topic.

        Parameters
        ----------
        topic : str or None
            Optional free-text topic, e.g. an experiment or reader name.
            When provided, the help browser's filter box is pre-filled so the
            relevant documentation entries are highlighted.
        """

        import importlib
        import pathlib
        try:
            try:
                help_plugin = importlib.import_module("chisurf.plugins.core.help")
            except Exception:
                help_plugin = importlib.import_module("chisurf.plugins.core.help")
            window = getattr(self, "_help_window", None)
            if window is None or not isinstance(window, help_plugin.HelpWidget):
                window = help_plugin.HelpWidget()
                try:
                    window.destroyed.connect(lambda _=None: setattr(self, "_help_window", None))
                except Exception:
                    pass
                self._help_window = window

            try:
                if topic:
                    txt = str(topic).strip()
                    if txt:
                        handled = False
                        # Allow topics like "some/doc.md#section-id" to open
                        # a specific Markdown file and subsection directly in
                        # the help browser. If parsing fails, fall back to the
                        # existing filter behavior.
                        try:
                            path_part = txt
                            anchor = None
                            if "#" in txt:
                                path_part, frag = txt.split("#", 1)
                                path_part = path_part.strip()
                                anchor = frag.strip() or None

                            if path_part.lower().endswith(".md"):
                                raw_path = pathlib.Path(path_part)

                                # Resolve relative paths against the project
                                # root (same root used by the help plugin for
                                # core docs discovery).
                                if not raw_path.is_absolute():
                                    try:
                                        base = pathlib.Path(cs.__file__).resolve().parent
                                        root = base.parent
                                        candidate = (root / raw_path).resolve()
                                    except Exception:
                                        candidate = raw_path
                                else:
                                    candidate = raw_path

                                if candidate.exists():
                                    try:
                                        window.open_markdown_path(candidate, anchor)
                                        handled = True
                                    except Exception:
                                        handled = False

                        except Exception:
                            handled = False

                        if not handled:
                            window.filter_line_edit.setText(txt)
            except Exception:
                pass

            window.show()
            try:
                window.raise_()
                window.activateWindow()
            except Exception:
                pass
        except Exception as e:
            cs.gui.widgets.general.MyMessageBox(
                label="Help Plugin Error",
                info=f"Error loading help plugin: {str(e)}",
                show_fortune=False
            )

    def onOpenUpdate(self):
        """Open the updater plugin."""
        # Import the updater plugin
        import importlib
        try:
            try:
                updater_plugin = importlib.import_module("chisurf.plugins.core.updater")
            except ImportError:
                updater_plugin = importlib.import_module("chisurf.plugins.core.updater")

            window = updater_plugin.UpdaterWidget()
            window.show()
        except Exception as e:
            # Show error message if plugin can't be loaded
            cs.gui.widgets.general.MyMessageBox(
                label="Updater Plugin Error",
                info=f"Error loading updater plugin: {str(e)}",
                show_fortune=False
            )

    def onOpenAbout(self):
        """Open the about plugin."""
        import importlib
        try:
            try:
                about_plugin = importlib.import_module("chisurf.plugins.core.about")
            except ImportError:
                about_plugin = importlib.import_module("chisurf.plugins.core.about")

            window = about_plugin.AboutDialog(parent=self)
            window.show()
        except Exception as e:
            cs.gui.widgets.general.MyMessageBox(
                label="About Plugin Error",
                info=f"Error opening About dialog: {str(e)}",
                show_fortune=False
            )

    def onClearLocalSettings(self):
        """Reset local settings and show a confirmation popup."""
        # Clear the settings folder
        cs.core.settings.clear_settings_folder()

        # Show a confirmation popup
        cs.gui.widgets.general.MyMessageBox(
            label="Settings Reset",
            info="Local settings have been reset successfully.",
            show_fortune=False
        )

    def onClearUserStyles(self):
        """Clear user style files (QSS) and show a confirmation popup."""
        # Get the path to the user styles folder
        user_styles_path = cs.core.settings.get_path('settings') / 'styles'

        # Check if the folder exists
        if user_styles_path.exists() and user_styles_path.is_dir():
            # Delete all QSS files in the folder
            for file in user_styles_path.glob('*.qss'):
                try:
                    file.unlink()
                except Exception as e:
                    cs.logging.warning(f"Could not delete style file {file}: {e}")

            # Show a confirmation popup
            cs.gui.widgets.general.MyMessageBox(
                label="Styles Reset",
                info="User style files have been cleared successfully. Restart the application to apply default styles.",
                show_fortune=False
            )
        else:
            # Show a message if the folder doesn't exist
            cs.gui.widgets.general.MyMessageBox(
                label="Styles Reset",
                info="No user style files found.",
                show_fortune=False
            )

    def onClearUserPlugins(self):
        """Clear user plugin folder and show a confirmation popup."""
        # Clear the user plugins folder
        cs.core.settings.clear_user_plugins_folder()

        # Show a confirmation popup
        cs.gui.widgets.general.MyMessageBox(
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
        toolbar_plugins = cs.core.settings.cs_settings.get('plugins', {}).get('toolbar_plugins', [])

        if not toolbar_plugins:
            return

        # Create a toolbar for plugins if it doesn't exist
        if not hasattr(self, 'plugins_toolbar'):
            self.plugins_toolbar = self.addToolBar("Plugins")
            self.plugins_toolbar.setObjectName("pluginsToolBar")
            # Set icon size to match standard toolbar (16x16)
            self.plugins_toolbar.setIconSize(QtCore.QSize(16, 16))

        # Determine built-in plugin directory
        plugin_root = pathlib.Path(cs.plugins.__file__).absolute().parent

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

        # Build an index of available plugins using cs.plugins.iter_plugins
        try:
            plugin_infos = list(cs.plugins.iter_plugins())
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
                    cs.logging.warning(f"Could not find module for plugin: {plugin_name}")
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
                cs.logging.info(f"Added plugin to toolbar: {plugin_name} (module: {module_name})")

            except Exception as e:
                cs.logging.error(f"Error loading toolbar plugin {plugin_name}: {e}")

    def load_and_show_plugin(self, module_path):
        """Load and show a plugin from its module path."""
        try:
            import pathlib

            plugin_dir_to_use = None

            # First try to resolve the plugin via cs.plugins.iter_plugins
            try:
                for info in cs.plugins.iter_plugins():
                    if info.get('module_path') == module_path:
                        plugin_dir_to_use = pathlib.Path(info.get('package_dir'))
                        break
            except Exception:
                plugin_dir_to_use = None

            # Fallback to legacy behavior using flat module names
            if plugin_dir_to_use is None:
                module_name = module_path.split('.')[-1]
                plugin_root = pathlib.Path(cs.plugins.__file__).absolute().parent
                plugin_dir = plugin_root / module_name
                user_plugin_root = pathlib.Path.home() / '.cs' / 'plugins'
                user_plugin_dir = user_plugin_root / module_name

                if plugin_dir.exists():
                    plugin_dir_to_use = plugin_dir
                elif user_plugin_dir.exists():
                    plugin_dir_to_use = user_plugin_dir
                else:
                    cs.logging.warning(f"Plugin directory not found in either built-in or user locations: {module_path}")
                    return

            misc_helpers.run_plugin_from_dir(self, plugin_dir_to_use)

        except Exception as e:
            cs.logging.error(f"Error loading plugin {module_path}: {e}")

    def init_console(self):
        self.verticalLayout_4.addWidget(cs.console)
        cs.console.pushVariables({'cs': self})
        cs.console.pushVariables({'cs': cs})
        try:
            cs.console.pushVariables({'history': cs.history})
        except Exception:
            pass
        cs.console.pushVariables({'np': np})
        cs.console.pushVariables({'os': os})
        cs.console.pushVariables({'QtCore': QtCore})
        cs.console.pushVariables({'QtGui': QtGui})
        cs.console.set_default_style('linux')

        def _run_with_history(code: str = None):
            if code is None:
                return None
            try:
                code_str = str(code)
            except Exception:
                return None
            try:
                lines = code_str.splitlines()
                first_line = lines[0] if lines else code_str
                record_action(
                    action_type="run_command",
                    summary=f"run command: {first_line[:120]}",
                    payload={"code": code_str},
                )
            except Exception:
                pass
            return cs.console.execute_on_gui_thread(code_str)

        cs.run = _run_with_history
        try:
            cs.log = cs.console.log_on_gui_thread
        except Exception:
            pass
        cs.run(str(cs.core.settings.gui['console_init']))


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(pathlib.Path(__file__).parent / "gui.ui", self)
        
        try:
            self._mdi_drop_filter = _MdiDropEventFilter(self.mdiarea)
            QtWidgets.QApplication.instance().installEventFilter(self._mdi_drop_filter)
        except Exception:
            pass


        # Set window icon to ChiSurf logo
        try:
            self.setWindowIcon(QtGui.QIcon(":/icons/icons/cs_logo.png"))
        except Exception:
            # Fallback to file-based icon if resource not available
            try:
                icon_path = pathlib.Path(__file__).parent.parent.parent / "icon.png"
                if icon_path.exists():
                    self.setWindowIcon(QtGui.QIcon(str(icon_path)))
            except Exception:
                pass
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
            misc_helpers.apply_window_tweaks(self)
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

        misc_helpers.setup_log_list_widget(self)
        self._init_history_browser()

        self.current_fit_widget = None
        self._current_fit = None
        self._current_model_class = None
        self._current_experiment_idx = 0
        self._fit_idx = 0
        self._current_setup_idx = 0
        self._system_info_watermark = None
        self._current_project_dir = None

        self.experiment_names = list()
        self.dataset_selector = cs.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=False,
            curve_types='all',
            change_event=self.onCurrentDatasetChanged,
            drag_enabled=True,
            experiment=None
        )

        # widget listing the existing fits
        self.fit_selector = cs.gui.widgets.fitting.ModelDataRepresentationSelector(parent=self)

        # Setup status bar with progress bar and message
        self.status = misc_helpers.TruncatingStatusBar(self)
        self.setStatusBar(self.status)

        # Create a QWidget to hold the progress bar and message
        status_widget = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_widget)

        # Set spacing and margins to zero
        status_layout.setSpacing(0)  # Set spacing between widgets to zero
        status_layout.setContentsMargins(0, 0, 0, 0)  # Set margins to zero

        # Create a progress bar — hidden initially, shown only during background loading
        self.progress_bar = QtWidgets.QProgressBar(self.status)
        self.progress_bar.setFixedWidth(150)  # Set a fixed width for the progress bar
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_bar.setFixedHeight(15)  # Adjust the height as needed
        self.progress_bar.setVisible(False)   # Hidden until background loading starts

        # Create a label for the status message
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setMaximumHeight(20)  # Set a maximum height for the status message

        # Add the progress bar and status message to the status layout
        status_layout.addWidget(self.status_label)
        status_layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        status_layout.addWidget(self.progress_bar)

        # Add the status widget to the status bar, aligning to the left
        self.status.addWidget(status_widget, 1)  # 1 gives the widget some stretch


        try:
            self._init_system_info_watermark()
            QtCore.QTimer.singleShot(0, self._update_system_info_watermark_geometry)
        except Exception:
            pass

    def update(self):
        super().update()
        self.fit_selector.update()
        self.dataset_selector.update()


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
                            for p in paths:
                                cs.core.actions.dispatch(
                                    name="dataset.add",
                                    payload={"filename": str(p), "experiment_reader": None},
                                )
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
        """Preload heavy modules to improve first-use responsiveness."""
        misc_helpers.warmup_imports()

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
            help_cfg = getattr(cs.core.settings, "help", {}) or {}
            rules = help_cfg.get("reader_rules", []) or []

            current_setup = getattr(cs.cs, "current_setup", None)
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
        self.dockWidgetScriptEdit.setVisible(cs.core.settings.gui['show_macro_edit'])
        self.dockWidget_console.setVisible(cs.core.settings.gui['show_console'])
        # Set the height of the console dock widget
        if 'console_height' in cs.core.settings.gui:
            from qtpy.QtCore import Qt
            self.resizeDocks([self.dockWidget_console], [cs.core.settings.gui['console_height']], Qt.Vertical)
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
        self.editor = cs.plugins.core.code_editor.CodeEditor()
        self.editor._on_editor_created = self._on_code_editor_created

        # --- Code dock navigation toolbar (same as FitSubWindow) ---
        self._code_toolbar = QtWidgets.QHBoxLayout()
        self._code_toolbar.setContentsMargins(5, 5, 5, 5)

        self._code_agent_btn = QtWidgets.QToolButton()
        self._code_agent_btn.setText("\U0001f916")
        self._code_agent_btn.setToolTip("Toggle AI agent panel")

        self._code_nav_back_btn = QtWidgets.QToolButton()
        self._code_nav_back_btn.setText("\u2190")
        self._code_nav_back_btn.setToolTip("Navigate back to previous cursor position")

        self._code_nav_forward_btn = QtWidgets.QToolButton()
        self._code_nav_forward_btn.setText("\u2192")
        self._code_nav_forward_btn.setToolTip("Navigate forward to next cursor position")

        self._code_file_combo = QtWidgets.QComboBox()
        self._code_func_combo = QtWidgets.QComboBox()

        self._code_save_btn = QtWidgets.QToolButton()
        self._code_save_btn.setText("Save/Apply")
        self._code_save_btn.setToolTip("Save the current editor content")

        self._code_settings_btn = self.editor.create_settings_button(self)

        self._code_toolbar.addWidget(self._code_agent_btn)
        self._code_toolbar.addWidget(self._code_nav_back_btn)
        self._code_toolbar.addWidget(self._code_nav_forward_btn)
        self._code_toolbar.addWidget(QtWidgets.QLabel("File:"))
        self._code_toolbar.addWidget(self._code_file_combo, 1)
        self._code_toolbar.addWidget(QtWidgets.QLabel("  Jump to:"))
        self._code_toolbar.addWidget(self._code_func_combo, 1)
        self._code_toolbar.addStretch()
        self._code_toolbar.addWidget(self._code_settings_btn)
        self._code_toolbar.addWidget(self._code_save_btn)

        self.verticalLayout_10.addLayout(self._code_toolbar)

        self.verticalLayout_10.addWidget(self.editor)

        # Wire toolbar signals
        self._code_nav_back_btn.clicked.connect(self._code_nav_back)
        self._code_nav_forward_btn.clicked.connect(self._code_nav_forward)
        self._code_file_combo.currentIndexChanged.connect(self._on_code_file_selected)
        self._code_func_combo.currentIndexChanged.connect(self._on_code_func_selected)
        self._code_save_btn.clicked.connect(self._code_save)
        self._code_agent_btn.clicked.connect(self.editor._toggle_agent_panel)
        self.editor.settings_changed.connect(self._on_code_editor_settings_changed)

        # Add data selector widget
        self.verticalLayout_8.addWidget(self.dataset_selector)

        # Add fit selector widget
        self.verticalLayout_5.addWidget(self.fit_selector)

        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidgetReadData.raise_()

        apply_dock_tab_colors(self)

        self._install_dev_mode_code_badges()

        # Restore persisted dock layout and window geometry, if available.
        # This must run after the default tabify setup so the saved layout
        # overrides the defaults when a previous session exists.
        try:
            self._restore_window_state()
        except Exception:
            pass

        QtCore.QTimer.singleShot(0, self._apply_read_data_dock_width)

    # ---- Code dock navigation callbacks ------------------------------------

    def _on_code_editor_created(self, editor):
        """Called when a new editor tab is created inside the code editor."""
        editor.file_load_callback = self._code_load_file

    def _code_nav_back(self):
        editor = self.editor._get_current_editor()
        if editor is not None:
            editor.navigate_back()

    def _code_nav_forward(self):
        editor = self.editor._get_current_editor()
        if editor is not None:
            editor.navigate_forward()

    def _code_load_file(self, file_path, line_number: int = 0):
        """Load a source file into the code editor and populate the function combo."""
        import re
        self.editor.open_file(file_path)
        editor = self.editor._get_current_editor()
        if editor is None:
            return

        if line_number > 0:
            doc = editor.document()
            block = doc.findBlockByNumber(line_number)
            if block.isValid():
                cursor = editor.textCursor()
                cursor.setPosition(block.position())
                editor.setTextCursor(cursor)
                editor.centerCursor()

        code = editor.toPlainText()
        self._code_func_combo.blockSignals(True)
        self._code_func_combo.clear()
        self._code_func_combo.addItem("Select...", -1)

        for i, line in enumerate(code.split('\n')):
            m = re.match(r'^ *(def |class )([a-zA-Z0-9_]+)', line)
            if m:
                indent = len(line) - len(line.lstrip())
                prefix = " " * indent
                self._code_func_combo.addItem(f"{prefix}{m.group(1)}{m.group(2)}", i)

        self._code_func_combo.blockSignals(False)
        if not editor._nav_history:
            editor.push_nav_history(file_path, 0)

    def _on_code_file_selected(self, idx):
        if idx < 0:
            return
        file_path = self._code_file_combo.itemData(idx)
        if file_path:
            self._code_load_file(file_path)

    def _on_code_func_selected(self, idx):
        if idx < 0:
            return
        line_num = self._code_func_combo.itemData(idx)
        if line_num is not None and line_num >= 0:
            editor = self.editor._get_current_editor()
            if editor is None:
                return
            doc = editor.document()
            block = doc.findBlockByNumber(line_num)
            cursor = editor.textCursor()
            cursor.setPosition(block.position())
            editor.setTextCursor(cursor)
            editor.centerCursor()
            editor.setFocus()

    def _code_save(self):
        """Save the current editor content to its file."""
        editor = self.editor._get_current_editor()
        if editor is None:
            return
        self.editor.save_text()

    def _on_code_editor_settings_changed(self, settings: dict) -> None:
        """Apply editor font settings to dependent code widgets."""
        font = QtGui.QFont()
        font.setFamily(str(settings.get("font_family", cs.core.settings.gui["editor"]["font_family"])))
        try:
            font.setPointSize(int(settings.get("font_size", cs.core.settings.gui["editor"]["font_size"])))
        except (TypeError, ValueError):
            font.setPointSize(int(cs.core.settings.gui["editor"]["font_size"]))

        try:
            if hasattr(cs, "console") and hasattr(cs.console, "set_editor_font"):
                cs.console.set_editor_font(font)
        except Exception as e:
            logging.log(1, f"Error updating console font: {e}")

    def filter_log_content(self):
        """
        Filter log content based on filter text and hide checkbox state.
        Matching rows are highlighted. If hide is enabled, non-matching rows
        are hidden as well.
        """
        misc_helpers.filter_log_content(self)
            
    def update_log_filter(self):
        """
        Update the log filter when new log entries are added.
        This method should be called after new log entries are added to plainTextEditLog.
        """
        misc_helpers.update_log_filter(self)

    def define_actions(self):
        ##########################################################
        # GUI ACTIONS
        ##########################################################
        # Connect log filter and hide checkbox
        log_widget = getattr(self, "plainTextEditLog", None)
        if log_widget is None or not hasattr(log_widget, "filter_log_content"):
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
        self.actionRecord.triggered.connect(cs.console.start_recording)
        self.actionStop.triggered.connect(cs.console.save_macro)
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
        try:
            self.actionSaveCurrentFit.setShortcut(QtGui.QKeySequence("Ctrl+S"))
            self.actionSaveCurrentFit.setShortcutContext(QtCore.Qt.ApplicationShortcut)
            self.addAction(self.actionSaveCurrentFit)
        except Exception:
            pass
        self.actionClose_Fit.triggered.connect(cs.macros.core_fit.close_fit)
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
        self.actionReinitialize.triggered.connect(self.reinitialize)
        self.actionReinitialize.setEnabled(True)

        try:
            self.actionHistoryUndo = QtWidgets.QAction("Undo", self)
            self.actionHistoryUndo.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
            self.actionHistoryUndo.setShortcutContext(QtCore.Qt.ApplicationShortcut)
            self.actionHistoryUndo.triggered.connect(self._history_undo)
            self.addAction(self.actionHistoryUndo)
            try:
                self.menuView.addAction(self.actionHistoryUndo)
            except Exception:
                pass

            self.actionHistoryRedo = QtWidgets.QAction("Redo", self)
            self.actionHistoryRedo.setShortcut(QtGui.QKeySequence("Ctrl+Y"))
            self.actionHistoryRedo.setShortcutContext(QtCore.Qt.ApplicationShortcut)
            self.actionHistoryRedo.triggered.connect(self._history_redo)
            self.addAction(self.actionHistoryRedo)
            try:
                self.menuView.addAction(self.actionHistoryRedo)
            except Exception:
                pass

            self._sync_history_navigation_actions()
        except Exception:
            pass

        try:
            self._init_recent_projects_menu()
        except Exception:
            pass

        self._init_developer_menu()

    def onOpenFretRdaAxisSettings(self):
        """Open a dialog for global FRET R_DA axis settings."""
        try:
            from chisurf.gui.widgets.models.pda.widgets import FretRdaAxisSettingsWidget
        except Exception as e:
            try:
                cs.logging.error(f"Could not load FretRdaAxisSettingsWidget: {e}")
            except Exception:
                pass
            try:
                QtWidgets.QMessageBox.warning(
                    self,
                    "FRET RDA axis settings",
                    (
                        "The RDA axis settings widget could not be loaded.\n"
                        "Please check that cs.gui.widgets.models.pda is available."
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
        ##########################################################
        #      Load toolbar plugins                              #
        ##########################################################
        self.load_toolbar_plugins()

        ##########################################################
        #      Settings                                          #
        ##########################################################
        # Configuration editor
        self.configuration = cs.gui.widgets.settings_editor.SettingsEditor(
            filename=cs.core.settings.chisurf_settings_file,
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
        self.actionClear_logging_files.triggered.connect(cs.core.settings.clear_logging_files)
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
        
        # Initialize ribbon interface (optional - can be enabled via settings)
        self._ribbon_integration = None
        
        # Restore ribbon interface state from settings
        try:
            gui_settings = cs.core.settings.cs_settings.get('gui', {})
            use_ribbon = gui_settings.get('use_ribbon_interface', True)
            
            if use_ribbon:
                # Enable ribbon if it was saved in settings
                self.toggle_ribbon_interface(True)
                cs.logging.info("Ribbon interface restored from settings")
        except Exception as e:
            cs.logging.warning(f"Failed to restore ribbon interface state: {e}")
        
        self.onExperimentChanged()

    def toggle_ribbon_interface(self, enabled=None):
        """
        Toggle or set the ribbon interface.
        
        Parameters
        ----------
        enabled : bool, optional
            If True, enable ribbon; if False, disable ribbon; 
            if None, toggle current state.
        """
        try:
            from chisurf.gui.widgets.ribbon import setup_chisurf_ribbon
            
            if enabled is None:
                # Toggle current state
                enabled = self._ribbon_integration is None
            
            if enabled and self._ribbon_integration is None:
                # Enable ribbon with style from settings
                gui_settings = cs.core.settings.cs_settings.get('gui', {})
                ribbon_style = gui_settings.get('ribbon_style', None)
                
                # Hide plugin toolbar when switching to ribbon
                if hasattr(self, 'plugins_toolbar'):
                    self.plugins_toolbar.hide()
                    cs.logging.info("Plugin toolbar hidden for ribbon mode")
                
                self._ribbon_integration = setup_chisurf_ribbon(self, ribbon_style=ribbon_style)
                if self._ribbon_integration:
                    cs.logging.info("Ribbon interface enabled")
                    # Save to settings persistently
                    from chisurf.core.settings.settings_utils import set_use_ribbon_interface
                    set_use_ribbon_interface(True)
                else:
                    cs.logging.warning("Failed to setup ribbon interface")
            elif not enabled and self._ribbon_integration is not None:
                # Disable ribbon
                self._ribbon_integration.restore_original_interface()
                self._ribbon_integration = None
                cs.logging.info("Ribbon interface disabled")
                
                # Show plugin toolbar when switching back to menu mode
                if hasattr(self, 'plugins_toolbar'):
                    self.plugins_toolbar.show()
                    cs.logging.info("Plugin toolbar restored for menu mode")
                
                # Save to settings persistently
                from chisurf.core.settings.settings_utils import set_use_ribbon_interface
                set_use_ribbon_interface(False)
            
        except Exception as e:
            cs.logging.error(f"Failed to toggle ribbon interface: {e}")
