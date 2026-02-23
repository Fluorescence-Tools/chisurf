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


import chisurf
import chisurf.decorators
import chisurf.base
import chisurf.fio
import chisurf.experiments
import chisurf.macros
import chisurf.settings
import chisurf.history_replay
from chisurf.runtime.actions import record_action

import chisurf.gui.widgets.settings_editor
import chisurf.gui.widgets
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.history_browser
import chisurf.gui.widgets.experiments.modelling

import chisurf.models
import chisurf.plugins
import chisurf.fitting
import chisurf.gui.resources
import chisurf.plugins.misc.code_editor


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

    def _restore_setup_defaults(self) -> None:
        """Restore saved setup defaults from user settings."""
        try:
            from chisurf.gui.widgets.experiments.setup_persistence import (
                load_setup_defaults,
                apply_setup_defaults,
            )
            defaults = load_setup_defaults()
            if defaults.get("experiments"):
                apply_setup_defaults(self, defaults)
                chisurf.logging.info("Restored setup defaults from user settings")
        except Exception as e:
            chisurf.logging.warning(f"Failed to restore setup defaults: {e}")

    def _save_setup_defaults(self) -> None:
        """Save current setup defaults to user settings."""
        try:
            from chisurf.gui.widgets.experiments.setup_persistence import (
                collect_setup_defaults,
                save_setup_defaults,
            )
            defaults = collect_setup_defaults(self)
            if save_setup_defaults(defaults):
                chisurf.logging.info("Saved setup defaults to user settings")
        except Exception as e:
            chisurf.logging.warning(f"Failed to save setup defaults: {e}")

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
        fit_helpers.add_fits_for_datasets(
            window=self,
            data_idx=data_idx,
            model_name=self.current_model_name,
        )

    def onExperimentChanged(self):
        experiment_name = self.comboBox_experimentSelect.currentText()
        chisurf.action_controller.execute(
            name="experiment.set",
            payload={"name": str(experiment_name)},
        )

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

    def _load_recent_projects(self) -> list[str]:
        return project_helpers.load_recent_projects()

    def _store_recent_projects(self, projects: list[str]) -> None:
        project_helpers.store_recent_projects(projects)

    def _set_recent_projects(self, projects: list[str]) -> None:
        project_helpers.set_recent_projects(self, projects)

    def add_recent_project(self, project_path) -> None:
        project_helpers.add_recent_project(self, project_path)

    def _clear_recent_projects(self) -> None:
        project_helpers.clear_recent_projects(self)

    def _open_recent_project(self, project_dir: str) -> None:
        project_helpers.open_recent_project(self, project_dir)

    def _refresh_recent_projects_menu(self) -> None:
        project_helpers.refresh_recent_projects_menu(self)

    def _init_recent_projects_menu(self) -> None:
        project_helpers.init_recent_projects_menu(self)

    def onSaveProject(self, event: QtCore.QEvent = None):
        """
        Save the current state of the application as a project.

        This method prompts the user for a directory and project name, then calls
        the save_project function to save the project.
        """
        current_dir = getattr(self, "_current_project_dir", None)
        if isinstance(current_dir, pathlib.Path) and current_dir.is_dir():
            try:
                chisurf.working_path = current_dir.parent
            except Exception:
                pass

            try:
                chisurf.action_controller.execute(
                    name="project.save",
                    payload={
                        "target_path": current_dir.parent.as_posix(),
                        "project_name": current_dir.name,
                    },
                )
            except Exception:
                return

            try:
                if (current_dir / "project.json").exists():
                    self.add_recent_project(current_dir)
            except Exception:
                pass
            return

        self.onSaveProjectAs(event=event)

    def onSaveProjectAs(self, event: QtCore.QEvent = None):
        path, _ = chisurf.gui.widgets.get_directory()
        if not path:
            return

        project_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Project As",
            "Project name:",
            QtWidgets.QLineEdit.Normal,
            "chisurf_project"
        )
        if not ok or not project_name:
            return

        project_dir = path / project_name
        try:
            needs_confirm = False
            if project_dir.exists():
                needs_confirm = True
            if (project_dir / "project.json").exists():
                needs_confirm = True
        except Exception:
            needs_confirm = False

        if needs_confirm:
            try:
                result = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite Project?",
                    f"The project folder already exists:\n\n{project_dir}\n\nOverwrite it?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
            except Exception:
                result = QtWidgets.QMessageBox.No
            if result != QtWidgets.QMessageBox.Yes:
                return

        try:
            chisurf.working_path = path
        except Exception:
            pass

        try:
            chisurf.action_controller.execute(
                name="project.save",
                payload={
                    "target_path": path.as_posix(),
                    "project_name": project_name,
                },
            )
        except Exception:
            return
        try:
            if (project_dir / "project.json").exists():
                self._current_project_dir = project_dir
        except Exception:
            pass

        try:
            self.add_recent_project(project_dir)
        except Exception:
            pass

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
            chisurf.action_controller.execute(
                name="project.load",
                payload={
                    "project_path": path.as_posix(),
                },
            )
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
        """Reinitialize ChiSurf application with user confirmation and feedback"""
        # Show confirmation dialog first
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Reinitialization",
            "This will completely reset ChiSurf and clear all data:\n\n"
            "• All loaded datasets will be removed\n"
            "• All fits and results will be deleted\n"
            "• All open windows will be closed\n"
            "• Memory will be cleaned up\n\n"
            "This action cannot be undone. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        # Show progress dialog to user
        progress_dialog = QtWidgets.QProgressDialog(
            "Reinitializing ChiSurf...", "Cancel", 0, 10, self
        )
        progress_dialog.setWindowTitle("Reinitializing")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)

        def progress_callback(step_name: str, progress_value: int):
            progress_dialog.setLabelText(f"Reinitializing ChiSurf...\n{step_name}")
            progress_dialog.setValue(progress_value)
            QtWidgets.QApplication.processEvents()

        try:
            # Call the helper function from macros
            import chisurf.macros
            chisurf.macros.reinitialize_application(
                main_window=self,
                progress_callback=progress_callback
            )

            # Close progress dialog and show completion message
            progress_dialog.close()
            QtWidgets.QMessageBox.information(
                self,
                "Reinitialization Complete",
                "ChiSurf has been successfully reinitialized.\nAll data has been cleared, memory freed, and the application reset to initial state."
            )
        except Exception as e:
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Reinitialization Error",
                f"An error occurred during reinitialization:\n{str(e)}"
            )

    def onCloseProject(self, event: QtCore.QEvent = None):
        try:
            chisurf.action_controller.execute(
                name="project.close",
                payload={
                    "main_window": self,
                    "current_project_dir": str(getattr(self, "_current_project_dir", "") or ""),
                },
            )
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
        # Close all existing fit windows directly, suppressing any per-fit
        # confirmation dialogs. This mirrors the original implementation and
        # avoids repeated cs.update() calls during shutdown.
        old_confirm = chisurf.settings.gui.get('confirm_close_fit', True)
        try:
            chisurf.settings.gui['confirm_close_fit'] = False
        except Exception:
            old_confirm = None

        try:
            for sub_window in list(chisurf.gui.fit_windows):
                try:
                    # Disable any per-window confirmation flags
                    setattr(sub_window, 'close_confirm', False)
                except Exception:
                    pass
                try:
                    w = sub_window.widget()
                    if w is not None:
                        setattr(w, 'close_confirm', False)
                except Exception:
                    pass
                try:
                    sub_window.close()
                except Exception:
                    pass

            # Clear Python-side tracking lists
            chisurf.fits.clear()
            chisurf.gui.fit_windows.clear()
        finally:
            if old_confirm is not None:
                try:
                    chisurf.settings.gui['confirm_close_fit'] = old_confirm
                except Exception:
                    pass

        # Clear the analysis dock layouts
        chisurf.gui.widgets.clear_layout(self.modelLayout)
        header_layout = getattr(self, "analysisHeaderLayout", None)
        if header_layout is not None:
            chisurf.gui.widgets.clear_layout(header_layout)
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
        chisurf.action_controller.execute(
            name="dataset.add",
            payload={"filename": s},
        )

    def onSaveFits(self, event: QtCore.QEvent = None):
        path, _ = chisurf.gui.widgets.get_directory()
        if not path:
            return
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.save_fits(target_path=r"{path.as_posix()}")')

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
            chisurf.logging.warning(f"onSaveFit: could not infer data folder from fit.data.filename: {e}")

        path, _ = chisurf.gui.widgets.get_directory(**kwargs)
        if not path:
            return
        # Keep behavior: user chooses where to save; update working path accordingly
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.save_fit(target_path=r"{path.as_posix()}")')

    def onOpenHelp(self):
        """Open the help plugin."""
        try:
            self.open_context_help_for_reader(None)
        except Exception as e:
            chisurf.gui.widgets.general.MyMessageBox(
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
                help_plugin = importlib.import_module("chisurf.plugins.chisurf.help")
            except Exception:
                help_plugin = importlib.import_module("chisurf.plugins.help")
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
                                        base = pathlib.Path(chisurf.__file__).resolve().parent
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
            chisurf.gui.widgets.general.MyMessageBox(
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
                updater_plugin = importlib.import_module("chisurf.plugins.chisurf.updater")
            except ImportError:
                updater_plugin = importlib.import_module("chisurf.plugins.updater")

            window = updater_plugin.UpdaterWidget()
            window.show()
        except Exception as e:
            # Show error message if plugin can't be loaded
            chisurf.gui.widgets.general.MyMessageBox(
                label="Updater Plugin Error",
                info=f"Error loading updater plugin: {str(e)}",
                show_fortune=False
            )

    def onOpenAbout(self):
        """Open the about plugin."""
        import importlib
        try:
            try:
                about_plugin = importlib.import_module("chisurf.plugins.chisurf.about")
            except ImportError:
                about_plugin = importlib.import_module("chisurf.plugins.about")

            window = about_plugin.AboutDialog(parent=self)
            window.show()
        except Exception as e:
            chisurf.gui.widgets.general.MyMessageBox(
                label="About Plugin Error",
                info=f"Error opening About dialog: {str(e)}",
                show_fortune=False
            )

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

            misc_helpers.run_plugin_from_dir(self, plugin_dir_to_use)

        except Exception as e:
            chisurf.logging.error(f"Error loading plugin {module_path}: {e}")

    def init_console(self):
        self.verticalLayout_4.addWidget(chisurf.console)
        chisurf.console.pushVariables({'cs': self})
        chisurf.console.pushVariables({'chisurf': chisurf})
        try:
            chisurf.console.pushVariables({'history': chisurf.history})
        except Exception:
            pass
        chisurf.console.pushVariables({'np': np})
        chisurf.console.pushVariables({'os': os})
        chisurf.console.pushVariables({'QtCore': QtCore})
        chisurf.console.pushVariables({'QtGui': QtGui})
        chisurf.console.set_default_style('linux')

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
            return chisurf.console.execute_on_gui_thread(code_str)

        chisurf.run = _run_with_history
        try:
            chisurf.log = chisurf.console.log_on_gui_thread
        except Exception:
            pass
        chisurf.run(str(chisurf.settings.gui['console_init']))

    def _init_history_browser(self) -> None:
        try:
            placeholder = getattr(self, "historyBrowserContainer", None)
            if placeholder is None:
                return
            parent = placeholder.parent()
            if parent is None:
                return
            parent_layout = parent.layout()
            if parent_layout is None:
                return
            idx = parent_layout.indexOf(placeholder)
            if idx < 0:
                return
            browser = chisurf.gui.widgets.history_browser.HistoryBrowserWidget(parent)
            browser.setObjectName("historyBrowser")
            browser.set_history(chisurf.history)
            try:
                browser.cursorChanged.connect(self._on_history_cursor_changed)
            except Exception:
                pass
            try:
                chisurf.history.set_checkpoint_capture(
                    chisurf.history_replay.capture_domain_snapshot
                )
            except Exception:
                pass
            parent_layout.removeWidget(placeholder)
            placeholder.setParent(None)
            parent_layout.insertWidget(idx, browser)
            self.historyBrowser = browser
            self._sync_history_navigation_actions()
        except Exception:
            self.historyBrowser = None

    @staticmethod
    def _focus_widget_has_native_undo_redo() -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        if widget is None:
            return False
        return isinstance(
            widget,
            (
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
            ),
        )

    def _history_undo(self) -> None:
        if self._focus_widget_has_native_undo_redo():
            chisurf.logging.info("HISTNAV: undo ignored (native text undo context)")
            return
        browser = getattr(self, "historyBrowser", None)
        if browser is None or not hasattr(browser, "undo_step"):
            chisurf.logging.info("HISTNAV: undo ignored (no history browser)")
            return
        try:
            event = browser.undo_step()
            if isinstance(event, dict):
                chisurf.logging.info(
                    f"HISTNAV: undo -> event={event.get('action_type','?')} id={str(event.get('event_id',''))[:8]}"
                )
            else:
                chisurf.logging.info("HISTNAV: undo produced no event")
        except Exception:
            pass
        self._sync_history_navigation_actions()

    def _history_redo(self) -> None:
        if self._focus_widget_has_native_undo_redo():
            chisurf.logging.info("HISTNAV: redo ignored (native text undo context)")
            return
        browser = getattr(self, "historyBrowser", None)
        if browser is None or not hasattr(browser, "redo_step"):
            chisurf.logging.info("HISTNAV: redo ignored (no history browser)")
            return
        try:
            event = browser.redo_step()
            if isinstance(event, dict):
                chisurf.logging.info(
                    f"HISTNAV: redo -> event={event.get('action_type','?')} id={str(event.get('event_id',''))[:8]}"
                )
            else:
                chisurf.logging.info("HISTNAV: redo produced no event")
        except Exception:
            pass
        self._sync_history_navigation_actions()

    def _sync_history_navigation_actions(self) -> None:
        browser = getattr(self, "historyBrowser", None)
        can_undo = bool(browser is not None and hasattr(browser, "can_undo") and browser.can_undo())
        can_redo = bool(browser is not None and hasattr(browser, "can_redo") and browser.can_redo())
        for name, enabled in (("actionHistoryUndo", can_undo), ("actionHistoryRedo", can_redo)):
            action = getattr(self, name, None)
            if action is not None:
                try:
                    action.setEnabled(bool(enabled))
                except Exception:
                    pass
        try:
            chisurf.logging.info(f"HISTNAV: action states undo={can_undo} redo={can_redo}")
        except Exception:
            pass

    def _apply_parameter_state(
            self,
            parameter_state: typing.Dict[typing.Tuple[str, str, str], typing.Dict[str, typing.Any]],
            force_unlink_keys: typing.Optional[typing.Set[typing.Tuple[str, str, str]]] = None,
    ) -> None:
        def format_key(key: typing.Tuple[str, str, str]) -> str:
            return f"{key[0]}/{key[1]}/{key[2]}"

        def resolve_param(
                key: typing.Tuple[str, str, str],
                state: typing.Optional[typing.Dict[str, typing.Any]] = None,
                link_uid: typing.Optional[typing.Tuple[str, str, str]] = None,
        ):
            state = state or {}
            src_param_uid = str(state.get("source_parameter_uid") or "")
            if src_param_uid:
                for fit_group in getattr(chisurf, "fits", []):
                    for local_fit in fit_group:
                        model = getattr(local_fit, "model", None)
                        if model is None:
                            continue
                        params = getattr(model, "parameters_all", [])
                        for p in params:
                            if str(getattr(p, "unique_identifier", "")) == src_param_uid:
                                return p

            if isinstance(link_uid, tuple):
                target_param_uid = str(link_uid[2] or "")
                if target_param_uid:
                    for fit_group in getattr(chisurf, "fits", []):
                        for local_fit in fit_group:
                            model = getattr(local_fit, "model", None)
                            if model is None:
                                continue
                            params = getattr(model, "parameters_all", [])
                            for p in params:
                                if str(getattr(p, "unique_identifier", "")) == target_param_uid:
                                    return p

            fit_group_name, local_fit_name, parameter_name = key
            for fit_group in getattr(chisurf, "fits", []):
                if str(getattr(fit_group, "name", "")) != str(fit_group_name):
                    continue
                for local_fit in fit_group:
                    if str(getattr(local_fit, "name", "")) != str(local_fit_name):
                        continue
                    model = getattr(local_fit, "model", None)
                    if model is None:
                        return None
                    params = getattr(model, "parameters_all_dict", {})
                    return params.get(parameter_name)
            return None

        force_unlink_keys = force_unlink_keys or set()
        touched_fit_groups: typing.Set[typing.Any] = set()
        applied_scalar = 0
        applied_links = 0
        applied_unlinks = 0
        unresolved = 0
        unresolved_keys: typing.List[str] = []

        # First apply scalar state
        for key, state in parameter_state.items():
            param = resolve_param(key, state=state)
            if param is None:
                unresolved += 1
                unresolved_keys.append(format_key(key))
                continue
            fit_group_name = key[0]
            for fg in getattr(chisurf, "fits", []):
                if str(getattr(fg, "name", "")) == fit_group_name:
                    touched_fit_groups.add(fg)
                    break
            try:
                if "fixed" in state:
                    param.fixed = bool(state["fixed"])
                    applied_scalar += 1
            except Exception:
                pass
            try:
                if "bounds_on" in state:
                    param.bounds_on = bool(state["bounds_on"])
                    applied_scalar += 1
            except Exception:
                pass
            try:
                if "bounds" in state:
                    lb, ub = state["bounds"]
                    param.bounds = (float(lb), float(ub))
                    applied_scalar += 1
            except Exception:
                pass
            try:
                if "value" in state:
                    was_fixed = bool(getattr(param, "fixed", False))
                    param.fixed = False
                    param.value = state["value"]
                    param.fixed = was_fixed
                    applied_scalar += 1
            except Exception:
                pass

        # Explicit unlink for touched keys not currently linked in replay state
        for key in force_unlink_keys:
            if key in parameter_state and "link" in parameter_state[key]:
                continue
            param = resolve_param(key, state=parameter_state.get(key, {}))
            if param is None:
                unresolved += 1
                unresolved_keys.append(format_key(key))
                continue
            try:
                param.link = None
                applied_unlinks += 1
            except Exception:
                pass

        # Then apply links after all parameters are available
        for key, state in parameter_state.items():
            if "link" not in state:
                continue
            param = resolve_param(key, state=state)
            if param is None:
                unresolved += 1
                unresolved_keys.append(format_key(key))
                continue
            target_key = state.get("link")
            target_uid = state.get("link_uid")
            try:
                if target_key is None:
                    param.link = None
                    applied_unlinks += 1
                else:
                    target_param = resolve_param(target_key, link_uid=target_uid)
                    if target_param is not None and target_param is not param:
                        param.link = target_param
                        applied_links += 1
            except Exception:
                pass

        # Refresh GUI/model for touched fit groups
        for fg in touched_fit_groups:
            try:
                finalize = getattr(fg, "finalize", None)
                if callable(finalize):
                    finalize()
            except Exception:
                pass

        try:
            unresolved_preview = ", ".join(unresolved_keys[:5])
            if len(unresolved_keys) > 5:
                unresolved_preview += ", ..."
            chisurf.logging.info(
                "HISTNAV: parameter replay apply "
                f"keys={len(parameter_state)} scalar_ops={applied_scalar} "
                f"links={applied_links} unlinks={applied_unlinks} "
                f"touched_fit_groups={len(touched_fit_groups)} unresolved={unresolved} "
                f"unresolved_keys=[{unresolved_preview}]"
            )
        except Exception:
            pass

        for fg in touched_fit_groups:
            try:
                update = getattr(fg, "update", None)
                if callable(update):
                    update()
            except Exception:
                pass

        self._refresh_parameter_widgets()
        self._refresh_plots()

    def _refresh_parameter_widgets(self) -> None:
        try:
            for fit_group in getattr(chisurf, "fits", []):
                for local_fit in fit_group:
                    model = getattr(local_fit, "model", None)
                    if model is None:
                        continue
                    for param in getattr(model, "parameters_all", []):
                        controller = getattr(param, "controller", None)
                        if controller is not None and hasattr(controller, "finalize"):
                            try:
                                controller.finalize()
                            except Exception:
                                pass
        except Exception:
            pass

    def _refresh_plots(self) -> None:
        try:
            for fit_window in getattr(chisurf.gui, "fit_windows", []):
                try:
                    plot = getattr(fit_window, "plot_tab_widget", None)
                    if plot is not None and hasattr(plot, "update"):
                        plot.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_fit_range_state(
            self,
            fit_range_state: typing.Dict[str, typing.Dict[str, typing.Any]],
    ) -> None:
        if not isinstance(fit_range_state, dict) or not fit_range_state:
            return

        applied = 0
        unresolved_fit_groups: typing.List[str] = []

        for fit_group_name, state in fit_range_state.items():
            try:
                xmin = int(state.get("xmin"))
                xmax = int(state.get("xmax"))
            except Exception:
                continue

            target_fit = None
            for fg in getattr(chisurf, "fits", []):
                if str(getattr(fg, "name", "")) == str(fit_group_name):
                    target_fit = fg
                    break
            if target_fit is None:
                unresolved_fit_groups.append(str(fit_group_name))
                continue

            try:
                target_fit.fit_range = (xmin, xmax)
                applied += 1
            except Exception:
                continue

            try:
                update = getattr(target_fit, "update", None)
                if callable(update):
                    update()
            except Exception:
                pass

            try:
                for fit_window in getattr(chisurf.gui, "fit_windows", []):
                    if getattr(fit_window, "fit", None) is not target_fit:
                        continue
                    fit_widget = getattr(fit_window, "fit_widget", None)
                    if fit_widget is None:
                        continue
                    fit_widget.blockSignals(True)
                    fit_widget.xmin = xmin
                    fit_widget.xmax = xmax
                    fit_widget.blockSignals(False)
                    break
            except Exception:
                pass

        try:
            unresolved_preview = ", ".join(unresolved_fit_groups[:5])
            if len(unresolved_fit_groups) > 5:
                unresolved_preview += ", ..."
            chisurf.logging.info(
                "HISTNAV: fit-range replay apply "
                f"keys={len(fit_range_state)} applied={applied} "
                f"unresolved={len(unresolved_fit_groups)} "
                f"unresolved_fit_groups=[{unresolved_preview}]"
            )
        except Exception:
            pass

    def _apply_setup_state(
            self,
            setup_state: typing.Dict[str, typing.Any],
    ) -> None:
        if not isinstance(setup_state, dict) or not setup_state:
            return

        experiment_name = str(setup_state.get("experiment") or "")
        setup_name = str(setup_state.get("setup") or "")
        params = setup_state.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        applied_params = 0

        if experiment_name:
            try:
                combo = self.comboBox_experimentSelect
                idx = combo.findText(experiment_name)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
                    self._current_experiment_idx = idx
                    self.comboBox_setupSelect.blockSignals(True)
                    self.comboBox_setupSelect.clear()
                    self.comboBox_setupSelect.addItems(self.current_experiment.reader_names)
                    self.comboBox_setupSelect.blockSignals(False)
            except Exception:
                pass

        if setup_name:
            try:
                combo = self.comboBox_setupSelect
                idx = combo.findText(setup_name)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
                    self._current_setup_idx = idx
                    self.onSetupChanged()
            except Exception:
                pass

        if params:
            try:
                setup_obj = self.current_setup
            except Exception:
                setup_obj = None
            if setup_obj is not None:
                for key, value in params.items():
                    path = str(key)
                    if not path:
                        continue
                    parts = path.split(".")
                    target = setup_obj
                    try:
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        setattr(target, parts[-1], value)
                        applied_params += 1
                    except Exception:
                        continue

        try:
            chisurf.logging.info(
                "HISTNAV: setup replay apply "
                f"experiment={experiment_name or '-'} setup={setup_name or '-'} "
                f"params={applied_params}/{len(params)}"
            )
        except Exception:
            pass

    def _on_history_cursor_changed(self, event: typing.Any) -> None:
        if not isinstance(event, dict):
            return

        try:
            chisurf.logging.info(
                f"HISTNAV: cursor changed to action={event.get('action_type','?')} id={str(event.get('event_id',''))[:8]}"
            )
        except Exception:
            pass

        try:
            browser = getattr(self, "historyBrowser", None)
            events = [event]
            all_events = [event]
            cursor_index = -1
            if browser is not None and hasattr(browser, "events_upto_cursor"):
                events = list(browser.events_upto_cursor())
            if browser is not None and hasattr(browser, "all_events"):
                all_events = list(browser.all_events())
            if browser is not None and hasattr(browser, "cursor_index"):
                cursor_index = browser.cursor_index()

            checkpoint_snapshot = None
            events_to_replay = events
            try:
                if cursor_index >= 0:
                    checkpoint_snapshot, events_to_replay = chisurf.history.get_events_from_checkpoint(cursor_index)
                    if checkpoint_snapshot is not None:
                        chisurf.logging.info(
                            f"HISTNAV: using checkpoint at index {cursor_index}, replaying {len(events_to_replay)} events"
                        )
            except Exception:
                pass

            try:
                chisurf.logging.info(
                    f"HISTNAV: replay window size upto={len(events)} total={len(all_events)} checkpoint={checkpoint_snapshot is not None}"
                )
            except Exception:
                pass

            if checkpoint_snapshot is not None:
                replay_state = chisurf.history_replay.snapshot_to_replay_state(checkpoint_snapshot)
                nav_state = replay_state.get("navigation", {})
                parameter_state = replay_state.get("parameters", {})
                fit_range_state = replay_state.get("fit_ranges", {})
                setup_state = replay_state.get("setup", {})
                nav_delta = chisurf.history_replay.reconstruct_navigation_state(events_to_replay)
                param_delta = chisurf.history_replay.reconstruct_parameter_state(events_to_replay)
                range_delta = chisurf.history_replay.reconstruct_fit_range_state(events_to_replay)
                setup_delta = chisurf.history_replay.reconstruct_setup_state(events_to_replay)
                for key in ["datasets", "dataset_uids", "fits", "fit_uids"]:
                    if key in nav_delta:
                        nav_state[key] = nav_delta[key]
                if nav_delta.get("selected_dataset"):
                    nav_state["selected_dataset"] = nav_delta["selected_dataset"]
                if nav_delta.get("selected_dataset_uid"):
                    nav_state["selected_dataset_uid"] = nav_delta["selected_dataset_uid"]
                if nav_delta.get("selected_fit"):
                    nav_state["selected_fit"] = nav_delta["selected_fit"]
                if nav_delta.get("selected_fit_uid"):
                    nav_state["selected_fit_uid"] = nav_delta["selected_fit_uid"]
                parameter_state.update(param_delta)
                fit_range_state.update(range_delta)
                setup_state.update(setup_delta)
            else:
                nav_state = chisurf.history_replay.reconstruct_navigation_state(events)
                parameter_state = chisurf.history_replay.reconstruct_parameter_state(events)
                fit_range_state = chisurf.history_replay.reconstruct_fit_range_state(events)
                setup_state = chisurf.history_replay.reconstruct_setup_state(events)
            link_touched = chisurf.history_replay.touched_parameter_keys(
                all_events,
                include_actions={"parameter_link", "parameter_unlink"},
            )
            selected_dataset = nav_state.get("selected_dataset")
            selected_dataset_uid = nav_state.get("selected_dataset_uid")
            selected_fit = nav_state.get("selected_fit")
            selected_fit_uid = nav_state.get("selected_fit_uid")
            self._apply_setup_state(setup_state)
            if selected_dataset or selected_dataset_uid:
                self._select_dataset_by_identity(
                    names=[str(selected_dataset)] if selected_dataset else [],
                    dataset_uid=str(selected_dataset_uid) if selected_dataset_uid else "",
                )
            if selected_fit or selected_fit_uid:
                self._select_fit_by_identity(
                    fit_name=str(selected_fit) if selected_fit else "",
                    fit_uid=str(selected_fit_uid) if selected_fit_uid else "",
                )
            self._apply_parameter_state(parameter_state, force_unlink_keys=link_touched)
            self._apply_fit_range_state(fit_range_state)
            try:
                chisurf.logging.info(
                    "HISTNAV: applied replay state "
                    f"selected_dataset={selected_dataset} selected_dataset_uid={selected_dataset_uid} selected_fit={selected_fit} selected_fit_uid={selected_fit_uid} "
                    f"parameter_keys={len(parameter_state)} fit_ranges={len(fit_range_state)} "
                    f"link_touched={len(link_touched)}"
                )
            except Exception:
                pass
            try:
                self.update()
            except Exception:
                pass
        except Exception:
            pass
        self._sync_history_navigation_actions()

    def _select_dataset_by_identity(self, names: typing.List[str], dataset_uid: str = "") -> None:
        try:
            target = [str(n) for n in names if n]
            target_uid = str(dataset_uid or "")
            if not target and not target_uid:
                return
            datasets = list(getattr(chisurf, "imported_datasets", []))
            for idx, dataset in enumerate(datasets):
                if target_uid and str(getattr(dataset, "unique_identifier", "")) == target_uid:
                    self.dataset_selector.selected_curve_index = idx
                    self.onCurrentDatasetChanged()
                    return
                dataset_name = str(getattr(dataset, "name", ""))
                if dataset_name in target:
                    self.dataset_selector.selected_curve_index = idx
                    self.onCurrentDatasetChanged()
                    return
        except Exception:
            pass

    def _select_fit_by_identity(self, fit_name: str = "", fit_uid: str = "") -> None:
        try:
            for idx, fit_group in enumerate(getattr(chisurf, "fits", [])):
                current_uid = str(getattr(fit_group, "unique_identifier", ""))
                name = str(getattr(fit_group, "name", ""))
                if fit_uid:
                    if current_uid != fit_uid:
                        continue
                elif name != fit_name:
                    continue
                try:
                    chisurf.logging.info(
                        f"HISTNAV: selecting fit '{name}' uid={current_uid} at index {idx}"
                    )
                except Exception:
                    pass
                try:
                    self.fit_selector.selected_fit_index = idx
                except Exception:
                    pass
                try:
                    self.current_fit = fit_group
                    self._fit_idx = idx
                    setattr(chisurf, "current_fit", fit_group)
                    setattr(chisurf, "current_fit_idx", idx)
                except Exception:
                    pass
                activated = False
                try:
                    for fit_window in getattr(chisurf.gui, "fit_windows", []):
                        if getattr(fit_window, "fit", None) is fit_group:
                            try:
                                fit_window.show()
                            except Exception:
                                pass
                            try:
                                fit_window.raise_()
                            except Exception:
                                pass
                            self.mdiarea.setActiveSubWindow(fit_window)
                            activated = True
                            break
                except Exception:
                    pass
                if activated:
                    try:
                        self.subWindowActivated()
                    except Exception:
                        pass
                try:
                    chisurf.logging.info(
                        f"HISTNAV: fit activation for '{fit_name}' success={activated} current_fit_idx={getattr(self, '_fit_idx', '?')}"
                    )
                except Exception:
                    pass
                return
            try:
                chisurf.logging.info(f"HISTNAV: fit not found name='{fit_name}' uid='{fit_uid}'")
            except Exception:
                pass
        except Exception:
            pass

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

        chisurf.action_controller.execute(
            name="dataset.add",
            payload={
                "experiment_reader": global_setup,
                "name": "Global Dataset",
            },
        )

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
        self.status = misc_helpers.TruncatingStatusBar(self)
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
        self._system_info_watermark = misc_helpers.init_system_info_watermark(parent, label)

    def _update_system_info_watermark_geometry(self) -> None:
        label = getattr(self, "_system_info_watermark", None)
        misc_helpers.update_system_info_watermark_geometry(label)

    def _install_dev_mode_code_badges(self) -> None:
        """Install code badge buttons on docks and key widgets when dev mode is enabled."""
        if not chisurf.settings.is_dev_mode():
            return

        try:
            from chisurf.gui.widgets.code_badge import (
                install_code_badge,
                make_widget_source_resolver,
            )
            from chisurf.gui.devtools.source_jump import (
                make_widget_resolver,
            )
        except ImportError:
            chisurf.logging.debug("Code badge module not available")
            return

        dev_settings = chisurf.settings.dev_mode_settings()
        badge_locations = dev_settings.get('badge_locations', {})

        if not dev_settings.get('show_code_badge', True):
            return

        dock_badges = badge_locations.get('docks', True)
        if dock_badges:
            dock_widgets = [
                ('dockWidgetReadData', 'Read Data'),
                ('dockWidgetDatasets', 'Datasets'),
                ('dockWidgetAnalysis', 'Analysis'),
                ('dockWidgetPlot', 'Plot Settings'),
                ('dockWidgetHistory', 'History'),
                ('dockWidgetScriptEdit', 'Code'),
            ]

            for attr_name, _label in dock_widgets:
                dock = getattr(self, attr_name, None)
                if dock is None:
                    continue
                widget = dock.widget()
                if widget is None:
                    continue
                resolver = make_widget_source_resolver(widget)
                install_code_badge(widget, resolver, corner='top-right', margin=4)

        if badge_locations.get('parameter_groups', True):
            try:
                self._install_parameter_badges()
            except Exception:
                pass

        if badge_locations.get('experiment_panels', True):
            try:
                self._install_experiment_panel_badges()
            except Exception:
                pass

        if badge_locations.get('mdi_windows', True):
            try:
                self.mdiarea.subWindowActivated.connect(self._on_mdi_window_activated_for_code_badge)
            except Exception:
                pass

    def _install_parameter_badges(self) -> None:
        """Install code badges on parameter group widgets."""
        try:
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import (
                resolve_parameter_group_source,
                make_widget_resolver,
            )
        except ImportError:
            return

        try:
            for fit in chisurf.fits:
                try:
                    model = getattr(fit, 'model', None)
                    if model is None:
                        continue
                    for param in getattr(model, 'parameters_all', []):
                        try:
                            widget = getattr(param, '_widget', None)
                            if widget is None or hasattr(widget, '_chisurf_code_badge_installed'):
                                continue
                            resolver = make_widget_resolver(widget)
                            install_code_badge(widget, resolver, corner='top-right', margin=4)
                            widget._chisurf_code_badge_installed = True
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass

    def _install_experiment_panel_badges(self) -> None:
        """Install code badges on experiment panel widgets."""
        try:
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import (
                resolve_experiment_panel_source,
                make_widget_resolver,
            )
        except ImportError:
            return

        try:
            experiment_panels = [
                'comboBox_experimentSelect',
                'comboBox_setupSelect',
                'comboBox_Model',
            ]
            for attr_name in experiment_panels:
                widget = getattr(self, attr_name, None)
                if widget is None or hasattr(widget, '_chisurf_code_badge_installed'):
                    continue
                resolver = make_widget_resolver(widget)
                install_code_badge(widget, resolver, corner='top-right', margin=4)
                widget._chisurf_code_badge_installed = True
        except Exception:
            pass

    def _on_mdi_window_activated_for_code_badge(self, sub_window) -> None:
        """Install code badge on newly activated MDI windows."""
        if not chisurf.settings.is_dev_mode():
            return

        if sub_window is None:
            return

        try:
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import resolve_fit_window_source
        except ImportError:
            return

        if hasattr(sub_window, '_chisurf_code_badge_installed'):
            return

        try:
            widget = sub_window.widget() if hasattr(sub_window, 'widget') else sub_window

            def resolver():
                return resolve_fit_window_source(sub_window)

            install_code_badge(widget, resolver, corner='top-right', margin=4)
            sub_window._chisurf_code_badge_installed = True
        except Exception:
            pass

    def _init_developer_menu(self) -> None:
        """Initialize the Developer menu with dev mode tools."""
        if not chisurf.settings.is_dev_mode():
            return

        try:
            dev_menu = QtWidgets.QMenu("Developer", self)
            self.menuBar().addMenu(dev_menu)

            action_open_source = QtWidgets.QAction("Open Source for Focus", self)
            action_open_source.setShortcut(QtGui.QKeySequence("Ctrl+Alt+J"))
            action_open_source.setShortcutContext(QtCore.Qt.ApplicationShortcut)
            action_open_source.triggered.connect(self._on_open_source_for_focus)
            action_open_source.setToolTip("Open source file for the currently focused widget")
            dev_menu.addAction(action_open_source)
            self.addAction(action_open_source)

            dev_menu.addSeparator()

            action_refresh_badges = QtWidgets.QAction("Refresh Code Badges", self)
            action_refresh_badges.triggered.connect(self._on_refresh_code_badges)
            dev_menu.addAction(action_refresh_badges)

            dev_menu.addSeparator()

            action_dev_settings = QtWidgets.QAction("Dev Mode Settings...", self)
            action_dev_settings.triggered.connect(self._on_open_dev_settings)
            dev_menu.addAction(action_dev_settings)

            self._dev_menu = dev_menu

        except Exception as e:
            chisurf.logging.debug(f"Could not initialize Developer menu: {e}")

    def _on_open_source_for_focus(self) -> None:
        """Open source for the currently focused widget."""
        try:
            from chisurf.gui.devtools.source_jump import (
                resolve_focused_widget_source,
                open_in_editor,
            )
            result = resolve_focused_widget_source()
            if result is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Source Target",
                    "Could not resolve a source file for the currently focused widget.",
                )
                return

            path, line = result
            open_in_editor(self, path, line)
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self,
                "Dev Mode Error",
                "Source jump module not available.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Dev Mode Error",
                f"Could not open source: {e}",
            )

    def _on_refresh_code_badges(self) -> None:
        """Refresh all code badges visibility."""
        try:
            from chisurf.gui.widgets.code_badge import get_badge_manager
            get_badge_manager().refresh_all()
            self._install_dev_mode_code_badges()
        except ImportError:
            pass

    def _on_open_dev_settings(self) -> None:
        """Open dev mode settings dialog."""
        try:
            from chisurf.gui.widgets.settings_editor import SettingsEditor
            if not hasattr(self, '_dev_settings_editor') or self._dev_settings_editor is None:
                self._dev_settings_editor = SettingsEditor(
                    filename=chisurf.settings.chisurf_settings_file,
                    window_title="Dev Mode Settings"
                )
            self._dev_settings_editor.show()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Dev Mode Settings",
                f"Could not open settings: {e}",
            )

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
                                chisurf.action_controller.execute(
                                    name="dataset.add",
                                    payload={"filename": str(p)},
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

        self._install_dev_mode_code_badges()

    def filter_log_content(self):
        """
        Filter log content based on filter text and hide checkbox state.
        If checkBox_filter_hide is checked, hide non-matching lines.
        If unchecked, highlight matching lines and gray out non-matching lines.
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
        try:
            self.actionSaveCurrentFit.setShortcut(QtGui.QKeySequence("Ctrl+S"))
            self.actionSaveCurrentFit.setShortcutContext(QtCore.Qt.ApplicationShortcut)
            self.addAction(self.actionSaveCurrentFit)
        except Exception:
            pass
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
        
        # Initialize ribbon interface (optional - can be enabled via settings)
        self._ribbon_integration = None
        
        # Restore ribbon interface state from settings
        try:
            import chisurf
            gui_settings = chisurf.settings.cs_settings.get('gui', {})
            use_ribbon = gui_settings.get('use_ribbon_interface', True)
            
            if use_ribbon:
                # Enable ribbon if it was saved in settings
                self.toggle_ribbon_interface(True)
                chisurf.logging.info("Ribbon interface restored from settings")
        except Exception as e:
            chisurf.logging.warning(f"Failed to restore ribbon interface state: {e}")
        
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
                import chisurf
                gui_settings = chisurf.settings.cs_settings.get('gui', {})
                ribbon_style = gui_settings.get('ribbon_style', None)
                
                # Hide plugin toolbar when switching to ribbon
                if hasattr(self, 'plugins_toolbar'):
                    self.plugins_toolbar.hide()
                    chisurf.logging.info("Plugin toolbar hidden for ribbon mode")
                
                self._ribbon_integration = setup_chisurf_ribbon(self, ribbon_style=ribbon_style)
                if self._ribbon_integration:
                    chisurf.logging.info("Ribbon interface enabled")
                    # Save to settings persistently
                    from chisurf.settings.settings_utils import set_use_ribbon_interface
                    set_use_ribbon_interface(True)
                else:
                    chisurf.logging.warning("Failed to setup ribbon interface")
            elif not enabled and self._ribbon_integration is not None:
                # Disable ribbon
                self._ribbon_integration.restore_original_interface()
                self._ribbon_integration = None
                chisurf.logging.info("Ribbon interface disabled")
                
                # Show plugin toolbar when switching back to menu mode
                if hasattr(self, 'plugins_toolbar'):
                    self.plugins_toolbar.show()
                    chisurf.logging.info("Plugin toolbar restored for menu mode")
                
                # Save to settings persistently
                from chisurf.settings.settings_utils import set_use_ribbon_interface
                set_use_ribbon_interface(False)
            
        except Exception as e:
            chisurf.logging.error(f"Failed to toggle ribbon interface: {e}")
