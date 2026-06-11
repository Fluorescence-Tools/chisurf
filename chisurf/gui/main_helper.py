from __future__ import annotations
import os
import pathlib
import typing
import yaml
import shutil
import copy
import importlib

import chisurf as cs
from chisurf.history import replay as _hr


if typing.TYPE_CHECKING:
    from chisurf.gui.main import Main

class ProjectMixin:
    def _load_recent_projects(self: Main) -> list[str]:
        return project_helpers.load_recent_projects()

    def _store_recent_projects(self: Main, projects: list[str]) -> None:
        project_helpers.store_recent_projects(projects)

    def _set_recent_projects(self: Main, projects: list[str]) -> None:
        project_helpers.set_recent_projects(self, projects)

    def add_recent_project(self: Main, project_path) -> None:
        project_helpers.add_recent_project(self, project_path)

    def _clear_recent_projects(self: Main) -> None:
        project_helpers.clear_recent_projects(self)

    def _open_recent_project(self: Main, project_dir: str) -> None:
        project_helpers.open_recent_project(self, project_dir)

    def _refresh_recent_projects_menu(self: Main) -> None:
        project_helpers.refresh_recent_projects_menu(self)

    def _init_recent_projects_menu(self: Main) -> None:
        project_helpers.init_recent_projects_menu(self)

    def onSaveProject(self: Main, event: QtCore.QEvent = None):
        current_dir = getattr(self, "_current_project_dir", None)
        if isinstance(current_dir, pathlib.Path) and current_dir.is_dir():
            try:
                cs.working_path = current_dir.parent
            except Exception:
                pass

            try:
                cs.core.actions.dispatch(
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

    def onSaveProjectAs(self: Main, event: QtCore.QEvent = None):
        path, _ = cs.gui.widgets.get_directory()
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
            cs.working_path = path
        except Exception:
            pass

        try:
            cs.core.actions.dispatch(
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

    def onLoadProject(self: Main, event: QtCore.QEvent = None):
        path, _ = cs.gui.widgets.get_directory(
            caption="Select Project Folder"
        )
        if not path:
            return

        project_file = path / "project.json"
        if not project_file.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Project",
                f"The selected folder does not contain a valid project file (project.json)."
            )
            return

        cs.working_path = path
        try:
            cs.core.actions.dispatch(
                name="project.load",
                payload={
                    "project_path": path.as_posix(),
                },
            )
        except Exception:
            try:
                cs.logging.exception(f"Project load failed: {path}")
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

    def onCloseProject(self: Main, event: QtCore.QEvent = None):
        try:
            cs.core.actions.dispatch(
                name="project.close",
                payload={
                    "main_window": self,
                    "current_project_dir": str(getattr(self, "_current_project_dir", "") or ""),
                },
            )
        except Exception:
            pass


class SetupMixin:
    def _restore_setup_defaults(self: Main) -> None:
        """Restore saved setup defaults from user settings."""
        try:
            from chisurf.gui.widgets.experiments.setup_persistence import (
                load_setup_defaults,
                apply_setup_defaults,
            )
            defaults = load_setup_defaults()
            if defaults.get("experiments"):
                apply_setup_defaults(self, defaults)
                cs.logging.info("Restored setup defaults from user settings")
        except Exception as e:
            cs.logging.warning(f"Failed to restore setup defaults: {e}")

    def _save_setup_defaults(self: Main) -> None:
        """Save current setup defaults to user settings."""
        try:
            from chisurf.gui.widgets.experiments.setup_persistence import (
                collect_setup_defaults,
                save_setup_defaults,
            )
            defaults = collect_setup_defaults(self)
            if save_setup_defaults(defaults):
                cs.logging.info("Saved setup defaults to user settings")
        except Exception as e:
            cs.logging.warning(f"Failed to save setup defaults: {e}")

    def set_current_setup_idx(self: Main, v: int):
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

    def onExperimentChanged(self: Main):
        if not cs.core.actions.is_dispatching():
            experiment_name = self.comboBox_experimentSelect.currentText()
            cs.core.actions.dispatch(
                name="experiment.set",
                payload={"name": str(experiment_name)},
            )

        self._refresh_experiment_ui()

    def _refresh_experiment_ui(self: Main):
        """Refresh the setup combo and trigger a setup refresh — no dispatch."""
        exp = self.current_experiment
        if exp is None:
            return

        # Add setups for selected experiment
        self.comboBox_setupSelect.blockSignals(True)
        self.comboBox_setupSelect.clear()
        self.comboBox_setupSelect.addItems(
            exp.reader_names
        )
        self.comboBox_setupSelect.blockSignals(False)
        self._current_experiment_idx = self.comboBox_experimentSelect.currentIndex()
        self._refresh_setup_ui()

    def onSetupChanged(self: Main):
        if not cs.core.actions.is_dispatching():
            setup_name = self.comboBox_setupSelect.currentText()
            cs.core.actions.dispatch(
                name="setup.select",
                payload={"name": str(setup_name)},
            )

        self._refresh_setup_ui()

    def _refresh_setup_ui(self: Main):
        """Show the reader widget for the current setup — no dispatch."""
        from qtpy import QtWidgets
        cs.gui.widgets.hide_items_in_layout(
            self.layout_experiment_reader
        )
        readers = self.current_experiment.readers
        if not readers:
            self._current_setup_idx = 0
            return
        widget = self.current_setup
        if not isinstance(widget, QtWidgets.QWidget):
            widget = widget.controller
        self.layout_experiment_reader.addWidget(widget)
        widget.show()
        if hasattr(widget, 'updateUI') and callable(widget.updateUI):
            widget.updateUI()
        self._current_setup_idx = self.comboBox_setupSelect.currentIndex()

        try:
            if hasattr(widget, 'set_help_callback') and callable(widget.set_help_callback):
                widget.set_help_callback(self.open_context_help_for_reader)
        except Exception:
            pass

    def _setup_experiment(self: Main, exp_type, config):
        """
        Set up an experiment based on its configuration.

        Args:
            exp_type (str): The experiment type key in cs.core.experiments.types
            config (dict): Configuration for the experiment with readers and models
        """
        import chisurf.core.experiments
        try:
            # Get the base experiment from registry or create a new one
            experiment = cs.core.experiments.types.get(exp_type)
            if experiment is None:
                experiment = cs.core.experiments.core.Experiment(
                    name=config.get('name', exp_type),
                    hidden=config.get('hidden', False)
                )

            # Add readers
            for reader_cfg in config.get("readers", []):
                try:
                    reader_class = self._resolve_class(reader_cfg.get('reader_class'))
                    if reader_class is None:
                        continue
                    
                    reader_params = reader_cfg.get('reader_params', {})
                    # Ensure experiment is passed to reader if it expects it
                    reader_params['experiment'] = experiment
                    reader = reader_class(**reader_params)
                    
                    # Resolve controller if present
                    controller_class_name = reader_cfg.get('controller_class')
                    controller = None
                    if controller_class_name:
                        controller_class = self._resolve_class(controller_class_name)
                        if controller_class:
                            controller_params = reader_cfg.get('controller_params', {})
                            controller_params['experiment_reader'] = reader
                            controller = controller_class(**controller_params)
                    
                    experiment.add_reader(reader, controller)
                except Exception as e:
                    cs.logging.error(f"Failed to setup reader {reader_cfg.get('reader_class')} in {exp_type}: {e}")

            # Add models
            for model_path in config.get("models", []):
                try:
                    model_class = self._resolve_class(model_path)
                    if model_class:
                        experiment.add_model_class(model_class)
                except Exception as e:
                    cs.logging.error(f"Failed to setup model {model_path} in {exp_type}: {e}")

            cs.experiment[experiment.name] = experiment
        except Exception as e:
            cs.logging.error(f"Failed to setup experiment {exp_type}: {e}")

    def _resolve_class(self: Main, class_path):
        """
        Resolve a class from its string path.

        Args:
            class_path (str): The full path to the class

        Returns:
            class: The resolved class
        """
        if not class_path or not isinstance(class_path, str):
            return None
        class_path = {
            "chisurf.models.global_model.GlobalFitModelWidget": (
                "chisurf.gui.widgets.models.global_model.GlobalFitModelWidget"
            ),
            "chisurf.models.global_model.ParameterTransformWidget": (
                "chisurf.gui.widgets.models.parameter_transform.ParameterTransformWidget"
            ),
        }.get(class_path, class_path)
        import importlib
        try:
            if '.' in class_path:
                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
            # Fallback for unqualified names — not used in current config
            return None
        except Exception as e:
            cs.logging.error(f"Failed to resolve class {class_path}: {e}")
            return None

    def init_setups(self: Main):
        """
        Initialize experiment setups based on configuration from YAML file.
        """
        import copy
        import yaml
        import pathlib
        import shutil
        import chisurf.core.experiments

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

        source_config_file = pathlib.Path(cs.core.settings.get_path('cs')) / "settings" / "experiment_configs.yaml"
        user_config_file = pathlib.Path(cs.core.settings.get_path('settings')) / "experiment_configs.yaml"

        check_updates = True
        try:
            check_updates = bool(cs.core.settings.cs_settings.get('check_experiment_config_updates_on_startup', True))
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
                            from chisurf.core.settings.settings_utils import set_check_experiment_config_updates_on_startup as _set_exp_flag
                            _set_exp_flag(False)
                            try:
                                cs.core.settings.cs_settings['check_experiment_config_updates_on_startup'] = False
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        if msg.clickedButton() is yes_button:
                            shutil.copyfile(source_config_file, user_config_file)
                    except Exception:
                        pass

        if not user_config_file.exists():
            if source_config_file.exists():
                shutil.copyfile(source_config_file, user_config_file)
            else:
                cs.logging.warning(f"Experiment configuration file not found: {source_config_file}")
                experiment_configs = {}

        default_configs = _load_yaml_config(source_config_file)
        user_configs = _load_yaml_config(user_config_file) if user_config_file.exists() else {}
        if default_configs and user_configs:
            experiment_configs = _deep_merge_dicts(default_configs, user_configs)
        elif default_configs:
            experiment_configs = default_configs
        else:
            experiment_configs = user_configs

        if experiment_configs:
            for exp_type, config in experiment_configs.items():
                if exp_type == 'global' or exp_type == 'experiment_types':
                    continue
                self._setup_experiment(exp_type, config)
        else:
            cs.logging.warning("Using default experiment configurations")
            for exp_type, experiment in cs.core.experiments.types.items():
                cs.experiment[experiment.name] = experiment

        global_config = experiment_configs.get('global', {})
        global_fit = cs.core.experiments.core.Experiment(
            name=global_config.get('name', 'Global'),
            hidden=global_config.get('hidden', True)
        )

        if 'readers' in global_config and global_config['readers']:
            reader_config = global_config['readers'][0]
            reader_class = self._resolve_class(reader_config['reader_class'])
            reader_params = reader_config.get('reader_params', {})
            reader_params['experiment'] = global_fit
            global_setup = reader_class(**reader_params)
            global_fit.add_reader(global_setup)
        else:
            global_setup = cs.core.experiments.globalfit.GlobalFitSetup(
                name='Global-Fit',
                experiment=global_fit
            )
            global_fit.add_reader(global_setup)

        if 'models' in global_config:
            model_classes = [self._resolve_class(model_class) for model_class in global_config['models']]
            global_fit.add_model_classes(models=model_classes)

        cs.experiment[global_fit.name] = global_fit

        cs.core.actions.dispatch(
            name="dataset.add",
            payload={
                "experiment_reader": global_setup,
                "name": "Global Dataset",
            },
        )

        self.experiment_names = [
            b.name for b in list(cs.experiment.values()) 
            if not b.hidden
        ]
        self.comboBox_experimentSelect.clear()
        self.comboBox_experimentSelect.addItems(
            self.experiment_names
        )

        # Chimol display config version check
        try:
            from chisurf.plugins.chimol.chimol.config import (
                check_for_display_config_update,
                get_user_display_config_path,
                get_package_display_config_path,
                DISPLAY_CONFIG_VERSION,
            )
            if check_for_display_config_update():
                user_path = get_user_display_config_path()
                package_path = get_package_display_config_path()
                msg = QtWidgets.QMessageBox(self)
                msg.setWindowTitle("Chimol display configuration update")
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText(
                    "The Chimol display configuration in your settings folder "
                    "is outdated."
                )
                msg.setInformativeText(
                    f"Your version is older than the current version "
                    f"(v{DISPLAY_CONFIG_VERSION}) shipped with the package.\n\n"
                    "Do you want to update? This will overwrite your current "
                    "user configuration."
                )
                yes_button = msg.addButton("Update", QtWidgets.QMessageBox.YesRole)
                msg.addButton("Skip", QtWidgets.QMessageBox.NoRole)
                msg.exec_()
                if msg.clickedButton() is yes_button:
                    if package_path.is_file() and user_path is not None:
                        try:
                            user_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(package_path, user_path)
                            from chisurf.plugins.chimol.chimol import config as _chimol_config
                            _chimol_config.reload_display_config()
                        except Exception as e:
                            cs.logging.error(
                                f"Failed to update chimol display config: {e}"
                            )
        except ImportError:
            pass
        except Exception as e:
            cs.logging.warning(
                f"Failed to check chimol display config update: {e}"
            )

    def reinitialize(self: Main):
        """Reinitialize ChiSurf application with user confirmation and feedback"""
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
            import chisurf.macros
            cs.macros.reinitialize_application(
                main_window=self,
                progress_callback=progress_callback
            )

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

# Placeholder for remaining Mixins
class HistoryMixin:
    def _init_history_browser(self: Main) -> None:
        try:
            from qtpy import QtCore, QtWidgets

            log_widget = getattr(self, "plainTextEditLog", None)
            filter_edit = getattr(self, "lineEdit_LogFilter", None)
            hide_checkbox = getattr(self, "checkBox_filter_hide", None)
            if log_widget is None:
                return

            placeholder = getattr(self, "historyBrowserContainer", None)
            if placeholder is None:
                placeholder = self.findChild(QtWidgets.QWidget, "historyBrowserContainer")
            parent = placeholder.parent() if placeholder is not None else log_widget.parent()
            if parent is None:
                return
            parent_layout = parent.layout()
            if parent_layout is None:
                return
            insert_index = parent_layout.indexOf(log_widget)

            widgets_to_remove = [
                log_widget,
                filter_edit,
                hide_checkbox,
                placeholder,
            ]
            for i in range(parent_layout.count()):
                item = parent_layout.itemAt(i)
                w = item.widget()
                if w is not None and isinstance(w, QtWidgets.QLabel) and w.text() in {"Logging", "History"}:
                    widgets_to_remove.append(w)

            seen = set()
            for widget in widgets_to_remove:
                if widget is None or widget in seen:
                    continue
                seen.add(widget)
                try:
                    parent_layout.removeWidget(widget)
                    widget.setParent(None)
                except Exception:
                    pass

            # Logging tab: wrap in a container with filter below the table
            log_container = QtWidgets.QWidget(parent)
            log_container.setObjectName("logContainer")
            log_layout = QtWidgets.QVBoxLayout(log_container)
            log_layout.setContentsMargins(0, 0, 0, 0)
            log_layout.setSpacing(4)
            log_layout.addWidget(log_widget, 1)

            filter_row = QtWidgets.QHBoxLayout()
            if filter_edit is None:
                filter_edit = QtWidgets.QLineEdit(log_container)
                filter_edit.setObjectName("lineEdit_LogFilter")
                self.lineEdit_LogFilter = filter_edit
            else:
                filter_edit.setParent(log_container)
            if hide_checkbox is None:
                hide_checkbox = QtWidgets.QCheckBox("hide", log_container)
                hide_checkbox.setObjectName("checkBox_filter_hide")
                self.checkBox_filter_hide = hide_checkbox
            else:
                hide_checkbox.setParent(log_container)
            filter_row.addWidget(filter_edit, 1)
            filter_row.addWidget(hide_checkbox)
            log_layout.addLayout(filter_row)

            shared_tabs = QtWidgets.QTabWidget(parent)
            shared_tabs.setObjectName("logRpcTabs")
            shared_tabs.addTab(log_container, "Logging")
            shared_tabs.tabBar().setDocumentMode(True)

            try:
                from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

                rpc_monitor = RPCMonitorWidget(shared_tabs)
                rpc_monitor.setObjectName("rpcMonitor")
                shared_tabs.addTab(rpc_monitor, "RPC Monitor")
                self.rpcMonitor = rpc_monitor
            except Exception as exc:
                try:
                    cs.logging.debug(f"RPC monitor disabled: {exc}")
                except Exception:
                    pass

            parent_layout.insertWidget(insert_index, shared_tabs)
            insert_index += 1

            history_label = QtWidgets.QLabel("History", parent)
            history_label.setObjectName("historyLabel")
            parent_layout.insertWidget(insert_index, history_label)
            insert_index += 1

            browser = cs.gui.widgets.history_browser.HistoryBrowserWidget(parent)
            browser.setObjectName("historyBrowser")
            browser.set_history(cs.history)
            try:
                browser.cursorChanged.connect(self._on_history_cursor_changed)
            except Exception:
                pass
            try:
                cs.history.set_checkpoint_capture(
                    _hr.capture_domain_snapshot
                )
            except Exception:
                pass
            parent_layout.insertWidget(insert_index, browser, 1)

            self.historyBrowser = browser
            self._sync_history_navigation_actions()
        except Exception as exc:
            try:
                cs.logging.debug(f"Failed to initialize history/log dock: {exc}")
            except Exception:
                pass
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

    def _history_undo(self: Main) -> None:
        if self._focus_widget_has_native_undo_redo():
            cs.logging.info("HISTNAV: undo ignored (native text undo context)")
            return
        browser = getattr(self, "historyBrowser", None)
        if browser is None or not hasattr(browser, "undo_step"):
            cs.logging.info("HISTNAV: undo ignored (no history browser)")
            return
        try:
            event = browser.undo_step()
            if isinstance(event, dict):
                cs.logging.info(
                    f"HISTNAV: undo -> event={event.get('action_type','?')} id={str(event.get('event_id',''))[:8]}"
                )
            else:
                cs.logging.info("HISTNAV: undo produced no event")
        except Exception:
            pass
        self._sync_history_navigation_actions()

    def _history_redo(self: Main) -> None:
        if self._focus_widget_has_native_undo_redo():
            cs.logging.info("HISTNAV: redo ignored (native text undo context)")
            return
        browser = getattr(self, "historyBrowser", None)
        if browser is None or not hasattr(browser, "redo_step"):
            cs.logging.info("HISTNAV: redo ignored (no history browser)")
            return
        try:
            event = browser.redo_step()
            if isinstance(event, dict):
                cs.logging.info(
                    f"HISTNAV: redo -> event={event.get('action_type','?')} id={str(event.get('event_id',''))[:8]}"
                )
            else:
                cs.logging.info("HISTNAV: redo produced no event")
        except Exception:
            pass
        self._sync_history_navigation_actions()

    def _sync_history_navigation_actions(self: Main) -> None:
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
            cs.logging.info(f"HISTNAV: action states undo={can_undo} redo={can_redo}")
        except Exception:
            pass

class StateMixin:
    def _apply_parameter_state(
            self: Main,
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
                for fit_group in getattr(cs, "fits", []):
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
                    for fit_group in getattr(cs, "fits", []):
                        for local_fit in fit_group:
                            model = getattr(local_fit, "model", None)
                            if model is None:
                                continue
                            params = getattr(model, "parameters_all", [])
                            for p in params:
                                if str(getattr(p, "unique_identifier", "")) == target_param_uid:
                                    return p

            fit_group_name, local_fit_name, parameter_name = key
            for fit_group in getattr(cs, "fits", []):
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
            for fg in getattr(cs, "fits", []):
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
            cs.logging.info(
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

    def _refresh_parameter_widgets(self: Main) -> None:
        try:
            for fit_group in getattr(cs, "fits", []):
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

    def _refresh_plots(self: Main) -> None:
        try:
            for fit_window in getattr(cs.gui, "fit_windows", []):
                try:
                    plot = getattr(fit_window, "plot_tab_widget", None)
                    if plot is not None and hasattr(plot, "update"):
                        plot.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_fit_range_state(
            self: Main,
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
            for fg in getattr(cs, "fits", []):
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
                for fit_window in getattr(cs.gui, "fit_windows", []):
                    if getattr(fit_window, "fit", None) is not target_fit:
                        continue
                    fit_widget = getattr(fit_window, "fit_widget", None)
                    if fit_widget is None:
                        continue
                    fit_widget.blockSignals(True)
                    fit_widget.xmin = xmin
                    fit_widget.xmax = xmax
                    fit_widget.blockSignals(False)
                    fit_widget.update()
                    break
            except Exception:
                pass

        try:
            cs.logging.info(f"HISTNAV: applied fit_range_state to {applied} fit groups; unresolved={unresolved_fit_groups}")
        except Exception:
            pass

    def _apply_model_state(self: Main, model_state: typing.Dict[str, typing.Any]) -> None:
        """Apply model state from history replay."""
        if not isinstance(model_state, dict) or not model_state:
            return

        applied = 0
        unresolved_fit_groups: typing.List[str] = []

        import chisurf.core.actions as actions

        for fit_group_uid, fg_data in model_state.items():
            if not isinstance(fg_data, dict):
                continue

            # Find the target fit group by UID
            target_fit_group = None
            for fg in getattr(cs, "fits", []):
                if str(getattr(fg, "unique_identifier", "")) == str(fit_group_uid):
                    target_fit_group = fg
                    break

            if target_fit_group is None:
                unresolved_fit_groups.append(str(fit_group_uid))
                continue

            local_fits_data = fg_data.get("local_fits", {})
            if not isinstance(local_fits_data, dict):
                continue

            # Find local fits by UID
            for local_fit_uid, local_data in local_fits_data.items():
                if not isinstance(local_data, dict):
                    continue

                target_local_fit = None
                for local_fit in getattr(target_fit_group, "local_fits", []):
                    if str(getattr(local_fit, "unique_identifier", "")) == str(local_fit_uid):
                        target_local_fit = local_fit
                        break

                if target_local_fit is None:
                    continue

                model = getattr(target_local_fit, "model", None)
                if model is None:
                    continue

                # Apply component changes
                components_data = local_data.get("components", [])
                if isinstance(components_data, list):
                    for comp_data in components_data:
                        if not isinstance(comp_data, dict):
                            continue

                        comp_name = str(comp_data.get("name", ""))
                        action = str(comp_data.get("action", ""))
                        
                        if action == "add":
                            try:
                                actions.dispatch("model.add_component", {
                                    "fit_index": target_fit_group.index,
                                    "local_fit_index": target_local_fit.index,
                                    "component_name": comp_name
                                })
                                applied += 1
                            except Exception:
                                pass
                        elif action == "remove":
                            try:
                                actions.dispatch("model.remove_component", {
                                    "fit_index": target_fit_group.index,
                                    "local_fit_index": target_local_fit.index,
                                    "component_name": comp_name
                                })
                                applied += 1
                            except Exception:
                                pass

                # Apply configuration changes
                config_data = local_data.get("config", {})
                if isinstance(config_data, dict):
                    # Handle specific model operations
                    if "normalize_amplitudes" in config_data:
                        try:
                            component_name = str(config_data["normalize_amplitudes"])
                            actions.dispatch("model.normalize_amplitudes", {
                                "fit_index": target_fit_group.index,
                                "local_fit_index": target_local_fit.index,
                                "component_name": component_name
                            })
                            applied += 1
                        except Exception:
                            pass

                    if "absolute_amplitudes" in config_data:
                        try:
                            component_name = str(config_data["absolute_amplitudes"])
                            actions.dispatch("model.absolute_amplitudes", {
                                "fit_index": target_fit_group.index,
                                "local_fit_index": target_local_fit.index,
                                "component_name": component_name
                            })
                            applied += 1
                        except Exception:
                            pass

                    if "unload_irf" in config_data:
                        try:
                            actions.dispatch("model.unload_irf", {
                                "fit_index": target_fit_group.index,
                                "local_fit_index": target_local_fit.index
                            })
                            applied += 1
                        except Exception:
                            pass

                    if "unload_lintable" in config_data:
                        try:
                            actions.dispatch("model.unload_lintable", {
                                "fit_index": target_fit_group.index,
                                "local_fit_index": target_local_fit.index
                            })
                            applied += 1
                        except Exception:
                            pass

                    if "unload_background_curve" in config_data:
                        try:
                            actions.dispatch("model.unload_background_curve", {
                                "fit_index": target_fit_group.index,
                                "local_fit_index": target_local_fit.index
                            })
                            applied += 1
                        except Exception:
                            pass

                    # Handle IRF changes
                    for key, value in config_data.items():
                        if key.startswith("irf_") and isinstance(value, str):
                            try:
                                irf_idx = int(key[4:])
                                actions.dispatch("model.change_irf", {
                                    "fit_index": target_fit_group.index,
                                    "local_fit_index": target_local_fit.index,
                                    "irf_idx": irf_idx,
                                    "irf_name": value
                                })
                                applied += 1
                            except Exception:
                                pass

                        if key.startswith("correction_") and isinstance(value, (int, float, str)):
                            try:
                                correction_type = key[11:]
                                actions.dispatch("model.set_correction", {
                                    "fit_index": target_fit_group.index,
                                    "local_fit_index": target_local_fit.index,
                                    "correction_type": correction_type,
                                    "value": value
                                })
                                applied += 1
                            except Exception:
                                pass

                        if key.startswith("linearization_") and isinstance(value, str):
                            try:
                                idx = int(key[15:])
                                actions.dispatch("model.set_linearization", {
                                    "fit_index": target_fit_group.index,
                                    "local_fit_index": target_local_fit.index,
                                    "idx": idx,
                                    "lin_name": value
                                })
                                applied += 1
                            except Exception:
                                pass

                    # Handle global parameters
                    if "global_parameters" in config_data:
                        try:
                            for param_name in config_data["global_parameters"]:
                                actions.dispatch("model.append_global_parameter", {
                                    "fit_index": target_fit_group.index,
                                    "local_fit_index": target_local_fit.index,
                                    "parameter_name": str(param_name)
                                })
                                applied += 1
                        except Exception:
                            pass

                    # Handle fit operations
                    if "remove_local_fit" in config_data:
                        try:
                            row = int(config_data["remove_local_fit"])
                            actions.dispatch("model.remove_local_fit", {
                                "fit_index": target_fit_group.index,
                                "row": row
                            })
                            applied += 1
                        except Exception:
                            pass

                    if "clear_local_fits" in config_data:
                        try:
                            actions.dispatch("model.clear_local_fits", {
                                "fit_index": target_fit_group.index
                            })
                            applied += 1
                        except Exception:
                            pass

                    if "append_fit" in config_data:
                        try:
                            fit_index = int(config_data["append_fit"])
                            actions.dispatch("model.append_fit", {
                                "fit_index": target_fit_group.index,
                                "fit_index_to_append": fit_index
                            })
                            applied += 1
                        except Exception:
                            pass

                    # Handle generic model updates
                    if "update" in config_data:
                        try:
                            actions.dispatch("model.update", {
                                "fit_index": target_fit_group.index,
                                "local_fit_index": target_local_fit.index
                            })
                            applied += 1
                        except Exception:
                            pass

                # Update the model to reflect changes
                try:
                    model.finalize()
                    if hasattr(target_local_fit, "update"):
                        target_local_fit.update()
                except Exception:
                    pass

        try:
            cs.logging.info(f"HISTNAV: applied model_state to {applied} model operations; unresolved_fit_groups={unresolved_fit_groups}")
        except Exception:
            pass

    def _apply_setup_state(
            self: Main,
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
            cs.logging.info(
                "HISTNAV: setup replay apply "
                f"experiment={experiment_name or '-'} setup={setup_name or '-'} "
                f"params={applied_params}/{len(params)}"
            )
        except Exception:
            pass

    def _on_history_cursor_changed(self: Main, event: typing.Any) -> None:
        if not isinstance(event, dict):
            return

        try:
            cs.logging.info(
                f"HISTNAV: cursor changed to action={event.get('action_type','?')} id={str(event.get('event_id',''))[:8]}"
            )
        except Exception:
            pass

        try:
            import chisurf.history
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
                    checkpoint_snapshot, events_to_replay = cs.history.get_events_from_checkpoint(cursor_index)
                    if checkpoint_snapshot is not None:
                        cs.logging.info(
                            f"HISTNAV: using checkpoint at index {cursor_index}, replaying {len(events_to_replay)} events"
                        )
            except Exception:
                pass

            if checkpoint_snapshot is not None:
                replay_state = _hr.snapshot_to_replay_state(checkpoint_snapshot)
                nav_state = replay_state.get("navigation", {})
                parameter_state = replay_state.get("parameters", {})
                fit_range_state = replay_state.get("fit_ranges", {})
                setup_state = replay_state.get("setup", {})
                model_state = replay_state.get("models", {})
                nav_delta = _hr.reconstruct_navigation_state(events_to_replay)
                param_delta = _hr.reconstruct_parameter_state(events_to_replay)
                range_delta = _hr.reconstruct_fit_range_state(events_to_replay)
                setup_delta = _hr.reconstruct_setup_state(events_to_replay)
                model_delta = _hr.reconstruct_model_state(events_to_replay)
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
                for fg_uid, fg_data in model_delta.items():
                    if fg_uid not in model_state:
                        model_state[fg_uid] = fg_data
                    else:
                        for local_uid, local_data in fg_data.get("local_fits", {}).items():
                            if local_uid not in model_state[fg_uid].get("local_fits", {}):
                                if "local_fits" not in model_state[fg_uid]:
                                    model_state[fg_uid]["local_fits"] = {}
                                model_state[fg_uid]["local_fits"][local_uid] = local_data
                            else:
                                existing = model_state[fg_uid]["local_fits"][local_uid]
                                if "components" in local_data:
                                    if "components" not in existing:
                                        existing["components"] = []
                                    for comp in local_data["components"]:
                                        comp_name = comp.get("name", "")
                                        found = False
                                        for i, existing_comp in enumerate(existing["components"]):
                                            if existing_comp.get("name") == comp_name:
                                                existing["components"][i] = comp
                                                found = True
                                                break
                                        if not found:
                                            existing["components"].append(comp)
                                if "config" in local_data:
                                    if "config" not in existing:
                                        existing["config"] = {}
                                    existing["config"].update(local_data["config"])
            else:
                nav_state = _hr.reconstruct_navigation_state(events)
                parameter_state = _hr.reconstruct_parameter_state(events)
                fit_range_state = _hr.reconstruct_fit_range_state(events)
                setup_state = _hr.reconstruct_setup_state(events)
                model_state = _hr.reconstruct_model_state(events)
            
            _hr.sync_domain_entities(nav_state, all_events)

            link_touched = _hr.touched_parameter_keys(
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
            self._apply_model_state(model_state)
            try:
                self.update()
            except Exception:
                pass
        except Exception:
            pass
        self._sync_history_navigation_actions()

    def _select_dataset_by_identity(self: Main, names: typing.List[str], dataset_uid: str = "") -> None:
        try:
            target = [str(n) for n in names if n]
            target_uid = str(dataset_uid or "")
            if not target and not target_uid:
                return
            datasets = list(getattr(cs, "imported_datasets", []))
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

    def _select_fit_by_identity(self: Main, fit_name: str = "", fit_uid: str = "") -> None:
        try:
            for idx, fit_group in enumerate(getattr(cs, "fits", [])):
                current_uid = str(getattr(fit_group, "unique_identifier", ""))
                name = str(getattr(fit_group, "name", ""))
                if fit_uid:
                    if current_uid != fit_uid:
                        continue
                elif name != fit_name:
                    continue
                try:
                    self.fit_selector.selected_fit_index = idx
                except Exception:
                    pass
                try:
                    self.current_fit = fit_group
                    self._fit_idx = idx
                    setattr(cs, "current_fit", fit_group)
                    setattr(cs, "current_fit_idx", idx)
                except Exception:
                    pass
                activated = False
                try:
                    for fit_window in getattr(cs.gui, "fit_windows", []):
                        if getattr(fit_window, "fit", None) is fit_group:
                            try:
                                fit_window.show()
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
                return
        except Exception:
            pass

class DevMixin:
    def _init_system_info_watermark(self: Main) -> None:
        try:
            parent = getattr(self, "mdiarea", None)
        except Exception:
            parent = None
        label = getattr(self, "_system_info_watermark", None)
        self._system_info_watermark = misc_helpers.init_system_info_watermark(parent, label)

    def _update_system_info_watermark_geometry(self: Main) -> None:
        label = getattr(self, "_system_info_watermark", None)
        misc_helpers.update_system_info_watermark_geometry(label)

    def _install_dev_mode_code_badges(self: Main) -> None:
        """Install code badge buttons on docks and key widgets when dev mode is enabled."""
        if not cs.core.settings.is_dev_mode():
            return

        try:
            from chisurf.gui.widgets.code_badge import (
                install_code_badge,
                make_widget_source_resolver,
            )
        except ImportError:
            cs.logging.debug("Code badge module not available")
            return

        dev_settings = cs.core.settings.dev_mode_settings()
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

    def _install_parameter_badges(self: Main) -> None:
        """Install code badges on parameter group widgets."""
        try:
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import (
                make_widget_resolver,
            )
        except ImportError:
            return

        try:
            for fit in cs.fits:
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

    def _install_experiment_panel_badges(self: Main) -> None:
        """Install code badges on experiment panel widgets."""
        try:
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import (
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

    def _on_mdi_window_activated_for_code_badge(self: Main, sub_window) -> None:
        """Install code badge on newly activated MDI windows."""
        if not cs.core.settings.is_dev_mode():
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


    def _init_developer_menu(self: Main) -> None:
        """Initialize the Developer menu with dev mode tools."""
        if not cs.core.settings.is_dev_mode():
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
            cs.logging.debug(f"Could not initialize Developer menu: {e}")

    def _on_open_source_for_focus(self: Main) -> None:
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

    def _on_refresh_code_badges(self: Main) -> None:
        """Refresh all code badges visibility."""
        try:
            from chisurf.gui.widgets.code_badge import get_badge_manager
            get_badge_manager().refresh_all()
            self._install_dev_mode_code_badges()
        except ImportError:
            pass

    def _on_open_dev_settings(self: Main) -> None:
        """Open dev mode settings dialog."""
        try:
            from chisurf.gui.widgets.settings_editor import SettingsEditor
            if not hasattr(self, '_dev_settings_editor') or self._dev_settings_editor is None:
                self._dev_settings_editor = SettingsEditor(
                    filename=cs.core.settings.chisurf_settings_file,
                    window_title="Dev Mode Settings"
                )
            self._dev_settings_editor.show()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Dev Mode Settings",
                f"Could not open settings: {e}",
            )
