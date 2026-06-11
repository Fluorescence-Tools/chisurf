from __future__ import annotations
from chisurf import typing
from chisurf.core.actions._decorator import action
import chisurf as cs


@action("experiment.set", schema={"name": str})
def set_experiment(name: str):
    """Switch the current experiment type and refresh the GUI."""
    gui = getattr(cs, "cs", None)
    if gui is None:
        return {}
    # Update combo selection without triggering a dispatch loop
    combo = getattr(gui, "comboBox_experimentSelect", None)
    if combo is not None:
        idx = combo.findText(name)
        if idx != -1 and combo.currentIndex() != idx:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            gui._current_experiment_idx = idx
    # Refresh GUI directly — no re-dispatch
    if hasattr(gui, "_refresh_experiment_ui"):
        gui._refresh_experiment_ui()
    return {}


@action("setup.select", schema={"name": str})
def select_setup(name: str):
    """Select a setup configuration and refresh the GUI."""
    gui = getattr(cs, "cs", None)
    if gui is None:
        return {}
    # Find the setup index from the current experiment's readers
    try:
        readers = gui.current_experiment.readers
        for j, s in enumerate(readers):
            if s.name == name:
                gui._current_setup_idx = j
                combo = getattr(gui, "comboBox_setupSelect", None)
                if combo is not None:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(j)
                    combo.blockSignals(False)
                break
    except Exception:
        pass
    # Refresh GUI directly — no re-dispatch
    if hasattr(gui, "_refresh_setup_ui"):
        gui._refresh_setup_ui()
    return {}


@action("setup.params.set", schema={"params": dict})
def set_setup_params(params: typing.Dict[str, typing.Any]):
    """Set parameters for the current setup."""
    setup = cs.cs.current_setup
    for key, value in params.items():
        if "." in key:
            parts = key.split(".")
            obj = setup
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(setup, key, value)
    return {}


@action("project.save", schema={"project_name": str})
def save_project(target_path: str, project_name: str):
    """Save the overall project."""
    from chisurf.macros import core_fit
    return core_fit.save_project(target_path=target_path, project_name=project_name)


@action("project.load", schema={"project_path": str})
def load_project(project_path: str):
    """Load a project from directory."""
    from chisurf.macros import core_fit
    return core_fit.load_project(project_path=project_path)


@action("project.close")
def close_project(main_window: typing.Any = None):
    """Close the current project."""
    if main_window:
        # Instead of calling reinitialize() (which might trigger confirmation
        # or another project.close dispatch), we perform a focused cleanup.
        try:
            # Close all fits
            if hasattr(main_window, 'onCloseAllFits'):
                main_window.onCloseAllFits()

            # Clear imported datasets
            if hasattr(cs, 'imported_datasets'):
                cs.imported_datasets.clear()
                from chisurf.macros.core_data import restore_global_fit_dataset
                try:
                    restore_global_fit_dataset(_from_controller=True, update_ui=False)
                except Exception:
                    pass

            # Reset project path
            main_window._current_project_dir = None

            # Refresh UI selectors
            if hasattr(main_window, 'dataset_selector'):
                main_window.dataset_selector.update()
            if hasattr(main_window, 'fit_selector'):
                main_window.fit_selector.update()
        except Exception as e:
            import chisurf.logging
            cs.logging.error(f"Error in project.close action: {e}")

    return {}


@action("action.catalog.export")
def export_action_catalog(target_path: str, file_type: str = "yaml"):
    """Export the list of available actions to a file."""
    from chisurf.core.actions._infra import get_action_catalog
    import yaml
    catalog = get_action_catalog()
    with open(target_path, "w") as f:
        yaml.dump(catalog, f)
    return {"target_path": target_path}


@action("app.reinitialize.start")
def app_reinitialize_start():
    """Signify start of application reinitialization."""
    return {}


@action("app.reinitialize.finish")
def app_reinitialize_finish():
    """Signify finish of application reinitialization."""
    return {}


@action("run_command", replayable=False, side_effect_class="diagnostic")
def run_command(command: str):
    """Run a shell or macro command."""
    return cs.run(command)
