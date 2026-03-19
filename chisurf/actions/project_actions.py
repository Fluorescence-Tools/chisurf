from __future__ import annotations
from chisurf import typing
from chisurf.runtime.action_decorator import action


@action("experiment.set", schema={"name": str})
def set_experiment(name: str):
    """Switch the current experiment type and refresh the GUI."""
    import chisurf
    cs = getattr(chisurf, "cs", None)
    if cs is None:
        return {}
    # Update combo selection without triggering a dispatch loop
    combo = getattr(cs, "comboBox_experimentSelect", None)
    if combo is not None:
        idx = combo.findText(name)
        if idx != -1 and combo.currentIndex() != idx:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            cs._current_experiment_idx = idx
    # Refresh GUI directly — no re-dispatch
    if hasattr(cs, "_refresh_experiment_ui"):
        cs._refresh_experiment_ui()
    return {}


@action("setup.select", schema={"name": str})
def select_setup(name: str):
    """Select a setup configuration and refresh the GUI."""
    import chisurf
    cs = getattr(chisurf, "cs", None)
    if cs is None:
        return {}
    # Find the setup index from the current experiment's readers
    try:
        readers = cs.current_experiment.readers
        for j, s in enumerate(readers):
            if s.name == name:
                cs._current_setup_idx = j
                combo = getattr(cs, "comboBox_setupSelect", None)
                if combo is not None:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(j)
                    combo.blockSignals(False)
                break
    except Exception:
        pass
    # Refresh GUI directly — no re-dispatch
    if hasattr(cs, "_refresh_setup_ui"):
        cs._refresh_setup_ui()
    return {}


@action("setup.params.set", schema={"params": dict})
def set_setup_params(params: typing.Dict[str, typing.Any]):
    """Set parameters for the current setup."""
    import chisurf
    setup = chisurf.cs.current_setup
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
    """Close the current project and reinitialize UI."""
    import chisurf
    if main_window:
        main_window.reinitialize()
        main_window._current_project_dir = None
    return {}


@action("action.catalog.export")
def export_action_catalog(target_path: str, file_type: str = "yaml"):
    """Export the list of available actions to a file."""
    from chisurf.runtime.actions import get_action_catalog
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
    import chisurf
    return chisurf.run(command)
