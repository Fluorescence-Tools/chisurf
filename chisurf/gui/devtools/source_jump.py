"""
Source Jump Utilities for Dev Mode.

Provides functions to resolve source locations from widgets/objects
and open them in the embedded code editor.
"""
from __future__ import annotations

import inspect
import pathlib
import typing
from typing import Optional, Tuple, Any, Callable

from chisurf.gui import QtWidgets

import chisurf.core.settings


def resolve_widget_source(widget: QtWidgets.QWidget) -> Optional[Tuple[str, int]]:
    """Resolve the source file and line for a widget.

    Priority:
    1. Widget class source via inspect (Python source preferred)
    2. Widget's _chisurf_ui_path attribute (for @init_with_ui widgets, fallback)
    3. Focused child widget under cursor (if any)

    Args:
        widget: The widget to resolve source for

    Returns:
        Tuple of (file_path, line_number) or None if not resolvable
    """
    if widget is None:
        return None

    # Prefer Python source over .ui files
    try:
        widget_class = widget.__class__
        file_path = inspect.getsourcefile(widget_class)
        if file_path and not file_path.endswith('.pyc'):
            try:
                _, start_line = inspect.getsourcelines(widget_class)
                return (file_path, start_line)
            except (TypeError, OSError):
                return (file_path, 1)
    except (TypeError, OSError):
        pass

    # Fallback to .ui file if Python source not available
    ui_path = getattr(widget, "_chisurf_ui_path", None)
    if ui_path:
        path = pathlib.Path(ui_path)
        if path.exists():
            return (str(path), 1)

    return None


def resolve_object_source(obj: Any) -> Optional[Tuple[str, int]]:
    """Resolve the source file and line for any Python object.

    Args:
        obj: The object to resolve source for

    Returns:
        Tuple of (file_path, line_number) or None if not resolvable
    """
    if obj is None:
        return None

    try:
        obj_class = obj.__class__
        file_path = inspect.getsourcefile(obj_class)
        if file_path:
            try:
                _, start_line = inspect.getsourcelines(obj_class)
                return (file_path, start_line)
            except (TypeError, OSError):
                return (file_path, 1)
    except (TypeError, OSError):
        pass

    return None


def resolve_focused_widget_source() -> Optional[Tuple[str, int]]:
    """Resolve source for the currently focused widget.

    Returns:
        Tuple of (file_path, line_number) or None if not resolvable
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None

    focused = app.focusWidget()
    if focused is None:
        return None

    return resolve_widget_source(focused)


def resolve_fit_window_source(fit_window: QtWidgets.QWidget) -> Optional[Tuple[str, int]]:
    """Resolve source for a fit window (MDI subwindow).

    Priority:
    1. Active plot widget class
    2. Model class (if available)
    3. FitSubWindow class

    Args:
        fit_window: The fit window widget

    Returns:
        Tuple of (file_path, line_number) or None if not resolvable
    """
    if fit_window is None:
        return None

    plot_widget = getattr(fit_window, "plot_widget", None)
    if plot_widget is not None:
        result = resolve_widget_source(plot_widget)
        if result:
            return result

    model = getattr(fit_window, "model", None)
    if model is not None:
        result = resolve_object_source(model)
        if result:
            return result

    return resolve_widget_source(fit_window)


def resolve_parameter_group_source(
    param_widget: QtWidgets.QWidget,
) -> Optional[Tuple[str, int]]:
    """Resolve source for a parameter group widget.

    Priority:
    1. Parameter widget class
    2. Associated fit/model (if available)

    Args:
        param_widget: The parameter widget

    Returns:
        Tuple of (file_path, line_number) or None if not resolvable
    """
    if param_widget is None:
        return None

    result = resolve_widget_source(param_widget)
    if result:
        return result

    fitting_parameter = getattr(param_widget, "fitting_parameter", None)
    if fitting_parameter is not None:
        result = resolve_object_source(fitting_parameter)
        if result:
            return result

    fit = getattr(param_widget, "fit", None)
    if fit is not None:
        model = getattr(fit, "model", None)
        if model is not None:
            result = resolve_object_source(model)
            if result:
                return result

    return None


def resolve_experiment_panel_source(
    experiment_widget: QtWidgets.QWidget,
) -> Optional[Tuple[str, int]]:
    """Resolve source for an experiment panel widget.

    Args:
        experiment_widget: The experiment widget

    Returns:
        Tuple of (file_path, line_number) or None if not resolvable
    """
    if experiment_widget is None:
        return None

    result = resolve_widget_source(experiment_widget)
    if result:
        return result

    setup = getattr(experiment_widget, "setup", None)
    if setup is not None:
        result = resolve_object_source(setup)
        if result:
            return result

    experiment = getattr(experiment_widget, "experiment", None)
    if experiment is not None:
        result = resolve_object_source(experiment)
        if result:
            return result

    return None


def open_in_editor(
    main_window: QtWidgets.QMainWindow,
    path: str,
    line: Optional[int] = None,
) -> bool:
    """Open a file in the standalone CodeEditorWindow plugin.

    Args:
        main_window: The main ChiSurf window
        path: Path to the file to open
        line: Optional line number to scroll to

    Returns:
        True if successful, False otherwise
    """
    if main_window is None:
        import logging
        logging.error("open_in_editor: main_window is None")
        return False

    try:
        from chisurf.plugins.core.code_editor.window import CodeEditorWindow

        # Check if an instance of CodeEditorWindow is already open
        editor_window = None
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(widget, CodeEditorWindow):
                editor_window = widget
                break

        if editor_window is None:
            # Instantiate and display the plugin's code editor window
            editor_window = CodeEditorWindow()
            # Apply statefulness from manifest if possible
            try:
                from chisurf.core.plugin.registry import apply_manifest_statefulness
                from chisurf.core.plugin.manifest import load_manifest
                import pathlib
                import chisurf.plugins.core.code_editor as ce
                manifest_path = pathlib.Path(ce.__file__).parent / "manifest.json"
                if manifest_path.exists():
                    manifest = load_manifest(manifest_path)
                    if manifest:
                        apply_manifest_statefulness(editor_window, manifest)
            except Exception as e:
                import logging
                logging.warning(f"Could not apply code editor statefulness: {e}")

            # Store reference in main_window._plugin_contexts to prevent garbage collection
            if not hasattr(main_window, "_plugin_contexts"):
                main_window._plugin_contexts = {}
            import chisurf.plugins.core.code_editor as ce
            import pathlib
            plugin_key = str(pathlib.Path(ce.__file__).parent)
            if plugin_key not in main_window._plugin_contexts:
                main_window._plugin_contexts[plugin_key] = {"__name__": "plugin"}
            main_window._plugin_contexts[plugin_key]["window"] = editor_window

            editor_window.show()

        editor_window.raise_()
        editor_window.activateWindow()

        if hasattr(editor_window, "editor") and editor_window.editor is not None:
            editor_window.editor.open_file(path, line=line)
            return True
        else:
            import logging
            logging.error("open_in_editor: CodeEditorWindow has no editor widget")
            return False

    except Exception as e:
        import logging
        logging.error(f"open_in_editor failed: {e}")
        return False


def make_resolver(
    *targets: Any,
) -> Callable[[], Optional[Tuple[str, int]]]:
    """Create a resolver that tries multiple targets in order.

    Args:
        *targets: Objects or widgets to try resolving source for

    Returns:
        A callable that returns (path, line) or None
    """
    def resolver() -> Optional[Tuple[str, int]]:
        for target in targets:
            if isinstance(target, QtWidgets.QWidget):
                result = resolve_widget_source(target)
            else:
                result = resolve_object_source(target)
            if result:
                return result
        return None

    return resolver


def make_widget_resolver(
    widget: QtWidgets.QWidget,
    include_ui: bool = True,
) -> Callable[[], Optional[Tuple[str, int]]]:
    """Create a resolver for a widget.

    Args:
        widget: The widget to resolve
        include_ui: Whether to check for _chisurf_ui_path first

    Returns:
        A callable that returns (path, line) or None
    """
    def resolver() -> Optional[Tuple[str, int]]:
        if include_ui:
            ui_path = getattr(widget, "_chisurf_ui_path", None)
            if ui_path:
                path = pathlib.Path(ui_path)
                if path.exists():
                    return (str(path), 1)
        return resolve_widget_source(widget)

    return resolver
