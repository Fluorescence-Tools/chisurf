"""GUI startup service entrypoints for the ChiSurf app lifecycle.

These functions are referenced by JSON config files in
``chisurf/startup/services.d/`` and receive an ``AppStartupContext``.
"""

from __future__ import annotations

import chisurf as cs


def _get_window(context):
    """Return the main window from context dependencies."""
    return context.dependencies.get("startup_interface")


def gui_imports(context) -> None:
    """Import core GUI modules needed for the main window scaffold."""
    import chisurf.core.settings
    import chisurf.core.base
    import chisurf.core.common
    import chisurf.core.curve
    import chisurf.core.decorators
    import chisurf.core.parameter
    import chisurf.core.experiments
    import chisurf.core.fio
    import chisurf.gui.decorators
    import chisurf.gui.widgets.ipython
    import chisurf.gui.widgets
    import chisurf.macros
    import chisurf.core.math
    if cs.core.settings.exceptions_on_gui:
        import chisurf.gui.exception_hook


def setup_ipython(context) -> None:
    """Create the IPython console widget."""
    import chisurf.gui.widgets
    cs.console = cs.gui.widgets.ipython.QIPythonWidget()
    cs.console.history_widget = None


def startup_interface(context) -> object:
    """Create the main window and return it via context.dependencies."""
    from chisurf.gui.main import Main
    window = Main()
    cs.cs = window
    import chisurf.core.base
    cs.core.base.set_safe_import_notify(
        lambda title, text: cs.gui.QtWidgets.QMessageBox.information(
            window, title, text, cs.gui.QtWidgets.QMessageBox.Ok
        )
    )
    return window


def setup_logging(context) -> None:
    """Attach logging widgets to the main window status bar."""
    from chisurf.gui import setup_logging_widgets
    window = _get_window(context)
    if window is not None:
        setup_logging_widgets(window)


def init_setups(context) -> None:
    """Initialize detector setups on the main window."""
    window = _get_window(context)
    if window is not None:
        window.init_setups()


def restore_setup_defaults(context) -> None:
    """Restore saved setup defaults."""
    window = _get_window(context)
    if window is not None:
        window._restore_setup_defaults()


def define_actions(context) -> None:
    """Define actions on the main window."""
    window = _get_window(context)
    if window is not None:
        window.define_actions()


def load_tools(context) -> None:
    """Load tools on the main window."""
    window = _get_window(context)
    if window is not None:
        window.load_tools()


def init_executors(context) -> None:
    """Initialize GUI executors."""
    try:
        from chisurf.gui import initialize_gui_executors
        initialize_gui_executors()
    except Exception:
        pass


def arrange_widgets(context) -> None:
    """Arrange widgets on the main window."""
    window = _get_window(context)
    if window is not None:
        window.arrange_widgets()


def setup_style(context) -> None:
    """Apply the configured stylesheet."""
    from chisurf.gui import set_app_style
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is not None:
        set_app_style(app)
    from chisurf.gui import setup_gui
    qt_app = QtWidgets.QApplication.instance()
    if qt_app is not None:
        try:
            setup_gui(app=qt_app, stage="setup_style")
        except Exception:
            pass


def deferred_gui_imports(context) -> None:
    """Import heavy functional submodules."""
    import chisurf.core.fitting
    import chisurf.core.fluorescence
    import chisurf.core.models
    import chisurf.gui.plots
    import chisurf.core.structure


def populate_plugins(context) -> None:
    """Populate the plugin menu."""
    from chisurf.gui import setup_gui
    window = _get_window(context)
    if window is not None:
        from qtpy import QtWidgets
        try:
            setup_gui(app=QtWidgets.QApplication.instance(), stage="populate_plugins", window=window)
        except Exception:
            pass


def check_updates(context) -> None:
    """Check for application updates."""
    from chisurf.gui import setup_gui
    window = _get_window(context)
    if window is not None:
        from qtpy import QtWidgets
        try:
            setup_gui(app=QtWidgets.QApplication.instance(), stage="check_updates", window=window)
        except Exception:
            pass


def start_jupyter(context) -> None:
    """Start the Jupyter notebook server.

    The ``enabled_if`` gate in the config prevents this function from
    being called when Jupyter is disabled in settings.
    """
    from chisurf.gui import setup_gui, launch_jupyter_process
    window = _get_window(context)
    if window is not None:
        from qtpy import QtWidgets
        try:
            setup_gui(app=QtWidgets.QApplication.instance(), stage="start_jupyter", window=window)
        except Exception:
            pass


def populate_notebooks(context) -> None:
    """Populate the notebooks menu.

    This service depends on ``gui.start_jupyter`` and is automatically
    skipped when Jupyter startup is disabled.
    """
    from chisurf.gui import setup_gui
    window = _get_window(context)
    if window is not None:
        from qtpy import QtWidgets
        try:
            setup_gui(app=QtWidgets.QApplication.instance(), stage="populate_notebooks", window=window)
        except Exception:
            pass


def warmup_imports(context) -> None:
    """Preload modules for a snappier first interaction."""
    try:
        from chisurf.gui.misc_helpers import warmup_imports as _warmup
        _warmup()
    except Exception:
        pass
