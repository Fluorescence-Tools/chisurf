from __future__ import annotations

import sys

from qtpy import QtWidgets, QtGui, QtCore

from chisurf import typing
import chisurf.settings


def setup_gui(
        app: QtWidgets.QApplication,
        window: chisurf.gui.main.Main = None,
        stage: str = None
) -> chisurf.gui.main.Main:

    def gui_imports():
        import chisurf.settings
        import chisurf.gui.widgets.ipython
        import chisurf.gui.decorators
        import chisurf.base
        import chisurf.common
        import chisurf.curve
        import chisurf.decorators
        import chisurf.parameter
        import chisurf.experiments
        import chisurf.fio
        import chisurf.fitting
        import chisurf.fluorescence
        import chisurf.gui.tools
        import chisurf.gui.widgets
        import chisurf.macros
        import chisurf.math
        import chisurf.models
        import chisurf.plots
        import chisurf.structure
        if chisurf.settings.exceptions_on_gui:
            import chisurf.gui.exception_hook

    def setup_ipython():
        import chisurf.gui.widgets
        chisurf.console = chisurf.gui.widgets.ipython.QIPythonWidget()

    def startup_interface():
        from chisurf.gui.main import Main
        import chisurf
        window = Main()
        chisurf.console.history_widget = window.plainTextEditHistory
        chisurf.cs = window
        return window

    def setup_style(app):
        import pathlib
        import chisurf
        app.setStyleSheet(
            open(
                pathlib.Path(chisurf.__file__).parent /
                "gui/styles/" /
                chisurf.settings.cs_settings['gui']['style_sheet'],
                mode='r'
            ).read()
        )

    if stage is None:
        gui_imports()
        setup_ipython()
        startup_interface()
        setup_style(
            app=app
        )
    elif stage == "gui_imports":
        gui_imports()
    elif stage == "setup_ipython":
        setup_ipython()
    elif stage == "setup_style":
        setup_style(
            app=app
        )
    elif stage == "startup_interface":
        return startup_interface()
    elif stage == "define_actions":
        window.define_actions()
    elif stage == "arrange_widgets":
        window.arrange_widgets()
    elif stage == "init_setups":
        window.init_setups()
    elif stage == "load_tools":
        window.load_tools()


def get_win(
        app: QtWidgets.QApplication
) -> chisurf.gui.main.Main:
    # import pyqtgraph at this stage to fix
    # Warning: QApplication was created before pyqtgraph was imported;
    import pyqtgraph
    import chisurf.gui.resources

    pixmap = QtGui.QPixmap(":/images/icons/splashscreen.png")
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.setContentsMargins(64, 0, 0, 64)
    splash.show()
    app.processEvents()
    align = QtCore.Qt.AlignTop
    offset = "\n" * 25 + " " * 5
    splash.showMessage(
        offset+"Loading modules",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        stage="gui_imports"
    )

    splash.showMessage(
        offset+"Setup ipython",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        stage="setup_ipython"
    )

    splash.showMessage(
        offset+"Starting interface",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    window = setup_gui(
        app=app,
        stage="startup_interface"
    )

    splash.showMessage(
        offset+"Initialize setups",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="init_setups"
    )

    splash.showMessage(
        offset+"Defining actions",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="define_actions"
    )

    splash.showMessage(
        offset+"Arrange widgets",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="arrange_widgets"
    )

    splash.showMessage(
        offset+"Loading tools",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="load_tools"
    )

    splash.showMessage(
        offset+"Styling up",
        alignment=align,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        stage="setup_style"
    )
    window.show()
    splash.hide()
    splash.finish(window)
    return window


def get_app():
    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()
    win = get_win(app=app)
    win.raise_()
    win.activateWindow()
    win.setFocus()
    return app

