from __future__ import annotations

import sys

from qtpy import QtWidgets, QtGui, QtCore

def setup_gui(
        app: QtWidgets.QApplication,
        stage: str = None
) -> chisurf.gui.main.Main:

    def gui_imports():
        import pyqtgraph
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
        import chisurf.settings
        import chisurf.structure

    def setup_ipython():
        import chisurf.gui.widgets
        chisurf.console = chisurf.gui.widgets.QIPythonWidget()

    def startup_interface():
        from chisurf.gui.main import Main
        import chisurf
        window = Main()
        chisurf.console.history_widget = window.plainTextEditHistory
        chisurf.cs = window
        window.init_setups()
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


def get_app() -> QtWidgets.QApplication:
    import chisurf.gui.resources

    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()

    pixmap = QtGui.QPixmap(":/icons/icons/cs_logo.png")
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    app.processEvents()

    splash.showMessage(
        "Loading modules",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        stage="gui_imports"
    )

    splash.showMessage(
        "Setup ipython",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        stage="setup_ipython"
    )

    splash.showMessage(
        "Starting interface",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    window = setup_gui(
        app=app,
        stage="startup_interface"
    )

    splash.showMessage(
        "Setup style",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        stage="setup_style"
    )
    window.show()
    splash.finish(window)
    return app

