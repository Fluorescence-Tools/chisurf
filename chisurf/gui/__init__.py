from __future__ import annotations

import sys

from qtpy import QtWidgets, QtGui, QtCore


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
        "Initialize setups",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="init_setups"
    )

    splash.showMessage(
        "Defining actions",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="define_actions"
    )

    splash.showMessage(
        "Arrange widgets",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="arrange_widgets"
    )

    splash.showMessage(
        "Loading tools",
        alignment=QtCore.Qt.AlignTop,
        color=QtCore.Qt.white
    )
    app.processEvents()
    setup_gui(
        app=app,
        window=window,
        stage="load_tools"
    )

    splash.showMessage(
        "Styling up",
        alignment=QtCore.Qt.AlignTop,
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
    get_win(
        app=app
    )
    return app

