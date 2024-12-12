from __future__ import annotations

import sys
import subprocess
import pathlib

from functools import partial
import pkgutil
import importlib

from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from qtpy import QtWidgets, QtGui, QtCore, uic

import chisurf.settings
import chisurf.gui.decorators


class CustomProgressBar(QtWidgets.QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)  # Ensure the default text is centered
        self.custom_text = ""  # Placeholder for custom text

    def set_custom_text(self, text: str):
        """Set custom text to display on the progress bar."""
        self.custom_text = text
        self.update()  # Request a repaint to update the display

    def paintEvent(self, event):
        """Custom paint event to render the progress bar and custom text on top."""
        # Do not call the default paint event to avoid drawing the percentage text
        painter = QtGui.QPainter(self)

        # Customize the progress bar appearance if needed (e.g., color, border, etc.)
        # painter.setPen(QtCore.Qt.green)  # Example for setting color
        # painter.setBrush(QtCore.Qt.blue)  # Example for setting fill color

        # Draw the progress bar manually
        rect = self.rect()
        progress = self.value() / self.maximum()  # Calculate the progress percentage
        progress_width = int(rect.width() * progress)  # Width based on progress
        progress_rect = QtCore.QRect(rect.x(), rect.y(), progress_width, rect.height())
        painter.fillRect(progress_rect, QtCore.Qt.green)  # Fill with desired color

        # Customize the text style and color
        painter.setPen(QtCore.Qt.white)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        # Draw the custom text on top of the progress bar
        painter.drawText(rect, QtCore.Qt.AlignCenter, self.custom_text)

        painter.end()


class SplashScreen(QtWidgets.QSplashScreen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use the CustomProgressBar to show text on top of the progress bar
        self.progress_bar = CustomProgressBar(self)
        self.progress_bar.setGeometry(130, self.height() - 40, self.width() - 300, 5)
        self.progress_bar.setRange(0, 100)  # Progress bar range 0 to 100
        self.progress_bar.setValue(0)  # Initial value

        # Initialize message attributes
        self.current_message = ""
        self.message_color = QtCore.Qt.white  # Default text color is white

    def update_progress(self, value):
        """Update progress bar value."""
        self.progress_bar.setValue(value)

    def update_message(self, message: str):
        """Update the message displayed on the splash screen."""
        self.current_message = message
        self.showMessage(
            self.current_message,
            QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
            self.message_color
        )
        self.repaint()  # Ensure the message is updated immediately

    def drawContents(self, painter):
        """Override the drawContents method to ensure text is drawn."""
        painter.setPen(self.message_color)

        # Get the geometry of the progress bar
        progress_bar_rect = self.progress_bar.geometry()

        # Calculate the position to draw the message above the progress bar
        message_y = progress_bar_rect.top() + 10  # Adjust as necessary to move up from the progress bar
        message_rect = self.rect().adjusted(0, 0, 0, 0)  # Full rect for alignment

        # Draw the message centered above the progress bar
        painter.drawText(message_rect.adjusted(0, message_y, 0, 0),
                         QtCore.Qt.AlignHCenter,
                         self.current_message)


def setup_gui(
        app: QtWidgets.QApplication,
        window: chisurf.gui.main.Main = None,
        stage: str = None
):
    def gui_imports():
        import chisurf.settings
        import chisurf.base
        import chisurf.common
        import chisurf.curve
        import chisurf.decorators
        import chisurf.parameter
        import chisurf.experiments
        import chisurf.fio
        import chisurf.fitting
        import chisurf.fluorescence
        import chisurf.gui.decorators
        import chisurf.gui.widgets.ipython
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

    def start_jupyter(
                      notebook_executable="jupyter-notebook",
                      port=8888,
                      directory=pathlib.Path().home()
    ):
        return subprocess.Popen([notebook_executable,
                                 "--port=%s" % port,
                                 "--browser=n",
                                 "--NotebookApp.token=''",  # Disable token
                                 "--NotebookApp.password=''",  # Disable password
                                 "--NotebookApp.disable_check_xsrf=True",
                                 # Disable cross-site request forgery protection (optional)
                                 "--notebook-dir=%s" % directory],
                                bufsize=1, stderr=subprocess.PIPE
                            )

    def populate_plugins():
        plugin_menu = window.menuBar.addMenu('Plugins')
        plugin_path = pathlib.Path(chisurf.plugins.__file__).absolute().parent
        for _, module_name, _ in pkgutil.iter_modules(chisurf.plugins.__path__):
            module_path = "chisurf.plugins." + module_name
            module = importlib.import_module(str(module_path))
            try:
                name = module.name
            except AttributeError as e:
                print(f"Failed to find plugin name: {e}")
                name = module_name
            p = partial(
                window.onRunMacro, plugin_path / module_name / "wizard.py",
                executor='exec',
                globals={'__name__': 'plugin'}
            )
            plugin_action = QtWidgets.QAction(f"{name}", window)
            plugin_action.triggered.connect(p)
            plugin_menu.addAction(plugin_action)

    def populate_notebooks():
        notebook_menu = window.menuBar.addMenu('Notebooks')

        home_dir = pathlib.Path.home()
        chisurf_path = pathlib.Path(chisurf.__file__).parent
        plugin_path = pathlib.Path(chisurf.plugins.browser.__file__).absolute().parent

        def add_notebook(notebook_file, base_addr='/notebooks/'):
            adr = str(chisurf.__jupyter_address__ + base_addr) + str(notebook_file.relative_to(home_dir))
            p = partial(
                window.onRunMacro, plugin_path / "wizard.py",
                executor='exec',
                globals={'__name__': 'plugin', 'adr': adr}
            )
            try:
                if notebook_file.exists():
                    menu_text = notebook_file.stem
                    action = QtWidgets.QAction(f"{menu_text}", window)
                else:
                    return
            except AttributeError as e:
                action = QtWidgets.QAction(f"{notebook_file}", window)
            action.triggered.connect(p)
            notebook_menu.addAction(action)

        # http://localhost:8888/tree
        add_notebook(pathlib.Path.home(), '/tree')

        notebook_path = chisurf.settings.cs_settings.get('notebook_path', chisurf_path / 'notebooks')
        for notebook_file in notebook_path.glob("*.ipynb"):
            add_notebook(notebook_file)


    if stage is None:
        gui_imports()
        setup_ipython()
        startup_interface()
        setup_style(app=app)
    elif stage == "gui_imports":
        gui_imports()
    elif stage == "setup_ipython":
        setup_ipython()
    elif stage == "setup_style":
        setup_style(app=app)
    elif stage == "populate_plugins":
        populate_plugins()
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
    elif stage == "start_jupyter":
        # start jupyter notebook and wait for line with the web address
        chisurf.logging.info("Starting Jupyter notebook process")
        chisurf.__jupyter_process__ = start_jupyter()
        while chisurf.__jupyter_address__ is None:
            line = str(chisurf.__jupyter_process__.stderr.readline())
            chisurf.logging.info(line)
            if "http://" in line:
                start = line.find("http://")
                end = line.find("/", start + len("http://"))
                chisurf.__jupyter_address__  = line[start:end]
        chisurf.logging.info("Server found at %s, migrating monitoring to listener thread" % chisurf.__jupyter_address__)
    elif stage == "populate_notebooks":
        chisurf.logging.info("Looking for ipynb in home folder")
        populate_notebooks()

def get_win(app: QtWidgets.QApplication) -> chisurf.gui.main.Main:
    import pyqtgraph
    import chisurf.gui.resources

    pixmap = QtGui.QPixmap(":/images/icons/splashscreen.png")
    splash = SplashScreen(pixmap)

    # move splashscreen to center of active window
    screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor().pos())
    fg = splash.frameGeometry()
    fg.moveCenter(screen.geometry().center())
    splash.move(fg.topLeft())

    getattr(splash, "raise")()
    splash.activateWindow()

    splash.setContentsMargins(0, 0, 0, 64)
    splash.show()
    app.processEvents()

    # Update progress as the setup progresses
    stages = [
        ("Loading modules", "gui_imports", 10),
        ("Setup ipython", "setup_ipython", 30),
        ("Starting interface", "startup_interface", 40),
        ("Initialize setups", "init_setups", 50),
        ("Defining actions", "define_actions", 55),
        ("Arrange widgets", "arrange_widgets", 65),
        ("Loading tools", "load_tools", 70),
        ("Initializing Jupyter", "start_jupyter", 85),
        ("Populate plugins", "populate_plugins", 90),
        ("Populate notebook", "populate_notebooks", 95),
        ("Styling up", "setup_style", 100),
    ]

    window = None
    for message, stage, progress_value in stages:
        splash.update_message(message)
        splash.update_progress(progress_value)
        app.processEvents()
        w2 = setup_gui(app=app, stage=stage, window=window)
        if w2 is not None:
            window = w2

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


fit_windows = list()
