from __future__ import annotations

import sys
import subprocess
import pathlib
import os
import signal
import threading
import time
import atexit

from functools import partial
import pkgutil
import importlib

#import os
#os.environ['QT_OPENGL'] = 'software'  # Use software rendering

from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from qtpy import QtWidgets, QtGui, QtCore, uic

import chisurf.settings
from chisurf import logging
import chisurf.gui.decorators


def launch_jupyter_process(
    notebook_executable="jupyter-notebook",
    port=8888,
    directory: pathlib.Path = pathlib.Path().home()
):
    """
    Launch Jupyter Notebook with a watchdog that kills it when this process dies.
    Cross-platform: uses os.killpg on Unix, CREATE_NEW_PROCESS_GROUP on Windows.
    """
    jupyter_cmd = [
        sys.executable, "-m", "notebook",
        f"--port={port}",
        "--no-browser",
        "--NotebookApp.token=''",
        "--NotebookApp.password=''",
        "--NotebookApp.disable_check_xsrf=True",
        f"--notebook-dir={directory}"
    ]

    # On Windows, put Jupyter into its own process group
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    start_new_session = (sys.platform != "win32")

    # Capture stdout (URL) and merge stderr
    jupyter_proc = subprocess.Popen(
        jupyter_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
        start_new_session=start_new_session
    )

    def terminate_jupyter():
        """Cleanly kill the notebook process (and its group)."""
        try:
            if sys.platform == "win32":
                # send CTRL_BREAK to the group, then kill if still alive
                jupyter_proc.send_signal(signal.CTRL_BREAK_EVENT)
                jupyter_proc.kill()
            else:
                os.killpg(jupyter_proc.pid, signal.SIGKILL)
        except Exception:
            pass

    def watchdog_unix():
        """Only on Unix: if parent vanishes, kill the notebook group."""
        parent_pid = os.getpid()
        while True:
            time.sleep(2)
            try:
                os.kill(parent_pid, 0)
            except OSError:
                terminate_jupyter()
                break

    # Register for normal shutdown
    atexit.register(terminate_jupyter)

    # Start a watcher **only on Unix**, since on Windows os.kill(pid,0) will terminate PID=0
    if sys.platform != "win32":
        thread = threading.Thread(target=watchdog_unix, daemon=True)
        thread.start()

    return jupyter_proc


class QTextEditLogger(logging.Handler):

    def __init__(
            self,
            widget,
            mode='set',
            log_string = "%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
    ):
        super().__init__()
        self.widget = widget
        self.mode = mode
        self.setFormatter(logging.Formatter(log_string))
        self.setLevel(level=level)

    def emit(self, record):
        print("Logger emit", self.format(record))
        msg = self.format(record)
        if self.mode == "set":
            self.widget.setText(msg)
        elif self.mode == "append":
            self.widget.appendPlainText(msg)


def setup_logging_widgets(window):

    # Create logger for status bar
    ##############################
    log_handler = QTextEditLogger(
        window.status_label,
        'set',
        log_string = "%(message)s",
        level = logging.INFO
    )
    window.status_log_handler = log_handler

    log_level = chisurf.settings.cs_settings.get('log_level', logging.INFO)

    # Attach logging to the root logger
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(log_level)

    ###########################

    # Create logger for text log field
    ##################################
    log_handler = QTextEditLogger(window.plainTextEditLog, 'append', level = logging.DEBUG)
    window.log_history_handler = log_handler

    # Attach logging to the root logger
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(log_level)

    # Example logging message
    logging.info("ChiSurf started.")


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

        # Initialize copyright, license, and contributors information
        import datetime
        current_year = datetime.datetime.now().year
        self.copyright_text = f"© 2015-{current_year} ChiSurf Team"
        self.license_text = "Licensed under GPL2.1"
        self.contributors_text = "Developers & Contributors: \nThomas-Otavio Peulen, Katherina Hemmen, Jakub Kubiak"

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

        # Set font for additional text boxes
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        # Draw copyright text at the bottom left
        copyright_rect = QtCore.QRect(10, self.height() - 95, self.width() - 20, 20)
        painter.drawText(copyright_rect, QtCore.Qt.AlignLeft, self.copyright_text)

        # Draw license text at the bottom center
        license_rect = QtCore.QRect(10, self.height() - 85, self.width() - 20, 20)
        painter.drawText(license_rect, QtCore.Qt.AlignLeft, self.license_text)

        # Draw contributors text at the bottom right
        contributors_rect = QtCore.QRect(240, self.height() - 80, self.width() - 20, 100)
        painter.drawText(contributors_rect, QtCore.Qt.AlignLeft, self.contributors_text)


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

    def populate_plugins():
        # Find the Help menu to insert the Plugins menu before it
        help_menu = None
        for action in window.menuBar.actions():
            if action.text() == 'Help':
                help_menu = action
                break

        # Insert the Plugins menu before the Help menu
        plugin_menu = QtWidgets.QMenu('Plugins', window)
        window.menuBar.insertMenu(help_menu, plugin_menu)

        # Store the plugin menu in a global variable so it can be accessed by populate_notebooks
        global plugin_menu_action
        plugin_menu_action = plugin_menu.menuAction()
        plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent

        # Dictionary to store submenus
        submenus = {}

        # Get plugin settings
        plugin_settings = chisurf.settings.cs_settings.get('plugins', {})
        disabled_plugins = plugin_settings.get('disabled_plugins', [])
        hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
        icons_enabled = plugin_settings.get('icons_enabled', True)

        # Check if we're in experimental mode
        experimental_mode = chisurf.settings.cs_settings.get('enable_experimental', False)

        for _, module_name, _ in pkgutil.iter_modules(chisurf.plugins.__path__):
            module_path = f"chisurf.plugins.{module_name}"
            module = importlib.import_module(module_path)
            name = getattr(module, 'name', module_name)

            # Check if this plugin is marked as broken
            # Handle broken plugins only by their name not location in menu
            clean_name = name.split(":")[-1]
            is_broken = name in disabled_plugins or module_name in disabled_plugins or clean_name in disabled_plugins

            # Skip broken plugins if they should be hidden and we're not in experimental mode
            if is_broken and hide_disabled_plugins and not experimental_mode:
                continue

            # Determine which file to run: wizard.py if it exists, else __init__.py
            plugin_dir = plugin_root / module_name
            wizard_file = plugin_dir / "wizard.py"
            script_file = wizard_file if wizard_file.is_file() else (plugin_dir / "__init__.py")

            # Build the callback
            callback = partial(
                window.onRunMacro,
                str(script_file),
                executor='exec',
                globals={'__name__': 'plugin'}
            )

            # Check for icon if enabled
            icon = None
            if icons_enabled:
                # Look for icon in module or in plugin directory
                icon_path = plugin_dir / "icon.png"
                if hasattr(module, 'icon'):
                    icon = module.icon
                elif icon_path.exists():
                    icon = QtGui.QIcon(str(icon_path))

            # Get plugin description if available
            description = getattr(module, '__doc__', "No description available.")

            # Check if the name contains a colon to determine if it should go in a submenu
            if ":" in name:
                # Split the name into submenu name and plugin name
                submenu_name, plugin_name = name.split(":", 1)

                # Create submenu if it doesn't exist
                if submenu_name not in submenus:
                    submenus[submenu_name] = plugin_menu.addMenu(submenu_name)

                # Add the plugin to the submenu
                plugin_action = QtWidgets.QAction(f"{plugin_name.strip()}", window)
                if icon:
                    plugin_action.setIcon(icon)
                plugin_action.triggered.connect(callback)
                # Set tooltip with plugin description
                plugin_action.setToolTip(description)
                # Add a visual indicator for broken plugins in experimental mode
                if is_broken and experimental_mode:
                    plugin_action.setText(f"{plugin_name.strip()} [BROKEN]")
                submenus[submenu_name].addAction(plugin_action)
            else:
                # Add the plugin directly to the main menu
                plugin_action = QtWidgets.QAction(f"{name}", window)
                if icon:
                    plugin_action.setIcon(icon)
                plugin_action.triggered.connect(callback)
                # Set tooltip with plugin description
                plugin_action.setToolTip(description)
                # Add a visual indicator for broken plugins in experimental mode
                if is_broken and experimental_mode:
                    plugin_action.setText(f"{name} [BROKEN]")
                plugin_menu.addAction(plugin_action)

    def populate_notebooks():
        # Find the Help menu to insert the Notebooks menu before it
        help_menu = None
        for action in window.menuBar.actions():
            if action.text() == 'Help':
                help_menu = action
                break

        # Create the Notebooks menu
        notebook_menu = QtWidgets.QMenu('Notebooks', window)

        # Get the next action after the Plugins menu
        next_action = None
        found_plugins = False
        for action in window.menuBar.actions():
            if found_plugins:
                next_action = action
                break
            if action == plugin_menu_action:
                found_plugins = True

        # Insert the Notebooks menu after the Plugins menu
        window.menuBar.insertMenu(next_action, notebook_menu)

        home_dir = pathlib.Path.home()
        chisurf_path = pathlib.Path(chisurf.__file__).parent
        plugin_path = pathlib.Path(chisurf.plugins.browser.__file__).absolute().parent

        # Define the target directory inside the home directory
        chisurf_notebooks_dir = home_dir / "notebooks"
        chisurf_notebooks_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        def copy_notebook(src, dest_dir):
            """Copy a notebook file using pathlib only."""
            dest_file = dest_dir / src.name
            dest_file.write_bytes(src.read_bytes())  # Read and write in binary mode
            return dest_file

        def add_notebook(notebook_file):
            if not notebook_file.exists():
                return

            try:
                notebook_file = notebook_file.resolve()  # Ensure absolute path

                # If the file is not inside home_dir, copy it to ~/notebooks/
                if not notebook_file.is_relative_to(home_dir):
                    notebook_file = copy_notebook(notebook_file, chisurf_notebooks_dir)

                # Convert path to POSIX format (to avoid Windows `\` issues in URL)
                notebook_path_str = notebook_file.relative_to(home_dir).as_posix()

                # Correct Jupyter notebook URL with `/tree/`
                # http://localhost:8932/notebooks/Links/smFRET_01_Burst_Search_ALEX.
                adr = f"{chisurf.__jupyter_address__}/notebooks/{notebook_path_str}"

                p = partial(
                    window.onRunMacro, plugin_path / "wizard.py",
                    executor='exec',
                    globals={'__name__': 'plugin', 'adr': adr}
                )

                menu_text = notebook_file.stem
                action = QtWidgets.QAction(f"{menu_text}", window)
                action.triggered.connect(p)
                notebook_menu.addAction(action)

            except AttributeError:
                action = QtWidgets.QAction(f"{notebook_file}", window)
                action.triggered.connect(p)
                notebook_menu.addAction(action)

        # Add the Jupyter root directory with `/tree/`
        add_notebook(home_dir)

        # Load notebooks from settings
        notebook_path = chisurf.settings.cs_settings.get('notebook_path', chisurf_path / 'notebooks')
        for notebook_file in sorted(notebook_path.glob("*.ipynb")):
            add_notebook(notebook_file)

    if stage is None:
        gui_imports()
        setup_ipython()
        window = startup_interface()
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
        # In your setup_gui function:
    elif stage == "start_jupyter":
        chisurf.logging.info("Starting Jupyter notebook process")
        # Start the notebook and capture the process
        chisurf.__jupyter_process__ = launch_jupyter_process()
        proc = chisurf.__jupyter_process__

        # Read lines until we see the HTTP address (or the process exits)
        while chisurf.__jupyter_address__ is None:
            line = proc.stdout.readline()
            if not line:
                # The process died or closed its output
                raise RuntimeError("Jupyter process exited before printing URL")
            chisurf.logging.info(line.strip())
            if "http://" in line:
                start = line.find("http://")
                end = line.find("/", start + len("http://"))
                chisurf.__jupyter_address__ = line[start:end]

        chisurf.logging.info(
            "Server found at %s, migrating monitoring to listener thread",
            chisurf.__jupyter_address__
        )
    elif stage == "setup_logging":
        setup_logging_widgets(window)  # Attach logging to status bar
    elif stage == "populate_notebooks":
        chisurf.logging.info("Looking for ipynb in home folder")
        populate_notebooks()
    return None

def get_win(app: QtWidgets.QApplication) -> chisurf.gui.main.Main:
    import pyqtgraph as pg
    pg.setConfigOptions(useOpenGL=False)  # Disable OpenGL in PyQtGraph

    import chisurf.gui.resources
    import pathlib

    # Load splash screen from file path instead of resource
    splash_path = pathlib.Path(chisurf.__file__).parent / "gui" / "resources" / "icons" / "splashscreen.png"
    pixmap = QtGui.QPixmap(str(splash_path))
    splash = SplashScreen(pixmap)

    # move splashscreen to center of active window
    screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor().pos())
    fg = splash.frameGeometry()
    fg.moveCenter(screen.geometry().center())
    splash.move(fg.topLeft())

    getattr(splash, "raise")()
    splash.activateWindow()

    splash.setContentsMargins(0, 0, 0, 100)
    splash.show()
    app.processEvents()

    # Update progress as the setup progresses
    stages = [
        ("Loading modules", "gui_imports", 10),
        ("Setup ipython", "setup_ipython", 30),
        ("Starting interface", "startup_interface", 40),
        ("Initialize setups", "init_setups", 50),
        ("Defining actions", "define_actions", 55),
        ("Loading tools", "load_tools", 65),
        ("Arrange widgets", "arrange_widgets", 70),
        ("Initializing Jupyter", "start_jupyter", 85),
        ("Populate plugins", "populate_plugins", 90),
        ("Populate notebook", "populate_notebooks", 95),
        ("Setup logging", "setup_logging", 98),
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

    def shutdown_jupyter():
        """Ensure the Jupyter notebook server is terminated when the application closes."""
        import chisurf
        jupyter_proc = getattr(chisurf, '__jupyter_process__', None)
        # Only terminate if it's still running.
        if jupyter_proc is not None and jupyter_proc.poll() is None:
            jupyter_proc.terminate()
            try:
                jupyter_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If it doesn't stop in time, force-kill it.
                jupyter_proc.kill()

    # Connect our shutdown function to the application's aboutToQuit signal.
    app.aboutToQuit.connect(shutdown_jupyter)

    return app


fit_windows = list()
