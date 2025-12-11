from __future__ import annotations

import sys
import subprocess
import pathlib
import os
import signal
import threading
import time
import atexit
import ast

from functools import partial
import pkgutil
import importlib
import chisurf.gui.gui_tweaks  # GUI tweaks (QT_OPENGL, etc.)

from qtpy import QtWidgets, QtGui, QtCore, uic
from qtpy.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
import pyqtgraph as pg

import chisurf  # Ensure chisurf is available module-wide
import chisurf.settings
from chisurf import logging
import chisurf.gui.decorators



class _GuiExecutor(QtCore.QObject):
    """Internal helper to execute callables on the GUI thread via queued signals."""

    runRequested = QtCore.Signal(object, tuple, dict)

    @QtCore.Slot(object, tuple, dict)
    def _run(self, func, args, kwargs):
        try:
            if func is not None:
                func(*args, **(kwargs or {}))
        except Exception:
            try:
                logging.exception("Error in GUI executor callback")
            except Exception:
                pass


_gui_executor = None


def run_on_gui_thread(func, *args, **kwargs):
    """Ensure *func* executes on the Qt GUI thread.

    - If called from the GUI thread, executes *func* synchronously and returns its
      result.
    - If called from another thread and a QApplication exists, schedules *func*
      via a queued signal and returns immediately.
    - If no QApplication exists or scheduling fails, falls back to direct call.
    """
    global _gui_executor

    if func is None:
        return None

    try:
        app = QtWidgets.QApplication.instance()
    except Exception:
        app = None

    # No Qt application: just run synchronously
    if app is None:
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    current = QtCore.QThread.currentThread()
    gui_thread = app.thread()

    # Already on GUI thread: execute directly
    if current is gui_thread:
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    # From a worker thread: use queued signal/slot via _GuiExecutor
    if _gui_executor is None:
        try:
            _executor = _GuiExecutor()
            _executor.moveToThread(gui_thread)
            # Ensure queued delivery onto GUI thread
            _executor.runRequested.connect(_executor._run, QtCore.Qt.QueuedConnection)
            _gui_executor = _executor
        except Exception:
            # Fallback to direct execution if setup fails
            try:
                return func(*args, **kwargs)
            except Exception:
                return None

    try:
        _gui_executor.runRequested.emit(func, args, kwargs or {})
        return None
    except Exception:
        # Last-resort fallback: direct execution
        try:
            return func(*args, **kwargs)
        except Exception:
            return None


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
        msg = self.format(record)
        if self.mode == "set":
            # Support label-like widgets and QStatusBar
            if hasattr(self.widget, 'setText') and callable(getattr(self.widget, 'setText')):
                self.widget.setText(msg)
            elif hasattr(self.widget, 'showMessage') and callable(getattr(self.widget, 'showMessage')):
                # For QStatusBar (including TruncatingStatusBar), use showMessage
                try:
                    self.widget.showMessage(msg)
                except Exception:
                    pass
        elif self.mode == "append":
            # Check if widget is QListWidget or QPlainTextEdit
            if hasattr(self.widget, 'addItem'):
                # QListWidget
                self.widget.addItem(msg)
                # Scroll to the bottom to show the latest entry
                self.widget.scrollToBottom()
            else:
                # QPlainTextEdit
                self.widget.appendPlainText(msg)
            
            # If this is the log widget and the parent has a filter method, call it
            if hasattr(self.widget.parent(), 'update_log_filter'):
                self.widget.parent().update_log_filter()


def setup_logging_widgets(window):

    # Create logger for status bar
    ##############################
    # Use the status bar itself for messages so its truncation logic applies
    log_handler = QTextEditLogger(
        window.status,
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
        self.message_color = QtCore.Qt.lightGray  # Light gray text color

        # Get version information
        from chisurf.info import __version__
        self.version_text = f"Version: {__version__}"

        # Initialize copyright, license, and contributors information
        import datetime
        current_year = datetime.datetime.now().year
        self.copyright_text = f"© 2014-{current_year} ChiSurf Team"
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
        copyright_rect = QtCore.QRect(10, self.height() - 100, self.width() - 20, 20)
        painter.drawText(copyright_rect, QtCore.Qt.AlignLeft, self.copyright_text)

        # Draw version text below copyright
        version_rect = QtCore.QRect(10, self.height() - 90, self.width() - 20, 20)
        painter.drawText(version_rect, QtCore.Qt.AlignLeft, self.version_text)

        # Draw license text below version
        license_rect = QtCore.QRect(10, self.height() - 80, self.width() - 20, 20)
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
        gui_settings = chisurf.settings.cs_settings.get('gui') or {}
        style_name = gui_settings.get('style_sheet')

        base_path = pathlib.Path(chisurf.__file__).parent
        package_styles_path = base_path / "gui" / "styles"

        try:
            user_styles_path = chisurf.settings.get_path('settings') / 'styles'
        except Exception:
            user_styles_path = None

        widgets_package_path = package_styles_path / "widgets"
        widgets_user_path = None
        if user_styles_path is not None:
            try:
                widgets_user_path = user_styles_path / "widgets"
            except Exception:
                widgets_user_path = None

        def _apply_stylesheet(qss_text, fallback_path=None):
            try:
                text = qss_text
                if (not text) and fallback_path is not None and fallback_path.is_file():
                    text = fallback_path.read_text(encoding="utf-8")
                if not text:
                    return False

                app.setStyleSheet(text)
                try:
                    chisurf.settings.style_sheet = text
                except Exception:
                    pass
                return True
            except Exception:
                return False

        def _load_shared_widget_styles():
            """Load shared widget styles from gui/styles/widgets.

            We first load package-provided fragments, then optional user
            overrides from the settings/styles/widgets folder. All .qss
            files are concatenated in sorted order so the user can split
            shared styles into multiple logical files if desired.
            """

            parts = []

            def _append_from_dir(d):
                try:
                    if d is None or not d.is_dir():
                        return
                except Exception:
                    return
                try:
                    for p in sorted(d.glob("*.qss")):
                        try:
                            parts.append(p.read_text(encoding="utf-8"))
                        except Exception:
                            # Ignore unreadable fragments but continue
                            continue
                except Exception:
                    pass

            _append_from_dir(widgets_package_path)
            _append_from_dir(widgets_user_path)

            return "\n\n".join([p for p in parts if p])

        def _resolve_style_path(name):
            if not name:
                return None
            name = str(name).strip()
            if not name:
                return None
            if user_styles_path is not None:
                try:
                    cand = user_styles_path / name
                    if cand.is_file():
                        return cand
                except Exception:
                    pass
            try:
                cand = package_styles_path / name
                if cand.is_file():
                    return cand
            except Exception:
                pass
            return None

        def _apply_style_by_name(name):
            style_path = _resolve_style_path(name)
            if style_path is None:
                return False
            try:
                theme_text = style_path.read_text(encoding="utf-8")
            except Exception:
                return False

            shared_text = _load_shared_widget_styles()
            combined = "\n\n".join(t for t in (shared_text, theme_text) if t)
            return _apply_stylesheet(combined, style_path)

        if not style_name:
            logging.warning("No GUI style_sheet configured; using default Qt theme.")
            return

        if _apply_style_by_name(style_name):
            return

        style_path = package_styles_path / style_name

        try:
            if not style_path.is_file():
                logging.warning(f"GUI style sheet not found: {style_path}")
                parent = None
                try:
                    parent = app.activeWindow()
                except Exception:
                    parent = None
                QtWidgets.QMessageBox.warning(
                    parent,
                    "Theme not found",
                    (
                        "The configured GUI theme file could not be loaded:\n"
                        f"{style_path}\n\n"
                        "The application will continue with the default theme.\n"
                        "Please open the settings and select an existing theme."
                    )
                )
        except Exception as e:
            logging.warning(f"Failed to load GUI style sheet '{style_path}': {e}")

    def read_module_docstring(package_path):
        """
        Given a path to a package directory, reads its __init__.py
        and returns the module docstring (or None if there isn't one).
        """
        init_py = package_path / "__init__.py"
        if not init_py.exists():
            return None

        # Read the source
        source = init_py.read_text(encoding="utf-8")

        # Parse into an AST and extract the docstring
        tree = ast.parse(source, filename=str(init_py))
        return ast.get_docstring(tree)

    def get_plugin_metadata(plugin_dir, module_name):
        """
        Extract plugin metadata without importing the module.
        Returns a tuple of (name, description)
        """
        # Default values
        name = module_name
        description = "No description available."

        # Path to the __init__.py file
        init_py = plugin_dir / module_name / "__init__.py"

        # Check if the file exists in the built-in directory
        if not init_py.exists():
            # Try to find it in the user plugins directory
            user_plugin_root = pathlib.Path.home() / '.chisurf' / 'plugins'
            user_init_py = user_plugin_root / module_name / "__init__.py"
            if user_init_py.exists():
                init_py = user_init_py
                chisurf.logging.info(f"Found user plugin: {module_name} at {init_py}")
            else:
                return name, description

        try:
            # Read the source
            source = init_py.read_text(encoding="utf-8")

            # Parse into an AST
            tree = ast.parse(source, filename=str(init_py))

            # Extract the docstring
            description = ast.get_docstring(tree) or description

            # Look for a name assignment
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'name':
                            if isinstance(node.value, ast.Str):
                                name = node.value.s
                            elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                name = node.value.value

            return name, description
        except Exception as e:
            chisurf.logging.warning(f"Error extracting metadata from {init_py}: {e}")
            return name, description

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

        # Dedicated submenu for development plugins that live under chisurf.plugins._dev
        dev_menu = plugin_menu.addMenu("Dev")

        # Dictionary to store submenus for non-dev plugins
        submenus = {}

        # Get plugin settings
        plugin_settings = chisurf.settings.cs_settings.get('plugins', {})
        disabled_plugins = plugin_settings.get('disabled_plugins', [])
        hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
        plugin_order = plugin_settings.get('plugin_order', {})

        # Check if we're in experimental mode
        experimental_mode = chisurf.settings.cs_settings.get('enable_experimental', False)

        # Discover plugins (built-in + user, including nested subpackages)
        try:
            plugin_infos = list(chisurf.plugins.iter_plugins())
        except Exception as e:
            chisurf.logging.error(f"Failed to enumerate plugins via chisurf.plugins.iter_plugins(): {e}")
            plugin_infos = []

        # Resolve the built-in plugins root so we can detect the _dev subtree
        try:
            plugins_root = pathlib.Path(chisurf.plugins.__file__).parent.resolve()
        except Exception:
            plugins_root = pathlib.Path(chisurf.plugins.__file__).parent

        # Resolve the built-in plugins root so we can detect the _dev subtree
        try:
            plugins_root = pathlib.Path(chisurf.plugins.__file__).parent.resolve()
        except Exception:
            plugins_root = pathlib.Path(chisurf.plugins.__file__).parent

        # Sort plugins by order (ascending) then by plugin name
        ordered = []
        for info in plugin_infos:
            plugin_name = info.get('plugin_name') or info.get('module_name') or ''
            order = plugin_order.get(plugin_name, 0)
            ordered.append((order, plugin_name, info))
        ordered.sort(key=lambda x: (x[0], x[1]))

        added_main = 0
        added_submenu = 0
        added_dev = 0
        skipped_hidden = 0
        marked_broken = 0

        chisurf.logging.info(
            f"Populating plugin menu: {len(ordered)} plugin(s) discovered "
            f"(experimental_mode={experimental_mode}, hide_disabled_plugins={hide_disabled_plugins})."
        )

        for _order, plugin_name, info in ordered:
            try:
                module_path = info.get('module_path')
                module_name = info.get('module_name') or ''
                package_dir = pathlib.Path(info.get('package_dir'))
                source = info.get('source') or 'built-in'

                # Check disabled/broken status
                clean_name = plugin_name.split(':')[-1].strip() if ':' in plugin_name else plugin_name
                is_broken = (
                    plugin_name in disabled_plugins
                    or module_name in disabled_plugins
                    or clean_name in disabled_plugins
                )

                # Detect built-in development plugins that live under the _dev package
                is_dev = False
                try:
                    rel = package_dir.resolve().relative_to(plugins_root)
                    if rel.parts and rel.parts[0] == "_dev":
                        is_dev = True
                except Exception:
                    is_dev = False

                # Detect built-in development plugins that live under the _dev package
                is_dev = False
                try:
                    rel = package_dir.resolve().relative_to(plugins_root)
                    if rel.parts and rel.parts[0] == "_dev":
                        is_dev = True
                except Exception:
                    is_dev = False

                # Skip broken plugins if they should be hidden and we're not in experimental mode
                if is_broken and hide_disabled_plugins and not experimental_mode:
                    skipped_hidden += 1
                    chisurf.logging.info(
                        f"Skipping disabled/broken plugin in menu: '{plugin_name}' "
                        f"(module='{module_name}', source='{source}', package_dir='{package_dir}')"
                    )
                    continue

                # Determine which file to run: wizard.py if it exists, else __init__.py
                plugin_dir = package_dir
                wizard_file = plugin_dir / "wizard.py"
                script_file = wizard_file if wizard_file.is_file() else (plugin_dir / "__init__.py")

                # Build the callback
                callback = partial(
                    window.onRunMacro,
                    str(script_file),
                    executor='exec',
                    globals={'__name__': 'plugin'}
                )

                # Check for icon
                icon = None
                for _icon_name in ("icon.png", "icon.svg"):
                    icon_path = plugin_dir / _icon_name
                    if icon_path.exists():
                        icon = QtGui.QIcon(str(icon_path))
                        break

                # Get plugin description from iter_plugins metadata or fallback to docstring
                description = info.get('description') or "No description available."

                status = "BROKEN" if is_broken else "ok"

                # Route development plugins into the dedicated Dev submenu, using the clean
                # name (without grouping prefix) as the visible label.
                if is_dev:
                    display_name = clean_name or plugin_name
                    chisurf.logging.info(
                        f"Adding plugin to Plugins->Dev menu: '{display_name}' "
                        f"(plugin='{plugin_name}', module='{module_name}', source='{source}', "
                        f"status={status}, script='{script_file}')"
                    )
                    plugin_action = QtWidgets.QAction(f"{display_name}", window)
                    if icon:
                        plugin_action.setIcon(icon)
                    plugin_action.triggered.connect(callback)
                    plugin_action.setToolTip(description)
                    if is_broken and experimental_mode:
                        plugin_action.setText(f"{display_name} [BROKEN]")
                        marked_broken += 1
                    added_dev += 1
                    dev_menu.addAction(plugin_action)
                    continue

                # Check if the name contains a colon to determine if it should go in a submenu
                if ":" in plugin_name:
                    submenu_name, short_name = plugin_name.split(":", 1)

                    # Create submenu if it doesn't exist
                    if submenu_name not in submenus:
                        submenus[submenu_name] = plugin_menu.addMenu(submenu_name)

                    # Add the plugin to the submenu
                    chisurf.logging.info(
                        f"Adding plugin to Plugins->{submenu_name} submenu: '{short_name.strip()}' "
                        f"(plugin='{plugin_name}', module='{module_name}', source='{source}', "
                        f"status={status}, script='{script_file}')"
                    )
                    plugin_action = QtWidgets.QAction(f"{short_name.strip()}", window)
                    if icon:
                        plugin_action.setIcon(icon)
                    plugin_action.triggered.connect(callback)
                    plugin_action.setToolTip(description)
                    if is_broken and experimental_mode:
                        plugin_action.setText(f"{short_name.strip()} [BROKEN]")
                        marked_broken += 1
                    added_submenu += 1
                    submenus[submenu_name].addAction(plugin_action)
                else:
                    # Add the plugin directly to the main menu
                    chisurf.logging.info(
                        f"Adding plugin to Plugins menu: '{plugin_name}' "
                        f"(module='{module_name}', source='{source}', status={status}, script='{script_file}')"
                    )
                    plugin_action = QtWidgets.QAction(f"{plugin_name}", window)
                    if icon:
                        plugin_action.setIcon(icon)
                    plugin_action.triggered.connect(callback)
                    plugin_action.setToolTip(description)
                    if is_broken and experimental_mode:
                        plugin_action.setText(f"{plugin_name} [BROKEN]")
                        marked_broken += 1
                    added_main += 1
                    plugin_menu.addAction(plugin_action)
            except Exception as e:
                chisurf.logging.error(
                    f"Error while adding plugin to plugin menu: '{plugin_name}' "
                    f"(module='{info.get('module_name')}', source='{info.get('source')}'): {e}"
                )
                continue

        chisurf.logging.info(
            f"Plugin menu populated: added {added_main} main, {added_submenu} submenu, "
            f"{added_dev} dev plugin(s); {skipped_hidden} disabled/broken plugin(s) hidden; "
            f"{marked_broken} plugin(s) marked as BROKEN."
        )

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
        plugin_path = pathlib.Path(chisurf.plugins.__file__).parent / "browser"

        # Define the target directory inside the home directory
        chisurf_notebooks_dir = home_dir / "notebooks"
        chisurf_notebooks_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        def copy_notebook(src, dest_dir):
            """Copy a notebook file using pathlib only."""
            dest_file = dest_dir / src.name
            if not dest_file.exists():  # Only copy if the file doesn't exist
                dest_file.write_bytes(src.read_bytes())  # Read and write in binary mode
                chisurf.logging.info(f"Copied notebook: {src.name} to {dest_file}")
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

        # Copy all notebooks from the package to the user's home directory
        # This ensures that all shipped notebooks are available to the user
        notebook_source_dirs = []
        # Note: deprecate latest on March 2026
        legacy_dir = chisurf_path / 'notebooks'
        if legacy_dir.is_dir():
            notebook_source_dirs.append(legacy_dir)

        repo_notebooks_dir = chisurf_path.parent / 'notebooks'
        if repo_notebooks_dir.is_dir() and repo_notebooks_dir not in notebook_source_dirs:
            notebook_source_dirs.append(repo_notebooks_dir)

        for src_dir in notebook_source_dirs:
            chisurf.logging.info(f"Checking for notebooks in: {src_dir}")
            for notebook_file in sorted(src_dir.glob("*.ipynb")):
                copy_notebook(notebook_file, chisurf_notebooks_dir)

        # Add the Jupyter root directory with `/tree/`
        add_notebook(home_dir)

        # Load notebooks from user's home directory for the menu
        for notebook_file in sorted(chisurf_notebooks_dir.glob("*.ipynb")):
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
    elif stage == "check_updates":
        # Respect user setting to ignore update prompts on startup
        try:
            _plugins = chisurf.settings.cs_settings.get('plugins') or {}
            _updater_settings = _plugins.get('updater') or {}
            _ignore_updates = bool(_updater_settings.get('ignore_updates_on_startup', False))
            _check_on_startup = bool(_updater_settings.get('check_on_startup', True))
        except Exception:
            _ignore_updates = False
            _check_on_startup = True

        if _ignore_updates or not _check_on_startup:
            chisurf.logging.info("Startup update prompt suppressed by user settings.")
        else:
            from chisurf.plugins.updater import updater as _updater_mod

            def _startup_update_check():
                update_available, latest_version, error = _updater_mod.check_for_updates()
                if error:
                    chisurf.logging.info(f"Update check skipped or failed: {error}")
                elif update_available:
                    chisurf.logging.info(f"Update available: {latest_version}")
                    # Prompt user to open the updater
                    try:
                        from chisurf.plugins.updater import build_installed_vs_latest_changelog as _build_changes
                        try:
                            _installed_ver, _changes = _build_changes(str(latest_version))
                        except Exception:
                            _installed_ver, _changes = None, None

                        _msg = (
                            f"A new version of ChiSurf ({latest_version}) is available.\n\n"
                            + (
                                f"Changes since your installed version ({_installed_ver}):\n\n{_changes}\n\n"
                                if _changes else ""
                            )
                            + "Do you want to open the Updater now?"
                        )
                        reply = QtWidgets.QMessageBox.question(
                            None,
                            "Update Available",
                            _msg,
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                            QtWidgets.QMessageBox.Yes,
                        )
                        if reply == QtWidgets.QMessageBox.Yes:
                            import importlib
                            updater_plugin = importlib.import_module("chisurf.plugins.updater")
                            # Keep a strong reference to prevent garbage collection from closing the window
                            import chisurf as _chisurf_mod
                            _chisurf_mod.__updater_window__ = updater_plugin.UpdaterWidget(suppress_initial_notification=True)
                            _chisurf_mod.__updater_window__.show()
                            try:
                                _chisurf_mod.__updater_window__.raise_()
                                _chisurf_mod.__updater_window__.activateWindow()
                            except Exception:
                                pass
                            # Signal startup should be interrupted so only the updater remains open
                            try:
                                _chisurf_mod.__startup_interrupt_for_updater__ = True
                            except Exception:
                                pass
                    except Exception as e:
                        chisurf.logging.debug(f"Failed to show update prompt: {e}")
                else:
                    chisurf.logging.info("ChiSurf is up to date.")
            _startup_update_check()
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
        try:
            _gui_cfg = chisurf.settings.cs_settings.get('gui') or {}
            _start_jupyter = bool(_gui_cfg.get('start_jupyter_on_startup', False))
        except Exception:
            _start_jupyter = False

        if not _start_jupyter:
            chisurf.logging.info("Skipping Jupyter notebook startup (disabled in settings).")
            return None

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
        try:
            _gui_cfg = chisurf.settings.cs_settings.get('gui') or {}
            _start_jupyter = bool(_gui_cfg.get('start_jupyter_on_startup', False))
        except Exception:
            _start_jupyter = False

        if not _start_jupyter or chisurf.__jupyter_address__ is None:
            chisurf.logging.info("Skipping notebook menu population (Jupyter disabled or not running).")
            return None

        chisurf.logging.info("Looking for ipynb in home folder")
        populate_notebooks()
    return None

def get_win(app: QtWidgets.QApplication) -> chisurf.gui.main.Main:
    logging.info("Starting GUI startup (get_win)")
    from chisurf.gui.gui_tweaks import apply_pyqtgraph_autorange_compat
    pg.setConfigOptions(useOpenGL=False)  # Disable OpenGL in PyQtGraph
    apply_pyqtgraph_autorange_compat(pg)

    import chisurf.gui.resources
    import pathlib

    # Load splash screen from file path instead of resource
    splash_path = pathlib.Path(chisurf.__file__).parent / "gui" / "resources" / "icons" / "splashscreen.png"
    pixmap = QtGui.QPixmap(str(splash_path))
    splash = SplashScreen(pixmap)

    # move splashscreen to center of active window; be robust if Qt cannot
    # determine the current screen (screenAt may return None in some setups)
    screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor().pos())
    if screen is None:
        try:
            screen = QtWidgets.QApplication.primaryScreen()
        except Exception:
            screen = None
    if screen is None:
        try:
            screens = QtGui.QGuiApplication.screens()
            if screens:
                screen = screens[0]
        except Exception:
            screen = None

    if screen is not None:
        fg = splash.frameGeometry()
        fg.moveCenter(screen.geometry().center())
        splash.move(fg.topLeft())

    splash.setContentsMargins(0, 0, 0, 100)
    splash.show()
    try:
        splash.raise_()
        splash.activateWindow()
    except Exception:
        pass
    app.processEvents()

    # Update progress as the setup progresses
    try:
        _gui_cfg = chisurf.settings.cs_settings.get('gui') or {}
        _start_jupyter = bool(_gui_cfg.get('start_jupyter_on_startup', False))
    except Exception:
        _start_jupyter = False

    stages = [
        ("Check for updates", "check_updates", 5),
        ("Loading modules", "gui_imports", 10),
        ("Setup ipython", "setup_ipython", 30),
        ("Starting interface", "startup_interface", 40),
        ("Setup logging", "setup_logging", 45),
        ("Initialize setups", "init_setups", 50),
        ("Defining actions", "define_actions", 55),
        ("Loading tools", "load_tools", 65),
        ("Arrange widgets", "arrange_widgets", 70),
    ]

    if _start_jupyter:
        stages.extend([
            ("Initializing Jupyter", "start_jupyter", 85),
            ("Populate plugins", "populate_plugins", 90),
            ("Populate notebook", "populate_notebooks", 95),
        ])
    else:
        stages.append(("Populate plugins", "populate_plugins", 90))

    stages.append(("Styling up", "setup_style", 100))

    window = None
    for message, stage, progress_value in stages:
        logging.info(f"Startup stage '{stage}' starting: {message}")
        splash.update_message(message)
        splash.update_progress(progress_value)
        app.processEvents()
        w2 = setup_gui(app=app, stage=stage, window=window)
        logging.info(f"Startup stage '{stage}' finished")
        if w2 is not None:
            window = w2
        # If user chose to open updater, interrupt startup immediately
        try:
            if getattr(chisurf, "__startup_interrupt_for_updater__", False):
                break
        except Exception:
            pass
        # After checking for updates, display version comparison on the splash
        if stage == "check_updates":
            try:
                from chisurf.plugins.updater import updater as _updater_mod
                from chisurf import info as _info
                import time as _time
                cur = getattr(_info, "__version__", "?")
                update_available, latest_version, error = _updater_mod.check_for_updates()
                if error:
                    text = f"v{cur} — Update check failed"
                else:
                    if update_available and latest_version:
                        text = f"{cur} vs. {latest_version} (Update available)"
                    else:
                        # If no update or latest unknown, assume up to date
                        latest_txt = latest_version or cur
                        text = f"v{cur} (Up to date)"
                splash.update_message(text)
                # Ensure the update info is visible for at least one second
                start_ts = _time.time()
                # Process events in small slices to keep UI responsive during the wait
                while _time.time() - start_ts < 2.0:
                    app.processEvents()
                    _time.sleep(0.05)
            except Exception as e:
                chisurf.logging.debug(f"Failed to update splash with version info: {e}")

    # If startup was interrupted for updater, do not show the main window
    try:
        if getattr(chisurf, "__startup_interrupt_for_updater__", False):
            splash.hide()
            return window
    except Exception:
        pass

    window.show()
    splash.hide()
    splash.finish(window)
    return window


def get_app():
    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()
    win = get_win(app=app)

    # If startup was interrupted to open the updater, do not touch/show the main window
    try:
        import chisurf as _chisurf_mod
        if getattr(_chisurf_mod, "__startup_interrupt_for_updater__", False):
            win = None  # We won't use the main window in this case
        else:
            win.raise_()
            win.activateWindow()
            win.setFocus()
    except Exception:
        # Fallback to showing the window if available
        if win is not None:
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
