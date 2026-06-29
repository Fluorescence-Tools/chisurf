#!/usr/bin/env python3
"""
Standalone SM Acquisition Application

This application can run the single-molecule acquisition plugin either as part of chisurf
or as a standalone application with its own MDI interface.
"""

import sys
import os
import logging

try:
    from qtpy.QtWidgets import (
        QApplication,
        QMainWindow,
        QMdiArea,
        QVBoxLayout,
        QWidget,
        QMenuBar,
        QMenu,
        QAction,
        QStatusBar,
    )
    from qtpy.QtCore import Qt
    PYQT_AVAILABLE = True
except Exception:
    # Allow this module to be imported in environments without a Qt stack so that
    # CLI-only mode (``standalone.py --cli ...``) continues to work.
    QApplication = None  # type: ignore
    QMainWindow = object  # type: ignore
    QMdiArea = QVBoxLayout = QWidget = QMenuBar = QMenu = QAction = QStatusBar = object  # type: ignore
    Qt = None  # type: ignore
    PYQT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_chisurf_available():
    """Check if chisurf is available."""
    try:
        import chisurf
        return True, chisurf
    except ImportError:
        return False, None

def run_standalone():
    """Run the SM acquisition as a standalone application."""
    if not PYQT_AVAILABLE:
        logger.error("PyQt5 is not available; standalone GUI mode cannot be used.")
        return 1
    logger.info("Running SM Acquisition in standalone mode")

    # Import our modules
    from .main import SMAcquisitionManager

    # Create the application
    app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()

    # Create standalone main window
    main_window = StandaloneMainWindow()

    # Create and initialize the acquisition manager
    try:
        manager = SMAcquisitionManager(main_window)
        main_window.acquisition_manager = manager
        logger.info("SM Acquisition manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SM Acquisition manager: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show the main window
    main_window.show()
    main_window.raise_()
    main_window.activateWindow()

    # Run the application
    return app.exec()

def run_as_plugin():
    """Run the SM acquisition as a chisurf plugin."""
    if not PYQT_AVAILABLE:
        logger.error("PyQt5 is not available; plugin GUI mode cannot be used.")
        return 1
    logger.info("Running SM Acquisition as chisurf plugin")

    chisurf_available, chisurf = is_chisurf_available()
    if not chisurf_available:
        logger.error("chisurf not available, cannot run as plugin")
        return 1

    # Import and run as normal plugin
    from .main import SMAcquisitionManager

    try:
        manager = SMAcquisitionManager()
        logger.info("SM Acquisition plugin loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SM Acquisition plugin: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Keep the plugin running
    return chisurf.run()

class StandaloneMainWindow(QMainWindow):
    """Standalone main window for SM Acquisition."""

    def __init__(self):
        super().__init__()
        self.acquisition_manager = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("SM Acquisition - Standalone")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget with MDI area
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create MDI area
        self.mdiarea = QMdiArea()
        self.mdiarea.setViewMode(QMdiArea.TabbedView)
        layout.addWidget(self.mdiarea)

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')
        exit_action = QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('&View')

        # Window menu
        window_menu = menubar.addMenu('&Window')

        # Help menu
        help_menu = menubar.addMenu('&Help')
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        """Show about dialog."""
        from qtpy.QtWidgets import QMessageBox
        QMessageBox.about(self, "About SM Acquisition",
                         "Single-Molecule Acquisition Plugin\n\n"
                         "Standalone version for testing and development.")

    def closeEvent(self, event):
        """Handle close event."""
        if self.acquisition_manager:
            try:
                self.acquisition_manager.close_acquisition_mode()
            except Exception as e:
                logger.error(f"Error closing acquisition manager: {e}")

        super().closeEvent(event)

def main():
    """Main entry point.

    Modes:

    - **CLI mode**: if ``--cli`` is present in the arguments, the remaining
      arguments are forwarded to the simulation CLI (config/run) and no GUI
      is started.
    - **GUI/plugin mode** (default): existing behaviour, either running as a
      chisurf plugin or as a standalone Qt application.
    """

    # First: check for CLI mode flag and delegate to the simulation CLI.
    args = sys.argv[1:]
    if '--cli' in args:
        cli_index = args.index('--cli')
        cli_args = args[cli_index + 1 :]

        # Import here to avoid unnecessary Click/CLI setup when running GUI.
        from .tcspc_devices.simulation.simulation_cli import main as simulation_cli_main

        # Synthesize argv for Click so that subcommands (config/run) see a
        # clean argument list. Example usage:
        #   python standalone.py --cli config --output ...
        sys.argv = [sys.argv[0]] + cli_args
        return simulation_cli_main()

    # Otherwise: run in GUI/plugin mode as before.
    # Check if chisurf is available
    chisurf_available, chisurf = is_chisurf_available()

    if chisurf_available:
        logger.info("chisurf detected, running as plugin")
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--standalone':
            logger.info("Forced standalone mode")
            return run_standalone()
        else:
            return run_as_plugin()
    else:
        logger.info("chisurf not detected, running standalone")
        return run_standalone()

if __name__ == '__main__':
    sys.exit(main())
