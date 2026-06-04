# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration - Base Module

This module contains the core ChiSurfRibbonIntegration class and basic setup functionality.
"""

import os
import sys
from pathlib import Path
import json
import functools
from math import ceil

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QObject

from .ribbonbar import RibbonBar
from .constants import RibbonStyle
from .logger import logging

# Monkey patch to disable problematic window dragging in ribbon title widget
def _disable_title_widget_dragging():
    """Monkey patch ribbon title widget to prevent window movement issues"""
    try:
        from .titlewidget import RibbonTitleWidget

        # Replace the problematic mouse event methods with no-ops
        def noop_mousePressEvent(self, e):
            pass

        def noop_mouseMoveEvent(self, e):
            pass

        def noop_mouseDoubleClickEvent(self, e):
            pass

        # Apply the monkey patch
        RibbonTitleWidget.mousePressEvent = noop_mousePressEvent
        RibbonTitleWidget.mouseMoveEvent = noop_mouseMoveEvent
        RibbonTitleWidget.mouseDoubleClickEvent = noop_mouseDoubleClickEvent

        logging.info("Applied monkey patch to disable ribbon title widget dragging")

    except ImportError as e:
        logging.warning(f"Failed to apply ribbon monkey patch: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error applying monkey patch: {e}")

# Apply the monkey patch immediately when the module is imported
_disable_title_widget_dragging()


class ChiSurfRibbonIntegration(QObject):
    """
    Integration class for adding ribbon interface to ChiSurf main window.

    This class provides methods to convert the existing menu/toolbar structure
    into a modern ribbon interface while maintaining all existing functionality.
    """

    def __init__(self, main_window):
        """
        Initialize ribbon integration for ChiSurf main window.

        Parameters
        ----------
        main_window : chisurf.gui.main.Main
            The ChiSurf main window instance
        """
        super().__init__()
        self.main_window = main_window
        self.ribbon_bar = None
        self.original_menubar = None
        self.original_toolbar = None
        # Create a hidden widget to preserve the menu bar
        from qtpy.QtWidgets import QWidget
        self.menu_preserve_widget = QWidget()
        self.menu_preserve_widget.hide()
        # Create a hidden widget to preserve the ribbon bar
        self.ribbon_preserve_widget = QWidget()
        self.ribbon_preserve_widget.hide()
        self.logger = logging.getLogger('chisurf.gui.widgets.ribbon')

        # Auto-fold functionality
        self.auto_fold_timer = QtCore.QTimer()
        self.auto_fold_timer.setSingleShot(True)
        self.auto_fold_timer.timeout.connect(self._auto_fold_ribbon)
        msg = "Auto-fold timer created and connected"
        self.logger.info(msg)
        self.last_activity_time = QtCore.QTimer()
        self.last_activity_time.start()
        self.is_folded = False

        # Auto-fold properties (initialized here, updated in _setup_auto_fold)
        self.auto_fold_enabled = True
        self.auto_fold_delay_ms = 1000
        self.auto_fold_speed_ms = 500
        self.mouse_over_ribbon = False

        # Pin functionality
        self.pin_button = None
        self.is_pinned = False

        # Note: Timer will be started by _setup_auto_fold() if auto-fold is enabled
        # and not overridden by pin state in _setup_pin_button()

    def eventFilter(self, obj, event):
        """
        Event filter for handling ribbon auto-fold and resize events.

        Parameters
        ----------
        obj : QObject
            The object being filtered
        event : QEvent
            The event to handle

        Returns
        -------
        bool
            True if event was handled, False otherwise
        """
        # Handle auto-fold events for ribbon and tab bar
        if (obj == self.ribbon_bar or (hasattr(self.ribbon_bar, 'tabBar') and obj == self.ribbon_bar.tabBar())) and self.auto_fold_enabled:
            if event.type() == QtCore.QEvent.Enter:
                # Mouse entered ribbon - stop auto-fold timer but DON'T auto-unfold
                self.mouse_over_ribbon = True
                msg = "MOUSE ENTERED RIBBON - stopping auto-fold timer (hover unfold disabled)"
                self.logger.debug(msg)
                self.auto_fold_timer.stop()
                # NOTE: Removed auto-unfold on hover - now requires click to uncollapse
                return False
            elif event.type() == QtCore.QEvent.Leave:
                # Mouse left ribbon - start auto-fold timer
                self.mouse_over_ribbon = False
                msg = "MOUSE LEFT RIBBON - starting auto-fold timer"
                self.logger.debug(msg)
                self._restart_auto_fold_timer()
                return False
            elif event.type() == QtCore.QEvent.MouseButtonPress:
                # Mouse clicked on ribbon - unfold if folded
                self.logger.debug(f"Mouse button press on {type(obj).__name__}: folded={self.is_folded}, pinned={self.is_pinned}")
                if self.is_folded and not self.is_pinned:
                    msg = "RIBBON CLICKED - unfolding ribbon"
                    self.logger.debug(msg)
                    self._unfold_ribbon()
                    return True
                else:
                    if self.is_folded:
                        self.logger.debug("Ribbon is folded but pinned - not unfolding")
                    elif not self.is_folded:
                        self.logger.debug("Ribbon is not folded - ignoring click")
                    return False
            else:
                # Log other events for debugging
                self.logger.debug(f"Other event on ribbon: {event.type()}")
                # Additional debugging for mouse events
                if event.type() in [QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseMove]:
                    self.logger.debug(f"Mouse event on ribbon: {event.type()}, folded={self.is_folded}, pinned={self.is_pinned}")

        # Handle resize events for main window
        elif obj == self.main_window and event.type() == QtCore.QEvent.Resize:
            # Fix white background issues on resize
            QtCore.QTimer.singleShot(100, self._apply_background_fix)
            QtCore.QTimer.singleShot(200, self._apply_title_widget_fix)
            return False

        return False  # Don't block the event

    def setup_ribbon_interface(self, ribbon_style=None):
        """
        Setup the ribbon interface by preserving menu bar in hidden widget.

        This approach moves the menu bar to a hidden widget to prevent deletion.

        Parameters
        ----------
        ribbon_style : int, optional
            Ribbon style to use (ribbon uses RibbonStyle constants)
            If None, uses default style
        """
        logging.info(f"DEBUG: setup_ribbon_interface called with ribbon_style={ribbon_style}")
        try:
            # Store original components for reference only
            # Handle both cases where menuBar is a method or property
            menubar_attr = self.main_window.menuBar
            if callable(menubar_attr):
                self.original_menubar = menubar_attr()  # Call if it's a method
            else:
                self.original_menubar = menubar_attr  # Use directly if it's a property

            self.original_toolbar = getattr(self.main_window, 'toolBar', None)

            self.logger.info("Ribbon setup starting...")

            # Move the menu bar to the hidden widget to preserve it
            if self.original_menubar:
                self.original_menubar.setParent(self.menu_preserve_widget)
                self.logger.info("Menu bar moved to hidden widget for preservation")

            # Hide original toolbar if it exists
            if self.original_toolbar:
                self.original_toolbar.hide()
                self.logger.info("Original toolbar hidden")

            # Hide plugin toolbar when switching to ribbon
            if hasattr(self.main_window, 'plugins_toolbar'):
                self.main_window.plugins_toolbar.hide()
                self.logger.info("Plugin toolbar hidden for ribbon mode")

            # Restore or create ribbon bar
            try:
                if self.ribbon_bar:
                    # Move ribbon bar back from hidden widget
                    self.ribbon_bar.setParent(self.main_window)
                    self.logger.info("Ribbon bar restored from hidden widget")
                else:
                    # Create ribbon bar
                    self.ribbon_bar = RibbonBar()

                    # Set ribbon style - ribbon uses different style constants
                    # For now, we'll use the default style
                    if ribbon_style is not None:
                        # Convert old style constants to new ones if needed
                        self.ribbon_bar.setRibbonStyle(RibbonStyle.Default)

                    self.logger.info(f"New ribbon bar created with style {ribbon_style}")
            except Exception as e:
                self.logger.error(f"Failed to initialize RibbonBar: {e}")
                self.restore_original_interface()
                return False

            # Set ribbon bar as menu widget (safe since menu bar is preserved)
            self.main_window.setMenuWidget(self.ribbon_bar)
            self.logger.info("Ribbon bar set as menu widget")

            # Apply immediate background fix to prevent white area
            self._apply_background_fix()

            # Additional fix for title widget and empty space issues
            self._apply_title_widget_fix()

            # Read max_rows and ribbon_height from settings
            import chisurf
            gui_settings = chisurf.settings.cs_settings.get('gui', {})
            ribbon_settings = gui_settings.get('ribbon', {})
            max_rows = ribbon_settings.get('max_rows', 3)  # Default to 3 rows for reduced height
            ribbon_height = ribbon_settings.get('ribbon_height', 110)  # Default to 110px for reduced height

            # Apply ribbon height setting
            if hasattr(self.ribbon_bar, 'setRibbonHeight'):
                self.ribbon_bar.setRibbonHeight(ribbon_height)
                self.logger.info(f"Set ribbon height to {ribbon_height}px")

            if hasattr(self.ribbon_bar, '_maxRows'):
                self.ribbon_bar._maxRows = max_rows
                self.logger.info(f"Set ribbon max rows to {max_rows}")

                # Update all existing categories to use the new max rows
                if hasattr(self.ribbon_bar, '_categories'):
                    for category in self.ribbon_bar._categories.values():
                        if hasattr(category, 'setMaximumRows'):
                            category.setMaximumRows(max_rows)
                    self.logger.info(f"Updated all categories to use {max_rows} max rows")

            # Set the application icon in the ribbon title widget
            try:
                # Use the same icon as the main window
                chisurf_icon = self.main_window.windowIcon()
                if not chisurf_icon.isNull():
                    self.ribbon_bar.setApplicationIcon(chisurf_icon)
                    self.logger.info("Set ChiSurf icon in ribbon title widget from main window")
                else:
                    self.logger.warning("Main window has no icon set")
            except Exception as e:
                self.logger.warning(f"Failed to set ribbon application icon: {e}")

            # Apply dark theme using palette approach like the demo
            self._apply_dark_palette()
            self.logger.debug("Applied dark palette to application")

            # Apply compact spacing to reduce ribbon item spacing
            self.apply_compact_spacing()
            self.logger.debug("Applied compact spacing to ribbon items")

            # Setup hover tab switching - DISABLED
            # if hasattr(self.ribbon_bar, 'tabBar'):
            #     tab_bar = self.ribbon_bar.tabBar()
            #     if tab_bar:
            #         tab_bar.setMouseTracking(True)  # Enable mouse tracking for hover events
            #         tab_bar.installEventFilter(self)
            #         self.logger.info("Installed event filter on tab bar for hover tab switching")

            # Setup auto-fold functionality
            self._setup_auto_fold()

            # Setup pin button
            self._setup_pin_button()

            # Setup quick access bar
            self._setup_quick_access_bar()

            # Create ribbon categories
            self.categories = {}
            # Create File category first (as the first tab)
            self.categories['File'] = self._create_file_category()
            # Create Main category with default actions
            self.categories['Main'] = self._create_main_category()
            # Plugin categories are created dynamically in _create_plugins_category
            self._create_plugins_category()
            # Create notebooks check directly - added here to be created on setup if and only if jupyter address exists
            self._create_notebooks_category()

            # Apply global alignment fix to all panels after everything is created
            from qtpy import QtCore
            QtCore.QTimer.singleShot(200, self._fix_all_panel_alignments)

            # Set the ribbon to always start on the Main tab
            QtCore.QTimer.singleShot(300, self._set_main_tab_as_default)

            self.logger.info("Ribbon interface setup completed successfully")
            return True

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to setup ribbon interface: {e}")
            self.logger.error(f"DEBUG: Exception traceback: {traceback.format_exc()}")
            return False

    def _apply_background_fix(self):
        """Apply background fix to prevent white area in MDI area and central widget"""
        try:
            # Fix the white area in MDI area by setting proper dark background
            if hasattr(self.main_window, 'mdiarea'):
                # Set MDI area background to match dark theme
                self.main_window.mdiarea.setStyleSheet("""
                    QMdiArea {
                        background-color: #353535;
                        border: none;
                    }
                    QMdiArea > QWidget {
                        background-color: #353535;
                    }
                """)
                self.logger.info("Applied immediate MDI area dark background fix")

            # Also fix central widget background if needed
            if hasattr(self.main_window, 'centralwidget'):
                self.main_window.centralwidget.setStyleSheet("""
                    QWidget#centralwidget {
                        background-color: #353535;
                    }
                """)
                self.logger.info("Applied immediate central widget dark background fix")

        except Exception as e:
            self.logger.warning(f"Failed to apply immediate background fix: {e}")

    def _apply_title_widget_fix(self):
        """Apply additional fixes for title widget and empty space issues"""
        try:
            # Fix title widget background if it exists
            if hasattr(self.ribbon_bar, 'titleWidget'):
                title_widget = self.ribbon_bar.titleWidget()
                if title_widget:
                    title_widget.setStyleSheet("""
                        RibbonTitleWidget {
                            background-color: #353535;
                            border: none;
                        }
                        RibbonTitleWidget * {
                            background-color: #353535;
                            border: none;
                        }
                        RibbonTitleLabel {
                            background-color: #353535;
                            color: #ffffff;
                        }
                        QToolBar {
                            background-color: #353535;
                            border: none;
                        }
                        QTabBar {
                            background-color: #353535;
                            border: none;
                        }
                    """)
                    self.logger.info("Applied title widget background fix")

            # Fix any potential empty space in the ribbon bar layout
            if hasattr(self.ribbon_bar, '_titleWidget'):
                title_widget = self.ribbon_bar._titleWidget
                if title_widget:
                    # Ensure the title label has proper background
                    if hasattr(title_widget, '_titleLabel'):
                        title_widget._titleLabel.setStyleSheet("""
                            QLabel {
                                background-color: #353535;
                                color: #ffffff;
                                border: none;
                            }
                        """)

                    # Ensure all child widgets have proper background
                    for child in title_widget.findChildren(QtWidgets.QWidget):
                        if child.objectName() == '' or 'title' in child.objectName().lower():
                            child.setStyleSheet("background-color: #353535; border: none;")

                    self.logger.info("Applied title widget child fixes")

            # Apply fix to the entire ribbon bar to catch any missed areas
            self.ribbon_bar.setStyleSheet(self.ribbon_bar.styleSheet() + """
                RibbonBar QWidget {
                    background-color: #353535;
                    border: none;
                }
                RibbonBar QLayout {
                    background-color: #353535;
                }
                RibbonBar QSpacerItem {
                    background-color: #353535;
                }
                /* Reduce spacing in ribbon items */
                RibbonPanel {
                    margin: 1px;
                    padding: 1px;
                }
                RibbonPanel > QWidget {
                    margin: 0px;
                    padding: 0px;
                }
                QToolButton {
                    margin: 1px;
                    padding: 2px;
                }
                QToolButton::button {
                    margin: 1px;
                    padding: 2px;
                }
                /* Reduce spacing between buttons in panels */
                RibbonPanel QLayout {
                    spacing: 1px;
                    margin: 1px;
                }
            """)

            self.logger.info("Applied comprehensive title widget and layout fixes")

        except Exception as e:
            self.logger.warning(f"Failed to apply title widget fix: {e}")

    def _setup_resize_handler(self):
        """Setup resize event handler to catch and fix white background issues"""
        try:
            # Install event filter on the main window to catch resize events
            self.main_window.installEventFilter(self)
            self.logger.info("Installed resize event filter")
        except Exception as e:
            self.logger.warning(f"Failed to setup resize handler: {e}")

    def _apply_dark_palette(self):
        """Apply dark palette to the application like the demo"""
        try:
            from qtpy.QtGui import QPalette, QColor
            from qtpy.QtCore import Qt

            app = QtWidgets.QApplication.instance()
            if app is None:
                self.logger.warning("No QApplication instance found")
                return

            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)

            app.setPalette(dark_palette)
            self.logger.debug("Applied dark palette successfully")

        except Exception as e:
            self.logger.error(f"Failed to apply dark palette: {e}")

    def set_ribbon_max_rows(self, row_count):
        """
        Set the maximum number of rows for ribbon categories.

        Parameters
        ----------
        row_count : int
            The maximum number of rows for ribbon panels (recommended: 1-3)
        """
        try:
            # Validate row_count
            if not isinstance(row_count, int) or row_count < 1 or row_count > 5:
                self.logger.warning(f"Invalid row_count {row_count}, must be integer between 1-5")
                return

            # Update the internal max rows value
            if hasattr(self.ribbon_bar, '_maxRows'):
                self.ribbon_bar._maxRows = row_count
                self.logger.info(f"Set ribbon max rows to {row_count}")

                # Update all existing categories to use the new max rows
                if hasattr(self.ribbon_bar, '_categories'):
                    for category in self.ribbon_bar._categories.values():
                        if hasattr(category, 'setMaximumRows'):
                            category.setMaximumRows(row_count)
                    self.logger.info(f"Updated all categories to use {row_count} max rows")

                    # Save to settings for persistence
                    import chisurf
                    gui_settings = chisurf.settings.cs_settings.get('gui', {})
                    ribbon_settings = gui_settings.get('ribbon', {})
                    ribbon_settings['max_rows'] = row_count
                    self.logger.info(f"Saved max_rows={row_count} to settings")
                else:
                    self.logger.warning("Ribbon bar has no _categories attribute")
            else:
                self.logger.warning("Ribbon bar has no _maxRows attribute")

        except Exception as e:
            self.logger.error(f"Failed to set ribbon max rows: {e}")

    def _set_main_tab_as_default(self):
        """Set the Main tab as the default ribbon tab"""
        try:
            if self.ribbon_bar and hasattr(self.ribbon_bar, '_titleWidget'):
                tab_bar = self.ribbon_bar._titleWidget.tabBar()
                if tab_bar:
                    # Find the index of the Main tab
                    main_tab_index = tab_bar.indexOf('Main')
                    if main_tab_index >= 0:
                        tab_bar.setCurrentIndex(main_tab_index)
                        self.ribbon_bar.showCategoryByIndex(main_tab_index)
                        self.logger.info(f"Set Main tab as default (index {main_tab_index})")
                    else:
                        self.logger.warning("Main tab not found in ribbon")
                else:
                    self.logger.warning("Ribbon tab bar not available")
            else:
                self.logger.warning("Ribbon bar not available")
        except Exception as e:
            self.logger.error(f"Failed to set Main tab as default: {e}")

    def set_ribbon_dark_theme(self, enabled=True):
        """
        Enable or disable dark theme for the ribbon using palette approach.

        Parameters
        ----------
        enabled : bool, optional
            Whether to enable dark theme (default: True)
        """
        try:
            if enabled:
                self._apply_dark_palette()
                self.logger.info("Applied dark theme using palette")
            else:
                # Restore default palette
                app = QtWidgets.QApplication.instance()
                if app:
                    app.setPalette(app.style().standardPalette())
                    self.logger.info("Restored default palette")
                else:
                    self.logger.warning("No QApplication instance found")

        except Exception as e:
            self.logger.error(f"Failed to set ribbon dark theme: {e}")

    def apply_compact_spacing(self):
        """
        Apply compact spacing to ribbon items to reduce space between buttons and panels
        while keeping button text readable.
        """
        try:
            if self.ribbon_bar:
                # Apply compact styling focused on spacing, not button size
                compact_stylesheet = """
                    /* Compact ribbon styling - focus on spacing gaps */
                    RibbonPanel {
                        margin: 0px;
                        padding: 1px;
                    }
                    RibbonPanel > QWidget {
                        margin: 0px;
                        padding: 0px;
                    }
                    RibbonPanel QLayout {
                        spacing: 1px;
                        margin: 0px;
                    }
                    QToolButton {
                        margin: 0px;
                        padding: 2px;
                    }
                    QToolButton::button {
                        margin: 0px;
                        padding: 2px;
                    }
                    /* Reduce category tab spacing */
                    QTabBar::tab {
                        padding: 3px 8px;
                        margin: 0px 1px;
                    }
                    /* Compact panel headers */
                    RibbonPanel > QLabel {
                        margin: 0px;
                        padding: 1px;
                    }
                """

                # Get existing stylesheet and append compact styles
                existing_stylesheet = self.ribbon_bar.styleSheet() or ""
                self.ribbon_bar.setStyleSheet(existing_stylesheet + compact_stylesheet)

                self.logger.debug("Applied compact spacing to ribbon (text readable)")

        except Exception as e:
            self.logger.error(f"Failed to apply compact spacing: {e}")

    def _restore_original_styling(self):
        """Restore original styling when switching back to menu mode"""
        try:
            # Restore MDI area original styling
            if hasattr(self.main_window, 'mdiarea'):
                # Clear the custom stylesheet to restore default appearance
                self.main_window.mdiarea.setStyleSheet("")
                self.logger.info("Restored MDI area original styling")

            # Restore central widget original styling
            if hasattr(self.main_window, 'centralwidget'):
                # Clear the custom stylesheet to restore default appearance
                self.main_window.centralwidget.setStyleSheet("")
                self.logger.info("Restored central widget original styling")

            # Ensure dock widgets are visible and properly styled
            dock_widgets = [
                'dockWidgetAnalysis', 'dockWidgetReadData', 'dockWidgetDatasets',
                'dockWidgetPlot', 'dockWidgetScriptEdit', 'dockWidget_console'
            ]

            for dock_name in dock_widgets:
                if hasattr(self.main_window, dock_name):
                    dock = getattr(self.main_window, dock_name)
                    # Clear any custom styling
                    dock.setStyleSheet("")
                    # Ensure dock is visible (it should be if it was visible before)
                    if dock.isVisible():
                        self.logger.info(f"Restored {dock_name} styling")

            # Ensure status bar is visible
            if hasattr(self.main_window, 'status') and self.main_window.statusBar():
                self.main_window.statusBar().show()
                self.logger.info("Ensured status bar is visible")

        except Exception as e:
            self.logger.warning(f"Failed to restore original styling: {e}")

    def restore_original_interface(self):
        """Restore the original interface by removing ribbon widget and showing original menu bar"""
        try:
            # Move ribbon bar to hidden widget to preserve it
            if self.ribbon_bar:
                self.ribbon_bar.setParent(self.ribbon_preserve_widget)
                self.logger.info("Ribbon bar moved to hidden widget for preservation")

            # Remove ribbon widget
            self.main_window.setMenuWidget(None)
            self.logger.info("Ribbon widget removed")

            # Move the menu bar back from the hidden widget and set it as the menu bar
            if self.original_menubar:
                # Move menu bar back to main window
                self.original_menubar.setParent(self.main_window)
                # Ensure it is native on macOS to restore system menu bar
                if sys.platform == 'darwin':
                    self.original_menubar.setNativeMenuBar(True)
                # Set it as the menu bar
                self.main_window.setMenuBar(self.original_menubar)
                self.original_menubar.show()
                self.logger.info("Menu bar restored from hidden widget and set as menu bar")
            else:
                self.logger.warning("No original menu bar reference found")

            # Show original toolbar if it exists
            if self.original_toolbar:
                self.original_toolbar.show()
                self.logger.info("Original toolbar restored")

            # Show plugin toolbar when switching back to menu mode
            if hasattr(self.main_window, 'plugins_toolbar'):
                self.main_window.plugins_toolbar.show()
                self.logger.info("Plugin toolbar restored for menu mode")

            # Restore original styling when switching back to menu mode
            self._restore_original_styling()

            # No geometry updates - removed to prevent window movement

            self.logger.info("Original interface restored - menu bar preserved")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore original interface: {e}")
            return False
