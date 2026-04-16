# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration - Auto-Fold Module

This module contains auto-fold functionality and pin button methods for the ribbon interface.
"""

from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolButton
from PyQt5 import QtWidgets

import chisurf
from chisurf import logging


class AutoFoldMethodsMixin:
    """Mixin class containing auto-fold and pin button methods for ChiSurfRibbonIntegration"""

    def _setup_auto_fold(self):
        """Setup auto-fold functionality based on settings"""
        try:
            import chisurf
            gui_settings = chisurf.settings.cs_settings.get('gui', {})
            ribbon_settings = gui_settings.get('ribbon', {})

            self.auto_fold_enabled = ribbon_settings.get('auto_fold', True)  # Default to True
            self.auto_fold_delay_ms = ribbon_settings.get('auto_fold_delay_ms', 1000)  # Default to 1 second
            self.auto_fold_speed_ms = ribbon_settings.get('auto_fold_speed_ms', 500)  # Default to 500ms

            self.mouse_over_ribbon = False  # Track if mouse is over ribbon

            self.logger.debug(f"Auto-fold setup: enabled={self.auto_fold_enabled}, delay={self.auto_fold_delay_ms}ms, speed={self.auto_fold_speed_ms}ms")

            if self.auto_fold_enabled and self.ribbon_bar:
                # Install event filter to track mouse enter/leave on ribbon
                self.ribbon_bar.installEventFilter(self)

                # Also install on tab bar to catch clicks
                tab_bar = self.ribbon_bar.tabBar()
                if tab_bar:
                    tab_bar.installEventFilter(self)
                    self.logger.debug("Installed event filter on tab bar")

                # IMPORTANT: Connect tab bar changes to show ribbon (like collapse button does)
                self.ribbon_bar.tabBar().currentChanged.connect(self._on_tab_changed)

                # Start/restart the timer with the correct delay
                self._restart_auto_fold_timer()

                self.logger.debug(f"Auto-fold enabled: {self.auto_fold_delay_ms}ms delay, {self.auto_fold_speed_ms}ms speed")
                self.logger.debug("Auto-fold will trigger when mouse leaves ribbon area")
            else:
                self.logger.debug("Auto-fold disabled")

        except Exception as e:
            self.logger.warning(f"Failed to setup auto-fold: {e}")
            # Set defaults even if setup fails
            self.auto_fold_enabled = True
            self.auto_fold_delay_ms = 1000
            self.auto_fold_speed_ms = 500
            self.logger.info("Using default auto-fold settings due to error")

    def _setup_pin_button(self):
        """Setup pin button to control auto-fold functionality"""
        try:
            if self.ribbon_bar:
                # Load settings
                import chisurf
                gui_settings = chisurf.settings.cs_settings.get('gui', {})
                ribbon_settings = gui_settings.get('ribbon', {})

                # Load pin state from settings, default to pinned
                self.is_pinned = ribbon_settings.get('pinned', True)

                # Create pin button
                from PyQt5.QtWidgets import QToolButton
                from PyQt5.QtGui import QIcon
                from PyQt5.QtCore import QSize

                self.pin_button = QToolButton()
                self.pin_button.setIconSize(QSize(20, 20))
                self.pin_button.setAutoRaise(True)
                self.pin_button.setCheckable(True)
                self.pin_button.setChecked(self.is_pinned)

                # Set initial pin icon (based on current state)
                self._update_pin_button_appearance()

                # Connect to toggle auto-fold
                self.pin_button.clicked.connect(self._toggle_pin)

                # Add pin button to the ribbon bar using the proper API
                self.ribbon_bar.addRightToolButton(self.pin_button)

                # Synchronize auto-fold state with pin state
                # If pinned (default), disable auto-fold; if unpinned, enable it
                if self.is_pinned:
                    self.auto_fold_timer.stop()
                    self.logger.debug("Ribbon starts pinned - auto-fold timer stopped")

                # Create and add help button
                help_button = QToolButton()
                help_button.setIconSize(QSize(20, 20))
                help_button.setAutoRaise(True)
                # Set question mark icon - try multiple standard icons
                icon_set = False
                help_icons = ['help-contents', 'help-about', 'question-mark', 'dialog-question']
                for icon_name in help_icons:
                    try:
                        icon = QIcon.fromTheme(icon_name)
                        if not icon.isNull():
                            help_button.setIcon(icon)
                            icon_set = True
                            # Apply red color styling for theme icons
                            help_button.setStyleSheet("QToolButton { color: red; }")
                            break
                    except Exception:
                        continue
                
                # If no theme icon works, create a simple text-based question mark
                if not icon_set:
                    help_button.setText("?")
                    help_button.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
                
                help_button.setToolTip("Open Help Plugin")
                help_button.clicked.connect(self.main_window.onOpenHelp)
                self.ribbon_bar.addRightToolButton(help_button)

                self.logger.debug(f"Pin button added to ribbon bar (pinned: {self.is_pinned})")
                self.logger.debug("Help button added to ribbon bar")
            else:
                self.logger.warning("Could not setup pin button - ribbon bar not available")

        except Exception as e:
            self.logger.error(f"Failed to setup pin button: {e}")

    def _update_pin_button_appearance(self):
        """Update pin button icon and tooltip based on pinned state"""
        if not self.pin_button:
            return

        try:
            from PyQt5.QtGui import QIcon
            from PyQt5.QtCore import QSize

            if self.is_pinned:
                # Pinned state - use a "pinned" icon or create one
                # For now, we'll use a simple approach with text
                self.pin_button.setText("P")
                self.pin_button.setToolTip("Unpin ribbon (enable auto-fold)")
                self.pin_button.setStyleSheet("QToolButton { color: #4CAF50; font-weight: bold; }")
            else:
                # Unpinned state
                self.pin_button.setText("L")
                self.pin_button.setToolTip("Pin ribbon (disable auto-fold)")
                self.pin_button.setStyleSheet("QToolButton { color: #CCCCCC; }")

        except Exception as e:
            self.logger.warning(f"Failed to update pin button appearance: {e}")

    def _toggle_pin(self, checked):
        """Toggle pin state and auto-fold functionality"""
        self.is_pinned = checked
        self._update_pin_button_appearance()

        # Save pin state to settings and persist to file
        try:
            import chisurf
            import yaml
            from pathlib import Path
            gui_settings = chisurf.settings.cs_settings.get('gui', {})
            ribbon_settings = gui_settings.get('ribbon', {})
            ribbon_settings['pinned'] = self.is_pinned

            # Persist to user settings file
            settings_file = chisurf.settings.chisurf_settings_path / 'settings_chisurf.yaml'
            with open(settings_file, 'w', encoding='utf-8') as fh:
                yaml.safe_dump(chisurf.settings.cs_settings, fh, default_flow_style=False)
            self.logger.debug(f"Saved pin state ({self.is_pinned}) to user settings file")
        except Exception as e:
            self.logger.warning(f"Failed to save pin state to settings: {e}")

        if self.is_pinned:
            # Pin the ribbon - disable auto-fold
            self.toggle_auto_fold(enabled=False)
            self.logger.debug("Ribbon pinned - auto-fold disabled")

            # Show ribbon if currently hidden
            if self.is_folded:
                self._unfold_ribbon()
        else:
            # Unpin the ribbon - enable auto-fold
            self.toggle_auto_fold(enabled=True)
            self.logger.debug("Ribbon unpinned - auto-fold enabled")

    def _restart_auto_fold_timer(self):
        """Restart the auto-fold timer"""
        if self.auto_fold_enabled:
            self.auto_fold_timer.stop()
            self.auto_fold_timer.start(self.auto_fold_delay_ms)
            self.last_activity_time.start()
            msg = f"Timer STARTED: {self.auto_fold_delay_ms}ms until auto-fold"
            self.logger.debug(msg)

    def _auto_fold_ribbon(self):
        """Automatically hide the ribbon (like collapse button)"""
        try:
            msg = f"Auto-fold timer expired - checking conditions"
            self.logger.debug(msg)

            # Don't auto-fold if ribbon is pinned
            if self.is_pinned:
                msg = "Auto-fold skipped - ribbon is pinned"
                self.logger.debug(msg)
                return

            if self.ribbon_bar and not self.is_folded and not self.mouse_over_ribbon:
                msg = "Auto-fold triggered - hiding ribbon"
                self.logger.debug(msg)
                # Use the same mechanism as the collapse button
                self.ribbon_bar.hideRibbon()
                self.is_folded = True
                msg = "Ribbon auto-folded (hidden)"
                self.logger.debug(msg)
            else:
                msg = f"Auto-fold skipped: ribbon_bar={self.ribbon_bar is not None}, is_folded={self.is_folded}, mouse_over_ribbon={self.mouse_over_ribbon}, is_pinned={self.is_pinned}"
                self.logger.debug(msg)
        except Exception as e:
            msg = f"Failed to auto-fold ribbon: {e}"
            self.logger.error(msg)

    def _unfold_ribbon(self):
        """Unfold the ribbon back to visible state"""
        try:
            if self.ribbon_bar and self.is_folded:
                # Use animation timer for smooth unfolding, then show ribbon
                QTimer.singleShot(self.auto_fold_speed_ms, self.ribbon_bar.showRibbon)
                self.is_folded = False
                self.logger.debug("Ribbon unfolded (shown)")
        except Exception as e:
            self.logger.error(f"Failed to unfold ribbon: {e}")

    def toggle_auto_fold(self, enabled=None):
        """Toggle or set auto-fold functionality

        Parameters
        ----------
        enabled : bool, optional
            If provided, set auto-fold to this state.
            If None, toggle current state.
        """
        if enabled is None:
            enabled = not self.auto_fold_enabled

        # If disabling, disconnect tab bar signal and remove event filter
        if not enabled and self.auto_fold_enabled and self.ribbon_bar:
            try:
                self.ribbon_bar.tabBar().currentChanged.disconnect(self._on_tab_changed)
                self.logger.debug("Disconnected tab bar signal")
            except Exception:
                pass  # Signal might not be connected
            self.ribbon_bar.removeEventFilter(self)
            # Also remove from tab bar
            tab_bar = self.ribbon_bar.tabBar()
            if tab_bar:
                tab_bar.removeEventFilter(self)
                self.logger.debug("Removed tab bar event filter")
            self.logger.debug("Removed ribbon event filter")

        self.auto_fold_enabled = enabled

        if enabled and self.ribbon_bar:
            # Install event filter on ribbon and tab bar
            self.ribbon_bar.installEventFilter(self)
            tab_bar = self.ribbon_bar.tabBar()
            if tab_bar:
                tab_bar.installEventFilter(self)
                self.logger.debug("Installed event filter on tab bar")
            # Reconnect tab bar signal
            try:
                self.ribbon_bar.tabBar().currentChanged.connect(self._on_tab_changed)
                self.logger.debug("Connected tab bar signal")
            except Exception:
                pass
            # Start the auto-fold timer
            self._restart_auto_fold_timer()
            self.logger.debug("Auto-fold enabled - tracking mouse enter/leave on ribbon")
        else:
            if self.ribbon_bar:
                self.ribbon_bar.removeEventFilter(self)
                tab_bar = self.ribbon_bar.tabBar()
                if tab_bar:
                    tab_bar.removeEventFilter(self)
            self.auto_fold_timer.stop()
            self.logger.debug("Auto-fold disabled")

            # Show ribbon if currently hidden
            if self.is_folded:
                self.ribbon_bar.showRibbon()
                self.is_folded = False

    def set_auto_fold_settings(self, delay_ms=None, speed_ms=None):
        """Update auto-fold settings

        Parameters
        ----------
        delay_ms : int, optional
            Delay before auto-folding starts in milliseconds
        speed_ms : int, optional
            Speed of folding animation in milliseconds
        """
        if delay_ms is not None:
            self.auto_fold_delay_ms = delay_ms
        if speed_ms is not None:
            self.auto_fold_speed_ms = speed_ms

        # Restart timer with new delay if enabled
        if self.auto_fold_enabled:
            self._restart_auto_fold_timer()

        self.logger.debug(f"Auto-fold settings updated: delay={self.auto_fold_delay_ms}ms, speed={self.auto_fold_speed_ms}ms")

    def _on_tab_changed(self, index):
        """Handle tab bar changes - show ribbon when user switches tabs (like collapse button)"""
        if self.is_folded:
            self.logger.debug(f"Tab changed to index {index}, showing ribbon")
            self._unfold_ribbon()

    def get_auto_fold_status(self):
        """Get current auto-fold status and settings"""
        return {
            'enabled': self.auto_fold_enabled,
            'is_folded': self.is_folded,
            'mouse_over_ribbon': self.mouse_over_ribbon,
            'is_pinned': self.is_pinned,
            'delay_ms': self.auto_fold_delay_ms,
            'speed_ms': self.auto_fold_speed_ms,
            'ribbon_visible': self.ribbon_bar._stackedWidget.isVisible() if self.ribbon_bar else None
        }
