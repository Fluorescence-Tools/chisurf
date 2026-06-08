# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration - Utilities Module

This module contains utility functions and convenience methods for the ribbon interface.
"""

from qtpy.QtCore import QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QMessageBox

import chisurf as cs
from chisurf import logging


class UtilityMethodsMixin:
    """Mixin class containing utility methods for ChiSurfRibbonIntegration"""

    def _backup_menu_actions(self):
        """Backup actions from original menus to prevent deletion"""
        try:
            self.menu_actions_backup = {}

            # Backup actions from all menus
            menu_names = ['menuFile', 'menuEdit', 'menuView', 'menuAnalysis', 'menuTools', 'menuHelp']

            for menu_name in menu_names:
                if hasattr(self.main_window, menu_name):
                    menu = getattr(self.main_window, menu_name)
                    if menu:
                        actions = []
                        for action in menu.actions():
                            if not action.isSeparator():
                                # Store action reference to prevent deletion
                                actions.append(action)
                        self.menu_actions_backup[menu_name] = actions
                        self.logger.info(f"Backed up {len(actions)} actions from {menu_name}")

            self.logger.info("Menu actions backup completed")

        except Exception as e:
            self.logger.warning(f"Failed to backup menu actions: {e}")
            self.menu_actions_backup = {}

    def _check_updates(self):
        """Check for ChiSurf updates"""
        QMessageBox.information(
            self.main_window,
            'Check Updates',
            'Update check functionality would be implemented here.\n\n'
            'This would check for new versions of ChiSurf\n'
            'and available plugin updates.'
        )

    def _fix_ribbon_fonts(self):
        """Fix font sizes for ribbon to match normal application fonts"""
        try:
            # Get the default application font
            app_font = QtWidgets.QApplication.font()

            # Keep ribbon text at the same readable size as parameter widgets.
            ribbon_font = QFont(app_font)

            # Apply font to ribbon bar
            self.ribbon_bar.setFont(ribbon_font)

            # Apply font to all child widgets recursively
            self._apply_font_recursively(self.ribbon_bar, ribbon_font)

            # Force font update
            self.ribbon_bar.update()

            self.logger.info("Applied font fixes to ribbon bar")

        except Exception as e:
            self.logger.warning(f"Failed to fix ribbon fonts: {e}")

    def _apply_font_recursively(self, widget, font):
        """Recursively apply font to widget and all its children"""
        try:
            # Apply font to current widget
            widget.setFont(font)

            # Apply to all children
            for child in widget.children():
                if isinstance(child, QtWidgets.QWidget):
                    self._apply_font_recursively(child, font)
        except Exception:
            # Skip widgets that don't support font setting
                pass

    def _add_ribbon_style_actions(self, panel):
        """Add ribbon style selection actions"""
        # Ribbon uses different style system - for now we'll skip style switching
        # This functionality can be added later if needed
        pass

    def _on_style_clicked(self, style):
        """Handle ribbon style button click"""
        if self.ribbon_bar:
            self.ribbon_bar.setRibbonStyle(style)

            # Save to settings
            gui_settings = cs.core.settings.cs_settings.get('gui', {})
            gui_settings['ribbon_style'] = style

            self.logger.info(f"Ribbon style changed to {style}")

    def _switch_theme(self, theme_name):
        """Switch application theme"""
        QMessageBox.information(
            self.main_window,
            'Theme Switch',
            f'Theme switching to {theme_name} would be implemented here.\n\n'
            'This would change the application color scheme\n'
            'and update all UI components accordingly.'
        )

    def switch_to_menu(self):
        """Switch back to traditional menu bar"""
        try:
            # Clean up ribbon actions to prevent SARibbon disconnect errors
            if self.ribbon_bar:
                self._cleanup_ribbon_actions()

            # Restore original interface
            if self.restore_original_interface():
                # Update the main window's ribbon integration reference
                self.main_window._ribbon_integration = None

                # Update the toggle action state if it exists
                if hasattr(self.main_window, 'actionToggle_Ribbon'):
                    self.main_window.actionToggle_Ribbon.setChecked(False)

                # Update settings
                gui_settings = cs.core.settings.cs_settings.get('gui', {})
                gui_settings['use_ribbon_interface'] = False

                self.logger.info("Switched back to traditional menu bar")

            else:
                raise Exception("Failed to restore original interface")

        except Exception as e:
            self.logger.error(f"Failed to switch to menu: {e}")
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.main_window,
                "Error",
                f"Failed to switch back to menu bar: {str(e)}"
            )

    def _cleanup_ribbon_actions(self):
        """Clean up ribbon actions to prevent disconnect errors"""
        try:
            # Ribbon doesn't seem to have the same cleanup API as SARibbon
            # For now, we'll skip the complex cleanup since Ribbon should handle this better
            self.logger.info("Ribbon actions cleanup skipped (ribbon handles this internally)")

        except Exception as e:
            self.logger.warning(f"Failed to cleanup ribbon actions: {e}")

    def _cleanup_category_actions(self, category):
        """Clean up actions in a specific category"""
        # Simplified cleanup for ribbon
        pass

    def _cleanup_panel_actions(self, panel):
        """Clean up actions in a specific panel"""
        try:
            # Remove all actions from the panel to prevent disconnect errors
            # This is a workaround for the SARibbon library disconnect bug
            actions = panel.actions()
            for action in actions:
                try:
                    # Try to disconnect the action safely
                    if hasattr(action, 'disconnect'):
                        action.disconnect()
                except Exception:
                    # Ignore disconnect errors - this is expected due to the SARibbon bug
                    pass

        except Exception as e:
            self.logger.warning(f"Failed to cleanup panel actions: {e}")

    def _show_plugin_manager(self):
        """Show plugin manager dialog"""
        QMessageBox.information(
            self.main_window,
            'Plugin Manager',
            'Plugin manager functionality would be implemented here.\n\n'
            'This would allow users to enable/disable plugins,\n'
            'configure plugin settings, and install new plugins.'
        )

    def _open_code_editor(self):
        """Open code editor"""
        try:
            import chisurf.plugins.core.code_editor
            # Code editor integration would go here
            QMessageBox.information(
                self.main_window,
                'Code Editor',
                'Code editor would be opened here.\n\n'
                'This provides a Python code editor for\n'
                'custom scripts and data analysis.'
            )
        except ImportError:
            QMessageBox.warning(
                self.main_window,
                'Code Editor',
                'Code editor plugin is not available.'
            )

    def _add_theme_actions(self, panel):
        """Add theme-related actions to a panel"""
        # Light theme
        action_light = QAction('Light Theme', self.main_window)
        action_light.setStatusTip('Switch to light theme')
        action_light.triggered.connect(lambda: self._switch_theme('light'))
        panel.addSmallButton(action_light.text(), icon=action_light.icon() if action_light.icon() else None, showText=True, slot=action_light.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Dark theme
        action_dark = QAction('Dark Theme', self.main_window)
        action_dark.setStatusTip('Switch to dark theme')
        action_dark.triggered.connect(lambda: self._switch_theme('dark'))
        panel.addSmallButton(action_dark.text(), icon=action_dark.icon() if action_dark.icon() else None, showText=True, slot=action_dark.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

    def _add_standard_help_actions(self, category):
        """Add standard help actions to Help category"""
        # Documentation panel
        panel = category.addPanel('Documentation', showPanelOptionButton=False)

        # About action
        if hasattr(self.main_window, 'actionAbout'):
            action = self.main_window.actionAbout
            # Add info icon
            try:
                # Use generic icon since ribbon doesn't have built-in icons
                action.setIcon(QIcon.fromTheme('help-about'))
            except Exception:
                pass
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Help action
        if hasattr(self.main_window, 'actionHelp'):
            action = self.main_window.actionHelp
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)
