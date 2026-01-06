# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration - Categories Module

This module contains category creation methods for the ribbon interface.
"""

import functools
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from PyQt5 import QtWidgets

import chisurf
from chisurf import logging


class CategoryMethodsMixin:
    """Mixin class containing category creation methods for ChiSurfRibbonIntegration"""

    def _setup_quick_access_bar(self):
        """Setup quick access bar with common actions following playground example"""
        try:
            quick_access_toolbar = self.ribbon_bar.quickAccessToolBar()

            # Add save action if available
            if hasattr(self.main_window, 'actionSave_Data'):
                action = self.main_window.actionSave_Data
                # Create button for quick access
                save_button = QtWidgets.QToolButton()
                save_button.setDefaultAction(action)
                save_button.setAutoRaise(True)
                self.ribbon_bar.addQuickAccessButton(save_button)

            # Add separator - pyqtribbon may handle this differently
            # For now, we'll skip separators as they may not be directly supported

            # Add undo action if available
            if hasattr(self.main_window, 'actionUndo'):
                action = self.main_window.actionUndo
                # Create button for quick access
                undo_button = QtWidgets.QToolButton()
                undo_button.setDefaultAction(action)
                undo_button.setAutoRaise(True)
                self.ribbon_bar.addQuickAccessButton(undo_button)

            # Add redo action if available
            if hasattr(self.main_window, 'actionRedo'):
                action = self.main_window.actionRedo
                # Create button for quick access
                redo_button = QtWidgets.QToolButton()
                redo_button.setDefaultAction(action)
                redo_button.setAutoRaise(True)
                self.ribbon_bar.addQuickAccessButton(redo_button)

            self.logger.info("Quick access bar setup completed")

        except Exception as e:
            self.logger.warning(f"Failed to setup quick access bar: {e}")

    def _create_main_category(self):
        """Create Main category with all actions from main window toolbar and setup/help plugins"""
        category = self.ribbon_bar.addCategory('Main')

        # Get all toolbar actions from main window
        toolbar_actions = []
        if hasattr(self.main_window, 'toolBar') and self.main_window.toolBar:
            toolbar_actions = self.main_window.toolBar.actions()
            self.logger.info(f"Found {len(toolbar_actions)} actions in main toolbar")
        else:
            self.logger.warning("No toolbar found in main window")

        # Create panels for organizing toolbar actions
        if toolbar_actions:
            # Define file-related actions that are already handled in File category
            file_actions = ['actionLoad_Data', 'actionSave_Data', 'actionImport', 'actionExport']

            # Group actions by type/size
            large_actions = []
            medium_actions = []
            small_actions = []

            for action in toolbar_actions:
                if action and not action.isSeparator():
                    # Determine action size based on icon and text
                    if action.icon() and (action.text() in ['New', 'Open', 'Save', 'Settings', 'Quit']):
                        large_actions.append(action)
                    elif action.icon():
                        medium_actions.append(action)
                    else:
                        small_actions.append(action)


            # Add remaining toolbar actions that weren't categorized yet
            remaining_actions = [action for action in toolbar_actions
                                if action and not action.isSeparator()]

            if remaining_actions:
                panel_other = category.addPanel('Other Tools')
                for action in remaining_actions[:8]:  # Limit to 8 actions
                    # Use small action for text below icon layout
                    btn = panel_other.addSmallButton(action.text(), icon=action.icon() if action.icon() else None, showText=True, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)
                    # Make icon smaller if it exists
                    if action.icon():
                        btn.setMaximumIconSize(14)
                    self.logger.info(f"Added remaining action '{action.text()}' to Other Tools panel")


        else:
            # Fallback: add basic actions without toolbar
            self._add_fallback_main_actions(category)

        # Add Setup plugins to Main category
        self._add_setup_plugins_to_main(category)

        # Add Help plugins to Main category
        self._add_help_plugins_to_main(category)

        return category

    def _add_setup_plugins_to_main(self, category):
        """Add Setup plugins to Main category with hierarchical submenu support"""
        try:
            import functools
            from pathlib import Path

            # Get plugin settings
            plugin_settings = chisurf.settings.cs_settings.get('plugins', {})
            disabled_plugins = plugin_settings.get('disabled_plugins', [])
            hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
            plugin_order = plugin_settings.get('plugin_order', {})
            experimental_mode = chisurf.settings.cs_settings.get('enable_experimental', False)

            # Discover plugins
            try:
                plugin_infos = list(chisurf.plugins.iter_plugins())
            except Exception as e:
                self.logger.error(f"Failed to enumerate plugins: {e}")
                return

            # Sort plugins
            ordered = []
            for info in plugin_infos:
                plugin_name = info.get('plugin_name') or info.get('module_name') or ''
                order = plugin_order.get(plugin_name, 0)
                ordered.append((order, plugin_name, info))
            ordered.sort(key=lambda x: (x[0], x[1]))

            # Resolve plugins root
            try:
                plugins_root = Path(chisurf.plugins.__file__).parent.resolve()
            except Exception:
                plugins_root = Path(chisurf.plugins.__file__).parent

            # Collect setup plugins with hierarchy support
            setup_plugins = []

            for _order, plugin_name, info in ordered:
                try:
                    if not plugin_name.startswith('Setup:'):
                        continue

                    module_path = info.get('module_path')
                    module_name = info.get('module_name') or ''
                    package_dir = Path(info.get('package_dir'))
                    source = info.get('source') or 'built-in'
                    is_cli_only = bool(info.get('cli_only'))

                    if bool(info.get('menu_hidden')):
                        continue

                    # Parse hierarchical name
                    hierarchy_parts, display_name = self._parse_hierarchical_plugin_name(plugin_name)

                    # Remove 'Setup' from hierarchy parts for cleaner organization
                    if hierarchy_parts and hierarchy_parts[0] == 'Setup':
                        hierarchy_parts = hierarchy_parts[1:]

                    # Check disabled/broken
                    clean_name = display_name
                    is_broken = (
                        plugin_name in disabled_plugins
                        or module_name in disabled_plugins
                        or clean_name in disabled_plugins
                    )

                    # Detect dev plugins
                    is_dev = False
                    try:
                        rel = package_dir.resolve().relative_to(plugins_root)
                        if rel.parts and rel.parts[0] == "_dev":
                            is_dev = True
                    except Exception:
                        is_dev = False

                    # Skip broken if hidden and not experimental
                    if is_broken and hide_disabled_plugins and not experimental_mode:
                        continue

                    # Determine script file
                    plugin_dir = package_dir
                    wizard_file = plugin_dir / "wizard.py"
                    script_file = wizard_file if wizard_file.is_file() else (plugin_dir / "__init__.py")

                    # Build callback
                    callback = functools.partial(
                        self.main_window.onRunMacro,
                        str(script_file),
                        executor='exec',
                        globals={'__name__': 'plugin'}
                    )

                    # Check for icon
                    icon = None
                    for _icon_name in ("icon.png", "icon.svg"):
                        icon_path = plugin_dir / _icon_name
                        if icon_path.exists():
                            icon = QIcon(str(icon_path))
                            break

                    # Get description
                    description = info.get('description') or "No description available."

                    # Determine display name
                    label_base = display_name
                    if is_cli_only:
                        label_base = f"{label_base} (CLI)"
                    label = f"{label_base} (BROKEN)" if is_broken else label_base

                    setup_plugins.append({
                        'label': label,
                        'callback': callback,
                        'icon': icon,
                        'description': description,
                        'enabled': not (is_broken or is_cli_only),
                        'plugin_name': plugin_name,
                        'hierarchy_parts': hierarchy_parts
                    })

                except Exception as e:
                    self.logger.error(f"Error processing setup plugin '{plugin_name}': {e}")
                    continue

            if setup_plugins:
                # Build hierarchical structure for setup plugins
                if any(plugin['hierarchy_parts'] for plugin in setup_plugins):
                    # Has hierarchy - use nested structure
                    plugin_structure = self._build_nested_plugin_structure(setup_plugins)

                    # Create hierarchical menu within the Setup Plugins panel
                    # Since we're within Main category, we'll organize by subpanels
                    self._add_hierarchical_plugins_to_panel(category, setup_plugins, 'Setup Plugins')
                else:
                    # No hierarchy - use simple flat gallery organization
                    panel_setup = category.addPanel('Setup Plugins')
                    # Add plugins directly as small buttons for better size control
                    for plugin_info in setup_plugins:
                        # Add as small button with text below icon for compact display
                        btn = panel_setup.addSmallButton(
                            plugin_info['label'],
                            icon=plugin_info['icon'],
                            showText=True,
                            slot=plugin_info['callback']
                        , alignment=Qt.AlignLeft | Qt.AlignTop)
                        btn.setEnabled(plugin_info['enabled'])
                        btn.setToolTip(plugin_info['description'])

                        # Make icon smaller if it exists
                        if plugin_info['icon']:
                            btn.setMaximumIconSize(14)

                    self.logger.info(f"Added {len(setup_plugins)} setup plugins to Main category (small button organization)")
            else:
                self.logger.info("No setup plugins found")

        except Exception as e:
            self.logger.error(f"Failed to add setup plugins to Main category: {e}")

    def _add_hierarchical_plugins_to_panel(self, category, plugins, base_panel_name):
        """
        Add hierarchical plugins to a category using galleries within panels for better organization.

        Parameters
        ----------
        category : RibbonCategory
            Ribbon category to add plugins to
        plugins : list
            List of plugin dictionaries with hierarchy information
        base_panel_name : str
            Base name for the main panel
        """
        if not plugins:
            return

        # Group plugins by their hierarchy
        hierarchy_groups = {}

        for plugin_info in plugins:
            hierarchy_parts = plugin_info['hierarchy_parts']

            if hierarchy_parts:
                # Use the first hierarchy level as the group
                group_name = hierarchy_parts[0]
            else:
                group_name = 'General'

            if group_name not in hierarchy_groups:
                hierarchy_groups[group_name] = []

            hierarchy_groups[group_name].append(plugin_info)

        # Create panels with galleries for each hierarchy group
        for group_name, group_plugins in hierarchy_groups.items():
            if len(hierarchy_groups) == 1:
                # Only one group, use the base panel name
                panel_name = base_panel_name
            else:
                # Multiple groups, use descriptive names
                panel_name = f"{base_panel_name} - {group_name}"

            # Create panel first
            panel = category.addPanel(panel_name)

            # Add plugins directly as small buttons for better size control
            for plugin_info in group_plugins:
                # Use the display label (should already be just the final name)
                display_label = plugin_info['label']

                # Add as small button with text below icon for compact display
                btn = panel.addSmallButton(
                    display_label,
                    icon=plugin_info['icon'],
                    showText=True,
                    slot=plugin_info['callback'],
                    alignment=Qt.AlignLeft | Qt.AlignTop
                )
                btn.setEnabled(plugin_info['enabled'])
                btn.setToolTip(plugin_info['description'])

                # Make icon smaller if it exists
                if plugin_info['icon']:
                    btn.setMaximumIconSize(14)

            self.logger.info(f"Added {len(group_plugins)} plugins to {panel_name} panel")

    def _add_help_plugins_to_main(self, category):
        """Add Help plugins to Main category with hierarchical submenu support"""
        try:
            import functools
            from pathlib import Path

            # Get plugin settings
            plugin_settings = chisurf.settings.cs_settings.get('plugins', {})
            disabled_plugins = plugin_settings.get('disabled_plugins', [])
            hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
            plugin_order = plugin_settings.get('plugin_order', {})
            experimental_mode = chisurf.settings.cs_settings.get('enable_experimental', False)

            # Discover plugins
            try:
                plugin_infos = list(chisurf.plugins.iter_plugins())
            except Exception as e:
                self.logger.error(f"Failed to enumerate plugins: {e}")
                return

            # Sort plugins
            ordered = []
            for info in plugin_infos:
                plugin_name = info.get('plugin_name') or info.get('module_name') or ''
                order = plugin_order.get(plugin_name, 0)
                ordered.append((order, plugin_name, info))
            ordered.sort(key=lambda x: (x[0], x[1]))

            # Resolve plugins root
            try:
                plugins_root = Path(chisurf.plugins.__file__).parent.resolve()
            except Exception:
                plugins_root = Path(chisurf.plugins.__file__).parent

            # Collect help plugins with hierarchy support
            help_plugins = []

            for _order, plugin_name, info in ordered:
                try:
                    if not plugin_name.startswith('Help:'):
                        continue

                    module_path = info.get('module_path')
                    module_name = info.get('module_name') or ''
                    package_dir = Path(info.get('package_dir'))
                    source = info.get('source') or 'built-in'
                    is_cli_only = bool(info.get('cli_only'))

                    if bool(info.get('menu_hidden')):
                        continue

                    # Parse hierarchical name
                    hierarchy_parts, display_name = self._parse_hierarchical_plugin_name(plugin_name)

                    # Remove 'Help' from hierarchy parts for cleaner organization
                    if hierarchy_parts and hierarchy_parts[0] == 'Help':
                        hierarchy_parts = hierarchy_parts[1:]

                    # Check disabled/broken
                    clean_name = display_name
                    is_broken = (
                        plugin_name in disabled_plugins
                        or module_name in disabled_plugins
                        or clean_name in disabled_plugins
                    )

                    # Detect dev plugins
                    is_dev = False
                    try:
                        rel = package_dir.resolve().relative_to(plugins_root)
                        if rel.parts and rel.parts[0] == "_dev":
                            is_dev = True
                    except Exception:
                        is_dev = False

                    # Skip broken if hidden and not experimental
                    if is_broken and hide_disabled_plugins and not experimental_mode:
                        continue

                    # Determine script file
                    plugin_dir = package_dir
                    wizard_file = plugin_dir / "wizard.py"
                    script_file = wizard_file if wizard_file.is_file() else (plugin_dir / "__init__.py")

                    # Build callback
                    callback = functools.partial(
                        self.main_window.onRunMacro,
                        str(script_file),
                        executor='exec',
                        globals={'__name__': 'plugin'}
                    )

                    # Check for icon
                    icon = None
                    for _icon_name in ("icon.png", "icon.svg"):
                        icon_path = plugin_dir / _icon_name
                        if icon_path.exists():
                            icon = QIcon(str(icon_path))
                            break

                    # Get description
                    description = info.get('description') or "No description available."

                    # Determine display name
                    label_base = display_name
                    if is_cli_only:
                        label_base = f"{label_base} (CLI)"
                    label = f"{label_base} (BROKEN)" if is_broken else label_base

                    help_plugins.append({
                        'label': label,
                        'callback': callback,
                        'icon': icon,
                        'description': description,
                        'enabled': not (is_broken or is_cli_only),
                        'plugin_name': plugin_name,
                        'hierarchy_parts': hierarchy_parts
                    })

                except Exception as e:
                    self.logger.error(f"Error processing help plugin '{plugin_name}': {e}")
                    continue

            if help_plugins:
                # Build hierarchical structure for help plugins
                if any(plugin['hierarchy_parts'] for plugin in help_plugins):
                    # Has hierarchy - use nested structure
                    plugin_structure = self._build_nested_plugin_structure(help_plugins)

                    # Create hierarchical menu within the Help Plugins panel
                    # Since we're within Main category, we'll organize by subpanels
                    self._add_hierarchical_plugins_to_panel(category, help_plugins, 'Help Plugins')
                else:
                    # No hierarchy - use simple flat gallery organization
                    panel_help = category.addPanel('Help Plugins')
                    # Add plugins directly as small buttons for better size control
                    for plugin_info in help_plugins:
                        # Add as small button with text below icon for compact display
                        btn = panel_help.addSmallButton(
                            plugin_info['label'],
                            icon=plugin_info['icon'],
                            showText=True,
                            slot=plugin_info['callback']
                        , alignment=Qt.AlignLeft | Qt.AlignTop)
                        btn.setEnabled(plugin_info['enabled'])
                        btn.setToolTip(plugin_info['description'])

                        # Make icon smaller if it exists
                        if plugin_info['icon']:
                            btn.setMaximumIconSize(14)

                        self.logger.info(f"Added {len(help_plugins)} help plugins to Main category (small button organization)")
            else:
                self.logger.info("No help plugins found")

        except Exception as e:
            self.logger.error(f"Failed to add help plugins to Main category: {e}")

    def _add_fallback_main_actions(self, category):
        """Add basic main actions when toolbar is not available"""

        # View panel
        panel_view = category.addPanel('View')
        # Theme actions removed - only window actions added

        # Add window actions if available
        view_actions = ['actionShow_Data_Editor', 'actionShow_Fit_Widget', 'actionShow_Plugin_Widget']
        for action_name in view_actions:
            if hasattr(self.main_window, action_name):
                action = getattr(self.main_window, action_name)
                # Use small action for text below icon layout
                panel_view.addSmallButton(action.text(), icon=action.icon() if action.icon() else None, showText=True, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

    def _create_file_category(self):
        """Create File category with file operations"""
        category = self.ribbon_bar.addCategory('File')

        # Common operations panel
        panel = category.addPanel('Common')

        # Load Data action
        if hasattr(self.main_window, 'actionLoad_Data'):
            action = self.main_window.actionLoad_Data
            # Add better icon if available
            try:
                # Use generic icon since pyqtribbon doesn't have built-in icons
                action.setIcon(QIcon.fromTheme('document-open'))
            except Exception:
                pass
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Save Data action
        if hasattr(self.main_window, 'actionSave_Data'):
            action = self.main_window.actionSave_Data
            # Add better icon if available
            try:
                # Use generic icon since pyqtribbon doesn't have built-in icons
                action.setIcon(QIcon.fromTheme('document-save'))
            except Exception:
                pass
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Import/Export panel
        panel_io = category.addPanel('Import/Export')

        # Import action
        if hasattr(self.main_window, 'actionImport'):
            action = self.main_window.actionImport
            # Add folder icon for import
            try:
                action.setIcon(QIcon.fromTheme('folder'))
            except Exception:
                pass
            panel_io.addMediumButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Export action
        if hasattr(self.main_window, 'actionExport'):
            action = self.main_window.actionExport
            # Add export icon
            try:
                action.setIcon(QIcon.fromTheme('document-save-as'))
            except Exception:
                pass
            panel_io.addMediumButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        return category

    def _create_edit_category(self):
        """Create Edit category with editing operations"""
        category = self.ribbon_bar.addCategory('Edit')

        # Basic operations panel
        panel = category.addPanel('Basic')

        # Undo action
        if hasattr(self.main_window, 'actionUndo'):
            action = self.main_window.actionUndo
            # Add undo icon
            try:
                action.setIcon(QIcon.fromTheme('edit-undo'))
            except Exception:
                pass
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Redo action
        if hasattr(self.main_window, 'actionRedo'):
            action = self.main_window.actionRedo
            # Add redo icon (rotate arrow)
            try:
                action.setIcon(QIcon.fromTheme('edit-redo'))
            except Exception:
                pass
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Clipboard panel
        panel_clip = category.addPanel('Clipboard')

        # Copy action
        if hasattr(self.main_window, 'actionCopy'):
            action = self.main_window.actionCopy
            panel.addMediumButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Paste action
        if hasattr(self.main_window, 'actionPaste'):
            action = self.main_window.actionPaste
            panel.addMediumButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        return category

    def _create_analysis_category(self):
        """Create Analysis category with ChiSurf-specific analysis tools"""
        category = self.ribbon_bar.addCategory('Analysis')

        # Check if we're in experimental mode
        experimental_mode = chisurf.settings.cs_settings.get('enable_experimental', False)

        # Discover plugins
        if hasattr(self.main_window, 'comboBox_modelSelect'):
            action = QAction('Select Model', self.main_window)
            action.setStatusTip('Change fitting model')
            # Add settings icon
            try:
                action.setIcon(QIcon.fromTheme('configure'))
            except Exception:
                pass
            panel.addMediumButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Fitting panel
        panel_fit = category.addPanel('Fitting')

        # Start fit action
        if hasattr(self.main_window, 'actionStart_Fit'):
            action = self.main_window.actionStart_Fit
            # Add play/start icon
            try:
                action.setIcon(QIcon.fromTheme('media-playback-start'))
            except Exception:
                pass
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Stop fit action
        if hasattr(self.main_window, 'actionStop_Fit'):
            action = self.main_window.actionStop_Fit
            panel.addLargeButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        return category

    def _create_tools_category(self):
        """Create Tools category with utility functions"""
        category = self.ribbon_bar.addCategory('Tools')

        # Utilities panel
        panel = category.addPanel('Utilities')

        # Settings action
        if hasattr(self.main_window, 'actionSettings'):
            action = self.main_window.actionSettings
            # Add settings icon
            try:
                action.setIcon(QIcon.fromTheme('configure'))
            except Exception:
                pass
            panel.addMediumButton(action.text(), icon=action.icon() if action.icon() else None, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop)

        return category
