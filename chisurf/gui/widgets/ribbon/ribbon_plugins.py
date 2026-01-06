# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration - Plugins Module

This module contains plugin-related functionality for the ribbon interface.
"""

import functools
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from PyQt5 import QtWidgets

import chisurf
from chisurf import logging


class PluginMethodsMixin:
    """Mixin class containing all plugin-related methods for ChiSurfRibbonIntegration"""

    def _parse_hierarchical_plugin_name(self, plugin_name):
        """
        Parse hierarchical plugin name into components.

        Parameters
        ----------
        plugin_name : str
            Plugin name in format "AA:BB:CC:Name" or "Category:Name"

        Returns
        -------
        tuple
        (hierarchy_parts, display_name) where hierarchy_parts is a list of the
        hierarchical components and display_name is the final plugin name
        """
        if ':' in plugin_name:
            parts = [part.strip() for part in plugin_name.split(':')]
            if len(parts) > 1:
                hierarchy_parts = parts[:-1]  # All parts except the last
                display_name = parts[-1]      # The last part is the display name
                return hierarchy_parts, display_name

        # No hierarchy, return as single category
        return ['Main'], plugin_name.strip()

    def _build_nested_plugin_structure(self, plugins):
        """
        Build nested structure for hierarchical plugin organization.

        Parameters
        ----------
        plugins : list
            List of plugin dictionaries with hierarchical names

        Returns
        -------
        dict
            Nested dictionary representing the plugin hierarchy
        """
        structure = {}

        for plugin_info in plugins:
            plugin_name = plugin_info['plugin_name']
            hierarchy_parts, display_name = self._parse_hierarchical_plugin_name(plugin_name)

            # Navigate/create the nested structure
            current_level = structure
            for part in hierarchy_parts[:-1]:  # All parts except the last
                if part not in current_level:
                    current_level[part] = {'_subcategories': {}, '_plugins': []}
                current_level = current_level[part]['_subcategories']

            # Add to the final category
            final_category = hierarchy_parts[-1]
            if final_category not in current_level:
                current_level[final_category] = {'_subcategories': {}, '_plugins': []}

            # Add the plugin to the final category
            current_level[final_category]['_plugins'].append(plugin_info)

        return structure

    def _create_hierarchical_menu_structure(self, structure, parent_category=None, parent_path=""):
        """
        Create hierarchical menu structure from nested plugin data.

        Since ribbon doesn't support true nested subcategories, this implementation
        uses the first hierarchy level as the main category and organizes remaining
        levels within panels and subpanels.

        Parameters
        ----------
        structure : dict
            Nested plugin structure from _build_nested_plugin_structure
        parent_category : RibbonCategory, optional
            Parent ribbon category for nesting (not used in current implementation)
        parent_path : str, optional
            Path of the parent for logging

        Returns
        -------
        list
            List of created categories
        """
        created_categories = []
        successful_plugins = 0
        failed_plugins = 0
        skipped_categories = 0

        # Sort categories to ensure 'Dev' is always last
        sorted_categories = sorted(
            [(category_name, category_data) for category_name, category_data in structure.items() 
             if not category_name.startswith('_')],
            key=lambda x: (1 if x[0] == 'Dev' else 0, x[0])
        )
        
        for category_name, category_data in sorted_categories:
            # Check if category already exists and use it, or create a new one
            if category_name in self.categories:
                category = self.categories[category_name]
                self.logger.debug(f"Using existing category '{category_name}' for hierarchical plugins")
                skipped_categories += 1
            else:
                try:
                    category = self.ribbon_bar.addCategory(category_name)
                    self.categories[category_name] = category
                    self.logger.debug(f"Created new category '{category_name}' for hierarchical plugins")
                except Exception as e:
                    self.logger.error(f"Failed to create category '{category_name}': {e}")
                    failed_plugins += len(category_data.get('_plugins', []))
                    continue

            created_categories.append(category)

            # Add plugins directly in this category
            plugins = category_data.get('_plugins', [])
            if plugins:
                added_count = self._add_plugins_to_category(category, plugins, category_name)
                successful_plugins += added_count
                failed_plugins += len(plugins) - added_count

            # Handle subcategories by organizing them into panels
            subcategories = category_data.get('_subcategories', {})
            if subcategories:
                added_count, sub_failed = self._add_subcategories_as_panels(category, subcategories, category_name)
                successful_plugins += added_count
                failed_plugins += sub_failed

        # Log summary instead of individual plugin details
        total_plugins = successful_plugins + failed_plugins
        if total_plugins > 0:
            self.logger.info(f"Hierarchical menu summary: {successful_plugins} plugins created successfully, {failed_plugins} failed, {skipped_categories} categories reused")
        
        # Ensure Dev category is always the last tab
        self._move_dev_category_to_end()

        return created_categories
    
    def _move_dev_category_to_end(self):
        """Move the Dev category to be the last tab in the ribbon."""
        try:
            if 'Dev' in self.categories and hasattr(self.ribbon_bar, '_titleWidget'):
                # Get the tab bar
                title_widget = self.ribbon_bar._titleWidget
                if hasattr(title_widget, 'tabBar'):
                    tab_bar = title_widget.tabBar()
                    
                    # Find the current index of the Dev tab
                    dev_index = tab_bar.indexOf('Dev')
                    
                    # Move Dev tab to the end if it's not already there
                    if dev_index >= 0 and dev_index < tab_bar.count() - 1:
                        # Move the tab
                        tab_bar.moveTab(dev_index, tab_bar.count() - 1)
                        
                        # Also need to move the corresponding widget in the stacked widget
                        if hasattr(self.ribbon_bar, '_stackedWidget'):
                            stacked_widget = self.ribbon_bar._stackedWidget
                            # Remove and re-insert the widget at the new position
                            widget = stacked_widget.widget(dev_index)
                            if widget:
                                stacked_widget.removeWidget(widget)
                                stacked_widget.insertWidget(tab_bar.count() - 1, widget)
                        
                        self.logger.info(f"Moved Dev category from index {dev_index} to the last position")
        except Exception as e:
            self.logger.warning(f"Failed to move Dev category to end: {e}")

    def _fix_panel_alignment(self, panel):
        """
        Fix panel alignment to ensure widgets align to top-left instead of center.

        Parameters
        ----------
        panel : RibbonPanel
            The panel to fix alignment for
        """
        def apply_alignment_fix():
            try:
                fixed_count = 0

                # Find all RibbonPanelItemWidget instances in the panel
                all_children = panel.findChildren(QtWidgets.QWidget)

                for child in all_children:
                    # Check by class name or object name
                    if (child.__class__.__name__ == "RibbonPanelItemWidget" or
                        "RibbonPanelItemWidget" in str(child.__class__)):

                        layout = child.layout()
                        if layout:
                            layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                            fixed_count += 1

                            # Also try to set alignment on all child layouts recursively
                            child_layouts = child.findChildren(QtWidgets.QLayout)
                            for child_layout in child_layouts:
                                child_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

                    # Also try to find any QVBoxLayout that might be causing centering
                    if hasattr(child, 'layout') and child.layout():
                        layout = child.layout()
                        if layout and hasattr(layout, 'setAlignment'):
                            layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                            fixed_count += 1

                # Also try to access the internal grid layout directly
                if hasattr(panel, '_actionsLayout'):
                    panel._actionsLayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                    fixed_count += 1

                self.logger.debug(f"Fixed alignment for panel: {fixed_count} layouts adjusted")

            except Exception as e:
                self.logger.warning(f"Failed to fix panel alignment: {e}")

        # Use a single-shot timer to ensure the fix runs after all widgets are constructed
        QTimer.singleShot(100, apply_alignment_fix)

    def _fix_all_panel_alignments(self):
        """
        Apply alignment fixes to all panels in the ribbon after complete setup.
        """
        try:
            fixed_panels = 0

            if self.ribbon_bar and hasattr(self.ribbon_bar, '_categories'):
                for category in self.ribbon_bar._categories.values():
                    if hasattr(category, '_panels'):
                        for panel in category._panels.values():
                            self._fix_panel_alignment(panel)
                            fixed_panels += 1

            self.logger.info(f"Applied global alignment fix to {fixed_panels} panels")

        except Exception as e:
            self.logger.warning(f"Failed to apply global alignment fix: {e}")

    def _add_subcategories_as_panels(self, category, subcategories, category_path):
        """
        Add subcategories as galleries within panels in the given category.

        Parameters
        ----------
        category : RibbonCategory
            Ribbon category to add galleries to
        subcategories : dict
            Dictionary of subcategories to add as galleries
        category_path : str
            Path of the parent category for logging

        Returns
        -------
        tuple
            (successful_count, failed_count) of plugins added
        """
        successful_count = 0
        failed_count = 0

        for subcat_name, subcat_data in subcategories.items():
            if subcat_name.startswith('_'):  # Skip metadata keys
                continue

            # Add plugins in this subcategory
            plugins = subcat_data.get('_plugins', [])
            if plugins:
                try:
                    # Create a panel for this subcategory
                    panel_name = subcat_name
                    panel = category.addPanel(panel_name, showPanelOptionButton=False)

                    # Add plugins directly as small buttons for better size control
                    for plugin_info in plugins:
                        try:
                            # Use the display label (should already be just the final name)
                            display_label = plugin_info['label']

                            # Add as small button with text below icon for compact display
                            btn = panel.addSmallButton(
                                display_label,
                                icon=plugin_info['icon'],
                                showText=True,
                                slot=plugin_info['callback']
                            , alignment=Qt.AlignLeft | Qt.AlignTop)
                            btn.setEnabled(plugin_info['enabled'])
                            btn.setToolTip(plugin_info['description'])

                            # Make icon smaller if it exists
                            if plugin_info['icon']:
                                btn.setMaximumIconSize(14)

                            successful_count += 1
                        except Exception as e:
                            self.logger.error(f"Failed to add plugin '{plugin_info.get('label', 'Unknown')}' to {category_path} > {panel_name} panel: {e}")
                            failed_count += 1

                    # Fix panel alignment after adding all plugins
                    self._fix_panel_alignment(panel)

                except Exception as e:
                    self.logger.error(f"Failed to create panel '{subcat_name}' in category '{category_path}': {e}")
                    failed_count += len(plugins)

            # Recursively handle deeper nesting
            deeper_subcats = subcat_data.get('_subcategories', {})
            if deeper_subcats:
                # For deeper levels, create panel names that include the hierarchy
                for deeper_name, deeper_data in deeper_subcats.items():
                    if deeper_name.startswith('_'):
                        continue

                    deeper_plugins = deeper_data.get('_plugins', [])
                    if deeper_plugins:
                        try:
                            # Create descriptive panel name
                            panel_name = f"{subcat_name} > {deeper_name}"
                            panel = category.addPanel(panel_name, showPanelOptionButton=False)

                            # Add plugins directly as small buttons for better size control
                            for plugin_info in deeper_plugins:
                                try:
                                    display_label = plugin_info['label']

                                    # Add as small button with text below icon for compact display
                                    btn = panel.addSmallButton(
                                        display_label,
                                        icon=plugin_info['icon'],
                                        showText=True,
                                        slot=plugin_info['callback']
                                    , alignment=Qt.AlignLeft | Qt.AlignTop)
                                    btn.setEnabled(plugin_info['enabled'])
                                    btn.setToolTip(plugin_info['description'])

                                    # Make icon smaller if it exists
                                    if plugin_info['icon']:
                                        btn.setMaximumIconSize(14)

                                    successful_count += 1
                                except Exception as e:
                                    self.logger.error(f"Failed to add plugin '{plugin_info.get('label', 'Unknown')}' to {category_path} > {panel_name} panel: {e}")
                                    failed_count += 1

                            # Fix panel alignment after adding all plugins
                            self._fix_panel_alignment(panel)

                        except Exception as e:
                            self.logger.error(f"Failed to create panel '{panel_name}' in category '{category_path}': {e}")
                            failed_count += len(deeper_plugins)

        return successful_count, failed_count

    def _add_plugins_to_category(self, category, plugins, category_path):
        """
        Add plugins to a ribbon category using galleries within panels for better organization.

        Parameters
        ----------
        category : RibbonCategory
            Ribbon category to add plugins to
        plugins : list
            List of plugin dictionaries
        category_path : str
            Path of the category for logging

        Returns
        -------
        int
            Number of successfully added plugins
        """
        if not plugins:
            return 0

        successful_count = 0

        # Group plugins into galleries of ~8 items each (smaller for better fit with 32px icons)
        gallery_size = 8
        for i in range(0, len(plugins), gallery_size):
            gallery_plugins = plugins[i:i + gallery_size]
            gallery_number = i // gallery_size + 1
            panel_name = f"Gallery {gallery_number}" if len(plugins) > gallery_size else "Plugins"

            # Create panel first
            panel = category.addPanel(panel_name, showPanelOptionButton=False)

            # Add plugins directly as small buttons for better size control
            for plugin_info in gallery_plugins:
                try:
                    # Use the label (which should already be the display name)
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

                    successful_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to add plugin '{plugin_info.get('label', 'Unknown')}' to {category_path} -> {panel_name}: {e}")

            # Fix panel alignment after adding all plugins to this gallery
            self._fix_panel_alignment(panel)

        return successful_count

    def _create_plugins_category(self):
        """Create dedicated Plugins category with hierarchical submenu support"""
        try:
            # Get plugin settings
            plugin_settings = chisurf.settings.cs_settings.get('plugins', {})
            disabled_plugins = plugin_settings.get('disabled_plugins', [])
            hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
            plugin_order = plugin_settings.get('plugin_order', {})

            # Check if we're in experimental mode
            experimental_mode = chisurf.settings.cs_settings.get('enable_experimental', False)

            # Discover plugins
            try:
                plugin_infos = list(chisurf.plugins.iter_plugins())
            except Exception as e:
                self.logger.error(f"Failed to enumerate plugins: {e}")
                plugin_infos = []

            # Prefer built-in updater
            try:
                has_builtin_updater = any(
                    (info.get('module_name') == 'updater' and info.get('source') == 'built-in')
                    for info in plugin_infos
                )
                if has_builtin_updater:
                    plugin_infos = [
                        info for info in plugin_infos
                        if not (
                            info.get('module_name') == 'updater'
                            and info.get('source') == 'user'
                        )
                    ]
            except Exception:
                pass

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

            # Collect all plugins (except Setup: and Help: which are handled separately)
            all_plugins = []

            for _order, plugin_name, info in ordered:
                try:
                    module_path = info.get('module_path')
                    module_name = info.get('module_name') or ''
                    package_dir = Path(info.get('package_dir'))
                    source = info.get('source') or 'built-in'
                    is_cli_only = bool(info.get('cli_only'))

                    if bool(info.get('menu_hidden')):
                        self.logger.info(f"Skipping plugin '{plugin_name}' (module='{module_name}', source='{source}'): marked as menu_hidden")
                        continue

                    # Skip Setup: and Help: plugins as they are handled separately
                    if plugin_name.startswith('Setup:') or plugin_name.startswith('Help:'):
                        continue

                    # Check disabled/broken
                    hierarchy_parts, display_name = self._parse_hierarchical_plugin_name(plugin_name)
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
                        self.logger.info(f"Skipping plugin '{plugin_name}' (module='{module_name}', source='{source}'): disabled/broken and hide_disabled_plugins=True")
                        continue

                    self.logger.info(f"Processing plugin '{plugin_name}' (module='{module_name}', source='{source}', is_dev={is_dev}, is_broken={is_broken}, is_cli_only={is_cli_only})")

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

                    # Determine display name with hierarchy support
                    if is_dev:
                        # For dev plugins, use the hierarchy but mark as dev
                        label_base = display_name
                        if is_cli_only:
                            label_base = f"{label_base} (CLI)"
                        label = f"{label_base} (BROKEN)" if is_broken else label_base
                    else:
                        # Use the display name from hierarchical parsing
                        label_base = display_name
                        if is_cli_only:
                            label_base = f"{label_base} (CLI)"
                        label = f"{label_base} (BROKEN)" if is_broken else label_base

                    all_plugins.append({
                        'label': label,
                        'callback': callback,
                        'icon': icon,
                        'description': description,
                        'enabled': not (is_broken or is_cli_only),
                        'plugin_name': plugin_name,
                        'hierarchy_parts': hierarchy_parts
                    })

                except Exception as e:
                    self.logger.error(f"Error processing plugin for hierarchical menu: '{plugin_name}': {e}")
                    continue

            # Build hierarchical structure and create menu
            if all_plugins:
                # Build nested structure
                plugin_structure = self._build_nested_plugin_structure(all_plugins)

                # Create hierarchical menu structure
                created_categories = self._create_hierarchical_menu_structure(plugin_structure)

                self.logger.info(f"Created hierarchical plugins menu with {len(all_plugins)} plugins in {len(created_categories)} categories")
            else:
                self.logger.info("No plugins found for hierarchical menu creation")

        except Exception as e:
            self.logger.error(f"Failed to create hierarchical plugins menu: {e}")

        return None
