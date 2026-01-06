# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration - File Category Module

This module contains the File category creation methods for the ribbon interface.
"""

from qtpy.QtWidgets import QStyle
from qtpy.QtCore import Qt
from chisurf import logging


class FileCategoryMixin:
    """Mixin class containing File category creation methods for ChiSurfRibbonIntegration"""

    def _create_file_category(self):
        """Create File category with all file menu actions organized into logical groups"""
        category = self.ribbon_bar.addCategory('File')
        
        # Define action groups with their standard icons
        action_groups = {
            'Project': [
                ('actionOpen_Project', QStyle.SP_DialogOpenButton),
                ('actionSave_Project', QStyle.SP_DialogSaveButton),
                ('actionClose_Project', QStyle.SP_DialogCloseButton),
            ],
            'Data': [
                ('actionLoad_Data', QStyle.SP_DialogOpenButton),
            ],
            'Fits': [
                ('actionLoad_Fit', QStyle.SP_DialogOpenButton),
                ('actionSaveCurrentFit', QStyle.SP_DialogSaveButton),
                ('actionSaveAllFits', QStyle.SP_DialogSaveButton),
                ('actionClose_Fit', QStyle.SP_DialogCloseButton),
                ('actionClose_all_fits', QStyle.SP_DialogCloseButton),
            ],
            'Application': [
                ('actionExit_2', QStyle.SP_DialogCloseButton),
                ('actionReinitialize', QStyle.SP_BrowserReload),
            ]
        }
        
        # Create panels and add buttons iteratively
        for panel_name, actions in action_groups.items():
            panel = category.addPanel(panel_name, showPanelOptionButton=False)
            
            for action_name, standard_icon in actions:
                if hasattr(self.main_window, action_name):
                    action = getattr(self.main_window, action_name)
                    self._add_action_button(panel, action, standard_icon)
        
        return category
    
    def _add_action_button(self, panel, action, standard_icon):
        """Add a button for an action to a panel with fallback icon handling"""
        try:
            icon = self.main_window.style().standardIcon(standard_icon)
        except Exception:
            icon = action.icon() if action.icon() else None
        
        panel.addSmallButton(
            action.text(),
            icon=icon,
            slot=action.trigger,
            alignment=Qt.AlignLeft | Qt.AlignTop
        )
