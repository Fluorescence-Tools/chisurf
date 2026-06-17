"""
ChiSurf Ribbon Integration - File Category Module.

This module contains the File category creation methods for the ribbon interface.
"""

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QStyle

from chisurf import logging

from .separator import RibbonSeparator


class FileCategoryMixin:
    """Mixin class containing File category creation methods for ChiSurfRibbonIntegration."""

    def _create_file_category(self):
        """Create File category with all file menu actions organized into logical groups."""
        category = self.ribbon_bar.addCategory("File")

        # Allow the scroll area contents to expand so stretch items take effect
        if hasattr(category, "_categoryScrollArea"):
            category._categoryScrollArea.setWidgetResizable(True)

        # Always add a dedicated Recent Projects panel.
        # This displays the most recent projects as small buttons.
        # Wrapped in try/except so that any failure here can never prevent the
        # other panels (Project, Fits, Application) from being created.
        try:
            self._panel_recent = category.addPanel("Recent", showPanelOptionButton=False)
            # Populate from the persisted user file
            self._refresh_ribbon_recent_projects()
        except Exception as _exc:
            try:
                logging.warning(f"Ribbon: failed to create Recent Projects panel: {_exc}")
            except Exception:
                pass

        # Define action groups with their standard icons
        action_groups = {
            "Project": [
                ("actionProject_Browser", QStyle.SP_DialogOpenButton),
                ("actionSave_Project", QStyle.SP_DialogSaveButton),
                ("actionExport_Project", QStyle.SP_DriveHDIcon),
                ("actionImport_Project", QStyle.SP_FileDialogListView),
                ("actionClose_Project", QStyle.SP_DialogCloseButton),
            ],
            "Fits": [
                ("actionLoad_Fit", QStyle.SP_DialogOpenButton),
                ("actionSaveCurrentFit", QStyle.SP_DialogSaveButton),
                ("actionSaveAllFits", QStyle.SP_DialogSaveButton),
                ("actionClose_Fit", QStyle.SP_DialogCloseButton),
                ("actionClose_all_fits", QStyle.SP_DialogCloseButton),
            ],
            "Application": [
                ("actionExit_2", QStyle.SP_DialogCloseButton),
                ("actionReinitialize", QStyle.SP_BrowserReload),
            ],
        }

        # Create panels and add buttons iteratively
        for panel_name, actions in action_groups.items():
            panel = category.addPanel(panel_name, showPanelOptionButton=False)

            for action_name, standard_icon in actions:
                if hasattr(self.main_window, action_name):
                    action = getattr(self.main_window, action_name)
                    self._add_action_button(panel, action, standard_icon)

        # Insert stretch + separator between Application panels and setup plugins
        if hasattr(category, "_categoryLayout"):
            category._categoryLayout.addItem(
                QtWidgets.QSpacerItem(
                    0,
                    0,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Minimum,
                )
            )
            separator = RibbonSeparator()
            category._categoryLayout.addWidget(separator)

        # Add Setup plugins to File category
        self._add_setup_plugins_to_main(category)

        return category

    def _refresh_ribbon_recent_projects(self):
        """Rebuild the ribbon-owned Recent Projects panel with small buttons for each project.

        Called once on ribbon setup and again whenever a project is
        opened/saved/cleared (via ``project_helpers.refresh_recent_projects_menu``).
        """
        panel_recent = getattr(self, "_panel_recent", None)
        if panel_recent is None:
            return

        # Clear existing widgets from the panel
        for widget in list(panel_recent._widgets):
            try:
                panel_recent.removeWidget(widget)
                widget.deleteLater()
            except Exception:
                pass
        panel_recent._widgets.clear()

        # Clean the grid layout
        for i in reversed(range(panel_recent._actionsLayout.count())):
            try:
                item = panel_recent._actionsLayout.takeAt(i)
                if item is not None:
                    container = item.widget()
                    if container is not None:
                        container.deleteLater()
            except Exception:
                pass

        # Reset grid layout manager
        try:
            from chisurf.gui.widgets.ribbon.panel import RibbonGridLayoutManager

            panel_recent._gridLayoutManager = RibbonGridLayoutManager(panel_recent._maxRows)
        except Exception:
            pass

        # Prefer the in-memory list; fall back to the persisted user file.
        try:
            projects = list(getattr(self.main_window, "_recent_projects", None) or [])
        except Exception:
            projects = []
        if not projects:
            try:
                from chisurf.gui.project_helpers import load_recent_projects

                projects = load_recent_projects()
            except Exception:
                projects = []

        if projects and not getattr(self.main_window, "_recent_projects", None):
            try:
                self.main_window._recent_projects = list(projects)
            except Exception:
                pass

        if not projects:
            try:
                # Add a disabled small button for "No recent projects"
                icon = self.main_window.style().standardIcon(QStyle.SP_MessageBoxInformation)
                btn = panel_recent.addSmallButton("No recent projects", icon=icon)
                btn.setEnabled(False)
            except Exception:
                pass
        else:
            import pathlib

            # Show up to 9 recent projects to fit neatly inside columns of 3 (maxRows is 6, Small rowSpan is 2)
            display_projects = projects[:9]
            for i, p in enumerate(display_projects):
                try:
                    label = pathlib.Path(p).name or p
                    icon = self.main_window.style().standardIcon(QStyle.SP_FileDialogListView)
                    btn = panel_recent.addSmallButton(
                        label,
                        icon=icon,
                        slot=lambda _chk=False, pp=p: self._open_recent_from_ribbon(pp),
                        tooltip=str(p),
                        alignment=Qt.AlignLeft | Qt.AlignTop
                    )
                except Exception:
                    continue

            # Add a clear button as a small button at the end
            try:
                clear_icon = self.main_window.style().standardIcon(QStyle.SP_DialogDiscardButton)
                btn = panel_recent.addSmallButton(
                    "Clear List",
                    icon=clear_icon,
                    slot=lambda _chk=False: self._clear_recent_from_ribbon(),
                    tooltip="Clear recent projects list",
                    alignment=Qt.AlignLeft | Qt.AlignTop
                )
            except Exception:
                pass

        # Explicitly show all newly created widgets/containers in the layout
        # since they are added after the panel/parent is already shown
        try:
            for i in range(panel_recent._actionsLayout.count()):
                item = panel_recent._actionsLayout.itemAt(i)
                if item is not None:
                    w = item.widget()
                    if w is not None:
                        w.show()
                        for child in w.findChildren(QtWidgets.QWidget):
                            child.show()
        except Exception:
            pass

        # Fix alignment to ensure everything aligns to top-left instead of center
        try:
            if hasattr(self, "_fix_panel_alignment"):
                self._fix_panel_alignment(panel_recent)
        except Exception:
            pass

    def _open_recent_from_ribbon(self, project_dir: str):
        """Open a recent project selected from the ribbon drop-down.

        Parameters
        ----------
        project_dir : str
            Path to the project directory.
        """
        try:
            from chisurf.gui.project_helpers import open_recent_project

            open_recent_project(self.main_window, project_dir)
        except Exception as exc:
            try:
                from chisurf import logging as _log

                _log.warning(f"Ribbon: failed to open recent project {project_dir!r}: {exc}")
            except Exception:
                pass

    def _clear_recent_from_ribbon(self):
        """Clear the recent-projects list from the ribbon drop-down."""
        try:
            from chisurf.gui.project_helpers import clear_recent_projects

            clear_recent_projects(self.main_window)
        except Exception as exc:
            try:
                from chisurf import logging as _log

                _log.warning(f"Ribbon: failed to clear recent projects: {exc}")
            except Exception:
                pass

    def _add_action_button(self, panel, action, standard_icon):
        """Add a button for an action to a panel with fallback icon handling.

        Parameters
        ----------
        panel : RibbonPanel
            The panel to add the button to.
        action : QAction
            The action whose text/icon/trigger will be used.
        standard_icon : QStyle.StandardPixmap
            Fallback standard-icon identifier.
        """
        try:
            icon = self.main_window.style().standardIcon(standard_icon)
        except Exception:
            icon = action.icon() if action.icon() else None

        panel.addSmallButton(
            action.text(), icon=icon, slot=action.trigger, alignment=Qt.AlignLeft | Qt.AlignTop
        )
