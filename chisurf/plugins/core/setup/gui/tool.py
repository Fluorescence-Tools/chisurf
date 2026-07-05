"""Unified Settings GUI tool."""

from __future__ import annotations

from qtpy import QtWidgets

from chisurf.gui.widgets.navigation import NavigationPanelTool

SETTINGS_PANELS = [
    {
        "name": "Getting Started",
        "icon": "🚀",
        "description": "First-run onboarding assistant (settings, detectors, FCS channels).",
        "class_path": "chisurf.plugins.core.boarding.wizard",
        "class_name": "BoardingAssistantWidget",
    },
    {
        "name": "ChiSurf Settings",
        "icon": "📄",
        "class_path": "chisurf.plugins.core.setup.gui.tool",
        "class_name": "ChiSurfSettingsEditorWidget",
    },
    {
        "name": "Acquisition",
        "icon": "📷",
        "class_path": "chisurf.plugins.core.acq.gui.settings_panel",
        "class_name": "AcquisitionSettingsWidget",
    },
    {
        "name": "Styles",
        "icon": "🎨",
        "class_path": "chisurf.plugins.core.style_manager.gui.tool",
        "class_name": "StyleManagerWidget",
    },
    {
        "name": "Models",
        "icon": "⚙️",
        "class_path": "chisurf.plugins.core.model_manager.gui.tool",
        "class_name": "ModelManagerWidget",
    },
    {
        "name": "User Editor",
        "icon": "👥",
        "class_path": "chisurf.plugins.core.user_editor.gui.tool",
        "class_name": "UserEditorWidget",
    },
    {
        "name": "AI Settings",
        "icon": "🤖",
        "class_path": "chisurf.plugins.ai_settings.gui.tool",
        "class_name": "AISettingsWidget",
    },
    {
        "name": "Plugins",
        "icon": "🔌",
        "class_path": "chisurf.plugins.core.plugin_manager.gui.tool",
        "class_name": "PluginManagerWidget",
    },
    {
        "name": "Updates",
        "icon": "⬆️",
        "class_path": "chisurf.plugins.core.setup.gui.tool",
        "class_name": "UpdatesSettingsPanel",
    },
    {
        "name": "Packages",
        "icon": "📦",
        "class_path": "chisurf.plugins.core.setup.gui.tool",
        "class_name": "PackagesSettingsPanel",
    },
    {
        "name": "Channel Definition",
        "icon": "🔢",
        "class_path": "chisurf.plugins.core.setup_channel_definition.gui.tool",
        "class_name": "SetupChannelDefinitionWidget",
    },
    {
        "name": "FCS Definitions",
        "icon": "📡",
        "class_path": "chisurf.plugins.fcs.fcs_channel_preset.gui.tool",
        "class_name": "FCSChannelWidget",
    },
    {
        "name": "TTTR LUT Tools",
        "icon": "🧮",
        "class_path": "chisurf.plugins.tttr.tttr_lut_tools.gui.tool",
        "class_name": "TTTRLUTSettingsPanel",
    },
    {
        "name": "Plugin Check",
        "icon": "🧪",
        "class_path": "chisurf.plugins.core.plugin_check.gui.tool",
        "class_name": "PluginCheckTool",
    },
]


class UnifiedSettingsTool(NavigationPanelTool):
    """Unified Settings dialog for ChiSurf."""

    def __init__(self, parent=None):
        """Create the unified settings tool."""
        super().__init__(
            title="Settings",
            panels=SETTINGS_PANELS,
            parent=parent,
            minimum_size=(850, 550),
            initial_size=(1020, 680),
            navigation_width=200,
        )


class ChiSurfSettingsEditorWidget(QtWidgets.QWidget):
    """Wrapper around the core ChiSurf settings editor."""

    def __init__(self, parent=None):
        """Create the settings editor panel."""
        super().__init__(parent)

        import chisurf as cs
        from chisurf.gui.widgets.settings_editor import SettingsEditor

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.editor = SettingsEditor(
            parent,
            filename=cs.core.settings.chisurf_settings_file,
            window_title="ChiSurf Settings",
        )
        layout.addWidget(self.editor)


class UpdatesSettingsPanel(QtWidgets.QWidget):
    """Settings panel hosting the ChiSurf update checker."""

    def __init__(self, parent=None):
        """Create the updates panel embedding the updater widget."""
        super().__init__(parent)

        from chisurf.plugins.core.updater import UpdaterWidget

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Suppress the startup popup when shown inside the settings dialog.
        self.updater = UpdaterWidget(self, suppress_initial_notification=True)
        layout.addWidget(self.updater)


class PackagesSettingsPanel(QtWidgets.QWidget):
    """Settings panel hosting the conda package manager."""

    def __init__(self, parent=None):
        """Create the packages panel embedding the package manager widget."""
        super().__init__(parent)

        from chisurf.plugins.core.updater.package_widget import PackageManagerWidget

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.packages = PackageManagerWidget(self)
        layout.addWidget(self.packages)
