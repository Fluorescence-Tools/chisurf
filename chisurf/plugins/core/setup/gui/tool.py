"""Unified Settings GUI tool."""

from qtpy import QtCore, QtGui, QtWidgets


class UnifiedSettingsTool(QtWidgets.QMainWindow):
    """
    Unified Settings dialog for ChiSurf.
    Hosts all settings panels (Styles, Models, User Editor, AI Settings, Plugins,
    Channel Definition, FCS Definitions, Plugin Check) in a single window with
    a left-side navigation list.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(1020, 680)
        self.setMinimumSize(850, 550)

        # Map list items to widget classes/factories
        self.panels = [
            {
                "name": "ChiSurf Settings",
                "icon": "📄",
                "class_path": "chisurf.plugins.core.setup.gui.tool",
                "class_name": "ChiSurfSettingsEditorWidget",
                "instance": None,
            },
            {
                "name": "Acquisition",
                "icon": "📷",
                "class_path": "chisurf.plugins.core.acq.gui.settings_panel",
                "class_name": "AcquisitionSettingsWidget",
                "instance": None,
            },
            {
                "name": "Styles",
                "icon": "🎨",
                "class_path": "chisurf.plugins.core.style_manager.gui.tool",
                "class_name": "StyleManagerWidget",
                "instance": None,
            },
            {
                "name": "Models",
                "icon": "⚙️",
                "class_path": "chisurf.plugins.core.model_manager.gui.tool",
                "class_name": "ModelManagerWidget",
                "instance": None,
            },
            {
                "name": "User Editor",
                "icon": "👥",
                "class_path": "chisurf.plugins.core.user_editor.gui.tool",
                "class_name": "UserEditorWidget",
                "instance": None,
            },
            {
                "name": "AI Settings",
                "icon": "🤖",
                "class_path": "chisurf.plugins.ai_settings.gui.tool",
                "class_name": "AISettingsWidget",
                "instance": None,
            },
            {
                "name": "Plugins",
                "icon": "🔌",
                "class_path": "chisurf.plugins.core.plugin_manager.gui.tool",
                "class_name": "PluginManagerWidget",
                "instance": None,
            },
            {
                "name": "Channel Definition",
                "icon": "🔢",
                "class_path": "chisurf.plugins.core.setup_channel_definition.gui.tool",
                "class_name": "SetupChannelDefinitionWidget",
                "instance": None,
            },
            {
                "name": "FCS Definitions",
                "icon": "📡",
                "class_path": "chisurf.plugins.fcs.fcs_channel_preset.gui.tool",
                "class_name": "FCSChannelWidget",
                "instance": None,
            },
            {
                "name": "Plugin Check",
                "icon": "🧪",
                "class_path": "chisurf.plugins.core.plugin_check.gui.tool",
                "class_name": "PluginCheckTool",
                "instance": None,
            },
        ]

        self._build_ui()

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left navigation menu (styled structure only; color is inherited from active QSS theme)
        self.nav_list = QtWidgets.QListWidget()
        self.nav_list.setIconSize(QtCore.QSize(20, 20))
        self.nav_list.setSpacing(4)
        self.nav_list.setStyleSheet("""
            QListWidget {
                border: none;
                border-right: 1px solid rgba(128, 128, 128, 0.3);
                padding-top: 5px;
            }
            QListWidget::item {
                height: 32px;
                padding-left: 10px;
                border-radius: 8px;
                margin: 2px 10px;
                font-weight: bold;
                font-size: 14px;
            }
        """)

        # Populate nav_list
        for panel in self.panels:
            item = QtWidgets.QListWidgetItem(f"{panel['icon']}  {panel['name']}")
            self.nav_list.addItem(item)

        splitter.addWidget(self.nav_list)

        # Right stacked widget (no background override; inherits theme background)
        self.stacked_widget = QtWidgets.QStackedWidget()
        
        # Pre-populate stacked widget with placeholder widgets
        for _ in self.panels:
            self.stacked_widget.addWidget(QtWidgets.QWidget())

        splitter.addWidget(self.stacked_widget)
        
        # Set splitter proportions (left 20%, right 80%)
        splitter.setSizes([200, 820])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Connect selection changes
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        # Select the first row by default
        self.nav_list.setCurrentRow(0)

    def _on_nav_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.panels):
            return

        panel = self.panels[index]
        if panel["instance"] is None:
            try:
                import importlib
                module = importlib.import_module(panel["class_path"])
                widget_class = getattr(module, panel["class_name"])
                widget = widget_class(parent=self)

                # Wrap it to provide consistent margins
                wrapper = QtWidgets.QWidget()
                wrapper_layout = QtWidgets.QVBoxLayout(wrapper)
                wrapper_layout.setContentsMargins(12, 12, 12, 12)
                wrapper_layout.addWidget(widget)

                panel["instance"] = wrapper
                
                # Replace the placeholder widget
                placeholder = self.stacked_widget.widget(index)
                self.stacked_widget.removeWidget(placeholder)
                self.stacked_widget.insertWidget(index, wrapper)
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_widget = QtWidgets.QWidget()
                err_layout = QtWidgets.QVBoxLayout(error_widget)
                err_label = QtWidgets.QLabel(
                    f"Failed to load settings panel '{panel['name']}':\n{str(e)}"
                )
                err_label.setStyleSheet("color: red; font-size: 13px; font-weight: bold;")
                err_label.setWordWrap(True)
                err_layout.addWidget(err_label)
                err_layout.addStretch()
                
                panel["instance"] = error_widget
                placeholder = self.stacked_widget.widget(index)
                self.stacked_widget.removeWidget(placeholder)
                self.stacked_widget.insertWidget(index, error_widget)

        self.stacked_widget.setCurrentWidget(panel["instance"])

class ChiSurfSettingsEditorWidget(QtWidgets.QWidget):
    """Wrapper for the core ChiSurf SettingsEditor."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        import chisurf as cs
        from chisurf.gui.widgets.settings_editor import SettingsEditor
        
        self.editor = SettingsEditor(
            parent,
            filename=cs.core.settings.chisurf_settings_file,
            window_title="ChiSurf Settings"
        )
        layout.addWidget(self.editor)
