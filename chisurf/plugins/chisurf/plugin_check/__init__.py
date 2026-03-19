"""Plugin Check Plugin

This plugin provides a GUI interface to run the plugin checker macro.
It adds a menu item to Help:Plugin-Check that opens the plugin testing dialog.

The UI logic is in this plugin, while the actual testing logic is in the macro.
"""

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QTextEdit, QSplitter, QProgressBar, QGroupBox,
    QHeaderView, QApplication, QCheckBox, QDoubleSpinBox, QScrollArea, QWidget,
    QSizePolicy
)
from qtpy.QtGui import QFont
from qtpy.QtCore import Qt
import chisurf

# Define the plugin name - this will appear in the Plugins menu
name = "Help:Plugin-Check"

description = (
    "This tool tests all plugins for startup errors.\n"
    "Green checkmarks indicate successful loading, red crosses indicate failures.\n"
    "It uses the same mechanism as ChiSurf's plugin system."
)


class PluginCheckWidget(QDialog):
    """Main dialog for plugin check system - UI only"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.test_runner = None
        self.plugin_results = {}
        self.setup_ui()
        self.setWindowTitle("ChiSurf Plugin Check")
        self.setModal(True)
        self.setFixedSize(940, 500)  # Reduced window size
        
        # Ensure cleanup when dialog is closed
        self.finished.connect(self.cleanup)

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        layout.setSpacing(3)  # Reduce spacing between widgets
        
        # Title
        title_label = QLabel("ChiSurf Plugin Check")
        title_font = QFont()
        title_font.setPointSize(12)  # Smaller font
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description - make more compact
        desc_label = QLabel(
            "Tests all plugins for startup errors. Green = success, red = failure, yellow = skipped."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 11px;")  # Smaller, gray text
        layout.addWidget(desc_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)  # Reduce button spacing
        
        self.test_button = QPushButton("Test All Plugins")
        self.test_button.clicked.connect(self.start_testing)
        button_layout.addWidget(self.test_button)
        
        self.test_safe_button = QPushButton("Test Safe Plugins")
        self.test_safe_button.clicked.connect(self.start_safe_testing)
        self.test_safe_button.setToolTip("Test a few plugins with aggressive filtering and short timeouts")
        button_layout.addWidget(self.test_safe_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_plugins)
        button_layout.addWidget(self.refresh_button)
        
        button_layout.addStretch()
        
        # Close button for modal dialog
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Blacklist controls - more compact
        blacklist_group = QGroupBox("Blacklist Settings")
        blacklist_group.setFixedHeight(50)  # Further reduced height
        blacklist_layout = QHBoxLayout(blacklist_group)
        blacklist_layout.setContentsMargins(5, 2, 5, 2)  # Reduced margins
        
        # Skip blacklisted checkbox
        self.skip_blacklisted_checkbox = QCheckBox("Skip blacklisted plugins")
        self.skip_blacklisted_checkbox.setChecked(True)
        self.skip_blacklisted_checkbox.setToolTip("Automatically skip plugins that have been blacklisted due to frequent failures")
        blacklist_layout.addWidget(self.skip_blacklisted_checkbox)
        
        blacklist_layout.addStretch()
        
        # Blacklist management buttons
        self.clear_blacklist_button = QPushButton("Clear Blacklist")
        self.clear_blacklist_button.clicked.connect(self.clear_blacklist)
        self.clear_blacklist_button.setToolTip("Remove all plugins from blacklist")
        blacklist_layout.addWidget(self.clear_blacklist_button)
        
        layout.addWidget(blacklist_group)
        
        # Advanced settings - more compact
        advanced_group = QGroupBox("Advanced Settings")
        advanced_group.setFixedHeight(40)  # Further reduced height
        advanced_layout = QHBoxLayout(advanced_group)
        advanced_layout.setContentsMargins(5, 2, 5, 2)  # Reduced margins
        
        # Delay control
        delay_label = QLabel("Delay:")
        delay_label.setToolTip("Delay between plugin tests to prevent GUI overload and access violations")
        advanced_layout.addWidget(delay_label)
        
        self.delay_spinbox = QDoubleSpinBox()
        self.delay_spinbox.setRange(0.0, 5.0)
        self.delay_spinbox.setSingleStep(0.1)
        self.delay_spinbox.setValue(0.5)
        self.delay_spinbox.setSuffix(" sec")
        self.delay_spinbox.setToolTip("Longer delays reduce GUI load but take more time")
        self.delay_spinbox.setMaximumWidth(80)  # Limit width
        advanced_layout.addWidget(self.delay_spinbox)
        
        advanced_layout.addStretch()
        
        layout.addWidget(advanced_group)
        
        # Progress bar - more compact
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(18)  # Further reduced height
        layout.addWidget(self.progress_bar)
        
        # Main content area - make plugin list expand vertically
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make splitter expand
        
        # Plugin tree (left side) - make it expand vertically
        self.plugin_tree = QTreeWidget()
        self.plugin_tree.setHeaderLabels(["Plugin", "Status", "Source", "Error"])
        self.plugin_tree.setColumnWidth(0, 250)
        self.plugin_tree.setColumnWidth(1, 80)
        self.plugin_tree.setColumnWidth(2, 80)
        self.plugin_tree.itemClicked.connect(self.on_plugin_selected)
        self.plugin_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make tree expand
        splitter.addWidget(self.plugin_tree)
        
        # Details panel (right side)
        details_group = QGroupBox("Plugin Details")
        details_group.setMinimumWidth(400)
        details_group.setMaximumWidth(400)  # Fixed width for details panel
        details_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)  # Make details expand vertically
        details_layout = QVBoxLayout(details_group)
        details_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        details_layout.setSpacing(3)  # Reduced spacing
        
        # Create scroll area for plugin details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make scroll area expand
        
        # Create content widget for scroll area
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make content expand
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        content_layout.setSpacing(3)  # Reduced spacing
        
        self.details_label = QLabel("Select a plugin to view details")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.RichText)
        self.details_label.setStyleSheet("font-size: 11px;")  # Smaller font
        self.details_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # Allow horizontal expansion
        content_layout.addWidget(self.details_label)
        
        self.error_text = QTextEdit()
        self.error_text.setMaximumHeight(120)  # Further reduced height
        self.error_text.setPlaceholderText("Error details will appear here...")
        self.error_text.setStyleSheet("font-size: 10px;")  # Smaller font
        self.error_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # Allow horizontal expansion
        content_layout.addWidget(self.error_text)
        
        # Set the content widget as the scroll area's widget
        scroll_area.setWidget(content_widget)
        details_layout.addWidget(scroll_area)
        
        splitter.addWidget(details_group)
        splitter.setSizes([540, 400])  # Give more space to plugin list
        splitter.setStretchFactor(0, 1)  # Plugin tree expands
        splitter.setStretchFactor(1, 0)  # Details panel doesn't expand
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make splitter expand
        
        # Create a container widget for the splitter to ensure proper expansion
        splitter_container = QWidget()
        splitter_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter_container_layout = QVBoxLayout(splitter_container)
        splitter_container_layout.setContentsMargins(0, 0, 0, 0)
        splitter_container_layout.addWidget(splitter)
        
        layout.addWidget(splitter_container)
        
        # Status bar - more compact
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")  # Smaller, gray text
        layout.addWidget(self.status_label)
        
        # Initial load
        self.refresh_plugins()

    def refresh_plugins(self):
        """Refresh the plugin list"""
        self.plugin_tree.clear()
        self.plugin_results = {}
        
        try:
            from chisurf.plugins import iter_plugins
            plugins = list(iter_plugins())
            
            for plugin_info in plugins:
                plugin_name = plugin_info.get('plugin_name', 'Unknown')
                source = plugin_info.get('source', 'Unknown')
                
                # Create tree item
                item = QTreeWidgetItem(self.plugin_tree)
                item.setText(0, plugin_name)
                item.setText(1, "?")  # Unknown status
                item.setText(2, source)
                item.setText(3, "")  # No error yet
                
                # Store plugin info for later use
                item.setData(0, Qt.UserRole, plugin_info)
                
                # Color based on source
                if source == 'user':
                    item.setForeground(2, Qt.blue)
                
        except Exception as e:
            self.status_label.setText(f"Error loading plugins: {e}")

    def start_safe_testing(self):
        """Start testing a few plugins with aggressive filtering and short timeouts"""
        if self.test_runner and self.test_runner.is_running:
            return
        
        # Collect a few plugins for safe testing
        plugins = []
        for i in range(self.plugin_tree.topLevelItemCount()):
            item = self.plugin_tree.topLevelItem(i)
            plugin_info = item.data(0, Qt.UserRole)
            if plugin_info:
                plugins.append(plugin_info)
                if len(plugins) >= 10:  # Just test first 10 plugins
                    break
        
        if not plugins:
            self.status_label.setText("No plugins to test")
            return
        
        # Setup runner with safe mode
        self._setup_runner(plugins, safe_mode=True)
        
        # Update UI
        self.test_button.setEnabled(False)
        self.test_safe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(plugins))
        self.progress_bar.setValue(0)
        self.status_label.setText("Testing safe plugins (aggressive filtering)...")
        
        # Start testing
        self.test_runner.start_testing()

    def start_testing(self):
        """Start testing all plugins"""
        if self.test_runner and self.test_runner.is_running:
            return
        
        # Collect all plugins
        plugins = []
        for i in range(self.plugin_tree.topLevelItemCount()):
            item = self.plugin_tree.topLevelItem(i)
            plugin_info = item.data(0, Qt.UserRole)
            if plugin_info:
                plugins.append(plugin_info)
        
        if not plugins:
            self.status_label.setText("No plugins to test")
            return
        
        # Setup runner
        self._setup_runner(plugins, safe_mode=False)
        
        # Update UI
        self.test_button.setEnabled(False)
        self.test_safe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(plugins))
        self.progress_bar.setValue(0)
        self.status_label.setText("Testing plugins...")
        
        # Start testing
        self.test_runner.start_testing()

    def _setup_runner(self, plugins, safe_mode=False):
        """Setup the test runner with callbacks"""
        # Import the test runner from macro directly to avoid QtWebEngine issues
        try:
            from chisurf.macros.plugin_check import PluginTestRunner
        except ImportError:
            # Fallback: import directly from file
            import importlib.util
            spec = importlib.util.spec_from_file_location('plugin_check', 'chisurf/macros/plugin_check.py')
            plugin_check_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_check_module)
            PluginTestRunner = plugin_check_module.PluginTestRunner
        
        self.test_runner = PluginTestRunner()
        self.test_runner.set_plugins(plugins)
        self.test_runner.set_safe_mode(safe_mode)
        self.test_runner.set_skip_blacklisted(self.skip_blacklisted_checkbox.isChecked())
        self.test_runner.set_delay_between_plugins(self.delay_spinbox.value())
        self.test_runner.set_callbacks(
            progress_callback=self.update_progress,
            result_callback=self.update_plugin_result,
            finished_callback=self.testing_finished
        )

    def update_progress(self, current, total):
        """Update progress bar"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Testing plugins... {current}/{total}")

    def update_plugin_result(self, plugin_name, success, error_message):
        """Update the result for a single plugin"""
        # Find the plugin item
        for i in range(self.plugin_tree.topLevelItemCount()):
            item = self.plugin_tree.topLevelItem(i)
            if item.text(0) == plugin_name:
                # Update status
                if success:
                    item.setText(1, "OK")
                    item.setForeground(1, Qt.green)
                else:
                    # Check if this is a skipped plugin - less strict detection
                    if any(skip_word in error_message.lower() for skip_word in ['skipped', 'gui execution blocked']):
                        item.setText(1, "Skipped")
                        item.setForeground(1, Qt.yellow)
                    else:
                        item.setText(1, "Failed")
                        item.setForeground(1, Qt.red)
                
                # Update error column
                item.setText(3, error_message[:50] + "..." if len(error_message) > 50 else error_message)
                
                # Store plugin info for later use
                plugin_info = item.data(0, Qt.UserRole)
                self.plugin_results[plugin_name] = {
                    'success': success,
                    'error': error_message,
                    'item': item
                }
                break

    def testing_finished(self):
        """Called when all plugins have been tested"""
        self.test_button.setEnabled(True)
        self.test_safe_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Count results
        total = len(self.plugin_results)
        successful = sum(1 for r in self.plugin_results.values() if r['success'])
        failed = total - successful
        
        self.status_label.setText(f"Testing complete: {successful} successful, {failed} failed")

    def cleanup(self):
        """Clean up runner when dialog is closed"""
        if self.test_runner and self.test_runner.is_running:
            self.test_runner.stop_testing()
            self.test_runner = None

    def clear_blacklist(self):
        """Clear all blacklisted plugins"""
        if self.test_runner:
            self.test_runner.blacklisted.clear()
            self.status_label.setText("Blacklist cleared")

    def on_plugin_selected(self, item, column):
        """Handle plugin selection to show details"""
        plugin_name = item.text(0)
        
        if plugin_name in self.plugin_results:
            result = self.plugin_results[plugin_name]
            plugin_info = item.data(0, Qt.UserRole)
            
            # Update details
            details = []
            details.append(f"<b>Plugin:</b> {plugin_name}")
            details.append(f"<b>Module:</b> {plugin_info.get('module_path', 'Unknown')}")
            details.append(f"<b>Source:</b> {plugin_info.get('source', 'Unknown')}")
            details.append(f"<b>Status:</b> {'✓ Success' if result['success'] else '✗ Failed'}")
            
            if plugin_info.get('description'):
                # Format description with proper word wrapping and line breaks
                description = plugin_info['description']
                # Add line breaks for better readability
                formatted_desc = description.replace('. ', '.<br><br>')  # Paragraph breaks
                formatted_desc = formatted_desc.replace(' - ', '<br>- ')  # Bullet points
                formatted_desc = formatted_desc.replace('Features:', '<br><br><b>Features:</b>')  # Features header
                details.append(f"<b>Description:</b><br>{formatted_desc}")
            
            self.details_label.setText("<br>".join(details))
            
            # Show error details if failed
            if not result['success'] and result['error']:
                self.error_text.setText(result['error'])
                self.error_text.setVisible(True)
            else:
                self.error_text.setVisible(False)
        else:
            self.details_label.setText("Select a plugin to view details")
            self.error_text.setVisible(False)


# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    try:
        parent = getattr(chisurf, 'cs', None)
        dialog = PluginCheckWidget(parent=parent)
        dialog.show()
    except Exception as e:
        print(f"Failed to open Plugin Check: {e}")
        import traceback
        traceback.print_exc()
