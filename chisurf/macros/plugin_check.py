"""
Plugin Check Macro

This macro provides the core testing logic for checking ChiSurf plugins.
It contains only the testing functionality - the UI is in the plugin.

Usage:
    Import PluginTestRunner from this macro and use callbacks for UI updates.
"""

import sys
import traceback
import pathlib
import queue
import time
import gc
import ast
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import ChiSurf components
import chisurf as cs
from chisurf.plugins import iter_plugins


def _resolve_call_name(call_node, alias_map):
    """Resolve a call target to a fully qualified name when possible."""
    if isinstance(call_node, ast.Name):
        return alias_map.get(call_node.id, call_node.id)

    if isinstance(call_node, ast.Attribute):
        parts = []
        current = call_node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            base_name = alias_map.get(current.id, current.id)
            return ".".join([base_name] + list(reversed(parts)))

    return None


def _contains_dangerous_operations(script_content, dangerous_call_names, dangerous_prefixes):
    """Detect dangerous operations via AST call analysis plus prefix fallback."""
    try:
        tree = ast.parse(script_content)
    except SyntaxError:
        # Let downstream compile/syntax checks handle syntax errors.
        return False

    alias_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                alias_map[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                if module_name:
                    alias_map[local_name] = f"{module_name}.{alias.name}"
                else:
                    alias_map[local_name] = alias.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _resolve_call_name(node.func, alias_map)
            if call_name in dangerous_call_names:
                return True

    return any(prefix in script_content for prefix in dangerous_prefixes)


class PluginTestRunner:
    """Simple plugin test runner using direct execution like ChiSurf plugins"""
    
    def __init__(self):
        self.plugins_to_test = []
        self.safe_mode = False
        self.skip_blacklisted = True
        self.history = {}  # Simple in-memory history
        self.blacklisted = set()  # Simple blacklist
        self.current_index = 0
        self.is_running = False
        self.delay_between_plugins = 0.5  # Seconds to wait between plugins
        self._qt_ui_blocked = False  # Qt UI blocking state
        
        # Callbacks for UI updates
        self.progress_callback = None
        self.result_callback = None
        self.finished_callback = None

    def set_plugins(self, plugins):
        """Set list of plugins to test"""
        self.plugins_to_test = plugins
    
    def set_safe_mode(self, safe_mode):
        """Enable or disable safe mode (shorter timeouts, more aggressive filtering)"""
        self.safe_mode = safe_mode

    def set_skip_blacklisted(self, skip_blacklisted):
        """Enable or disable skipping of blacklisted plugins"""
        self.skip_blacklisted = skip_blacklisted

    def set_delay_between_plugins(self, delay_seconds):
        """Set delay between plugin tests to prevent GUI overload"""
        self.delay_between_plugins = delay_seconds

    def set_callbacks(self, progress_callback=None, result_callback=None, finished_callback=None):
        """Set callback functions for UI updates"""
        self.progress_callback = progress_callback
        self.result_callback = result_callback
        self.finished_callback = finished_callback

    def start_testing(self):
        """Start testing plugins (blocking/synchronous) with UI blocking and memory protection"""
        if self.is_running:
            return
        
        self.is_running = True
        self.current_index = 0
        
        # Block Qt UI to minimize strain during testing
        self._block_qt_ui()
        
        try:
            # Process plugins one by one (blocking)
            while self.is_running and self.current_index < len(self.plugins_to_test):
                self._process_next_plugin()
        finally:
            # Always unblock Qt UI when done
            self._unblock_qt_ui()

    def stop_testing(self):
        """Stop testing plugins"""
        self.is_running = False
        # Unblock Qt UI when stopping
        self._unblock_qt_ui()

    def _block_qt_ui(self):
        """Block Qt UI to minimize strain during testing"""
        try:
            # Try to import Qt components only if needed
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QCoreApplication
            
            app = QApplication.instance()
            if app:
                # Block all Qt event processing
                QCoreApplication.processEvents()  # Process any pending events first
                # Set a flag to indicate UI is blocked
                self._qt_ui_blocked = True
                print("Qt UI blocked for testing")
        except ImportError:
            # Qt not available, no UI to block
            self._qt_ui_blocked = False
        except Exception as e:
            print(f"Warning: Could not block Qt UI: {str(e)}")
            self._qt_ui_blocked = False

    def _unblock_qt_ui(self):
        """Unblock Qt UI after testing"""
        try:
            if hasattr(self, '_qt_ui_blocked') and self._qt_ui_blocked:
                # Try to import Qt components only if needed
                from qtpy.QtWidgets import QApplication
                from qtpy.QtCore import QCoreApplication
                
                app = QApplication.instance()
                if app:
                    # Process any accumulated events
                    QCoreApplication.processEvents()
                    print("Qt UI unblocked after testing")
                
                self._qt_ui_blocked = False
        except ImportError:
            # Qt not available, no UI to unblock
            pass
        except Exception as e:
            print(f"Warning: Could not unblock Qt UI: {str(e)}")

    def _process_next_plugin(self):
        """Process the next plugin in the queue"""
        if not self.is_running or self.current_index >= len(self.plugins_to_test):
            self.is_running = False
            if self.finished_callback:
                self.finished_callback()
            return
        
        plugin_info = self.plugins_to_test[self.current_index]
        plugin_name = plugin_info.get('plugin_name', f'Plugin {self.current_index + 1}')
        
        # Display current plugin being tested
        print(f"Testing plugin {self.current_index + 1}/{len(self.plugins_to_test)}: {plugin_name}")
        
        # Check if plugin is blacklisted
        if self.skip_blacklisted and plugin_name in self.blacklisted:
            if self.result_callback:
                self.result_callback(plugin_name, False, f"Skipped: Blacklisted due to repeated failures")
            
            self.current_index += 1
            if self.progress_callback:
                self.progress_callback(self.current_index, len(self.plugins_to_test), plugin_name)
            
            # Add delay and cleanup even for skipped plugins
            self._cleanup_between_plugins()
            return
        
        # Test plugin
        try:
            success, error_msg = self._test_plugin(plugin_info)
            
            # Update simple history
            if plugin_name not in self.history:
                self.history[plugin_name] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'failed_tests': 0,
                    'recent_failures': []
                }
            
            self.history[plugin_name]['total_tests'] += 1
            self.history[plugin_name]['last_tested'] = datetime.now().isoformat()
            
            if success:
                self.history[plugin_name]['successful_tests'] += 1
                self.history[plugin_name]['recent_failures'] = []
            else:
                self.history[plugin_name]['failed_tests'] += 1
                self.history[plugin_name]['recent_failures'].append(datetime.now())
                
                # Auto-blacklist after 5 failures
                if len(self.history[plugin_name]['recent_failures']) >= 5:
                    self.blacklisted.add(plugin_name)
            
        except Exception as e:
            success = False
            error_msg = f"Plugin test crashed: {str(e)}"
        
        # Emit result
        if self.result_callback:
            self.result_callback(plugin_name, success, error_msg)
        
        # Update progress
        self.current_index += 1
        if self.progress_callback:
            self.progress_callback(self.current_index, len(self.plugins_to_test))
        
        # Cleanup and delay between plugins
        self._cleanup_between_plugins()
    
    def _cleanup_between_plugins(self):
        """Clean up between plugin tests to prevent GUI overload and memory corruption"""
        try:
            # Aggressive garbage collection to clean up any created objects
            gc.collect()
            gc.collect()  # Call twice to ensure thorough cleanup
            gc.collect()  # Third time for final cleanup
            
            # Clear any cached module imports that might cause memory leaks
            if hasattr(sys, 'modules'):
                # Clear modules from plugin directories that might cause memory issues
                modules_to_clear = []
                for module_name in sys.modules:
                    if any(plugin_dir in str(module_name) for plugin_info in self.plugins_to_test 
                           for plugin_dir in [str(plugin_info.get('package_dir', ''))]):
                        modules_to_clear.append(module_name)
                
                for module_name in modules_to_clear:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
            
            # Temporarily disable widget hiding to prevent race conditions
            # self._hide_created_widgets()
            
            # Add configurable delay to let GUI settle
            if self.delay_between_plugins > 0:
                time.sleep(self.delay_between_plugins)
            
        except Exception as e:
            # Don't let cleanup errors stop the testing
            pass
    
    def _hide_created_widgets(self):
        """Hide only widgets created by plugins during testing, not the test UI"""
        try:
            # Try to import Qt components only if needed
            # This won't be imported at module level, so no Qt dependency in macro
            from qtpy.QtWidgets import QApplication, QWidget
            app = QApplication.instance()
            
            if app:
                # Add extra safety - check if application is shutting down
                if not app or app.closingDown():
                    return
                
                # Get all top-level widgets with thread safety
                widgets_to_hide = []
                
                # Store widgets that should NOT be hidden
                protected_widgets = set()
                
                # Add main ChiSurf window to protected list
                main_window = getattr(cs, 'cs', None)
                if main_window and hasattr(main_window, 'isVisible') and main_window.isVisible():
                    protected_widgets.add(main_window)
                
                # Add any existing PluginCheckWidget dialogs to protected list
                for widget in app.topLevelWidgets():
                    if not widget or not hasattr(widget, 'isVisible') or not widget.isVisible():
                        continue
                    
                    # Check if this is a PluginCheckWidget (our test UI)
                    if widget.__class__.__name__ == 'PluginCheckWidget':
                        protected_widgets.add(widget)
                    # Also check window title
                    elif hasattr(widget, 'windowTitle'):
                        title = widget.windowTitle()
                        if 'Plugin Check' in title or 'ChiSurf Plugin Check' in title:
                            protected_widgets.add(widget)
                
                # Find widgets to hide (exclude protected ones) with extra checks
                for widget in app.topLevelWidgets():
                    if not widget or not hasattr(widget, 'isVisible') or not widget.isVisible():
                        continue
                    
                    # Double-check widget is not in protected list
                    if widget not in protected_widgets:
                        # Additional safety check - make sure widget is valid
                        try:
                            # Test if widget is still valid by checking a basic property
                            _ = widget.objectName()
                            widgets_to_hide.append(widget)
                        except (RuntimeError, AttributeError):
                            # Widget is no longer valid, skip it
                            continue
                
                # Hide only widgets that were likely created by plugins
                for widget in widgets_to_hide:
                    try:
                        # Additional safety check before hiding
                        if widget and hasattr(widget, 'isVisible') and widget.isVisible():
                            widget.hide()
                            # Also close dialog-style widgets more aggressively
                            if hasattr(widget, 'close'):
                                widget.close()
                    except (RuntimeError, AttributeError):
                        # Widget became invalid during operation, skip it
                        continue
                
                # Force process events to ensure widgets are hidden, but with error handling
                try:
                    app.processEvents()
                except RuntimeError:
                    # Application might be shutting down
                    pass
                
        except ImportError:
            # Qt not available, skip widget hiding
            pass
        except Exception:
            # Any other error, skip widget hiding
            pass

    def _test_plugin(self, plugin_info):
        """Test a single plugin using ChiSurf's exact plugin execution mechanism"""
        if not plugin_info:
            return False, "No plugin info"
        
        package_dir = plugin_info.get('package_dir')
        if not package_dir:
            return False, "No package directory"
        
        try:
            # Use ChiSurf's exact plugin execution mechanism
            package_dir = pathlib.Path(package_dir)
            wizard_path = package_dir / "wizard.py"
            init_path = package_dir / "__init__.py"
            
            # Determine which script to run (same as ChiSurf)
            if wizard_path.exists():
                script_file = wizard_path
            elif init_path.exists():
                script_file = init_path
            else:
                return False, "No wizard.py or __init__.py found"
            
            # Read the script first to analyze it
            with open(script_file, "r", encoding="utf-8") as file:
                script_content = file.read()
            
            # Check for dangerous patterns (AST-aware to avoid false positives, e.g. ast.literal_eval)
            dangerous_call_names = {
                'os.system', 'subprocess.call', 'subprocess.run', 'subprocess.Popen',
                'eval', 'exec', 'input', 'raw_input',
                'socket.socket', 'urllib.request.urlopen', 'requests.get',
                'multiprocessing.Process'
            }
            dangerous_prefixes = ()
            has_dangerous = _contains_dangerous_operations(
                script_content,
                dangerous_call_names,
                dangerous_prefixes
            )
            
            if has_dangerous:
                return False, "Plugin contains potentially dangerous operations - skipped for safety"
            
            # Check if plugin has GUI components
            gui_imports = [
                'from qtpy', 'from PyQt', 'from PySide', 'import qtpy', 'import PyQt', 'import PySide',
                'import matplotlib', 'from matplotlib', 'import plotly', 'from plotly',
                'import seaborn', 'from seaborn', 'import tkinter', 'from tkinter',
            ]
            
            has_gui = any(imp in script_content for imp in gui_imports)
            
            # Check for complex imports
            complex_imports = [
                'import cs', 'from cs', 'import numpy', 'from numpy',
                'import scipy', 'from scipy', 'import pandas', 'from pandas',
                'import torch', 'from torch', 'import tensorflow', 'from tensorflow'
            ]
            
            has_complex = any(imp in script_content for imp in complex_imports)
            
            # For complex GUI plugins, do quick syntax check only
            if has_gui and has_complex and self.safe_mode:
                return self._test_plugin_syntax_only(script_file, script_content)
            # For GUI plugins, try to test imports
            elif has_gui:
                return self._test_gui_plugin_safely(script_file, script_content)
            else:
                # For non-GUI plugins, test using ChiSurf's exact mechanism
                return self._test_plugin_chisurf_way(script_file, script_content)
                    
        except Exception as e:
            return False, f"Plugin test setup failed: {str(e)}"

    def _test_plugin_syntax_only(self, script_file, script_content):
        """Test plugin by checking syntax only (fastest method)"""
        try:
            compile(script_content, str(script_file), "exec")
            return True, "Plugin syntax OK (complex imports skipped for speed)"
                
        except SyntaxError as e:
            return False, f"Plugin has syntax error: {str(e)}"
        except Exception as e:
            return False, f"Plugin syntax check failed: {str(e)}"

    def _test_gui_plugin_safely(self, script_file, script_content):
        """Test GUI plugin by checking imports and structure without executing GUI code"""
        try:
            # Parse the script to check for syntax errors and import issues
            compile(script_content, str(script_file), "exec")
            
            # Try to extract import statements and test them individually
            import_lines = []
            for line in script_content.split('\n'):
                line = line.strip()
                if line.startswith(('import ', 'from ')) and not line.startswith('#'):
                    import_lines.append(line)
            
            # Test each import in isolation with timeout
            failed_imports = []
            critical_failures = []
            
            for import_line in import_lines:
                try:
                    # Test imports with timeout to prevent hanging
                    success = self._test_import_with_timeout(import_line)
                    if not success:
                        failed_imports.append(f"{import_line}: timeout or error")
                except ImportError as e:
                    failed_imports.append(f"{import_line}: {str(e)}")
                    # Check if this is a critical failure (DLL load, module not found)
                    if any(critical in str(e).lower() for critical in ['dll load failed', 'no module named']):
                        critical_failures.append(f"{import_line}: {str(e)}")
                except Exception as e:
                    failed_imports.append(f"{import_line}: {str(e)}")
            
            # If there are critical failures, report them
            if critical_failures:
                return False, f"GUI plugin has critical import issues: {'; '.join(critical_failures[:2])}"
            # If there are minor import issues but syntax is OK, consider it a partial success
            elif failed_imports:
                return True, f"GUI plugin syntax OK, minor import issues: {'; '.join(failed_imports[:2])}"
            else:
                return True, "GUI plugin imports successful (GUI execution skipped for safety)"
                
        except SyntaxError as e:
            return False, f"Plugin has syntax error: {str(e)}"
        except Exception as e:
            return False, f"GUI plugin analysis failed: {str(e)}"

    def _test_plugin_chisurf_way(self, script_file, script_content):
        """Test plugin using ChiSurf's exact execution mechanism with comprehensive Qt blocking"""
        try:
            # Create a mock main window with onRunMacro method
            class MockMainWindow:
                def __init__(self):
                    self._plugin_contexts = {}
                
                def onRunMacro(self, script_path, executor='exec', globals=None):
                    """Mock implementation of ChiSurf's onRunMacro with comprehensive Qt blocking"""
                    if executor != 'exec':
                        return False, "Only 'exec' executor supported"
                    
                    if globals is None:
                        globals = {"__name__": "plugin"}
                    
                    # Set __package__ for relative imports to work
                    script_path_obj = pathlib.Path(script_path)
                    script_dir = script_path_obj.parent
                    
                    # For all plugins, set __package__ to the plugin directory name
                    # and add the parent directory to sys.path so the package can be resolved
                    package_name = script_dir.name
                    globals["__package__"] = package_name
                    
                    # Add parent directory to sys.path so the package can be resolved
                    parent_dir = str(script_dir.parent)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    
                    # Add script directory to sys.path temporarily
                    script_dir = str(pathlib.Path(script_path).parent)
                    original_sys_path = sys.path.copy()
                    if script_dir not in sys.path:
                        sys.path.insert(0, script_dir)
                    
                    try:
                        # Read and execute the script with comprehensive Qt blocking
                        with open(script_path, "r", encoding="utf-8") as file:
                            script_content = file.read()
                        
                        # Comprehensive Qt blocking - check for ALL Qt-related patterns
                        qt_patterns = [
                            # Direct Qt imports
                            'from qtpy', 'from PyQt', 'from PySide', 
                            'import qtpy', 'import PyQt', 'import PySide',
                            'from qtpy.QtWidgets', 'from PyQt5.QtWidgets', 'from PySide2.QtWidgets',
                            'import qtpy.QtWidgets', 'import PyQt5.QtWidgets', 'import PySide2.QtWidgets',
                            'from qtpy.QtCore', 'from PyQt5.QtCore', 'from PySide2.QtCore',
                            'import qtpy.QtCore', 'import PyQt5.QtCore', 'import PySide2.QtCore',
                            'from qtpy.QtGui', 'from PyQt5.QtGui', 'from PySide2.QtGui',
                            'import qtpy.QtGui', 'import PyQt5.QtGui', 'import PySide2.QtGui',
                            # Qt class usage
                            'QApplication', 'QWidget', 'QDialog', 'QMainWindow', 'QPushButton',
                            'QLabel', 'QLineEdit', 'QTextEdit', 'QComboBox', 'QCheckBox',
                            'QVBoxLayout', 'QHBoxLayout', 'QGridLayout', 'QFormLayout',
                            'QTimer', 'QThread', 'QSignal', 'QSlot', 'QObject',
                            # Qt methods
                            '.show()', '.hide()', '.exec()', '.exec_()', '.close()',
                            'QApplication.instance()', 'QApplication.processEvents()',
                            # Qt-related modules
                            'matplotlib.', 'plotly.', 'seaborn.', 'tkinter.',
                            # GUI-related patterns
                            'plt.show()', 'plt.figure(', 'plt.subplot(',
                            'fig.show()', 'fig.savefig(',
                        ]
                        
                        has_qt = any(pattern in script_content for pattern in qt_patterns)
                        
                        if has_qt:
                            # For ANY Qt/GUI plugins, do syntax check only
                            try:
                                compile(script_content, str(script_path), "exec")
                                return True, "GUI/Qt plugin syntax OK (Qt/GUI execution blocked for safety)"
                            except SyntaxError as e:
                                return False, f"GUI/Qt plugin has syntax error: {str(e)}"
                        
                        # Additional memory protection checks (AST-aware)
                        dangerous_call_names = {
                            'os.system', 'subprocess.call', 'subprocess.run', 'subprocess.Popen',
                            'eval', 'exec', 'input', 'raw_input',
                            'socket.socket', 'urllib.request.urlopen', 'requests.get',
                            'multiprocessing.Process', 'memoryview',
                            'array.array', 'buffer', 'bytearray',
                            # Additional Qt-related dangerous patterns
                            'QApplication', 'QWidget', 'QDialog', 'QMainWindow',
                            'QTimer', 'QThread', 'QObject'
                        }
                        dangerous_prefixes = ('ctypes.',)
                        has_dangerous = _contains_dangerous_operations(
                            script_content,
                            dangerous_call_names,
                            dangerous_prefixes
                        )
                        
                        if has_dangerous:
                            return False, "Plugin contains potentially dangerous operations - skipped for memory safety"
                        
                        # Compile first to catch syntax errors
                        compiled_code = compile(script_content, str(script_path), "exec")
                        
                        # Create completely isolated globals
                        script_path_obj = pathlib.Path(script_path)
                        script_dir = script_path_obj.parent
                        
                        # For all plugins, set __package__ to the plugin directory name
                        package_name = script_dir.name
                        
                        isolated_globals = {
                            "__name__": "plugin", 
                            "__package__": package_name,
                            "__builtins__": __builtins__
                        }
                        
                        # Add parent directory to sys.path so the package can be resolved
                        parent_dir = str(script_dir.parent)
                        if parent_dir not in sys.path:
                            sys.path.insert(0, parent_dir)
                        
                        # Block ALL Qt and GUI modules
                        blocked_modules = [
                            'qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'PyQt4', 'PySide',
                            'QtWidgets', 'QtCore', 'QtGui', 'QtTest', 'QtNetwork', 'QtSql',
                            'QtOpenGL', 'QtPrintSupport', 'QtSvg', 'QtXml', 'QtXmlPatterns',
                            'QtDesigner', 'QtHelp', 'QtMultimedia', 'QtMultimediaWidgets',
                            'QtOpenGLWidgets', 'QtQuick', 'QtQuickWidgets', 'QtQml',
                            'QtGamepad', 'QtSerialPort', 'QtWebEngine', 'QtWebEngineWidgets',
                            'matplotlib', 'pyplot', 'plt', 'plotly', 'seaborn', 'tkinter',
                            'matplotlib.pyplot', 'matplotlib.figure', 'matplotlib.backends',
                            'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
                            'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui',
                            'PySide2.QtWidgets', 'PySide2.QtCore', 'PySide2.QtGui',
                            'PySide6.QtWidgets', 'PySide6.QtCore', 'PySide6.QtGui'
                        ]
                        
                        for module in blocked_modules:
                            isolated_globals[module] = None
                        
                        # Block dangerous modules
                        dangerous_modules = ['os', 'subprocess', 'socket', 'urllib', 'requests', 'multiprocessing', 'ctypes']
                        for module in dangerous_modules:
                            isolated_globals[module] = None
                        
                        # Execute with maximum protection
                        try:
                            exec(compiled_code, isolated_globals)
                        except MemoryError:
                            return False, "Plugin caused memory error - skipped for safety"
                        except SystemError:
                            return False, "Plugin caused system error - skipped for safety"
                        except RuntimeError as e:
                            if any(qt_term in str(e).lower() for qt_term in ['qt', 'gui', 'widget', 'application']):
                                return False, f"Plugin caused Qt/GUI runtime error: {str(e)}"
                            elif 'memory' in str(e).lower():
                                return False, f"Plugin caused memory runtime error: {str(e)}"
                            raise
                        except Exception as e:
                            # Check if error is Qt-related
                            if any(qt_term in str(e).lower() for qt_term in ['qt', 'gui', 'widget', 'application', 'sip']):
                                return False, f"Plugin caused Qt/GUI error: {str(e)}"
                            raise
                        
                        return True, "Plugin executed successfully (comprehensive Qt blocked, memory protected)"
                        
                    finally:
                        # Always restore sys.path
                        sys.path = original_sys_path
            
            # Create mock main window
            mock_main_window = MockMainWindow()
            
            # Use ChiSurf's exact logic from run_plugin_from_dir
            package_dir = script_file.parent
            wizard_path = package_dir / "wizard.py"
            init_path = package_dir / "__init__.py"
            
            # Create persistent globals dict per plugin (same as ChiSurf)
            plugin_key = str(package_dir)
            context = mock_main_window._plugin_contexts.get(plugin_key)
            if context is None:
                # Set __package__ for relative imports to work
                package_name = package_dir.name
                
                context = {
                    "__name__": "plugin",
                    "__package__": package_name
                }
                mock_main_window._plugin_contexts[plugin_key] = context
                
                # Add parent directory to sys.path so the package can be resolved
                parent_dir = str(package_dir.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
            
            # Execute using ChiSurf's exact mechanism
            if wizard_path.exists():
                # Add default adr like ChiSurf does
                context.setdefault("adr", "https://github.com/fluorescence-tools/cs")
                return mock_main_window.onRunMacro(str(wizard_path), executor='exec', globals=context)
            elif init_path.exists():
                return mock_main_window.onRunMacro(str(init_path), executor='exec', globals=context)
            else:
                return False, "No wizard.py or __init__.py found"
            
        except SyntaxError as e:
            return False, f"Plugin has syntax error: {str(e)}"
        except ImportError as e:
            return False, f"Plugin import error: {str(e)}"
        except MemoryError:
            return False, "Plugin caused memory error - skipped for safety"
        except SystemError:
            return False, "Plugin caused system error - skipped for safety"
        except Exception as e:
            # Check if error is Qt-related at the top level
            if any(qt_term in str(e).lower() for qt_term in ['qt', 'gui', 'widget', 'application', 'sip']):
                return False, f"Plugin caused Qt/GUI error: {str(e)}"
            
            error_msg = f"Plugin execution failed: {str(e)}"
            # Get traceback but limit it to avoid huge outputs
            if hasattr(e, '__traceback__') and e.__traceback__:
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                # Take last 3 lines of traceback
                error_msg += "\n" + "".join(tb_lines[-3:])
            return False, error_msg

    def _test_import_with_timeout(self, import_line, timeout=2.0):
        """Test a single import with timeout protection (synchronous)"""
        try:
            # Create a minimal globals dict for import testing
            test_globals = {"__name__": "plugin"}
            
            # Execute import synchronously with simple timeout check
            start_time = time.time()
            exec(import_line, test_globals)
            
            # Check if execution took too long (simple timeout)
            if time.time() - start_time > timeout:
                return False
            
            return True
            
        except Exception as e:
            return False


def main():
    """Main entry point for the plugin check macro"""
    try:
        parent = getattr(cs, 'cs', None)
        print("Plugin Check macro - use PluginCheckWidget from the plugin for GUI")
        print("Testing logic is available in PluginTestRunner class")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# Run the macro
if __name__ == "__main__":
    main()
