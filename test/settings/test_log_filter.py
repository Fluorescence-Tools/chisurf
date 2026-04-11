# Consolidated test file: test_log_filter.py


# --- FROM test_log_filter.py ---

# --- FROM test_log_filter_fix.py ---
"""
Test script to verify the fix for the log filter not restoring content issue.
This script simulates logging activity and tests the filter functionality.
"""

import logging
import time
import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QLineEdit, QVBoxLayout, QWidget, QPushButton, QLabel
from qtpy.QtCore import Qt, QTimer

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log Filter Fix Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create log text area
        self.plainTextEditLog = QPlainTextEdit()
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.plainTextEditLog)
        
        # Create filter input
        layout.addWidget(QLabel("Filter:"))
        self.lineEdit_LogFilter = QLineEdit()
        self.lineEdit_LogFilter.textChanged.connect(self.filter_log_content)
        layout.addWidget(self.lineEdit_LogFilter)
        
        # Add buttons for testing
        add_log_button = QPushButton("Add Log Entry")
        add_log_button.clicked.connect(self.add_log_entry)
        layout.addWidget(add_log_button)
        
        clear_filter_button = QPushButton("Clear Filter")
        clear_filter_button.clicked.connect(self.clear_filter)
        layout.addWidget(clear_filter_button)
        
        # Initialize original content storage
        self._original_log_content = ""
        
        # Add some initial log entries
        for i in range(5):
            self.plainTextEditLog.appendPlainText(f"Initial log entry {i+1}")
        
        # Store the initial content
        self._original_log_content = self.plainTextEditLog.toPlainText()
        
        # Set up a timer to add log entries automatically
        self.timer = QTimer()
        self.timer.timeout.connect(self.add_log_entry)
        self.timer.start(5000)  # Add a log entry every 5 seconds
    
    def filter_log_content(self):
        """
        Filter the content of plainTextEditLog based on the text in lineEdit_LogFilter.
        """
        filter_text = self.lineEdit_LogFilter.text().strip().lower()
        
        # Initialize _original_log_content if it doesn't exist
        if not hasattr(self, '_original_log_content'):
            self._original_log_content = ""
        
        # Get the current content
        current_content = self.plainTextEditLog.toPlainText()
        
        # If we're filtering, store the original content only if we don't have a filter active
        # or if we're adding new content (current content is longer than original)
        if not filter_text or len(current_content) > len(self._original_log_content):
            self._original_log_content = current_content
        
        # If there's no filter text, show all content
        if not filter_text:
            # Restore the original content
            self.plainTextEditLog.setPlainText(self._original_log_content)
            return
            
        # Split the original log text into lines
        lines = self._original_log_content.split('\n')
        
        # Filter lines that contain the filter text
        filtered_lines = [line for line in lines if filter_text in line.lower()]
        
        # Clear the current content
        self.plainTextEditLog.clear()
        
        # Add the filtered lines back to the log
        if filtered_lines:
            self.plainTextEditLog.setPlainText('\n'.join(filtered_lines))
        else:
            self.plainTextEditLog.setPlainText("No matching log entries found.")
    
    def add_log_entry(self):
        """Add a new log entry for testing."""
        import random
        prefixes = ["INFO", "DEBUG", "WARNING", "ERROR", "TEST"]
        prefix = random.choice(prefixes)
        self.plainTextEditLog.appendPlainText(f"{prefix}: New log entry at {time.strftime('%H:%M:%S')}")
        
        # Update the filter if needed
        self.update_log_filter()
    
    def update_log_filter(self):
        """
        Update the log filter when new log entries are added.
        """
        # Only apply filtering if there's a filter text
        if hasattr(self, 'lineEdit_LogFilter') and self.lineEdit_LogFilter.text().strip():
            self.filter_log_content()
    
    def clear_filter(self):
        """Clear the filter text."""
        self.lineEdit_LogFilter.clear()


# --- FROM test_log_filter_restore.py ---
"""
Test script to reproduce and verify the issue with log filter not restoring content.
"""

import logging
import time
import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QLineEdit, QVBoxLayout, QWidget, QPushButton, QLabel
from qtpy.QtCore import Qt

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log Filter Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create log text area
        self.plainTextEditLog = QPlainTextEdit()
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.plainTextEditLog)
        
        # Create filter input
        layout.addWidget(QLabel("Filter:"))
        self.lineEdit_LogFilter = QLineEdit()
        self.lineEdit_LogFilter.textChanged.connect(self.filter_log_content)
        layout.addWidget(self.lineEdit_LogFilter)
        
        # Add buttons for testing
        add_log_button = QPushButton("Add Log Entry")
        add_log_button.clicked.connect(self.add_log_entry)
        layout.addWidget(add_log_button)
        
        # Initialize original content storage
        self._original_log_content = ""
        
        # Add some initial log entries
        for i in range(5):
            self.plainTextEditLog.appendPlainText(f"Initial log entry {i+1}")
        
        # Store the initial content
        self._original_log_content = self.plainTextEditLog.toPlainText()
    
    def filter_log_content(self):
        """
        Filter the content of plainTextEditLog based on the text in lineEdit_LogFilter.
        """
        filter_text = self.lineEdit_LogFilter.text().strip().lower()
        
        # Always update the original log content to capture new entries
        self._original_log_content = self.plainTextEditLog.toPlainText()
        
        # If there's no filter text, show all content
        if not filter_text:
            # Restore the original content
            self.plainTextEditLog.setPlainText(self._original_log_content)
            return
            
        # Split the original log text into lines
        lines = self._original_log_content.split('\n')
        
        # Filter lines that contain the filter text
        filtered_lines = [line for line in lines if filter_text in line.lower()]
        
        # Clear the current content
        self.plainTextEditLog.clear()
        
        # Add the filtered lines back to the log
        if filtered_lines:
            self.plainTextEditLog.setPlainText('\n'.join(filtered_lines))
        else:
            self.plainTextEditLog.setPlainText("No matching log entries found.")
    
    def add_log_entry(self):
        """Add a new log entry for testing."""
        import random
        prefixes = ["INFO", "DEBUG", "WARNING", "ERROR", "TEST"]
        prefix = random.choice(prefixes)
        self.plainTextEditLog.appendPlainText(f"{prefix}: New log entry at {time.strftime('%H:%M:%S')}")
        
        # Update the original content
        self._original_log_content = self.plainTextEditLog.toPlainText()
        
        # If filter is active, reapply it
        if self.lineEdit_LogFilter.text().strip():
            self.filter_log_content()

