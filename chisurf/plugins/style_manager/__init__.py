"""
Style Manager Plugin for ChiSurf

This plugin provides tools for managing QSS styles in ChiSurf. It includes:
- A QSS style editor for creating and editing style files
- Functions for copying style files to the user's .chisurf folder
- Functions for clearing style files from the user's .chisurf folder

The plugin integrates with ChiSurf's settings system to ensure styles are properly
managed and applied.
"""

import os
import pathlib
import shutil

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QMessageBox, QStatusBar,
    QTextEdit, QPlainTextEdit, QInputDialog
)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QTextCharFormat, QFont, QColor, QSyntaxHighlighter

import chisurf
import chisurf.settings

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Style Manager"


def copy_styles_to_user_folder():
    """Copies all style files from the gui/styles directory to the user folder,
    ensuring that existing files are not overwritten."""
    package_path = pathlib.Path(chisurf.__file__).parent / 'gui' / 'styles'
    user_settings_path = chisurf.settings.get_path('settings') / 'styles'
    user_settings_path.mkdir(parents=True, exist_ok=True)

    for file in package_path.iterdir():
        if file.is_file() and file.suffix == '.qss':
            destination_file = user_settings_path / file.name
            if not destination_file.exists():  # Avoid overwriting existing files
                shutil.copyfile(file, destination_file)


def clear_style_files():
    """
    Remove style files from the styles subdirectory in the user settings folder.

    This function deletes all .qss files in the styles subdirectory of the user's
    .chisurf folder. After deletion, it copies the original style files from the
    package directory to ensure the application has default styles available.

    Raises:
        None. All deletion errors are caught and logged.
    """
    styles_dir = chisurf.settings.get_path('settings') / 'styles'

    # If the styles directory doesn't exist, nothing to do
    if not os.path.isdir(styles_dir):
        return

    # Iterate through all files in the styles directory
    for entry in os.scandir(styles_dir):
        path = entry.path
        try:
            if not entry.is_dir(follow_symlinks=False):
                # Only remove QSS files (files ending with .qss)
                if str(path).endswith('.qss'):
                    os.unlink(path)
        except PermissionError as e:
            chisurf.logging.warning(f"Skipping locked style file: {path}")
        except OSError as e:
            chisurf.logging.warning(f"Couldn't remove style file: {path}")

    # Copy original style files back to ensure defaults are available
    copy_styles_to_user_folder()


class QssSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for QSS files."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.highlighting_rules = []
        
        # Property format
        property_format = QTextCharFormat()
        property_format.setForeground(QColor("#2980b9"))  # Blue
        property_format.setFontWeight(QFont.Bold)
        property_pattern = r'\b[a-z-]+\s*:'
        self.highlighting_rules.append((QRegExp(property_pattern), property_format))
        
        # Value format
        value_format = QTextCharFormat()
        value_format.setForeground(QColor("#27ae60"))  # Green
        value_pattern = r':\s*[^;]+;'
        self.highlighting_rules.append((QRegExp(value_pattern), value_format))
        
        # Selector format
        selector_format = QTextCharFormat()
        selector_format.setForeground(QColor("#c0392b"))  # Red
        selector_format.setFontWeight(QFont.Bold)
        selector_pattern = r'[#.]?[A-Za-z0-9_-]+\s*[,{]'
        self.highlighting_rules.append((QRegExp(selector_pattern), selector_format))
        
        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#7f8c8d"))  # Gray
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegExp(r'/\*.*\*/'), comment_format))
        
    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)


class StyleManagerWidget(QMainWindow):
    """
    A widget for managing QSS styles in ChiSurf.
    
    This widget provides tools for:
    - Editing QSS style files
    - Applying styles to the application
    - Clearing style files from the user's .chisurf folder
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Style Manager")
        self.resize(800, 600)
        
        # Get the styles directory
        self.styles_dir = chisurf.settings.get_path('settings') / 'styles'
        self.styles_dir.mkdir(parents=True, exist_ok=True)
        
        # Current file being edited
        self.current_file = None
        
        self.setup_ui()
        self.load_style_files()
        
    def setup_ui(self):
        """Set up the user interface components."""
        main_layout = QVBoxLayout()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)
        
        # Top section with file selection and buttons
        top_layout = QHBoxLayout()
        
        # Style file selector
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(200)
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        
        # Buttons
        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.on_new_file)
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.on_save_file)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.on_apply_style)
        
        self.clear_button = QPushButton("Clear All Styles")
        self.clear_button.clicked.connect(self.on_clear_styles)
        
        # Add widgets to top layout
        top_layout.addWidget(QLabel("Style File:"))
        top_layout.addWidget(self.file_combo)
        top_layout.addWidget(self.new_button)
        top_layout.addWidget(self.save_button)
        top_layout.addWidget(self.apply_button)
        top_layout.addWidget(self.clear_button)
        top_layout.addStretch()
        
        # Text editor for QSS
        self.editor = QPlainTextEdit()
        font = QFont("Courier New", 10)
        self.editor.setFont(font)
        
        # Set up syntax highlighting
        self.highlighter = QssSyntaxHighlighter(self.editor.document())
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        
        # Add all components to main layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.editor)
        main_layout.addWidget(self.status_bar)
        
    def load_style_files(self):
        """Load available QSS files from the styles directory."""
        self.file_combo.clear()
        
        # First check if styles directory exists
        if not self.styles_dir.exists():
            self.styles_dir.mkdir(parents=True, exist_ok=True)
            
        # Add files from user styles directory
        qss_files = list(self.styles_dir.glob("*.qss"))
        
        # If no user styles, check package styles
        if not qss_files:
            package_styles_dir = pathlib.Path(chisurf.__file__).parent / 'gui' / 'styles'
            qss_files = list(package_styles_dir.glob("*.qss"))
            
            # Copy these files to user directory
            for file in qss_files:
                dest_file = self.styles_dir / file.name
                if not dest_file.exists():
                    with open(file, 'r') as src, open(dest_file, 'w') as dst:
                        dst.write(src.read())
            
            # Now use the user directory files
            qss_files = list(self.styles_dir.glob("*.qss"))
        
        # Add files to combo box
        for file in sorted(qss_files):
            self.file_combo.addItem(file.name, str(file))
            
        if self.file_combo.count() > 0:
            self.file_combo.setCurrentIndex(0)
            
    def on_file_selected(self, index):
        """Handle file selection from the combo box."""
        if index < 0:
            return
            
        file_path = self.file_combo.itemData(index)
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path):
        """Load a QSS file into the editor."""
        self.current_file = file_path
        try:
            with open(file_path, 'r') as f:
                self.editor.setPlainText(f.read())
            self.status_bar.showMessage(f"Loaded {os.path.basename(file_path)}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading file: {str(e)}")
            
    def on_new_file(self):
        """Create a new QSS file."""
        name, ok = QInputDialog.getText(
            self, "New Style File", "Enter file name (without .qss extension):"
        )
        
        if ok and name:
            if not name.endswith('.qss'):
                name += '.qss'
                
            file_path = self.styles_dir / name
            
            # Check if file already exists
            if file_path.exists():
                reply = QMessageBox.question(
                    self, "File exists", 
                    f"File {name} already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
                    
            # Create empty file
            with open(file_path, 'w') as f:
                f.write("/* QSS Style Sheet */\n\n")
                
            # Reload file list and select the new file
            self.load_style_files()
            index = self.file_combo.findText(name)
            if index >= 0:
                self.file_combo.setCurrentIndex(index)
                
    def on_save_file(self):
        """Save the current file."""
        if not self.current_file:
            self.on_new_file()
            return
            
        try:
            with open(self.current_file, 'w') as f:
                f.write(self.editor.toPlainText())
            self.status_bar.showMessage(f"Saved {os.path.basename(self.current_file)}")
        except Exception as e:
            self.status_bar.showMessage(f"Error saving file: {str(e)}")
            
    def on_apply_style(self):
        """Apply the current style to the application."""
        if not self.current_file:
            return
            
        try:
            # Save first to ensure changes are written
            self.on_save_file()
            
            # Apply the style
            style_sheet = self.editor.toPlainText()
            QApplication.instance().setStyleSheet(style_sheet)
            
            # Update the style sheet in chisurf settings
            chisurf.settings.style_sheet = style_sheet
            
            # Update the current style sheet file name in settings
            file_name = os.path.basename(self.current_file)
            chisurf.settings.gui['style_sheet'] = file_name
            
            self.status_bar.showMessage(f"Applied style: {file_name}")
        except Exception as e:
            self.status_bar.showMessage(f"Error applying style: {str(e)}")
            
    def on_clear_styles(self):
        """Clear all style files from the user's .chisurf folder."""
        reply = QMessageBox.question(
            self, "Clear Styles", 
            "Are you sure you want to clear all style files? This will reset them to defaults.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            clear_style_files()
            self.load_style_files()
            self.status_bar.showMessage("Style files have been reset to defaults")


# When the plugin is loaded, this code will be executed
if __name__ == "plugin":
    # Create an instance of the StyleManagerWidget class
    window = StyleManagerWidget()
    # Show the window
    window.show()