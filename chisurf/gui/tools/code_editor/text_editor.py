from __future__ import annotations

import pathlib
import sys
from enum import Enum, auto

from qtpy import QtCore, QtGui, QtWidgets

import chisurf
import chisurf.fio as io
from chisurf import logging
import chisurf.gui.widgets
import chisurf.settings


class SyntaxHighlighter(QtGui.QSyntaxHighlighter):
    """Base class for syntax highlighters."""

    def __init__(self, parent=None, font_family=None, font_point_size=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Set up default formats
        self.formats = {}

        if font_family is None:
            font_family = chisurf.settings.gui['editor']['font_family']
        if font_point_size is None:
            font_point_size = chisurf.settings.gui['editor']['font_size']

        self.default_font = QtGui.QFont(font_family, int(font_point_size))

    def add_rule(self, pattern, format_name):
        """Add a highlighting rule with the given pattern and format name."""
        if format_name in self.formats:
            self.highlighting_rules.append((QtCore.QRegExp(pattern), self.formats[format_name]))

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        for pattern, format in self.highlighting_rules:
            expression = QtCore.QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)


class PythonHighlighter(SyntaxHighlighter):
    """Syntax highlighter for Python code."""

    def __init__(self, parent=None, font_family=None, font_point_size=None, 
                 paper_color=None, default_color=None):
        super().__init__(parent, font_family, font_point_size)

        if paper_color is None:
            paper_color = chisurf.settings.gui['editor']['paper_color']
        if default_color is None:
            default_color = chisurf.settings.gui['editor']['default_color']

        # Create formats for different syntax elements
        keyword_format = QtGui.QTextCharFormat()
        keyword_format.setForeground(QtGui.QColor("#569CD6"))
        keyword_format.setFontWeight(QtGui.QFont.Bold)

        class_format = QtGui.QTextCharFormat()
        class_format.setForeground(QtGui.QColor("#4EC9B0"))
        class_format.setFontWeight(QtGui.QFont.Bold)

        function_format = QtGui.QTextCharFormat()
        function_format.setForeground(QtGui.QColor("#DCDCAA"))

        string_format = QtGui.QTextCharFormat()
        string_format.setForeground(QtGui.QColor("#CE9178"))

        comment_format = QtGui.QTextCharFormat()
        comment_format.setForeground(QtGui.QColor("#6A9955"))

        number_format = QtGui.QTextCharFormat()
        number_format.setForeground(QtGui.QColor("#B5CEA8"))

        self.formats = {
            "keyword": keyword_format,
            "class": class_format,
            "function": function_format,
            "string": string_format,
            "comment": comment_format,
            "number": number_format
        }

        # Python keywords
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "exec", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda",
            "not", "or", "pass", "print", "raise", "return", "try",
            "while", "with", "yield", "None", "True", "False"
        ]

        # Add rules for keywords
        keyword_patterns = [r'\b' + word + r'\b' for word in keywords]
        for pattern in keyword_patterns:
            self.add_rule(pattern, "keyword")

        # Add rule for classes
        self.add_rule(r'\bclass\b\s*(\w+)', "class")

        # Add rule for functions
        self.add_rule(r'\bdef\b\s*(\w+)', "function")

        # Add rule for strings
        self.add_rule(r'"[^"\\]*(\\.[^"\\]*)*"', "string")
        self.add_rule(r"'[^'\\]*(\\.[^'\\]*)*'", "string")

        # Add rule for comments
        self.add_rule(r'#[^\n]*', "comment")

        # Add rule for numbers
        self.add_rule(r'\b[0-9]+\b', "number")


class JSONHighlighter(SyntaxHighlighter):
    """Syntax highlighter for JSON."""

    def __init__(self, parent=None, font_family=None, font_point_size=None,
                 paper_color=None, default_color=None):
        super().__init__(parent, font_family, font_point_size)

        if paper_color is None:
            paper_color = chisurf.settings.gui['editor']['paper_color']
        if default_color is None:
            default_color = chisurf.settings.gui['editor']['default_color']

        # Create formats for different syntax elements
        property_format = QtGui.QTextCharFormat()
        property_format.setForeground(QtGui.QColor("#9CDCFE"))

        string_format = QtGui.QTextCharFormat()
        string_format.setForeground(QtGui.QColor("#CE9178"))

        number_format = QtGui.QTextCharFormat()
        number_format.setForeground(QtGui.QColor("#B5CEA8"))

        keyword_format = QtGui.QTextCharFormat()
        keyword_format.setForeground(QtGui.QColor("#569CD6"))
        keyword_format.setFontWeight(QtGui.QFont.Bold)

        self.formats = {
            "property": property_format,
            "string": string_format,
            "number": number_format,
            "keyword": keyword_format
        }

        # Add rule for properties
        self.add_rule(r'"[^"\\]*(\\.[^"\\]*)*"\s*:', "property")

        # Add rule for strings
        self.add_rule(r':\s*"[^"\\]*(\\.[^"\\]*)*"', "string")

        # Add rule for numbers
        self.add_rule(r':\s*-?\b\d+(\.\d+)?([eE][+-]?\d+)?\b', "number")

        # Add rule for keywords
        keywords = ["true", "false", "null"]
        keyword_patterns = [r':\s*\b' + word + r'\b' for word in keywords]
        for pattern in keyword_patterns:
            self.add_rule(pattern, "keyword")


class YAMLHighlighter(SyntaxHighlighter):
    """Syntax highlighter for YAML."""

    def __init__(self, parent=None, font_family=None, font_point_size=None,
                 paper_color=None, default_color=None):
        super().__init__(parent, font_family, font_point_size)

        if paper_color is None:
            paper_color = chisurf.settings.gui['editor']['paper_color']
        if default_color is None:
            default_color = chisurf.settings.gui['editor']['default_color']

        # Create formats for different syntax elements
        key_format = QtGui.QTextCharFormat()
        key_format.setForeground(QtGui.QColor("#9CDCFE"))

        value_format = QtGui.QTextCharFormat()
        value_format.setForeground(QtGui.QColor("#CE9178"))

        comment_format = QtGui.QTextCharFormat()
        comment_format.setForeground(QtGui.QColor("#6A9955"))

        number_format = QtGui.QTextCharFormat()
        number_format.setForeground(QtGui.QColor("#B5CEA8"))

        keyword_format = QtGui.QTextCharFormat()
        keyword_format.setForeground(QtGui.QColor("#569CD6"))
        keyword_format.setFontWeight(QtGui.QFont.Bold)

        self.formats = {
            "key": key_format,
            "value": value_format,
            "comment": comment_format,
            "number": number_format,
            "keyword": keyword_format
        }

        # Add rule for keys
        self.add_rule(r'^\s*[^:]+:', "key")

        # Add rule for values
        self.add_rule(r':\s*[^#\n]+', "value")

        # Add rule for comments
        self.add_rule(r'#[^\n]*', "comment")

        # Add rule for numbers
        self.add_rule(r':\s*-?\b\d+(\.\d+)?([eE][+-]?\d+)?\b', "number")

        # Add rule for keywords
        keywords = ["true", "false", "null", "yes", "no", "on", "off"]
        keyword_patterns = [r':\s*\b' + word + r'\b' for word in keywords]
        for pattern in keyword_patterns:
            self.add_rule(pattern, "keyword")


class LineNumberArea(QtWidgets.QWidget):
    """Widget for displaying line numbers."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QtCore.QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class TextEditor(QtWidgets.QPlainTextEdit):
    """Text editor with syntax highlighting and line numbers."""

    ARROW_MARKER_NUM = 8

    def __init__(
            self,
            parent=None,
            font_family: str = None,
            font_point_size: float = 10.0,
            margins_background_color: str = None,
            marker_background_color: str = None,
            caret_line_background_color: str = None,
            caret_line_visible: bool = False,
            language: str = None,
            **kwargs
    ):
        """
        Initialize the text editor.

        :param parent: Parent widget
        :param font_family: Font family to use
        :param font_point_size: Font size
        :param margins_background_color: Background color for margins
        :param marker_background_color: Background color for markers
        :param caret_line_background_color: Background color for current line
        :param caret_line_visible: Whether to highlight the current line
        :param language: Language for syntax highlighting (Python, JSON, or YAML)
        :param kwargs: Additional keyword arguments
        """
        super().__init__(parent)

        if font_point_size is None:
            font_point_size = chisurf.settings.gui['editor']['font_size']
        if font_family is None:
            font_family = chisurf.settings.gui['editor']['font_family']
        if margins_background_color is None:
            margins_background_color = chisurf.settings.gui['editor']['margins_background_color']
        if marker_background_color is None:
            marker_background_color = chisurf.settings.gui['editor']['marker_background_color']
        if caret_line_background_color is None:
            caret_line_background_color = chisurf.settings.gui['editor']['caret_line_background_color']

        paper_color = kwargs.get("paper_color", chisurf.settings.gui['editor']['paper_color'])
        default_color = kwargs.get("default_color", chisurf.settings.gui['editor']['default_color'])

        # Set the default font
        font = QtGui.QFont()
        font.setFamily(font_family)
        font.setPointSize(int(font_point_size))
        self.setFont(font)

        # Set up line numbers
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)

        # Set up current line highlighting
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.current_line_color = QtGui.QColor(caret_line_background_color)
        self.caret_line_visible = caret_line_visible

        # Set up syntax highlighting
        if language:
            language = language.lower()
            if language == "python":
                self.highlighter = PythonHighlighter(
                    self.document(), 
                    font_family, 
                    font_point_size,
                    paper_color,
                    default_color
                )
            elif language == "json":
                self.highlighter = JSONHighlighter(
                    self.document(), 
                    font_family, 
                    font_point_size,
                    paper_color,
                    default_color
                )
            else:  # Default to YAML
                self.highlighter = YAMLHighlighter(
                    self.document(), 
                    font_family, 
                    font_point_size,
                    paper_color,
                    default_color
                )

        # Set up colors
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(paper_color))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(default_color))
        self.setPalette(palette)

        # Set minimum size
        self.setMinimumSize(400, 200)

    def line_number_area_width(self):
        """Calculate the width of the line number area."""
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1

        space = 3 + self.fontMetrics().width('9') * digits
        return space

    def update_line_number_area_width(self, _):
        """Update the width of the line number area."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        """Update the line number area when the editor's viewport is scrolled."""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        """Handle resize events to adjust the line number area."""
        super().resizeEvent(event)

        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        """Paint the line number area."""
        painter = QtGui.QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QtGui.QColor(chisurf.settings.gui['editor']['margins_background_color']))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtCore.Qt.darkGray)
                rect = QtCore.QRect(0, int(top), self.line_number_area.width(), self.fontMetrics().height())
                painter.drawText(rect, QtCore.Qt.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    def paintEvent(self, event):
        """Paint the editor, including the current line highlight."""
        super().paintEvent(event)

        if self.caret_line_visible:
            # Highlight current line
            selection = QtWidgets.QTextEdit.ExtraSelection()
            selection.format.setBackground(self.current_line_color)
            selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()

            extra_selections = [selection]
            self.setExtraSelections(extra_selections)

    def text(self):
        """Get the text content of the editor."""
        return self.toPlainText()

    def setText(self, text):
        """Set the text content of the editor."""
        self.setPlainText(text)


class CodeEditor(QtWidgets.QWidget):
    """Widget that combines a text editor with load, save, and run buttons."""

    def load_file_event(self, event, filename: str = None, **kwargs):
        self.load_file(filename)

    def load_file(self, filename: str = None, **kwargs):
        """Load a file into the editor."""
        filename = filename or chisurf.gui.widgets.get_filename()
        if not filename:
            return
        try:
            logging.log(0, f"Loading file: {filename}")
            with open(filename, encoding="utf-8") as file:
                self.editor.setText(file.read())
            self.line_edit.setText(str(filename))
            self.filename = filename
        except IOError as e:
            logging.log(1, f"Error loading file {filename}: {e}")

    def run_macro(self, event):
        """Execute the currently loaded Python script."""
        if not self.filename:
            logging.log(1, "No file to run.")
            return
        self.save_text()
        chisurf.console.run_macro(filename=self.filename)

    def save_text(self, event = None):
        """Save the current text to a file."""
        if not self.filename:
            self.filename = chisurf.gui.widgets.save_file(file_type="Python script (*.py)")
            if not self.filename:
                return
        try:
            with io.zipped.open_maybe_zipped(self.filename, "w") as file:
                file.write(self.editor.text())
            self.line_edit.setText(str(pathlib.Path(self.filename).as_posix()))
        except IOError as e:
            logging.log(1, f"Error saving file {self.filename}: {e}")

    def __init__(
        self,
        *args,
        filename: str = 'None',
        language: str = "Python",
        can_load: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.filename = filename
        self.setLayout(layout)
        self.line_edit = QtWidgets.QLineEdit()
        self.editor = TextEditor(
            parent=self,
            language=language
        )
        layout.addWidget(self.editor)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load")
        self.save_button = QtWidgets.QPushButton("Save")
        self.run_button = QtWidgets.QPushButton("Run")

        button_layout.addWidget(self.line_edit)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.run_button)
        layout.addLayout(button_layout)

        # Connect buttons to actions
        self.save_button.clicked.connect(self.save_text)
        self.load_button.clicked.connect(self.load_file_event)
        self.run_button.clicked.connect(self.run_macro)

        # Handle initial file loading
        if pathlib.Path(filename).is_file():
            self.load_file(filename=filename)

        if isinstance(language, str) and language.lower() != "python":
            self.run_button.hide()

        if not can_load:
            self.load_button.hide()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    editor = CodeEditor()
    editor.show()
    app.exec_()
