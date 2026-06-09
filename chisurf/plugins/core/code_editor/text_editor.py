from __future__ import annotations

import pathlib
import sys
from enum import Enum, auto

from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.core.fio as io
from chisurf import logging
import chisurf.gui.widgets
import chisurf.core.settings
from chisurf.plugins.core.code_editor.agent_panel import AgentPanelWidget
from chisurf.gui.widgets.dock_area import DockArea


class SyntaxHighlighter(QtGui.QSyntaxHighlighter):
    """Base class for syntax highlighters."""

    def __init__(self, parent=None, font_family=None, font_point_size=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Set up default formats
        self.formats = {}

        if font_family is None:
            font_family = cs.core.settings.gui['editor']['font_family']
        if font_point_size is None:
            font_point_size = cs.core.settings.gui['editor']['font_size']

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
            paper_color = cs.core.settings.gui['editor']['paper_color']
        if default_color is None:
            default_color = cs.core.settings.gui['editor']['default_color']

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
            paper_color = cs.core.settings.gui['editor']['paper_color']
        if default_color is None:
            default_color = cs.core.settings.gui['editor']['default_color']

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
            paper_color = cs.core.settings.gui['editor']['paper_color']
        if default_color is None:
            default_color = cs.core.settings.gui['editor']['default_color']

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
            font_point_size = cs.core.settings.gui['editor']['font_size']
        if font_family is None:
            font_family = cs.core.settings.gui['editor']['font_family']
        if margins_background_color is None:
            margins_background_color = cs.core.settings.gui['editor']['margins_background_color']
        if marker_background_color is None:
            marker_background_color = cs.core.settings.gui['editor']['marker_background_color']
        if caret_line_background_color is None:
            caret_line_background_color = cs.core.settings.gui['editor']['caret_line_background_color']

        paper_color = kwargs.get("paper_color", cs.core.settings.gui['editor']['paper_color'])
        default_color = kwargs.get("default_color", cs.core.settings.gui['editor']['default_color'])

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

        # Navigation history
        self._nav_history = []
        self._nav_index = -1
        self.current_file = None
        self.external_definition_callback = None

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

        space = 10 + self.fontMetrics().horizontalAdvance('9') * digits
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
        painter.fillRect(event.rect(), QtGui.QColor(cs.core.settings.gui['editor']['margins_background_color']))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtGui.QColor('#D0D0D0'))
                painter.setFont(self.font())
                rect = QtCore.QRect(0, int(top), self.line_number_area.width() - 4, self.fontMetrics().height())
                painter.drawText(rect, QtCore.Qt.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # Ctrl+Click or Cmd+Click for jump to definition
        if event.modifiers() & QtCore.Qt.ControlModifier or event.modifiers() & QtCore.Qt.MetaModifier:
            cursor = self.cursorForPosition(event.pos())
            cursor.select(QtGui.QTextCursor.WordUnderCursor)
            word = cursor.selectedText()
            if word:
                self.jump_to_definition(word)


    def navigate_back(self):
        if self._nav_index > 0:
            self._nav_index -= 1
            file_path, line_number = self._nav_history[self._nav_index]
            self.goto_file_line(file_path, line_number)

    def navigate_forward(self):
        if self._nav_index < len(self._nav_history) - 1:
            self._nav_index += 1
            file_path, line_number = self._nav_history[self._nav_index]
            self.goto_file_line(file_path, line_number)

    def goto_file_line(self, file_path, line_number):
        # Notify parent to load file if different; the callback handles
        # navigation when the file changes (including cursor positioning).
        if hasattr(self, "file_load_callback") and getattr(self, "current_file", "") != file_path:
            self.file_load_callback(file_path, line_number=line_number)
            return

        doc = self.document()
        block = doc.findBlockByNumber(line_number)
        if block.isValid():
            cursor = self.textCursor()
            cursor.setPosition(block.position())
            self.setTextCursor(cursor)
            self.centerCursor()

    def push_nav_history(self, file_path=None, line_number=None):
        if file_path is None:
            # We assume the parent or holder sets a property, or we just store line number
            file_path = getattr(self, "current_file", "")
        if line_number is None:
            line_number = self.textCursor().blockNumber()
            
        # truncate future history if we are in the past
        if self._nav_index < len(self._nav_history) - 1:
            self._nav_history = self._nav_history[:self._nav_index + 1]
            
        # don't push if it's the exact same as last
        if self._nav_history and self._nav_history[-1] == (file_path, line_number):
            return
            
        self._nav_history.append((file_path, line_number))
        self._nav_index = len(self._nav_history) - 1

    def jump_to_definition(self, word):
        import re
        content = self.toPlainText()
        lines = content.split('\n')
        pattern = re.compile(r'^ *(def |class )' + re.escape(word) + r'\b')
        for i, line in enumerate(lines):
            if pattern.match(line):
                self.push_nav_history()
                doc = self.document()
                block = doc.findBlockByNumber(i)
                cursor = self.textCursor()
                cursor.setPosition(block.position())
                self.setTextCursor(cursor)
                self.centerCursor()
                self.push_nav_history()
                return

        # Not found in current file — try external modules
        if self.external_definition_callback is not None:
            self._jump_to_external_definition(word)

    def _jump_to_external_definition(self, word):
        import re
        import inspect
        import importlib

        content = self.toPlainText()
        lines = content.split('\n')

        imports = []
        for line in lines:
            m = re.match(r'^\s*import\s+(\S+)', line)
            if m:
                imports.append(m.group(1).split('.')[0])
            m = re.match(r'^\s*from\s+(\S+)\s+import', line)
            if m:
                imports.append(m.group(1).split('.')[0])

        for mod_name in set(imports):
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            try:
                obj = getattr(mod, word, None)
            except Exception:
                continue
            if obj is None:
                continue
            try:
                source_file = inspect.getsourcefile(obj)
                if source_file is None:
                    continue
                _, line_num = inspect.getsourcelines(obj)
                self.push_nav_history()
                self.external_definition_callback(source_file, line_num)
                return
            except Exception:
                continue

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
    """Tabbed text editor with DockArea tabs and an AI agent side panel."""

    def __init__(
        self,
        *args,
        filename: str = None,
        language: str = "Python",
        can_load: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.filename = filename
        self._open_files: dict[str, QtWidgets.QWidget] = {}
        self._agent_panel_visible = False
        self.setLayout(main_layout)

        self.tab_widget = DockArea()
        self.tab_widget.tabActionRequested.connect(self._on_tab_action)
        self.tab_widget.newTabRequested.connect(self._add_new_editor_tab)
        self.tab_widget.setCloseTabCallback(self._close_tab)
        main_layout.addWidget(self.tab_widget)

        self._create_editor_tab(filename=filename, language=language)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setNewTabButtonVisible(True)
        self.tab_widget.setContextMenuEnabled(True)

        self.agent_panel = AgentPanelWidget(
            parent=None,
            get_context_callback=self._get_editor_context
        )

    def _add_new_editor_tab(self):
        """Create a new blank editor tab with a unique name."""
        base = "Untitled"
        used = set()
        for i in range(self.tab_widget.count()):
            if self.tab_widget.widget(i) is self.agent_panel:
                continue
            text = self.tab_widget.tabText(i)
            used.add(text[:-2] if text.endswith(" *") else text)
        if base not in used:
            name = base
        else:
            n = 1
            while f"{base}-{n}" in used:
                n += 1
            name = f"{base}-{n}"
        self._create_editor_tab(filename=name)

    def _create_editor_tab(self, filename: str = None, language: str = "Python"):
        """Create a new editor tab."""
        editor = TextEditor(parent=self, language=language)
        self.tab_widget.addTab(editor, filename or "Untitled")
        editor.document().modificationChanged.connect(
            lambda modified, e=editor: self._on_modification_changed(e, modified)
        )
        tab_index = self.tab_widget.indexOf(editor)
        if hasattr(self, '_on_editor_created'):
            self._on_editor_created(editor)
        return editor, tab_index

    def _get_current_editor(self):
        """Get the current editor widget."""
        w = self.tab_widget.currentWidget()
        if w is self.agent_panel:
            return None
        return w

    def _get_current_filename(self):
        """Get the filename of the current tab (without dirty marker)."""
        idx = self.tab_widget.currentIndex()
        if idx >= 0:
            text = self.tab_widget.tabText(idx)
            return text[:-2] if text.endswith(" *") else text
        return None

    def _on_modification_changed(self, editor: QtWidgets.QWidget, modified: bool) -> None:
        """Update the `` *`` marker on the tab when the document's modified state changes."""
        idx = self.tab_widget.indexOf(editor)
        if idx < 0:
            return
        text = self.tab_widget.tabText(idx)
        base = text[:-2] if text.endswith(" *") else text
        self.tab_widget.setTabText(idx, base + " *" if modified else base)

    def _toggle_agent_panel(self):
        """Toggle the AI agent panel visibility."""
        if self._agent_panel_visible:
            idx = self.tab_widget.indexOf(self.agent_panel)
            if idx >= 0:
                self.tab_widget.removeTab(idx)
            self._agent_panel_visible = False
        else:
            idx = self.tab_widget.indexOf(self.agent_panel)
            if idx < 0:
                self.tab_widget.addTab(self.agent_panel, "Agent")
            self.tab_widget.setCurrentWidget(self.agent_panel)
            self.agent_panel.show()
            self._agent_panel_visible = True

    def _get_editor_context(self) -> str:
        """Get the current editor content for the agent context."""
        editor = self._get_current_editor()
        if editor is None:
            return ""

        filename = self._get_current_filename() or "Untitled"
        content = editor.toPlainText()

        return f"File: {filename}\n\n```{content}\n```"

    def _close_tab(self, index: int):
        """Close a tab at the given absolute index."""
        widget = self.tab_widget.widget(index)

        # Agent panel close = toggle off
        if widget is self.agent_panel:
            self._agent_panel_visible = False
            self.tab_widget.removeTab(index)
            return

        # Confirm close if dirty (check the tab-text marker which is always in sync)
        if widget is not self.agent_panel:
            tab_text = self.tab_widget.tabText(index)
            if tab_text.endswith(" *"):
                name = tab_text[:-2]
                msg = QtWidgets.QMessageBox(self)
                msg.setWindowTitle("Unsaved Changes")
                msg.setText(f"Do you want to save changes to {name}?")
                msg.setIcon(QtWidgets.QMessageBox.Question)
                msg.setStandardButtons(
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel
                )
                msg.setDefaultButton(QtWidgets.QMessageBox.Save)
                reply = msg.exec_()
                if reply == QtWidgets.QMessageBox.Save:
                    self._save_tab(widget, name, index)
                elif reply == QtWidgets.QMessageBox.Cancel:
                    return

        # Remove from _open_files by matching the widget
        to_remove = [k for k, v in self._open_files.items() if v is widget]
        for k in to_remove:
            del self._open_files[k]

        # Compute editor count BEFORE removing
        editor_count = sum(
            1 for i in range(self.tab_widget.count())
            if self.tab_widget.widget(i) not in (self.agent_panel, None)
        )

        self.tab_widget.removeTab(index)
        if widget:
            widget.deleteLater()

        # Open a blank Untitled tab if the last editor was just closed
        if editor_count <= 1:
            self._add_new_editor_tab()

    def _on_tab_action(self, action: str, index: int):
        """Handle context-menu actions on tabs.

        Parameters
        ----------
        action : str
            One of ``"save"``, ``"save_as"``, ``"rename"``, ``"reload"``,
            ``"copy_path"``, ``"copy_name"``, ``"copy_dir"``.
        index : int
            The absolute tab index.
        """
        editor = self.tab_widget.widget(index)
        if editor is None or editor is self.agent_panel:
            return
        tab_text = self.tab_widget.tabText(index)
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text

        if action == "save":
            self._save_tab(editor, clean, index)
        elif action == "save_as":
            self._save_tab_as(editor, clean, index)
        elif action == "rename":
            self._rename_tab(editor, clean, index)
        elif action == "reload":
            self._reload_tab(editor, clean, index)
        elif action == "copy_path":
            self._copy_to_clipboard(clean)
        elif action == "copy_name":
            self._copy_to_clipboard(pathlib.Path(clean).name)
        elif action == "copy_dir":
            self._copy_to_clipboard(str(pathlib.Path(clean).parent))

    def _save_tab(self, editor, tab_text: str, index: int):
        """Save the tab content to its file."""
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text
        if clean and clean != "Untitled":
            try:
                with io.zipped.open_maybe_zipped(clean, "w") as f:
                    f.write(editor.text())
            except IOError as e:
                logging.log(1, f"Error saving {clean}: {e}")
                return
        else:
            self._save_tab_as(editor, clean, index)
            return
        editor.document().setModified(False)

    def _save_tab_as(self, editor, tab_text: str, index: int):
        """Open a save-as dialog and save the tab content."""
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text
        new_filename = cs.gui.widgets.save_file(file_type="Python script (*.py)")
        if not new_filename:
            return
        new_path = str(new_filename)
        try:
            with io.zipped.open_maybe_zipped(new_path, "w") as f:
                f.write(editor.text())
        except IOError as e:
            logging.log(1, f"Error saving {new_path}: {e}")
            return
        self.tab_widget.setTabText(index, new_path)
        old_key = next((k for k, v in self._open_files.items() if v is editor), None)
        if old_key:
            del self._open_files[old_key]
        self._open_files[new_path] = editor
        editor.document().setModified(False)

    def _rename_tab(self, editor, tab_text: str, index: int):
        """Prompt for a new tab name and update accordingly."""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Tab", "New name:", text=tab_text
        )
        if not ok or not new_name or new_name == tab_text:
            return
        self.tab_widget.setTabText(index, new_name)
        if editor.document().isModified():
            self.tab_widget.setTabText(index, new_name + " *")
        old_key = next((k for k, v in self._open_files.items() if v is editor), None)
        if old_key:
            del self._open_files[old_key]
        self._open_files[new_name] = editor

    def _reload_tab(self, editor, tab_text: str, index: int):
        """Re-read the file from disk and replace editor content."""
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text
        if clean == "Untitled":
            return
        try:
            with open(clean, encoding="utf-8") as f:
                editor.blockSignals(True)
                editor.setText(f.read())
                editor.blockSignals(False)
        except IOError as e:
            logging.log(1, f"Error reloading {clean}: {e}")
            return
        editor.document().setModified(False)

    @staticmethod
    def _copy_to_clipboard(text: str):
        """Copy *text* to the system clipboard."""
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(text)

    def load_file_event(self, event, filename: str = None, **kwargs):
        self.load_file(filename)

    def load_file(self, filename: str = None, **kwargs):
        """Load a file into the current or a new tab."""
        filename = filename or cs.gui.widgets.get_filename()
        if not filename:
            return

        filename_str = str(filename)

        if filename_str in self._open_files:
            editor = self._open_files[filename_str]
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))
            return

        try:
            logging.log(0, f"Loading file: {filename_str}")
            with open(filename_str, encoding="utf-8") as file:
                content = file.read()
        except IOError as e:
            logging.log(1, f"Error loading file {filename_str}: {e}")
            return

        editor, _ = self._create_editor_tab(filename=filename_str)
        editor.blockSignals(True)
        editor.setText(content)
        editor.blockSignals(False)
        editor.document().setModified(False)
        self._open_files[filename_str] = editor
        self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))

    def open_file(self, path: str, line: int = None, col: int = None):
        """Open a file in a new tab or switch to existing tab, optionally jump to line."""
        path_str = str(path)

        if path_str in self._open_files:
            editor = self._open_files[path_str]
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))
        else:
            try:
                with open(path_str, encoding="utf-8") as file:
                    content = file.read()
            except IOError as e:
                logging.log(1, f"Error opening file {path_str}: {e}")
                return

            editor, _ = self._create_editor_tab(filename=path_str)
            editor.blockSignals(True)
            editor.setText(content)
            editor.blockSignals(False)
            editor.document().setModified(False)
            self._open_files[path_str] = editor
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))

        if line and line > 0:
            self.goto_line(line)

    def goto_line(self, line: int):
        """Move the cursor to a specific line number in the current editor."""
        editor = self._get_current_editor()
        if editor is None:
            return

        if line < 1:
            return

        doc = editor.document()
        block = doc.findBlockByNumber(line - 1)
        if block.isValid():
            cursor = editor.textCursor()
            cursor.setPosition(block.position())
            editor.setTextCursor(cursor)
            editor.ensureCursorVisible()

            self._highlight_line_temporarily(editor, line)

    def _highlight_line_temporarily(self, editor, line: int, duration_ms: int = 2000):
        """Temporarily highlight a line in the given editor."""
        doc = editor.document()
        block = doc.findBlockByNumber(line - 1)
        if not block.isValid():
            return

        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.format.setBackground(QtGui.QColor(255, 255, 0, 100))
        selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
        selection.cursor = editor.textCursor()
        selection.cursor.setPosition(block.position())
        selection.cursor.movePosition(
            QtGui.QTextCursor.EndOfBlock,
            QtGui.QTextCursor.KeepAnchor
        )

        extra_selections = editor.extraSelections() + [selection]
        editor.setExtraSelections(extra_selections)

        QtCore.QTimer.singleShot(
            duration_ms,
            lambda: self._clear_temporary_highlight(editor, selection)
        )

    def _clear_temporary_highlight(self, editor, selection: QtWidgets.QTextEdit.ExtraSelection):
        """Clear a temporary line highlight."""
        extra_selections = editor.extraSelections()
        if selection in extra_selections:
            extra_selections.remove(selection)
            editor.setExtraSelections(extra_selections)

    def run_macro(self, event):
        """Execute the currently loaded Python script."""
        filename = self._get_current_filename()
        if not filename or filename == "Untitled":
            logging.log(1, "No file to run. Save the file first.")
            return
        self.save_text()
        cs.console.run_macro(filename=filename)

    def save_text(self, event=None):
        """Save the current tab's text to a file."""
        editor = self._get_current_editor()
        if editor is None:
            return

        filename = self._get_current_filename()
        if filename == "Untitled" or not filename:
            new_filename = cs.gui.widgets.save_file(file_type="Python script (*.py)")
            if not new_filename:
                return
            filename = new_filename
            idx = self.tab_widget.currentIndex()
            self.tab_widget.setTabText(idx, filename)
            self._open_files[str(filename)] = editor

        try:
            with io.zipped.open_maybe_zipped(filename, "w") as file:
                file.write(editor.text())
        except IOError as e:
            logging.log(1, f"Error saving file {filename}: {e}")
            return
        editor.document().setModified(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    editor = CodeEditor()
    editor.show()
    app.exec_()
