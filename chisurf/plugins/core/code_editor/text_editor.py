from __future__ import annotations

import re
import sys

from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
from chisurf.plugins.core.code_editor.settings import (
    EDITOR_COLOR_SCHEMES,
    EDITOR_SETTINGS_KEYS,
    EditorSettingsDialog,
    default_editor_settings,
    editor_language_key,
    get_editor_settings,
    make_editor_font,
    normalize_editor_language,
    save_editor_settings,
)
from chisurf.plugins.core.code_editor.symbols import (
    CodeSymbol,
    extract_python_symbols,
)

__all__ = [
    "CodeSymbol",
    "EditorSettingsDialog",
    "JSONHighlighter",
    "PythonHighlighter",
    "SyntaxHighlighter",
    "TextEditor",
    "YAMLHighlighter",
    "default_editor_settings",
    "editor_language_key",
    "extract_python_symbols",
    "get_editor_settings",
    "make_editor_font",
    "normalize_editor_language",
    "save_editor_settings",
]


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


class FindBar(QtWidgets.QWidget):
    """Notepad++-style find bar floating at the bottom of the editor viewport."""

    def __init__(self, editor: "TextEditor") -> None:
        super().__init__(editor.viewport())
        self._editor = editor
        self._matches: list[tuple[int, int]] = []
        self._current = -1

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        close_btn = QtWidgets.QToolButton(self)
        close_btn.setText("✕")
        close_btn.setFixedSize(18, 18)
        close_btn.setToolTip("Close (Esc)")
        close_btn.clicked.connect(self.close_bar)

        self.input = QtWidgets.QLineEdit(self)
        self.input.setPlaceholderText("Find…")
        self.input.setMinimumWidth(180)
        self.input.setMaximumWidth(300)
        self.input.textChanged.connect(self._search)
        self.input.installEventFilter(self)

        prev_btn = QtWidgets.QToolButton(self)
        prev_btn.setText("▲")
        prev_btn.setToolTip("Previous match (Shift+F3)")
        prev_btn.clicked.connect(self.find_prev)

        next_btn = QtWidgets.QToolButton(self)
        next_btn.setText("▼")
        next_btn.setToolTip("Next match (F3 / Enter)")
        next_btn.clicked.connect(self.find_next)

        self.case_btn = QtWidgets.QToolButton(self)
        self.case_btn.setText("Aa")
        self.case_btn.setToolTip("Match case")
        self.case_btn.setCheckable(True)
        self.case_btn.toggled.connect(lambda _: self._search(self.input.text()))

        self.word_btn = QtWidgets.QToolButton(self)
        self.word_btn.setText("\\b")
        self.word_btn.setToolTip("Whole word")
        self.word_btn.setCheckable(True)
        self.word_btn.toggled.connect(lambda _: self._search(self.input.text()))

        self.regex_btn = QtWidgets.QToolButton(self)
        self.regex_btn.setText(".*")
        self.regex_btn.setToolTip("Regular expression")
        self.regex_btn.setCheckable(True)
        self.regex_btn.toggled.connect(lambda _: self._search(self.input.text()))

        self.count_label = QtWidgets.QLabel("")
        self.count_label.setMinimumWidth(72)
        self.count_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        layout.addWidget(close_btn)
        layout.addWidget(self.input)
        layout.addWidget(prev_btn)
        layout.addWidget(next_btn)
        layout.addWidget(self.case_btn)
        layout.addWidget(self.word_btn)
        layout.addWidget(self.regex_btn)
        layout.addWidget(self.count_label)
        layout.addStretch()

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, pal.color(QtGui.QPalette.AlternateBase))
        self.setPalette(pal)

        editor.viewport().installEventFilter(self)
        self.hide()

    # ------------------------------------------------------------------
    # Event filter: viewport resize → reposition; input keys → nav/close
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._editor.viewport():
            if event.type() == QtCore.QEvent.Resize and self.isVisible():
                self._reposition()
            return False
        if obj is self.input and event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if key == QtCore.Qt.Key_Escape:
                self.close_bar()
                return True
            if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.find_prev() if mods & QtCore.Qt.ShiftModifier else self.find_next()
                return True
            if key == QtCore.Qt.Key_F3:
                self.find_prev() if mods & QtCore.Qt.ShiftModifier else self.find_next()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_bar(self) -> None:
        self._reposition()
        self.show()
        self.raise_()
        self.input.setFocus()
        self.input.selectAll()
        self._search(self.input.text())

    def close_bar(self) -> None:
        self.hide()
        self._clear_highlights()
        self._editor.setFocus()

    def find_next(self) -> None:
        if not self._matches:
            self._search(self.input.text())
            return
        self._current = (self._current + 1) % len(self._matches)
        self._select_current()

    def find_prev(self) -> None:
        if not self._matches:
            self._search(self.input.text())
            return
        self._current = (self._current - 1) % len(self._matches)
        self._select_current()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reposition(self) -> None:
        vp = self._editor.viewport()
        h = self.sizeHint().height()
        self.setGeometry(0, vp.height() - h, vp.width(), h)

    def _build_pattern(self, text: str):
        if not text:
            return None
        pat = text if self.regex_btn.isChecked() else re.escape(text)
        if self.word_btn.isChecked():
            pat = r"\b" + pat + r"\b"
        flags = 0 if self.case_btn.isChecked() else re.IGNORECASE
        try:
            return re.compile(pat, flags)
        except re.error:
            return None

    def _search(self, text: str) -> None:
        self._clear_highlights()
        self._matches = []
        self._current = -1

        pattern = self._build_pattern(text)
        if pattern is None:
            self.count_label.setText("")
            self._set_input_state(None)
            return

        content = self._editor.toPlainText()
        self._matches = [(m.start(), m.end()) for m in pattern.finditer(content)]

        if not self._matches:
            self.count_label.setText("No results")
            self._set_input_state("none")
            return

        self._set_input_state("found")
        cursor_pos = self._editor.textCursor().position()
        self._current = next(
            (i for i, (s, _) in enumerate(self._matches) if s >= cursor_pos), 0
        )
        self._select_current()

    def _build_selections(self, current: int) -> list:
        all_fmt = QtGui.QTextCharFormat()
        all_fmt.setBackground(QtGui.QColor("#ffe066"))
        all_fmt.setForeground(QtGui.QColor("#000000"))

        cur_fmt = QtGui.QTextCharFormat()
        cur_fmt.setBackground(QtGui.QColor("#ff8c00"))
        cur_fmt.setForeground(QtGui.QColor("#ffffff"))

        doc = self._editor.document()
        selections = []
        for i, (s, e) in enumerate(self._matches):
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.format = cur_fmt if i == current else all_fmt
            c = QtGui.QTextCursor(doc)
            c.setPosition(s)
            c.setPosition(e, QtGui.QTextCursor.KeepAnchor)
            sel.cursor = c
            selections.append(sel)
        return selections

    def _select_current(self) -> None:
        if not self._matches or self._current < 0:
            return
        self._editor._find_extra_selections = self._build_selections(self._current)
        self._editor._update_extra_selections()

        start, end = self._matches[self._current]
        doc = self._editor.document()
        cur = QtGui.QTextCursor(doc)
        cur.setPosition(start)
        cur.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        self._editor.setTextCursor(cur)
        self._editor.ensureCursorVisible()
        self.count_label.setText(f"{self._current + 1} / {len(self._matches)}")

    def _clear_highlights(self) -> None:
        self._editor._find_extra_selections = []
        self._editor._update_extra_selections()

    def _set_input_state(self, state) -> None:
        if state == "none":
            self.input.setStyleSheet("background-color: #ffcccc;")
        elif state == "found":
            self.input.setStyleSheet("background-color: #ccffcc;")
        else:
            self.input.setStyleSheet("")


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
    statusChanged = QtCore.Signal(dict)
    filePathChanged = QtCore.Signal(str)
    symbolsChanged = QtCore.Signal(list)
    definitionRequested = QtCore.Signal(str, int, int)

    def __init__(
            self,
            parent=None,
            font_family: str = None,
            font_point_size: float = None,
            margins_background_color: str = None,
            marker_background_color: str = None,
            caret_line_background_color: str = None,
            caret_line_visible: bool = None,
            line_numbers_visible: bool = None,
            language: str = None,
            **kwargs
    ):
        """
        Initialize the text editor.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        font_family : str, optional
            Font family to use.
        font_point_size : float, optional
            Font size in points.
        margins_background_color : str, optional
            Background color for line-number margins.
        marker_background_color : str, optional
            Background color for selection markers.
        caret_line_background_color : str, optional
            Background color for the current line highlight.
        caret_line_visible : bool, optional
            Whether to highlight the current line.
        line_numbers_visible : bool, optional
            Whether to show the line-number margin.
        language : str, optional
            Language for syntax highlighting: Python, JSON, YAML, or Plain text.
        kwargs : dict
            Additional color options: ``paper_color`` and ``default_color``.
        """
        super().__init__(parent)

        settings = get_editor_settings()
        if font_family is not None:
            settings["font_family"] = font_family
        if font_point_size is not None:
            settings["font_size"] = font_point_size
        if margins_background_color is not None:
            settings["margins_background_color"] = margins_background_color
        if marker_background_color is not None:
            settings["marker_background_color"] = marker_background_color
        if caret_line_background_color is not None:
            settings["caret_line_background_color"] = caret_line_background_color
        if caret_line_visible is not None:
            settings["caret_line_visible"] = caret_line_visible
        if line_numbers_visible is not None:
            settings["line_numbers_visible"] = line_numbers_visible
        if language is not None:
            settings["language"] = language
        if "paper_color" in kwargs:
            settings["paper_color"] = kwargs["paper_color"]
        if "default_color" in kwargs:
            settings["default_color"] = kwargs["default_color"]

        self._editor_settings = settings
        self.language = normalize_editor_language(settings.get("language"))
        self.paper_color = settings["paper_color"]
        self.default_color = settings["default_color"]
        self.margins_background_color = settings["margins_background_color"]
        self.marker_background_color = settings["marker_background_color"]
        self.current_line_color = QtGui.QColor(settings["caret_line_background_color"])
        self.caret_line_visible = bool(settings["caret_line_visible"])
        self.line_numbers_visible = bool(settings.get("line_numbers_visible", True))
        self.show_whitespace = bool(settings.get("show_whitespace", False))
        self.highlighter = None
        self._find_extra_selections: list = []
        self._caret_selection = None

        self.setFont(make_editor_font(settings))

        self.line_number_area = LineNumberArea(self)
        self.line_number_area.setVisible(self.line_numbers_visible)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)

        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        self._nav_history = []
        self._nav_index = -1
        self.current_file = None
        self.external_definition_callback = None
        self._definition_uses_host = False
        self._symbols: list[CodeSymbol] = []
        self._symbol_timer = QtCore.QTimer(self)
        self._symbol_timer.setSingleShot(True)
        self._symbol_timer.setInterval(250)
        self._symbol_timer.timeout.connect(self.refresh_symbols)
        self.cursorPositionChanged.connect(self._emit_status_changed)
        self.cursorPositionChanged.connect(self._update_caret_selection)
        self.textChanged.connect(self._schedule_symbol_refresh)
        self.document().modificationChanged.connect(lambda _modified: self._emit_status_changed())

        self._rebuild_highlighter()
        self._apply_palette()
        self.set_show_whitespace(self.show_whitespace)
        self.refresh_symbols()

        self._find_bar = FindBar(self)
        for key, slot in [
            ("Ctrl+F", self._find_bar.open_bar),
            ("F3",      self._find_bar.find_next),
            ("Shift+F3", self._find_bar.find_prev),
        ]:
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        self.setMinimumSize(400, 200)

    def set_current_file(self, path: str | None) -> None:
        """Set the file path represented by this editor."""
        path_str = str(path) if path else ""
        if self.current_file == path_str:
            return
        self.current_file = path_str
        self.filePathChanged.emit(path_str)
        self.refresh_symbols()
        self._emit_status_changed()

    def line_column(self) -> tuple[int, int]:
        """Return the current one-based line and zero-based column."""
        cursor = self.textCursor()
        return cursor.blockNumber() + 1, cursor.positionInBlock()

    def current_word(self) -> str:
        """Return the word under the cursor."""
        cursor = self.textCursor()
        cursor.select(QtGui.QTextCursor.WordUnderCursor)
        return cursor.selectedText()

    def symbols(self) -> list[CodeSymbol]:
        """Return the current document symbols."""
        return list(self._symbols)

    def refresh_symbols(self) -> list[CodeSymbol]:
        """Refresh and emit the symbol list for the current document."""
        if editor_language_key(self.language) == "python":
            self._symbols = extract_python_symbols(self.toPlainText(), self.current_file or "")
        else:
            self._symbols = []
        self.symbolsChanged.emit(self.symbols())
        return self.symbols()

    def goto_symbol(self, symbol: CodeSymbol | dict) -> None:
        """Move the cursor to *symbol* and record navigation history."""
        if isinstance(symbol, dict):
            line = int(symbol.get("line", 1))
            column = int(symbol.get("column", 0))
        else:
            line = symbol.line
            column = symbol.column
        self.goto_line_column(line, column)

    def goto_line_column(self, line: int, column: int = 0, record: bool = True) -> None:
        """Move the cursor to a one-based line and zero-based column."""
        if line < 1:
            return
        if record:
            self.push_nav_history()
        doc = self.document()
        block = doc.findBlockByNumber(line - 1)
        if not block.isValid():
            return
        cursor = self.textCursor()
        cursor.setPosition(block.position() + max(0, min(column, block.length() - 1)))
        self.setTextCursor(cursor)
        self.centerCursor()
        if record:
            self.push_nav_history()
        self._emit_status_changed()

    def _schedule_symbol_refresh(self) -> None:
        """Schedule a debounced symbol refresh."""
        self._symbol_timer.start()
        self._emit_status_changed()

    def _emit_status_changed(self) -> None:
        """Emit reusable editor status for host widgets."""
        line, column = self.line_column()
        self.statusChanged.emit(
            {
                "file": self.current_file or "",
                "line": line,
                "column": column,
                "modified": self.document().isModified(),
                "language": self.language,
            }
        )

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
        bg_color = QtGui.QColor('green')
        text_color = QtGui.QColor('white')
        painter.fillRect(event.rect(), bg_color)

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(text_color)
                painter.setFont(self.font())
                rect = QtCore.QRect(
                    0,
                    int(top),
                    self.line_number_area.width() - 4,
                    self.fontMetrics().height(),
                )
                painter.drawText(rect, QtCore.Qt.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    def mousePressEvent(self, event):
        """Handle Ctrl/Cmd-click definition navigation."""
        super().mousePressEvent(event)
        if (
            event.modifiers() & QtCore.Qt.ControlModifier
            or event.modifiers() & QtCore.Qt.MetaModifier
        ):
            cursor = self.cursorForPosition(event.pos())
            cursor.select(QtGui.QTextCursor.WordUnderCursor)
            word = cursor.selectedText()
            if word:
                line, column = self.line_column()
                self.definitionRequested.emit(word, line, column)
                if not self._definition_uses_host:
                    self.jump_to_definition(word)

    def navigate_back(self):
        """Navigate backward in this editor's cursor history."""
        if self._nav_index > 0:
            self._nav_index -= 1
            file_path, line_number = self._nav_history[self._nav_index]
            self.goto_file_line(file_path, line_number)

    def navigate_forward(self):
        """Navigate forward in this editor's cursor history."""
        if self._nav_index < len(self._nav_history) - 1:
            self._nav_index += 1
            file_path, line_number = self._nav_history[self._nav_index]
            self.goto_file_line(file_path, line_number)

    def goto_file_line(self, file_path, line_number):
        """Open or focus *file_path* and move to a zero-based line number."""
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
            self._emit_status_changed()

    def push_nav_history(self, file_path=None, line_number=None):
        """Record the current file and zero-based line in navigation history."""
        if file_path is None:
            file_path = getattr(self, "current_file", "")
        if line_number is None:
            line_number = self.textCursor().blockNumber()

        if self._nav_index < len(self._nav_history) - 1:
            self._nav_history = self._nav_history[:self._nav_index + 1]

        if self._nav_history and self._nav_history[-1] == (file_path, line_number):
            return

        self._nav_history.append((file_path, line_number))
        self._nav_index = len(self._nav_history) - 1

    def jump_to_definition(self, word):
        """Jump to the local or externally resolved definition for *word*."""
        for symbol in self.refresh_symbols():
            if symbol.name == word:
                self.push_nav_history()
                self.goto_line_column(symbol.line, symbol.column, record=False)
                self.push_nav_history()
                return

        if self.external_definition_callback is not None:
            self._jump_to_external_definition(word)

    def _jump_to_external_definition(self, word):
        import importlib
        import inspect
        import re

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
        """Paint the editor."""
        super().paintEvent(event)

    def keyPressEvent(self, event):
        """Close the find bar on Escape."""
        if event.key() == QtCore.Qt.Key_Escape and self._find_bar.isVisible():
            self._find_bar.close_bar()
            return
        super().keyPressEvent(event)

    def _update_caret_selection(self) -> None:
        """Rebuild the current-line extra selection and refresh."""
        if self.caret_line_visible:
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.format.setBackground(self.current_line_color)
            sel.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            self._caret_selection = sel
        else:
            self._caret_selection = None
        self._update_extra_selections()

    def _update_extra_selections(self) -> None:
        """Merge find highlights with the caret-line selection."""
        selections = list(self._find_extra_selections)
        if self._caret_selection is not None:
            selections.append(self._caret_selection)
        self.setExtraSelections(selections)

    def text(self):
        """Get the text content of the editor."""
        return self.toPlainText()

    def setText(self, text):
        """Set the text content of the editor."""
        self.setPlainText(text)

    def set_editor_settings(self, settings: dict) -> None:
        """Apply editor settings to this widget."""
        if not isinstance(settings, dict):
            return

        merged = dict(self._editor_settings)
        scheme = settings.get("color_scheme")
        if scheme in EDITOR_COLOR_SCHEMES:
            merged.update(EDITOR_COLOR_SCHEMES[scheme])
        merged.update({
            key: value
            for key, value in settings.items()
            if key in EDITOR_SETTINGS_KEYS
        })

        self._editor_settings = merged
        self.language = normalize_editor_language(merged.get("language"))
        self.paper_color = merged["paper_color"]
        self.default_color = merged["default_color"]
        self.margins_background_color = merged["margins_background_color"]
        self.marker_background_color = merged["marker_background_color"]
        self.current_line_color = QtGui.QColor(merged["caret_line_background_color"])
        self.caret_line_visible = bool(merged["caret_line_visible"])
        self.line_numbers_visible = bool(merged.get("line_numbers_visible", True))

        self.setFont(make_editor_font(merged))
        self._rebuild_highlighter()
        self._apply_palette()
        self.set_line_numbers_visible(self.line_numbers_visible)
        self.set_show_whitespace(bool(merged.get("show_whitespace", False)))
        self._update_caret_selection()
        self.viewport().update()

    def get_editor_settings(self) -> dict:
        """Return the current editor settings."""
        settings = dict(self._editor_settings)
        settings["font_family"] = self.font().family() or settings.get("font_family", "Courier New")
        settings["font_size"] = self.font().pointSize()
        settings["language"] = self.language
        settings["line_numbers_visible"] = self.line_numbers_visible
        settings["show_whitespace"] = self.show_whitespace
        return settings

    def set_font_family(self, font_family: str) -> None:
        """Set the editor font family."""
        self.set_editor_settings({"font_family": font_family})

    def set_font_size(self, font_size: int) -> None:
        """Set the editor font size in points."""
        self.set_editor_settings({"font_size": font_size})

    def set_language(self, language: str) -> None:
        """Set the syntax highlighting language."""
        self.set_editor_settings({"language": language})

    def set_color_scheme(self, color_scheme: str) -> None:
        """Set the editor color scheme."""
        self.set_editor_settings({"color_scheme": color_scheme})

    def set_caret_line_visible(self, visible: bool) -> None:
        """Enable or disable current-line highlighting."""
        self.set_editor_settings({"caret_line_visible": visible})

    def set_line_numbers_visible(self, visible: bool) -> None:
        """Enable or disable the line-number margin."""
        self.line_numbers_visible = bool(visible)
        self.line_number_area.setVisible(self.line_numbers_visible)
        self.update_line_number_area_width(0)

    def set_show_whitespace(self, visible: bool) -> None:
        """Show or hide whitespace characters (spaces, tabs, line endings)."""
        self.show_whitespace = bool(visible)
        opt = self.document().defaultTextOption()
        flags = opt.flags()
        ws_flags = (
            QtGui.QTextOption.ShowTabsAndSpaces
            | QtGui.QTextOption.ShowLineAndParagraphSeparators
        )
        if self.show_whitespace:
            flags |= ws_flags
        else:
            flags &= ~ws_flags
        opt.setFlags(flags)
        self.document().setDefaultTextOption(opt)
        self.viewport().update()

    def _rebuild_highlighter(self) -> None:
        """Recreate the syntax highlighter for the current language."""
        if hasattr(self, "highlighter") and self.highlighter is not None:
            self.highlighter.deleteLater()

        key = editor_language_key(self.language)
        if key == "plain":
            self.highlighter = None
            return

        highlighter_classes = {
            "python": PythonHighlighter,
            "json": JSONHighlighter,
            "yaml": YAMLHighlighter,
            "yml": YAMLHighlighter,
        }
        highlighter_class = highlighter_classes.get(key, YAMLHighlighter)
        self.highlighter = highlighter_class(
            self.document(),
            self.font().family(),
            int(self._editor_settings.get("font_size", 9)),
            self.paper_color,
            self.default_color,
        )

    def _apply_palette(self) -> None:
        """Apply editor colors to the widget palette."""
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(self.paper_color))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(self.default_color))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(self.marker_background_color))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(self.default_color))
        self.setPalette(palette)



def __getattr__(name: str):
    """Lazily expose compatibility imports from split editor modules."""
    if name == "CodeEditor":
        from chisurf.plugins.core.code_editor.editor import CodeEditor
        return CodeEditor
    if name == "CodeEditorWindow":
        from chisurf.plugins.core.code_editor.window import CodeEditorWindow
        return CodeEditorWindow
    raise AttributeError(name)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    from chisurf.plugins.core.code_editor.window import CodeEditorWindow
    editor = CodeEditorWindow()
    editor.show()
    app.exec_()
