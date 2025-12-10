from __future__ import annotations

"""Lightweight Python syntax highlighter used by the node editor.

This is intentionally self-contained so that the ``node_editor`` package can be
run as ``python -m node_editor`` without importing the full chisurf GUI stack.
"""

from qtpy import QtCore, QtGui, QtWidgets

try:  # Optional dependency: use Pygments when available
    from pygments.lexers import PythonLexer  # type: ignore
    from pygments.token import Token  # type: ignore

    _HAS_PYGMENTS = True
except Exception:  # pragma: no cover - fallback when pygments is missing
    PythonLexer = None  # type: ignore
    Token = None  # type: ignore
    _HAS_PYGMENTS = False


class SyntaxHighlighter(QtGui.QSyntaxHighlighter):
    """Minimal base class for regex-based syntax highlighters."""

    def __init__(self, parent=None, font_family: str | None = None, font_point_size: float | None = None):
        super().__init__(parent)
        self._rules: list[tuple[QtCore.QRegExp, QtGui.QTextCharFormat]] = []

        # Optional default font configuration (not strictly required for the
        # PT Transform editor, which sets its own font).
        self.default_font = QtGui.QFont()
        if font_family:
            self.default_font.setFamily(font_family)
        if font_point_size is not None:
            self.default_font.setPointSize(int(font_point_size))

    def add_rule(self, pattern: str, fmt: QtGui.QTextCharFormat) -> None:
        """Register a simple regex rule with an associated text format."""

        self._rules.append((QtCore.QRegExp(pattern), fmt))

    # Qt override
    def highlightBlock(self, text: str) -> None:  # type: ignore[override]
        for regex, fmt in self._rules:
            expression = QtCore.QRegExp(regex)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, fmt)
                index = expression.indexIn(text, index + length)


class LineNumberArea(QtWidgets.QWidget):
    """Side widget that paints line numbers for CodeEditor.

    This is a lightweight variant of chisurf's line-number implementation and
    is intentionally self-contained.
    """

    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        self._editor.line_number_area_paint_event(event)


class CodeEditor(QtWidgets.QPlainTextEdit):
    """Simple QPlainTextEdit with a line-number margin on the left."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        self._line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        self.update_line_number_area_width(0)

        self._current_line_color = QtGui.QColor(45, 45, 55, 120)

    # ----- Line number support -------------------------------------------
    def line_number_area_width(self) -> int:
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1
        space = 6 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def update_line_number_area_width(self, _new_block_count: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect: QtCore.QRect, dy: int) -> None:
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QtGui.QColor(30, 30, 35))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtGui.QColor(150, 150, 160))
                painter.drawText(
                    0,
                    top,
                    self._line_number_area.width() - 4,
                    self.fontMetrics().height(),
                    QtCore.Qt.AlignRight,
                    number,
                )

            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def highlight_current_line(self) -> None:
        extra_selections: list[QtWidgets.QTextEdit.ExtraSelection] = []

        if not self.isReadOnly():
            selection = QtWidgets.QTextEdit.ExtraSelection()
            selection.format.setBackground(self._current_line_color)
            selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)

        self.setExtraSelections(extra_selections)
class PythonHighlighter(SyntaxHighlighter):
    """Python syntax highlighter.

    If Pygments is installed, use its ``PythonLexer`` to tokenize the code and
    map tokens to QTextCharFormats for high-quality highlighting. Otherwise,
    fall back to a small regex-based highlighter similar to chisurf's
    lightweight editor.
    """

    def __init__(self, parent=None, font_family: str | None = None, font_point_size: float | None = None):
        super().__init__(parent, font_family, font_point_size)

        self._use_pygments = bool(_HAS_PYGMENTS and PythonLexer is not None)

        if self._use_pygments:
            # Pygments-based lexer
            self._lexer = PythonLexer(stripall=False)  # type: ignore[call-arg]

            # Map a few broad token families to colors inspired by dark themes
            self._token_formats: dict[object, QtGui.QTextCharFormat] = {}

            def _fmt(color: str, bold: bool = False, italic: bool = False) -> QtGui.QTextCharFormat:
                f = QtGui.QTextCharFormat()
                f.setForeground(QtGui.QColor(color))
                if bold:
                    f.setFontWeight(QtGui.QFont.Bold)
                if italic:
                    f.setFontItalic(True)
                return f

            if Token is not None:  # type: ignore[truthy-function]
                # Core mappings (keywords, names, strings, comments, numbers)
                self._token_formats[Token.Keyword] = _fmt("#569CD6", bold=True)
                self._token_formats[Token.Name.Class] = _fmt("#4EC9B0", bold=True)
                self._token_formats[Token.Name.Function] = _fmt("#DCDCAA")
                self._token_formats[Token.String] = _fmt("#CE9178")
                self._token_formats[Token.Comment] = _fmt("#6A9955", italic=True)
                self._token_formats[Token.Number] = _fmt("#B5CEA8")
        else:
            # Fallback: regex-based highlighting (simple but dependency-free)
            # Colors loosely inspired by common dark themes
            keyword_fmt = QtGui.QTextCharFormat()
            keyword_fmt.setForeground(QtGui.QColor("#569CD6"))
            keyword_fmt.setFontWeight(QtGui.QFont.Bold)

            class_fmt = QtGui.QTextCharFormat()
            class_fmt.setForeground(QtGui.QColor("#4EC9B0"))
            class_fmt.setFontWeight(QtGui.QFont.Bold)

            func_fmt = QtGui.QTextCharFormat()
            func_fmt.setForeground(QtGui.QColor("#DCDCAA"))

            string_fmt = QtGui.QTextCharFormat()
            string_fmt.setForeground(QtGui.QColor("#CE9178"))

            comment_fmt = QtGui.QTextCharFormat()
            comment_fmt.setForeground(QtGui.QColor("#6A9955"))

            number_fmt = QtGui.QTextCharFormat()
            number_fmt.setForeground(QtGui.QColor("#B5CEA8"))

            # Keywords
            keywords = [
                "and", "as", "assert", "break", "class", "continue", "def",
                "del", "elif", "else", "except", "finally", "for",
                "from", "global", "if", "import", "in", "is", "lambda",
                "not", "or", "pass", "raise", "return", "try",
                "while", "with", "yield", "None", "True", "False",
            ]
            for kw in keywords:
                # \b word-boundary on both sides of the keyword
                self.add_rule(r"\b" + kw + r"\b", keyword_fmt)

            # Classes and functions
            self.add_rule(r"\bclass\b\s+(\w+)", class_fmt)
            self.add_rule(r"\bdef\b\s+(\w+)", func_fmt)

            # Simple string patterns (single- and double-quoted)
            self.add_rule(r'"[^"\\]*(\\.[^"\\]*)*"', string_fmt)
            self.add_rule(r"'[^'\\]*(\\.[^'\\]*)*'", string_fmt)

            # Comments
            self.add_rule(r"#[^\n]*", comment_fmt)

            # Integers
            self.add_rule(r"\b[0-9]+\b", number_fmt)

    def _format_for_token(self, token) -> QtGui.QTextCharFormat | None:
        """Return the best-matching format for a Pygments token by walking up
        the token hierarchy until we find a registered style.
        """

        if not self._use_pygments:
            return None
        if Token is None:
            return None
        # Walk up the token hierarchy (e.g. Name.Function -> Name -> Token)
        t = token
        while t not in self._token_formats and getattr(t, "parent", None) is not None:
            t = t.parent  # type: ignore[assignment]
        return self._token_formats.get(t)

    # Qt override
    def highlightBlock(self, text: str) -> None:  # type: ignore[override]
        if self._use_pygments and self._lexer is not None:
            # Use Pygments to tokenize this single line of code
            try:
                for index, token, value in self._lexer.get_tokens_unprocessed(text):  # type: ignore[attr-defined]
                    fmt = self._format_for_token(token)
                    if fmt is not None:
                        self.setFormat(index, len(value), fmt)
            except Exception:
                # Fallback to no highlighting on error
                pass
        else:
            # Use the simple regex-based rules from the base class
            super().highlightBlock(text)


__all__ = ["SyntaxHighlighter", "PythonHighlighter", "CodeEditor"]
