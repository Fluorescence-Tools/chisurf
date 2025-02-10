from __future__ import annotations

import pathlib
import sys

from PyQt5.Qsci import QsciScintilla
try:
    # For <=2.10.0 QScintilla versions not all Lexer are available
    from PyQt5.Qsci import QsciLexerPython, QsciLexerJSON, QsciLexerYAML
    lexers_available = True
except ImportError:
    QsciLexerJSON = None
    QsciLexerYAML = None
    QsciLexerPython = None
    lexers_available = False

from qtpy import QtGui
from qtpy import QtWidgets

import chisurf
import chisurf.fio as io
from chisurf import logging
import chisurf.gui.widgets
import chisurf.settings


class SimpleCodeEditor(QsciScintilla):

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

        :param parent:
        :param font_family:
        :param font_point_size:
        :param margins_background_color:
        :param marker_background_color:
        :param caret_line_background_color:
        :param caret_line_visible:
        :param language: a string that is set to select the lexer of the
        editor (either Python or JSON) the default lexer is a YAML lexer
        :param kwargs:
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
        self.setMarginsFont(font)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)

        # Margin 0 is used for line numbers
        fontmetrics = QtGui.QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, fontmetrics.width("000") + 4)
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(QtGui.QColor(margins_background_color))

        # Clickable margin 1 for showing markers
        self.setMarginSensitivity(1, True)
#        self.connect(self,
#            SIGNAL('marginClicked(int, int, Qt::KeyboardModifiers)'),
#            self.on_margin_clicked)
        self.markerDefine(QsciScintilla.RightArrow, self.ARROW_MARKER_NUM)

        self.setMarkerBackgroundColor(QtGui.QColor(marker_background_color), self.ARROW_MARKER_NUM)

        # Brace matching: enable for a brace immediately before or after
        # the current position
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Current line visible with special background color
        self.setCaretLineVisible(caret_line_visible)
        self.setCaretLineBackgroundColor(QtGui.QColor(caret_line_background_color))

        # Set Python lexer
        # Set style for Python comments (style number 1) to a fixed-width
        # courier.
        if lexers_available and isinstance(language, str):
            if language.lower() == "python":
                lexer = QsciLexerPython()
            elif language.lower() == "json":
                lexer = QsciLexerJSON()
            else:
                lexer = QsciLexerYAML()
            lexer.setDefaultPaper(QtGui.QColor(paper_color))
            lexer.setDefaultColor(QtGui.QColor(default_color))
            lexer.setDefaultFont(font)
            self.setLexer(lexer)
        #
        # text = bytearray(str.encode("Arial"))
        # # 32, "Courier New"
        # self.SendScintilla(QsciScintilla.SCI_STYLESETFONT, 1, text)

        # Don't want to see the horizontal scrollbar at all
        # Use raw message to Scintilla here (all messages are documented
        # here: http://www.scintilla.org/ScintillaDoc.html)
        self.SendScintilla(QsciScintilla.SCI_SETHSCROLLBAR, 0)

        # Set the color of the fold boxes
        fold_margin_color = QtGui.QColor("#006400")  # Replace with your desired color
        self.setFoldMarginColors(fold_margin_color, fold_margin_color)

        # not too small
        self.setMinimumSize(400, 200)

    def on_margin_clicked(self, nmargin, nline, modifiers):
        # Toggle marker for the line the margin was clicked on
        if self.markersAtLine(nline) != 0:
            self.markerDelete(nline, self.ARROW_MARKER_NUM)
        else:
            self.markerAdd(nline, self.ARROW_MARKER_NUM)


class CodeEditor(QtWidgets.QWidget):

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
        filename: str = None,
        language: str = "Python",
        can_load: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.filename = None
        self.setLayout(layout)
        self.line_edit = QtWidgets.QLineEdit()
        self.editor = SimpleCodeEditor(
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
        if isinstance(filename, str) and pathlib.Path(filename).is_file():
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
