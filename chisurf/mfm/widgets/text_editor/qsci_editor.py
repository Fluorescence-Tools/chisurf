"""

"""
from __future__ import annotations

import sys

from PyQt5.Qsci import QsciScintilla
from PyQt5.Qsci import QsciLexerPython, QsciLexerJSON, QsciLexerYAML
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import qdarkstyle

import mfm


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
            caret_line_visible: bool = True,
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
            font_point_size = mfm.settings.gui['editor']['font_size']
        if font_family is None:
            font_family = mfm.settings.gui['editor']['font_family']
        if margins_background_color is None:
            margins_background_color = mfm.settings.gui[
                'editor'
            ]['margins_background_color']
        if marker_background_color is None:
            marker_background_color = mfm.settings.gui[
                'editor'
            ]['marker_background_color']
        if caret_line_background_color is None:
            caret_line_background_color = mfm.settings.gui[
                'editor'
            ]['caret_line_background_color']

        # Set the default font
        font = QFont()
        font.setFamily(font_family)
        font.setFixedPitch(True)
        font.setPointSize(font_point_size)

        self.setFont(font)
        self.setMarginsFont(font)

        # Margin 0 is used for line numbers
        fontmetrics = QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, fontmetrics.width("0000") + 6)
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(
            QColor(margins_background_color)
        )

        # Clickable margin 1 for showing markers
        self.setMarginSensitivity(1, True)
#        self.connect(self,
#            SIGNAL('marginClicked(int, int, Qt::KeyboardModifiers)'),
#            self.on_margin_clicked)
        self.markerDefine(QsciScintilla.RightArrow, self.ARROW_MARKER_NUM)

        self.setMarkerBackgroundColor(
            QColor(marker_background_color),
            self.ARROW_MARKER_NUM
        )

        # Brace matching: enable for a brace immediately before or after
        # the current position
        #
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Current line visible with special background color
        self.setCaretLineVisible(caret_line_visible)
        self.setCaretLineBackgroundColor(
            QColor(caret_line_background_color)
        )

        # Set Python lexer
        # Set style for Python comments (style number 1) to a fixed-width
        # courier.
        if language.lower() == "python":
            lexer = QsciLexerPython()
        elif language.lower() == "json":
            lexer = QsciLexerJSON()
        else:
            lexer = QsciLexerYAML()
        lexer.setDefaultFont(font)
        self.setLexer(lexer)

        text = bytearray(str.encode("Arial"))
        # 32, "Courier New"
        self.SendScintilla(QsciScintilla.SCI_STYLESETFONT, 1, text)

        # Don't want to see the horizontal scrollbar at all
        # Use raw message to Scintilla here (all messages are documented
        # here: http://www.scintilla.org/ScintillaDoc.html)
        self.SendScintilla(QsciScintilla.SCI_SETHSCROLLBAR, 0)

        # not too small
        # self.setMinimumSize(400, 200)

    def on_margin_clicked(self, nmargin, nline, modifiers):
        # Toggle marker for the line the margin was clicked on
        if self.markersAtLine(nline) != 0:
            self.markerDelete(nline, self.ARROW_MARKER_NUM)
        else:
            self.markerAdd(nline, self.ARROW_MARKER_NUM)


class CodeEditor(QWidget):

    def load_file(
            self,
            filename: str = None,
            **kwargs
    ):
        if filename is None:
            filename = mfm.widgets.get_filename()
        self.filename = filename
        try:
            print('loading filename: ', filename)
            text = ""
            with open(filename) as fp:
                text = fp.read()
            self.editor.setText(text)
            self.line_edit.setText(filename)
        except IOError:
            print("Not a valid filename.")

    def run_macro(self):
        self.save_text()
        print("running macros %s" % self.filename)
        mfm.console.run_macro(filename=self.filename)

    def save_text(self):
        print("saving macros")
        if self.filename is None or self.filename == '':
            self.filename = mfm.widgets.save_file(file_type='Python script (*.py)')
        with mfm.io.zipped.open_maybe_zipped(
                filename=self.filename,
                mode='w'
        ) as fp:
            text = str(self.editor.text())
            fp.write(text)
            self.line_edit.setText(self.filename)

    def __init__(
            self,
            *args,
            filename: str = None,
            language: str = 'Python',
            can_load: bool = True,
            **kwargs
    ):
        super(CodeEditor, self).__init__(*args, **kwargs)

        layout = QVBoxLayout()
        self.filename = None
        self.setLayout(layout)
        self.line_edit = QLineEdit()
        self.editor = SimpleCodeEditor(
            parent=self,
            language=language
        )
        layout.addWidget(self.editor)
        h_layout = QHBoxLayout()

        load_button = QPushButton('load')
        save_button = QPushButton('save')
        run_button = QPushButton('run')

        h_layout.addWidget(self.line_edit)
        h_layout.addWidget(load_button)
        h_layout.addWidget(save_button)
        h_layout.addWidget(run_button)
        layout.addLayout(h_layout)

        save_button.clicked.connect(self.save_text)
        load_button.clicked.connect(self.load_file)
        run_button.clicked.connect(self.run_macro)

        # Load the file
        if filename is not None and filename is not '':
            self.load_file(filename=filename)
        if language != 'Python':
            run_button.hide()
        if not can_load:
            load_button.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    editor = CodeEditor()
    editor.show()
    app.exec_()
