#-------------------------------------------------------------------------
# qsci_simple_pythoneditor.pyw
#
# QScintilla sample with PyQt
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------

import sys

from PyQt5.Qsci import QsciScintilla, QsciLexerPython
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import mfm

#editor_settings = mfm.settings.cs_settings['gui']['editor']


class SimplePythonEditor(QsciScintilla):
    ARROW_MARKER_NUM = 8

    def __init__(self, parent=None, **kwargs):
        super(SimplePythonEditor, self).__init__(parent)

        # Set the default font
        font = QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(
            mfm.settings.cs_settings['gui']['editor']['font_size']
        )
        self.setFont(font)
        self.setMarginsFont(font)

        # Margin 0 is used for line numbers
        fontmetrics = QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, fontmetrics.width("0000") + 6)
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(
            QColor(
                mfm.settings.cs_settings['gui']['editor']['MarginsBackgroundColor']
            )
        )

        # Clickable margin 1 for showing markers
        self.setMarginSensitivity(1, True)
#        self.connect(self,
#            SIGNAL('marginClicked(int, int, Qt::KeyboardModifiers)'),
#            self.on_margin_clicked)
        self.markerDefine(QsciScintilla.RightArrow,
            self.ARROW_MARKER_NUM)
        self.setMarkerBackgroundColor(QColor(mfm.settings.cs_settings['gui']['editor']['MarkerBackgroundColor']),
            self.ARROW_MARKER_NUM)

        # Brace matching: enable for a brace immediately before or after
        # the current position
        #
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Current line visible with special background color
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QColor(mfm.settings.cs_settings['gui']['editor']['CaretLineBackgroundColor']))

        # Set Python lexer
        # Set style for Python comments (style number 1) to a fixed-width
        # courier.
        #
        lexer = QsciLexerPython()
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

    def load_file(self, **kwargs):
        filename = kwargs.get('filename', None)
        if filename is None or filename is False:
            filename = mfm.widgets.get_filename()
        self.filename = filename
        try:
            print('loading filename: ', filename)
            text = open(filename).read()
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
        with open(self.filename, 'w') as fp:
            text = str(self.editor.text())
            fp.write(text)
            self.line_edit.setText(self.filename)

    def __init__(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        language = kwargs.pop('language', 'Python')
        can_load = kwargs.pop('can_load', True)
        QWidget.__init__(self, *args, **kwargs)
        layout = QVBoxLayout()

        self.filename = None
        self.setLayout(layout)
        self.line_edit = QLineEdit()
        self.editor = SimplePythonEditor(parent=self, language=language)

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
    editor = CodeEditor()
    editor.show()
    app.exec_()
