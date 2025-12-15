"""
Code editor module providing syntax highlighting and editing capabilities.

This module provides a text editor with syntax highlighting for Python, JSON, and YAML.
It uses QSyntaxHighlighter and QPlainTextEdit instead of QScintilla2.
"""
from chisurf.plugins.misc.code_editor.text_editor import (
    SyntaxHighlighter,
    PythonHighlighter,
    JSONHighlighter,
    YAMLHighlighter,
    TextEditor,
    CodeEditor,
)

# For backward compatibility
SimpleCodeEditor = TextEditor

name = "Miscellaneous:Code Editor"

if __name__ == "plugin":
    window = CodeEditor()
    window.show()
