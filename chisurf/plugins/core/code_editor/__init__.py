"""
Code editor module providing syntax highlighting and editing capabilities.

This module provides a text editor with syntax highlighting for Python, JSON, and YAML.
It uses QSyntaxHighlighter and QPlainTextEdit instead of QScintilla2.
"""
from chisurf.plugins.core.code_editor.text_editor import (
    SyntaxHighlighter,
    PythonHighlighter,
    JSONHighlighter,
    YAMLHighlighter,
    TextEditor,
    CodeEditor,
)

# For backward compatibility
SimpleCodeEditor = TextEditor

name = "Tools:Miscellaneous:Code Editor"
icon = "📝"  # Memo/notepad emoji for editor

def load():
    """Return the plugin's main widget instance."""
    from .text_editor import CodeEditor
    return CodeEditor()

__all__ = ["name", "load", "icon", "CodeEditor", "TextEditor", "SyntaxHighlighter"]

if __name__ == "plugin":
    window = CodeEditor()
    window.show()
