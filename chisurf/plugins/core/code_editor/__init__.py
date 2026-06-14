"""
Code editor module providing syntax highlighting and editing capabilities.

This module provides a text editor with syntax highlighting for Python, JSON, and YAML.
It uses QSyntaxHighlighter and QPlainTextEdit instead of QScintilla2.
"""
from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.core.code_editor.document_store import DocumentSnapshot, DocumentStore
from chisurf.plugins.core.code_editor.editor import CodeEditor
from chisurf.plugins.core.code_editor.symbols import CodeSymbol, extract_python_symbols
from chisurf.plugins.core.code_editor.text_editor import (
    JSONHighlighter,
    PythonHighlighter,
    SyntaxHighlighter,
    TextEditor,
    YAMLHighlighter,
)
from chisurf.plugins.core.code_editor.window import CodeEditorWindow

# For backward compatibility
SimpleCodeEditor = TextEditor

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Tools:Miscellaneous:Code Editor"
icon = "📝"  # Memo/notepad emoji for editor

def load():
    """Return the plugin's full editor window."""
    from .text_editor import CodeEditorWindow
    return CodeEditorWindow()

__all__ = [
    "name",
    "load",
    "icon",
    "CodeEditor",
    "CodeEditorWindow",
    "DocumentSnapshot",
    "DocumentStore",
    "CodeSymbol",
    "JSONHighlighter",
    "PythonHighlighter",
    "TextEditor",
    "SyntaxHighlighter",
    "YAMLHighlighter",
    "extract_python_symbols",
]

if __name__ == "plugin":
    window = CodeEditorWindow()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
