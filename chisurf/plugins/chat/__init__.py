"""
ChiSurf Assistant plugin (local RAG + Tools + Agent), self-contained under chisurf.plugins.chat

This plugin provides a PyQt Dock widget that connects to a local FastAPI backend (http://localhost:8000/)
for RAG-augmented chat and safe tool execution. It can also run in Agent mode.

Modules:
- index_docs: ingestion and retrieval
- server: FastAPI backend
- agent: tool registry and agent loop
- tools: anisotropy and utility tools
- qt_chat_dock: PyQt dock widget
"""

# Plugin menu entry name (appears under Plugins menu; category before colon creates a submenu)
name = "Help:Assistant Chat"

# Optional: provide a loader function similar to some other plugins
# Not required by the runner, but can be useful programmatically

def load():
    """Return an instance of the chat dock (not added to any window)."""
    from .qt_chat_dock import create_dock
    return create_dock()


# When executed by ChiSurf's plugin runner, __name__ is set to "plugin".
# Follow the same pattern used by other plugins: construct the widget and show it.
if __name__ == "plugin":
    import chisurf
    from PyQt5 import QtCore
    from .qt_chat_dock import create_dock

    try:
        parent = getattr(chisurf, 'cs', None)
        dock = create_dock(parent=parent)
        # If a main window is available and the dock is not already added, add as a dock on the right
        if parent is not None and hasattr(parent, 'addDockWidget'):
            try:
                # Attempt to add; Qt will ignore if already managed
                parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            except Exception:
                pass
        dock.show()
    except Exception as e:
        # Fallback: try showing as a top-level window
        try:
            w = create_dock()
            w.show()
        except Exception:
            # Last resort: print to console (will also appear in ChiSurf logs)
            print(f"Failed to open Assistant Chat dock: {e}")
