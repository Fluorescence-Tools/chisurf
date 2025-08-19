"""
Chato plugin: Chat dock using cja.py backend (local Ollama + RAG over Python code).

UI/UX is based on the existing chat plugin, but functionality (RAG, embeddings, chat)
comes from the top-level cja.py module.
"""

# Display name in the Plugins menu
name = "Help:Chato"



def load():
    """Return an instance of the Chato dock (not added to any window)."""
    from .qt_chato_dock import create_dock
    return create_dock()


# Standard plugin runner pattern used across ChiSurf
if __name__ == "plugin":
    import chisurf
    from PyQt5 import QtCore
    from .qt_chato_dock import create_dock

    try:
        parent = getattr(chisurf, 'cs', None)
        dock = create_dock(parent=parent)
        if parent is not None and hasattr(parent, 'addDockWidget'):
            try:
                parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            except Exception:
                pass
        dock.show()
    except Exception as e:
        try:
            w = create_dock()
            w.show()
        except Exception:
            print(f"Failed to open Chato dock: {e}")
