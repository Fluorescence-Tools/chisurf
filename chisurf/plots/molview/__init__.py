MolView = None

def __getattr__(name: str):
    """Lazy-load MolView on first access to avoid PyMOL import at startup."""
    global MolView
    if name == "MolView":
        if MolView is None:
            try:
                from .MolView import MolView as _MolView
                MolView = _MolView
            except Exception:
                MolView = None
        return MolView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MolView"]