try:
    from .MolView import MolView
except Exception:
    MolView = None

__all__ = ["MolView"]