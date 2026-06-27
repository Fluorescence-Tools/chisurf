"""Fluorophore Database Plugin.

This plugin manages a centralized database of fluorophore spectra and optical properties.
"""

from .mfdb_adapter import DEFAULT_DATABASE_PATH, FluorophoreDatabase

name = "Dev:Spectroscopy:Fluorophore DB"

# Shared instance
_db_instance = None

def get_db():
    """Get the singleton instance of the FluorophoreDatabase."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FluorophoreDatabase(DEFAULT_DATABASE_PATH)
    return _db_instance

class FluorophoreDBWidget:
    """Placeholder for the curation UI widget implemented in db_manager_widget."""

    pass

if __name__ == "plugin":
    from .db_manager_widget import FluorophoreDBWidget as _Widget
    window = _Widget()
    window.show()
