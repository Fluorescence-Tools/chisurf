"""
Fluorophore Database Plugin

This plugin manages a centralized database of fluorophore spectra and optical properties.
"""

import os
from chisurf.fio.mmcif.db import FluorophoreDatabase

name = "Dev:Spectroscopy:Fluorophore DB"

# Shared instance
_db_instance = None

def get_db():
    """Get the singleton instance of the FluorophoreDatabase."""
    global _db_instance
    if _db_instance is None:
        db_path = os.path.join(os.path.dirname(__file__), "spectra.db")
        _db_instance = FluorophoreDatabase(db_path)
    return _db_instance

class FluorophoreDBWidget:
    # Placeholder for the curation UI widget
    # This will be implemented in db_manager_widget.py
    pass

if __name__ == "plugin":
    from .db_manager_widget import FluorophoreDBWidget as _Widget
    window = _Widget()
    window.show()
