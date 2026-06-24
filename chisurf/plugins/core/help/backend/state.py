"""Help plugin state namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HelpPluginState:
    """Persistent state for the Help plugin."""

    current_path: Optional[str] = None
    filter_text: str = ""
    edit_mode: bool = False
    window_geometry: Optional[Dict[str, Any]] = None
