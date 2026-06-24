"""Data models for the Help plugin API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HelpState:
    """Persistent plugin state."""

    current_path: Optional[str] = None
    filter_text: str = ""
    edit_mode: bool = False


@dataclass
class HelpRequest:
    """Generic request envelope for help RPC calls."""

    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HelpResponse:
    """Generic response envelope from help RPC calls."""

    ok: bool = True
    result: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
