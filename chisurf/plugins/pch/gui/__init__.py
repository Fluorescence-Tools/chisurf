"""Client-side GUI components for PCH analysis.

* :class:`PCHClient` — typed RPC wrapper communicating with backend services
* :class:`PCHApp` — main analysis window (QMainWindow)
"""

from .client import PCHClient
from .tool import PCHApp

__all__ = ["PCHClient", "PCHApp"]

