"""API endpoints for FRET modelling.

``operations`` is the transport-agnostic seam (used by the CLI and RPC
services). ``router``/``app`` provide an optional FastAPI surface and are only
available when ``fastapi`` is installed.
"""

from __future__ import annotations

from . import models, operations

try:  # optional HTTP surface
    from .router import app, router
except Exception:  # pragma: no cover - fastapi not installed
    router = None
    app = None
