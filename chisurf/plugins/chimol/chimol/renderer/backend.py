from __future__ import annotations

import os

try:  # optional OpenGL dependency
    import pyqtgraph.opengl as gl  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - GL may be unavailable
    gl = None


_HARD_DISABLE_GL = os.environ.get("QT_OPENGL", "").lower() == "software"

_HAVE_GL = (gl is not None) and not _HARD_DISABLE_GL
