"""New-style GUI entrypoint for the PSF Determination plugin.

Exposes :class:`PsfDeterminationTool` — a thin alias for the existing
:class:`~chisurf.plugins.microscopy.psf_determination.PSFDeterminationWidget`
which already carries the ``@persist_plugin_state`` decorator.

Importing this module requires Qt; it is only loaded lazily via the
``__getattr__`` hook in the package ``__init__.py``.
"""

from __future__ import annotations

from chisurf.plugins.microscopy.psf_determination import PSFDeterminationWidget  # noqa: F401

#: New-style entrypoint alias.
PsfDeterminationTool = PSFDeterminationWidget

__all__ = ["PsfDeterminationTool"]
