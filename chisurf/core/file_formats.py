"""File format registry — loaded from ``file_formats.json``.

Provides ``FILE_FORMATS``, a dict mapping file extensions (e.g. ``.ptu``,
``.ht3``) to metadata dicts with keys ``name``, ``description``,
``tttrlib_container``, ``reading_routine``, and ``experiments``.

Usage::

    from chisurf.core.file_formats import FILE_FORMATS
    info = FILE_FORMATS.get('.ptu')
    # -> {'name': 'PicoQuant PTU', 'reading_routine': 'PTU', ...}
"""

from __future__ import annotations

import json
import pathlib
from chisurf import typing

_FILE = pathlib.Path(__file__).resolve().parent / 'file_formats.json'

FILE_FORMATS: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
try:
    with open(_FILE, 'r') as fh:
        raw = json.load(fh)
    FILE_FORMATS = {k: v for k, v in raw.items() if not k.startswith('_')}
except Exception:
    pass
