"""Open a registered burst selection from MFDB in ndXplorer (PRD-28).

The chisurf side of the ndXplorer ↔ MFDB round trip. Reuses the existing dataset
picker (the sample/measurement selection widget) and ``mfdb.datasets.open`` to
resolve an artifact to a local path, then hands it to ndXplorer via its drop-style
opener. ``modules/ndxplorer`` stays chisurf-free; this module is the only glue.

Usable directly (e.g. from the Code Editor) for a manual round-trip test::

    from chisurf.plugins.ndxplorer.mfdb_launcher import open_burst_selection_from_mfdb
    open_burst_selection_from_mfdb()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Artifact kinds that can be opened in ndXplorer (burst outputs).
BURST_KINDS = ["burst_table", "external_reference", "processed_data"]


def resolve_dataset_path(client: Any, artifact_id: str) -> str | None:
    """Resolve an MFDB artifact to a local readable path via ``mfdb.datasets.open``."""
    if not artifact_id:
        return None
    result = client.call("mfdb.datasets.open", {"artifact_id": artifact_id}) or {}
    return result.get("local_path") or result.get("path")


def open_path_in_ndxplorer(path: str) -> Any:
    """Launch (a new) ndXplorer and open ``path`` using its drop-style opener.

    Best-effort: ndXplorer is an optional external module. Returns the ndXplorer
    instance or ``None`` if it could not be launched.
    """
    try:
        import ndxplorer
        from ndxplorer.__main__ import open_path_like_drop
    except Exception as exc:  # pragma: no cover - depends on optional module
        logger.error("ndXplorer is not available: %s", exc)
        return None
    ndx = ndxplorer.NDXplorer()
    open_path_like_drop(ndx, str(path))
    ndx.show()
    ndx.raise_()
    ndx.activateWindow()
    return ndx


def open_burst_selection_from_mfdb(parent: Any = None, scope: str = "all") -> Any:
    """Pick a registered burst selection from MFDB and open it in ndXplorer.

    Pick (the sample/measurement selection widget) → resolve the artifact to a
    local path → open in ndXplorer. Returns the ndXplorer instance, or ``None`` if
    nothing was selected / it could not be opened.
    """
    from chisurf.gui.widgets.mfdb.dataset_browser import MfdbDatasetPickerDialog
    from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

    client = MFDBClient(inprocess=True)
    sel = MfdbDatasetPickerDialog.pick_dataset(
        parent=parent,
        kinds=BURST_KINDS,
        scope=scope,
        client=client,
    )
    if sel is None:
        return None
    path = resolve_dataset_path(client, sel.artifact_id)
    if not path:
        logger.error("Could not resolve a local path for artifact %s", sel.artifact_id)
        return None
    return open_path_in_ndxplorer(path)


def send_path_to_ndxplorer(path: str, parent: Any = None) -> Any:
    """Open a burst-selection output path (e.g. just produced) in ndXplorer."""
    if not path:
        return None
    return open_path_in_ndxplorer(path)
