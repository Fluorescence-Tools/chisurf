from __future__ import annotations

import pathlib
from typing import List

from chisurf.gui import QtWidgets
from chisurf.plugins.fcs_correlator.wizard import FileListWidget


class BurstFileListWidget(FileListWidget):
    """File list that expands burstwise folders into BUR/BST files.

    When a user drops a burst analysis folder (e.g. ``burstwise_*``), this
    widget replaces the folder entry by the ``.bur`` files found in its
    ``bi4_bur``/``bur`` subfolders, or, if none exist, by ``BID/*.bst``
    files. This way the list explicitly shows all index files that will
    be processed.
    """

    def add_files(self, file_paths: List[str]):  # type: ignore[override]
        if not file_paths:
            return

        expanded: List[str] = []
        for fp in file_paths:
            try:
                p = pathlib.Path(fp)
            except Exception:
                continue
            if p.is_dir():
                expanded.extend(self._expand_burst_folder(p))
            else:
                expanded.append(fp)

        # Delegate to the base implementation with the expanded list
        super().add_files(expanded)

    @staticmethod
    def _expand_burst_folder(folder: pathlib.Path) -> List[str]:
        """Return BUR (preferred) or BID/BST files inside a burstwise folder."""

        out: List[str] = []
        if not folder.is_dir():
            return [folder.as_posix()]

        # Prefer bi4_bur/bur with .bur files
        bur_files: List[pathlib.Path] = []
        for sub_name in ("bi4_bur", "bur"):
            subdir = folder / sub_name
            if not subdir.is_dir():
                continue
            try:
                for bur in sorted(subdir.glob("*.bur")):
                    bur_files.append(bur)
            except Exception:
                continue

        if bur_files:
            return [b.as_posix() for b in bur_files]

        # Fallback: BID/*.bst
        bid_dir = folder / "BID"
        if bid_dir.is_dir():
            try:
                for bst in sorted(bid_dir.glob("*.bst")):
                    out.append(bst.as_posix())
            except Exception:
                pass

        # If nothing was found at all, keep the folder so the user sees it
        return out or [folder.as_posix()]
