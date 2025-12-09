from __future__ import annotations

from pathlib import Path
import re
import tempfile
import urllib.request
from typing import List

from .base import BaseCmd


class LoaderCommands(BaseCmd):
    """Loading and remote fetch commands."""

    def _mixin_commands(self):
        return {
            "load": self._cmd_load,
            "open": self._cmd_load,
            "fetch": self._cmd_fetch,
            "fetch_emdb": self._cmd_fetch_emdb,
            "fetch_ihm": self._cmd_fetch_ihm,
        }

    def _cmd_load(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: load <path> [more paths...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        for raw in args:
            path = Path(raw).expanduser()
            try:
                window._load_structure_from_path(path)
            except Exception as exc:
                self._emit_error(f"Failed to load '{path}': {exc}")
            else:
                self._emit_message(f"Loaded: {path}")

    def _cmd_fetch(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: fetch <pdb_id> [more ids...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        tmp_root = Path(tempfile.gettempdir())

        for raw in args:
            code = (raw or "").strip()
            if not code:
                continue
            pdb_id = code.lower()
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            dest = tmp_root / f"chimol_{pdb_id}.pdb"

            try:
                with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                self._emit_error(f"Failed to fetch '{pdb_id}' from RCSB: {exc}")
                continue

            try:
                window._load_structure_from_path(dest, name=code)
            except Exception as exc:
                self._emit_error(f"Failed to load fetched PDB '{pdb_id}': {exc}")
            else:
                self._emit_message(f"Fetched and loaded PDB: {pdb_id}")

    def _cmd_fetch_emdb(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: fetch_emdb <emdb_id> [more ids...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        tmp_root = Path(tempfile.gettempdir())

        for raw in args:
            code = (raw or "").strip()
            if not code:
                continue
            m = re.search(r"(\\d+)", code)
            if not m:
                self._emit_error(f"Could not parse EMDB id from {code!r}")
                continue
            emdb_num = m.group(1)
            folder = f"EMD-{emdb_num}"
            fname = f"emd_{emdb_num}.map.gz"
            url = (
                "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
                f"{folder}/map/{fname}"
            )
            dest = tmp_root / f"chimol_emd_{emdb_num}.map.gz"

            try:
                with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                self._emit_error(f"Failed to fetch EMDB map '{code}': {exc}")
                continue

            try:
                window._load_structure_from_path(dest, name=f"EMD-{emdb_num}")
            except Exception as exc:
                self._emit_error(f"Failed to load EMDB map '{code}': {exc}")
            else:
                self._emit_message(f"Fetched and loaded EMDB map: EMD-{emdb_num}")

    def _cmd_fetch_ihm(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: fetch_ihm <entry_id> [more ids...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        tmp_root = Path(tempfile.gettempdir())

        base_url = "https://pdb-ihm.org/cif"

        for raw in args:
            code = (raw or "").strip()
            if not code:
                continue
            entry_id = code.lower()
            url = f"{base_url}/{entry_id}.cif"
            dest = tmp_root / f"chimol_ihm_{entry_id}.cif"

            try:
                with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                self._emit_error(
                    f"Failed to fetch IHM CIF '{entry_id}' from pdb-ihm.org: {exc}"
                )
                continue

            try:
                window._load_structure_from_path(dest, name=code)
            except Exception as exc:
                self._emit_error(f"Failed to load fetched IHM CIF '{entry_id}': {exc}")
            else:
                self._emit_message(f"Fetched and loaded IHM CIF: {entry_id}")
