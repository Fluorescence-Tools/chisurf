"""PDB file helpers for the FPS JSON Editor plugin."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..api.contract import normalize_pdb_id

RCSB_PDB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"


def default_cache_dir() -> Path:
    """Return the default PDB cache directory for FPS JSON Editor."""
    return Path.home() / ".chisurf" / "fps_json_editor" / "pdb"


def pdb_source_url(pdb_id: str) -> str:
    """Return the RCSB download URL for a normalized PDB ID."""
    return RCSB_PDB_URL_TEMPLATE.format(pdb_id=normalize_pdb_id(pdb_id))


def download_pdb_file(
    pdb_id: str,
    output_dir: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a PDB file from RCSB and return the local path."""
    normalized_id = normalize_pdb_id(pdb_id)
    target_dir = Path(output_dir) if output_dir else (cache_dir or default_cache_dir())
    target_path = target_dir / f"{normalized_id}.pdb"
    if target_path.exists():
        return target_path

    target_dir.mkdir(parents=True, exist_ok=True)
    source_url = pdb_source_url(normalized_id)
    request = Request(source_url, headers={"User-Agent": "ChiSurf fps_json_editor"})
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"RCSB returned {exc.code} for PDB ID {normalized_id!r}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download PDB ID {normalized_id!r}: {exc}") from exc

    if not data.strip():
        raise RuntimeError(f"RCSB returned an empty PDB file for {normalized_id!r}")

    target_path.write_bytes(data)
    return target_path
