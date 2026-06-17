"""ServiceDispatcher-compatible RPC handlers for FPS JSON Editor."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..api.contract import (
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FETCH_PDB,
    contract_descriptor,
    normalize_pdb_id,
    service_success,
)

RCSB_PDB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"


def default_cache_dir() -> Path:
    """Return the default PDB cache directory for FPS JSON Editor."""
    return Path.home() / ".chisurf" / "fps_json_editor" / "pdb"


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
    source_url = RCSB_PDB_URL_TEMPLATE.format(pdb_id=normalized_id)
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


def register_services(dispatcher: Any) -> None:
    """Register FPS JSON Editor RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.

    """
    dispatcher.register(
        METHOD_FETCH_PDB,
        lambda params: fetch_pdb_handler(**params),
    )
    dispatcher.register(
        METHOD_DESCRIBE_CONTRACT,
        lambda params: contract_handler(),
    )


def list_methods() -> dict[str, str]:
    """Return the FPS JSON Editor RPC method catalogue."""
    return {
        METHOD_FETCH_PDB: "Download a PDB file from RCSB by four-character PDB ID.",
        METHOD_DESCRIBE_CONTRACT: "Return the FPS JSON Editor workflow contract.",
    }


def fetch_pdb_handler(
    pdb_id: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Download a PDB file and return its local path.

    Parameters
    ----------
    pdb_id : str
        Four-character RCSB PDB ID.
    output_dir : str, optional
        Directory where the downloaded PDB file should be written. If omitted,
        the plugin cache directory is used.

    Returns
    -------
    dict
        JSON-serializable ServiceResult with ``pdb_id``, ``path`` and ``source``.

    """
    try:
        normalized_id = normalize_pdb_id(pdb_id)
        path = download_pdb_file(normalized_id, output_dir=output_dir)
        source_url = RCSB_PDB_URL_TEMPLATE.format(pdb_id=normalized_id)
        return service_success(
            {
                "pdb_id": normalized_id,
                "path": str(path),
                "source": source_url,
            }
        )
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)


def contract_handler() -> dict[str, Any]:
    """Return the FPS JSON Editor workflow contract descriptor."""
    return service_success(contract_descriptor())
