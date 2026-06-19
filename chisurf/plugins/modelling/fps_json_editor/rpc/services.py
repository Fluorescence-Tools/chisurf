"""ServiceDispatcher-compatible RPC handlers for FPS JSON Editor."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..api.contract import (
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FETCH_PDB,
    METHOD_NORMALIZE_PAYLOAD,
    METHOD_SAVE_AV_MRC,
    METHOD_SUMMARIZE_PAYLOAD,
    METHOD_VALIDATE_PAYLOAD,
    contract_descriptor,
    normalize_pdb_id,
    service_success,
)
from ..core.mrc import save_av_mrc
from ..core.payload import normalize_payload, summarize_payload, validate_payload
from ..core.pdb import download_pdb_file, pdb_source_url


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
    dispatcher.register(
        METHOD_VALIDATE_PAYLOAD,
        lambda params: validate_payload_handler(**params),
    )
    dispatcher.register(
        METHOD_SUMMARIZE_PAYLOAD,
        lambda params: summarize_payload_handler(**params),
    )
    dispatcher.register(
        METHOD_NORMALIZE_PAYLOAD,
        lambda params: normalize_payload_handler(**params),
    )
    dispatcher.register(
        METHOD_SAVE_AV_MRC,
        lambda params: save_av_mrc_handler(**params),
    )


def list_methods() -> dict[str, str]:
    """Return the FPS JSON Editor RPC method catalogue."""
    return {
        METHOD_FETCH_PDB: "Download a PDB file from RCSB by four-character PDB ID.",
        METHOD_DESCRIBE_CONTRACT: "Return the FPS JSON Editor workflow contract.",
        METHOD_VALIDATE_PAYLOAD: "Validate an fps.json payload and return errors/warnings.",
        METHOD_SUMMARIZE_PAYLOAD: "Summarize positions, distances, score sets and references.",
        METHOD_NORMALIZE_PAYLOAD: "Normalize an fps.json payload through the core model.",
        METHOD_SAVE_AV_MRC: "Save AV points as an IMP-backed MRC density map.",
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
        return service_success(
            {
                "pdb_id": normalized_id,
                "path": str(path),
                "source": pdb_source_url(normalized_id),
            }
        )
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)


def contract_handler() -> dict[str, Any]:
    """Return the FPS JSON Editor workflow contract descriptor."""
    return service_success(contract_descriptor())


def validate_payload_handler(payload: Any) -> dict[str, Any]:
    """Validate an fps.json payload."""
    return service_success(validate_payload(payload))


def summarize_payload_handler(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize an fps.json payload."""
    return service_success(summarize_payload(payload))


def normalize_payload_handler(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an fps.json payload through the core model."""
    return service_success({"payload": normalize_payload(payload)})


def save_av_mrc_handler(
    path: str,
    points: list[list[float]],
    grid_step: float,
) -> dict[str, Any]:
    """Save AV points as an MRC map and return the written path."""
    try:
        out_path = save_av_mrc(path, np.asarray(points, dtype=np.float64), grid_step)
        return service_success({"path": str(out_path)})
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)
