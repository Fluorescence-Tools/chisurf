"""ServiceDispatcher-compatible RPC handlers for Time Window Bins.

Thin adapters that accept JSON-compatible params, delegate to the
``api/`` layer, and return JSON-safe results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..api.contract import (
    METHOD_ANALYZE_FILES,
    METHOD_DESCRIBE_CONTRACT,
    contract_descriptor,
    request_from_payload,
    service_success,
)
from ..api.io import compute_and_save
from ..api.models import TimeWindowResult


def register_services(dispatcher: Any) -> None:
    """Register Time Window Bins RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.
    """
    dispatcher.register(
        METHOD_ANALYZE_FILES,
        lambda params: analyze_files_handler(**params),
    )
    dispatcher.register(
        METHOD_DESCRIBE_CONTRACT,
        lambda params: contract_handler(**(params or {})),
    )


def list_methods() -> dict[str, str]:
    """Return the Time Window Bins RPC method catalogue."""
    return {
        METHOD_ANALYZE_FILES: "Split TTTR files into fixed-duration time-window BIDs.",
        METHOD_DESCRIBE_CONTRACT: "Return the Time Window Bins workflow contract.",
    }


def analyze_files_handler(
    files: list[str],
    time_window_ms: float = 10.0,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Split TTTR files into fixed-duration time-window BIDs.

    Parameters
    ----------
    files : list of str
        TTTR file paths.
    time_window_ms : float
        Time window duration in milliseconds.
    output_dir : str, optional
        Output directory for ``.bst`` files. If omitted, an output
        directory is derived from the first input file.

    Returns
    -------
    dict
        JSON-serializable ServiceResult with per-file window counts
        and output paths.
    """
    from chisurf.server.services import OPERATION_FAILED, service_error

    try:
        if not files:
            return service_error("No files provided", error_code=OPERATION_FAILED)
        request = request_from_payload(
            {
                "files": files,
                "time_window_ms": time_window_ms,
                "output_dir": output_dir,
            }
        )
        time_window_s = request.time_window_ms / 1000.0
        resolved_output_dir = _resolve_output_dir(request)

        n_windows: dict[str, int] = {}
        output_paths: dict[str, str] = {}

        for fp in request.files:
            cnt, out_path = compute_and_save(fp, time_window_s, resolved_output_dir)
            n_windows[str(fp)] = cnt
            output_paths[str(fp)] = out_path

        result = TimeWindowResult(
            files=list(request.files),
            n_windows=n_windows,
            output_paths=output_paths,
            metadata={
                "n_files": len(request.files),
                "total_windows": sum(n_windows.values()),
                "output_dir": str(resolved_output_dir),
                "time_window_ms": request.time_window_ms,
            },
        )
        return service_success(result)
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)


def contract_handler() -> dict[str, Any]:
    """Return the Time Window Bins workflow contract descriptor."""
    return service_success(contract_descriptor())


def _resolve_output_dir(request: Any) -> Path:
    """Resolve or derive the output directory for a request."""
    if request.output_dir:
        out_dir = Path(request.output_dir)
    elif request.files:
        first = Path(request.files[0])
        folder_name = f"{first.stem}_TW_{request.time_window_ms:.0f}ms"
        out_dir = first.parent / folder_name
    else:
        out_dir = Path("time_window_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
