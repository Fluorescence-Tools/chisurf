"""Dedicated ZMQ server adapter for the Burst Selection API.

This module exists for backwards compatibility with the historical
``chisurf.plugins.burst.burst_selection.server`` import path.  New code should
use ``backend.services`` or the public ``api`` package directly.
"""

from __future__ import annotations

from typing import Any

from ..api.contract import (
    LEGACY_METHOD_ANALYZE_FILES,
    LEGACY_METHOD_FIT_GMM,
    LEGACY_METHOD_INSPECT_BUR,
    METHOD_ANALYZE_FILES,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FIT_GMM,
    METHOD_INSPECT_BUR,
    METHOD_LOAD_DIAGNOSTICS,
)
from ..backend.services import (
    analyze_files_handler,
    contract_handler,
    diagnostics_handler,
    fit_gmm_handler,
    inspect_bur_handler,
)


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a dedicated ZMQ method call to the shared backend handler."""
    handlers = {
        METHOD_ANALYZE_FILES: lambda payload: analyze_files_handler(**payload),
        METHOD_INSPECT_BUR: lambda payload: inspect_bur_handler(**payload),
        METHOD_FIT_GMM: lambda payload: fit_gmm_handler(**payload),
        METHOD_LOAD_DIAGNOSTICS: lambda payload: diagnostics_handler(**payload),
        METHOD_DESCRIBE_CONTRACT: lambda _payload: contract_handler(),
        LEGACY_METHOD_ANALYZE_FILES: lambda payload: analyze_files_handler(**payload),
        LEGACY_METHOD_INSPECT_BUR: lambda payload: inspect_bur_handler(**payload),
        LEGACY_METHOD_FIT_GMM: lambda payload: fit_gmm_handler(**payload),
    }
    try:
        return handlers[method](params or {})
    except KeyError as exc:
        raise KeyError(f"unknown burst-selection method: {method}") from exc


def serve(host: str = "127.0.0.1", cmd_port: int = 8765, pub_port: int = 8766) -> None:
    """Run a dedicated Burst Selection ZMQ server."""
    from chisurf.server.transport.zmq import ZmqServer

    server = ZmqServer(
        handler=lambda method, params: _dispatch(method, params or {}),
        cmd_port=cmd_port,
        pub_port=pub_port,
        host=host,
    )
    server.serve_forever()


def analyze_files(
    files: list[str],
    filetype: str | None = None,
    windows: dict[str, list[int]] | None = None,
    detectors: dict[str, dict[str, Any]] | None = None,
    settings: dict[str, Any] | None = None,
    output_dir: str | None = None,
    legacy_output: bool = False,
    legacy_output_folder_name: str | None = None,
    selected_setup: str | None = None,
    legacy_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Burst Selection analysis over TTTR files."""
    return analyze_files_handler(
        files=files,
        filetype=filetype,
        windows=windows,
        detectors=detectors,
        settings=settings,
        output_dir=output_dir,
        legacy_output=legacy_output,
        legacy_output_folder_name=legacy_output_folder_name,
        selected_setup=selected_setup,
        legacy_parameters=legacy_parameters,
    )


def inspect_bur(path: str) -> dict[str, Any]:
    """Inspect a saved ``.bur`` file."""
    return inspect_bur_handler(path=path)


def fit_gmm_from_bur(path: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit a GMM to features extracted from a saved ``.bur`` file."""
    return fit_gmm_handler(path=path, settings=settings)
