"""Workflow contract for the Burst Selection plugin.

This module is the stable boundary for non-GUI integrations.  GUI, CLI and
RPC layers may collect parameters differently, but they should all normalize
to :class:`AnalysisRequest` and return :class:`AnalysisResult` payloads through
the helpers defined here.
"""

from __future__ import annotations

from typing import Any

from .models import AnalysisRequest, AnalysisResult, AnalysisSettings
from .serialization import settings_from_dict, to_jsonable

PLUGIN_ID = "burst_selection"
CONTRACT_VERSION = "1.0.0"

METHOD_ANALYZE_FILES = "burst_selection.jobs.analyze_files"
METHOD_INSPECT_BUR = "burst_selection.results.inspect_bur"
METHOD_FIT_GMM = "burst_selection.gmm.fit"
METHOD_LOAD_DIAGNOSTICS = "burst_selection.diagnostics.load"
METHOD_DESCRIBE_CONTRACT = "burst_selection.contract.describe"

LEGACY_METHOD_ANALYZE_FILES = "burst_selection.analyze_files"
LEGACY_METHOD_INSPECT_BUR = "burst_selection.inspect_bur"
LEGACY_METHOD_FIT_GMM = "burst_selection.fit_gmm_from_bur"

CANONICAL_METHODS = (
    METHOD_ANALYZE_FILES,
    METHOD_INSPECT_BUR,
    METHOD_FIT_GMM,
    METHOD_LOAD_DIAGNOSTICS,
    METHOD_DESCRIBE_CONTRACT,
)

LEGACY_METHODS = (
    LEGACY_METHOD_ANALYZE_FILES,
    LEGACY_METHOD_INSPECT_BUR,
    LEGACY_METHOD_FIT_GMM,
)


def normalize_windows(windows: dict[str, Any] | None) -> dict[str, tuple[int, int]]:
    """Return PIE windows as ``{name: (start, stop)}``.

    Parameters
    ----------
    windows : dict, optional
        JSON-compatible window mapping. Values may be two-item lists or tuples.

    Returns
    -------
    dict
        Normalized window mapping.

    Raises
    ------
    ValueError
        If a window value is not a two-item sequence.

    """
    normalized: dict[str, tuple[int, int]] = {}
    for name, bounds in (windows or {}).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"window {name!r} must be a two-item range")
        normalized[str(name)] = (int(bounds[0]), int(bounds[1]))
    return normalized


def analysis_request_from_payload(payload: dict[str, Any]) -> AnalysisRequest:
    """Build an :class:`AnalysisRequest` from a JSON-compatible payload.

    Parameters
    ----------
    payload : dict
        Workflow/RPC request payload.

    Returns
    -------
    AnalysisRequest
        Normalized request object used by the API layer.

    """
    settings_payload = payload.get("settings")
    if isinstance(settings_payload, AnalysisSettings):
        settings = settings_payload
    elif isinstance(settings_payload, dict):
        settings = settings_from_dict(settings_payload)
    else:
        settings = AnalysisSettings()
    return AnalysisRequest(
        files=[str(path) for path in payload.get("files", [])],
        filetype=payload.get("filetype"),
        windows=normalize_windows(payload.get("windows")),
        detectors=payload.get("detectors") or {},
        settings=settings,
        output_dir=payload.get("output_dir"),
        legacy_output=bool(payload.get("legacy_output", False)),
        legacy_output_folder_name=payload.get("legacy_output_folder_name"),
        selected_setup=payload.get("selected_setup"),
        legacy_parameters=payload.get("legacy_parameters") or {},
    )


def analysis_request_to_payload(request: AnalysisRequest) -> dict[str, Any]:
    """Return a JSON-compatible analysis request payload."""
    return to_jsonable(request)


def analysis_result_to_payload(result: AnalysisResult) -> dict[str, Any]:
    """Return a JSON-compatible analysis result payload."""
    return to_jsonable(result)


def service_success(result: AnalysisResult | dict[str, Any]) -> dict[str, Any]:
    """Wrap an API result in the standard JSON-RPC service envelope."""
    payload = analysis_result_to_payload(result) if isinstance(result, AnalysisResult) else to_jsonable(result)
    return {"ok": True, "result": payload}


def contract_descriptor() -> dict[str, Any]:
    """Return the JSON-compatible Burst Selection workflow contract.

    The descriptor intentionally stays dependency-free and JSON-schema-like so
    desktop GUI, ZMQ JSON-RPC, CLI callers and future node/workflow runtimes can
    validate their own payloads without importing Qt or tttrlib.
    """
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "cli": "chisurf burst-selection",
            "api": "chisurf.plugins.burst.burst_selection.api",
        },
        "inputs": {
            "AnalyzeFiles": {
                "type": "object",
                "required": ["files"],
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "filetype": {"type": ["string", "null"]},
                    "windows": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "detectors": {"type": "object"},
                    "settings": {"$ref": "#/definitions/AnalysisSettings"},
                    "output_dir": {"type": ["string", "null"]},
                    "legacy_output": {"type": "boolean"},
                    "legacy_output_folder_name": {"type": ["string", "null"]},
                    "selected_setup": {"type": ["string", "null"]},
                    "legacy_parameters": {"type": "object"},
                },
            },
            "InspectBur": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
            "FitGmm": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "settings": {"$ref": "#/definitions/GMMSettings"},
                },
            },
            "DiagnosticsLoad": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "settings": {"$ref": "#/definitions/AnalysisSettings"},
                },
            },
        },
        "outputs": {
            "ServiceResult": {
                "type": "object",
                "required": ["ok"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"},
                    "error_code": {"type": "string"},
                },
            },
            "AnalysisResult": {
                "type": "object",
                "required": ["files", "dataframes", "output_paths", "metadata"],
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "dataframes": {"type": "object"},
                    "feature_dataframe": {"type": ["object", "null"]},
                    "gmm_fit": {"type": ["object", "null"]},
                    "output_paths": {"type": "object"},
                    "metadata": {"type": "object"},
                },
            },
        },
        "definitions": {
            "AnalysisSettings": {
                "type": "object",
                "properties": {
                    "photon_filter": {"type": "object"},
                    "burst_detection": {"type": "object"},
                    "gmm": {"$ref": "#/definitions/GMMSettings"},
                    "output_formats": {"type": "array", "items": {"type": "string"}},
                    "zip_output": {"type": "boolean"},
                    "remove_folder": {"type": "boolean"},
                },
            },
            "GMMSettings": {
                "type": "object",
                "properties": {
                    "covariance_type": {"type": "string"},
                    "random_state": {"type": "integer"},
                    "max_iter": {"type": "integer"},
                    "n_init": {"type": "integer"},
                    "tol": {"type": "number"},
                    "max_components": {"type": "integer"},
                    "reg_covar": {"type": "number"},
                    "auto_components": {"type": "boolean"},
                },
            },
        },
        "rpc_methods": {
            METHOD_ANALYZE_FILES: {
                "input": "AnalyzeFiles",
                "output": "ServiceResult<AnalysisResult>",
                "long_running": True,
            },
            METHOD_INSPECT_BUR: {"input": "InspectBur", "output": "ServiceResult"},
            METHOD_FIT_GMM: {"input": "FitGmm", "output": "ServiceResult"},
            METHOD_LOAD_DIAGNOSTICS: {"input": "DiagnosticsLoad", "output": "ServiceResult"},
            METHOD_DESCRIBE_CONTRACT: {"input": "{}", "output": "ServiceResult<Contract>"},
        },
    }
