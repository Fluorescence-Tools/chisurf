"""Workflow contract for the ALEX Creator plugin.

GUI, CLI and RPC layers normalize to :class:`AlexRequest` and return
:class:`AlexResult` payloads through the helpers here, then delegate to the
Qt-free :mod:`..core` implementation.
"""

from __future__ import annotations

import pathlib
from typing import Any

from .. import core
from .models import AlexRequest, AlexResult

PLUGIN_ID = "ptu_alex_creator"
CONTRACT_VERSION = "1.0.0"

METHOD_CONVERT = "alex.convert"
METHOD_MERGE = "alex.merge"
METHOD_HISTOGRAM = "alex.histogram"
METHOD_DESCRIBE_CONTRACT = "alex.contract.describe"

CANONICAL_METHODS = (
    METHOD_CONVERT,
    METHOD_MERGE,
    METHOD_HISTOGRAM,
    METHOD_DESCRIBE_CONTRACT,
)


def request_from_payload(payload: dict[str, Any]) -> AlexRequest:
    """Normalize a JSON payload into an :class:`AlexRequest`."""
    payload = payload or {}
    return AlexRequest(
        files=[str(f) for f in (payload.get("files") or [])],
        alex_period=int(payload.get("alex_period", 8000)),
        period_shift=int(payload.get("period_shift", 0)),
        output_format=str(payload.get("output_format", "PTU")),
        input_format=str(payload.get("input_format", "Auto")),
        mode=str(payload.get("mode", "convert")),
        output_dir=str(payload.get("output_dir", "")),
        output_path=str(payload.get("output_path", "")),
    )


def result_to_payload(result: AlexResult) -> dict[str, Any]:
    """Serialize an :class:`AlexResult` to a JSON-safe dict."""
    return {"output_paths": list(result.output_paths), "mode": result.mode}


def run(request: AlexRequest) -> AlexResult:
    """Execute *request* through :mod:`..core` and return the produced paths."""
    if not request.files:
        raise ValueError("No input files specified.")

    if request.mode == "merge":
        out_path = request.output_path
        if not out_path:
            base_dir = request.output_dir or str(pathlib.Path(request.files[0]).parent)
            name = core.default_output_name(
                request.files[0], request.output_format, suffix="_merged_alex"
            )
            out_path = str(pathlib.Path(base_dir) / name)
        core.merge_files(
            request.files,
            out_path,
            request.alex_period,
            request.period_shift,
            request.output_format,
            request.input_format,
        )
        return AlexResult(output_paths=[out_path], mode="merge")

    outputs: list[str] = []
    for in_path in request.files:
        base_dir = request.output_dir or str(pathlib.Path(in_path).parent)
        out_path = str(
            pathlib.Path(base_dir) / core.default_output_name(in_path, request.output_format)
        )
        core.convert_file(
            in_path,
            out_path,
            request.alex_period,
            request.period_shift,
            request.output_format,
            request.input_format,
        )
        outputs.append(out_path)
    return AlexResult(output_paths=outputs, mode="convert")


def contract_descriptor() -> dict[str, Any]:
    """Describe the ALEX Creator workflow contract for discovery."""
    return {
        "plugin_id": PLUGIN_ID,
        "version": CONTRACT_VERSION,
        "methods": list(CANONICAL_METHODS),
        "modes": ["convert", "merge"],
        "output_formats": core.supported_containers(),
    }


__all__ = [
    "PLUGIN_ID",
    "CONTRACT_VERSION",
    "METHOD_CONVERT",
    "METHOD_MERGE",
    "METHOD_HISTOGRAM",
    "METHOD_DESCRIBE_CONTRACT",
    "CANONICAL_METHODS",
    "request_from_payload",
    "result_to_payload",
    "run",
    "contract_descriptor",
]
