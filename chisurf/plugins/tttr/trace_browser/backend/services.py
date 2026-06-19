"""RPC handlers for Trace Browser."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from chisurf.plugins.tttr.trace_browser.api.contract import (
    LEGACY_METHODS,
    METHOD_CONTRACT,
    METHOD_EXPORT_CSV,
    METHOD_GET_METADATA,
    METHOD_LIST_FILES,
    METHOD_LOAD_TRACE,
    METHOD_SET_METADATA,
    contract_descriptor,
    service_error,
    service_success,
)
from chisurf.plugins.tttr.trace_browser.api.io import export_csv, list_files
from chisurf.plugins.tttr.trace_browser.core.metadata import load_meta, save_meta
from chisurf.plugins.tttr.trace_browser.core.trace import load_trace

_log = logging.getLogger(__name__)


def _list_files_handler(params: dict[str, Any]) -> dict[str, Any]:
    try:
        folder = params.get("folder")
        if not folder:
            return service_error("folder is required", "INVALID_PARAMS")
        rows = list_files(
            str(folder),
            recursive=bool(params.get("recursive", False)),
            setup_settings=params.get("setup_settings"),
            selected_channels=params.get("selected_channels"),
        )
        return service_success({"files": rows})
    except Exception as exc:
        _log.exception("trace_browser.files.list failed")
        return service_error(str(exc), "LIST_FAILED")


def _get_metadata_handler(params: dict[str, Any]) -> dict[str, Any]:
    try:
        folder = params.get("folder")
        if not folder:
            return service_error("folder is required", "INVALID_PARAMS")
        return service_success({"metadata": load_meta(pathlib.Path(str(folder)))})
    except Exception as exc:
        _log.exception("trace_browser.metadata.get failed")
        return service_error(str(exc), "METADATA_GET_FAILED")


def _set_metadata_handler(params: dict[str, Any]) -> dict[str, Any]:
    try:
        folder = params.get("folder")
        metadata = params.get("metadata", {})
        if not folder or not isinstance(metadata, dict):
            return service_error("folder and metadata are required", "INVALID_PARAMS")
        save_meta(pathlib.Path(str(folder)), metadata)
        return service_success({"folder": str(folder)})
    except Exception as exc:
        _log.exception("trace_browser.metadata.set failed")
        return service_error(str(exc), "METADATA_SET_FAILED")


def _load_trace_handler(params: dict[str, Any]) -> dict[str, Any]:
    try:
        path = params.get("path")
        if not path:
            return service_error("path is required", "INVALID_PARAMS")
        data = load_trace(
            str(path),
            time_window_ms=float(params.get("time_window_ms", 10.0)),
            setup_settings=params.get("setup_settings"),
            selected_channels=params.get("selected_channels"),
            cache_folder=params.get("cache_folder"),
        )
        return service_success({"trace": data})
    except Exception as exc:
        _log.exception("trace_browser.traces.load failed")
        return service_error(str(exc), "TRACE_LOAD_FAILED")


def _export_csv_handler(params: dict[str, Any]) -> dict[str, Any]:
    try:
        paths = params.get("paths", [])
        output_dir = params.get("output_dir")
        if not paths or not output_dir:
            return service_error("paths and output_dir are required", "INVALID_PARAMS")
        result = export_csv(
            [str(path) for path in paths],
            str(output_dir),
            time_window_ms=float(params.get("time_window_ms", 10.0)),
            setup_settings=params.get("setup_settings"),
            selected_channels=params.get("selected_channels"),
        )
        return service_success(result)
    except Exception as exc:
        _log.exception("trace_browser.export.csv failed")
        return service_error(str(exc), "EXPORT_CSV_FAILED")


def _contract_handler() -> dict[str, Any]:
    return service_success(contract_descriptor())


def register_services(dispatcher: Any) -> None:
    """Register Trace Browser RPC handlers."""
    handlers = {
        METHOD_LIST_FILES: _list_files_handler,
        METHOD_GET_METADATA: _get_metadata_handler,
        METHOD_SET_METADATA: _set_metadata_handler,
        METHOD_LOAD_TRACE: _load_trace_handler,
        METHOD_EXPORT_CSV: _export_csv_handler,
        METHOD_CONTRACT: lambda _params: _contract_handler(),
    }
    for method, handler in handlers.items():
        dispatcher.register(method, handler)
    for legacy, canonical in LEGACY_METHODS.items():
        dispatcher.register(legacy, handlers[canonical])
