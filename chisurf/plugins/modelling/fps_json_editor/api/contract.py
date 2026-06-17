"""Workflow contract for the FPS JSON Editor plugin."""

from __future__ import annotations

import re
from typing import Any

PLUGIN_ID = "fps_json_editor"
CONTRACT_VERSION = "2.0.0"

METHOD_FETCH_PDB = "fps_json_editor.pdb.fetch"
METHOD_DESCRIBE_CONTRACT = "fps_json_editor.contract.describe"

_PDB_ID_RE = re.compile(r"^[0-9][a-z0-9]{3}$", re.IGNORECASE)


def normalize_pdb_id(pdb_id: str) -> str:
    """Normalize and validate a four-character RCSB PDB ID."""
    if not isinstance(pdb_id, str):
        raise ValueError("pdb_id must be a string")
    normalized = pdb_id.strip().lower()
    if not _PDB_ID_RE.fullmatch(normalized):
        raise ValueError("pdb_id must be a four-character RCSB PDB ID")
    return normalized


def service_success(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON-compatible result in the standard service envelope."""
    return {"ok": True, "result": result}


def contract_descriptor() -> dict[str, Any]:
    """Return the FPS JSON Editor workflow contract descriptor."""
    return {
        "plugin_id": PLUGIN_ID,
        "contract_version": CONTRACT_VERSION,
        "transport": {
            "rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ",
            "gui": "chisurf.plugins.modelling.fps_json_editor.gui.tool",
            "api": "chisurf.plugins.modelling.fps_json_editor.api",
        },
        "inputs": {
            "FetchPdb": {
                "type": "object",
                "required": ["pdb_id"],
                "properties": {
                    "pdb_id": {"type": "string", "pattern": "^[0-9][A-Za-z0-9]{3}$"},
                    "output_dir": {"type": ["string", "null"]},
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
            "FetchPdbResult": {
                "type": "object",
                "required": ["pdb_id", "path", "source"],
                "properties": {
                    "pdb_id": {"type": "string"},
                    "path": {"type": "string"},
                    "source": {"type": "string"},
                },
            },
        },
        "rpc_methods": {
            METHOD_FETCH_PDB: {
                "input": "FetchPdb",
                "output": "ServiceResult<FetchPdbResult>",
                "long_running": False,
            },
            METHOD_DESCRIBE_CONTRACT: {
                "input": "{}",
                "output": "ServiceResult<Contract>",
                "long_running": False,
            },
        },
    }
