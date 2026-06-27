"""ServiceDispatcher-compatible RPC handlers for Burst MLE handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

METHOD_PREPARE_WORKFLOW = "burst_mle.workflow.prepare"
METHOD_DESCRIBE_CONTRACT = "burst_mle.contract.describe"


def register_services(dispatcher: Any) -> None:
    """Register Burst MLE workflow RPC handlers."""
    dispatcher.register(METHOD_PREPARE_WORKFLOW, lambda params: prepare_workflow_handler(**(params or {})))
    dispatcher.register(METHOD_DESCRIBE_CONTRACT, lambda params: contract_handler(**(params or {})))


def list_methods() -> dict[str, str]:
    """Return Burst MLE RPC method descriptions."""
    return {
        METHOD_PREPARE_WORKFLOW: "Resolve MLE burst files and channel definitions from workflow context.",
        METHOD_DESCRIBE_CONTRACT: "Return the Burst MLE workflow contract.",
    }


def prepare_workflow_handler(
    workflow_context: dict[str, Any] | None = None,
    bur_files: list[str] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve MLE inputs from explicit files plus workflow context."""
    try:
        context = workflow_context or {}
        files = [Path(path) for path in (bur_files or [])]
        if not files:
            files = [Path(path) for path in context.get("bur_files", [])]
        burst_folder = context.get("burst_folder")
        if not files and isinstance(burst_folder, str):
            folder = Path(burst_folder)
            if folder.exists():
                files = sorted(folder.glob("**/*.bur"))

        return {
            "ok": True,
            "result": {
                "bur_files": [str(path) for path in files],
                "raw_files": list(context.get("raw_files", [])),
                "channel_settings": context.get("channel_settings", {}),
                "mfdb_artifacts": context.get("mfdb_artifacts", {}),
                "settings": settings or {},
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def contract_handler() -> dict[str, Any]:
    """Return the Burst MLE workflow contract descriptor."""
    return {
        "ok": True,
        "result": {
            "plugin_id": "burst_mle_analysis",
            "contract_version": "1.0.0",
            "transport": {"rpc": "JSON-RPC over ChiSurf ServiceDispatcher/ZMQ"},
            "rpc_methods": {
                METHOD_PREPARE_WORKFLOW: {
                    "input": "WorkflowContext",
                    "output": "ServiceResult",
                    "long_running": False,
                },
                METHOD_DESCRIBE_CONTRACT: {
                    "input": {},
                    "output": "ServiceResult",
                    "long_running": False,
                },
            },
        },
    }
