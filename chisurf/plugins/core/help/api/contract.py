"""Workflow contract for the Help plugin."""

from __future__ import annotations

from typing import Any, Dict

METHOD_LIST_DOCS = "help.docs.list"
METHOD_READ_DOC = "help.docs.read"
METHOD_SAVE_DOC = "help.docs.save"
METHOD_SEARCH_DOCS = "help.docs.search"
METHOD_CONTRACT = "help.docs.contract"


def contract_descriptor() -> Dict[str, Any]:
    """Return the full contract descriptor for the Help plugin.

    Returns
    -------
    dict
        JSON-serializable contract with input/output schemas for each RPC method.

    """
    return {
        "methods": {
            METHOD_LIST_DOCS: {
                "summary": "List all available documentation files.",
                "params": {},
                "result": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "result": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "title": {"type": "string"},
                                    "category": {"type": "string"},
                                    "file_name": {"type": "string"},
                                    "size": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
            METHOD_READ_DOC: {
                "summary": "Read a documentation file and return its content.",
                "params": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "result": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "html": {"type": "string"},
                                "title": {"type": "string"},
                            },
                        },
                    },
                },
            },
            METHOD_SAVE_DOC: {
                "summary": "Save content to a documentation file.",
                "params": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "result": {"type": "object"},
                    },
                },
            },
            METHOD_SEARCH_DOCS: {
                "summary": "Search documentation files.",
                "params": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "result": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "title": {"type": "string"},
                                    "match_type": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def service_success(result: Any = None) -> Dict[str, Any]:
    """Return a success envelope."""
    return {"ok": True, "result": result}


def service_error(
    message: str,
    error_code: str = "UNKNOWN",
) -> Dict[str, Any]:
    """Return an error envelope."""
    return {"ok": False, "error": message, "error_code": error_code}
