"""ZMQ RPC handlers for the Help plugin."""

from __future__ import annotations

import logging
from typing import Any

from chisurf.plugins.core.help.api.contract import (
    METHOD_CONTRACT,
    METHOD_LIST_DOCS,
    METHOD_READ_DOC,
    METHOD_SAVE_DOC,
    METHOD_SEARCH_DOCS,
    contract_descriptor,
    service_error,
    service_success,
)
from chisurf.plugins.core.help.api.io import (
    discover_docs,
    read_doc,
    save_doc,
    search_docs,
)
from chisurf.plugins.core.help.api.markdown import extract_title, render_markdown

_log = logging.getLogger(__name__)


def _list_docs_handler() -> dict:
    try:
        info = discover_docs()
        results = [
            {
                "path": e.path,
                "title": e.title,
                "category": e.category,
                "file_name": e.file_name,
                "size": e.size,
            }
            for e in info.entries
        ]
        return service_success({"entries": results, "tree": info.tree})
    except Exception as exc:
        _log.exception("help.docs.list failed")
        return service_error(str(exc), "LIST_FAILED")


def _read_doc_handler(params: dict) -> dict:
    path = params.get("path", "")
    if not path:
        return service_error("path is required", "INVALID_PARAMS")
    try:
        content = read_doc(path)
        if content is None:
            return service_error(f"cannot read {path}", "READ_FAILED")
        html = render_markdown(content)
        title = extract_title(content) or path
        return service_success(
            {"content": content, "html": html, "title": title}
        )
    except Exception as exc:
        _log.exception("help.docs.read failed")
        return service_error(str(exc), "READ_FAILED")


def _save_doc_handler(params: dict) -> dict:
    path = params.get("path", "")
    content = params.get("content", "")
    if not path or content is None:
        return service_error("path and content are required", "INVALID_PARAMS")
    try:
        ok = save_doc(path, content)
        if not ok:
            return service_error(f"cannot save {path}", "SAVE_FAILED")
        return service_success({"path": path})
    except Exception as exc:
        _log.exception("help.docs.save failed")
        return service_error(str(exc), "SAVE_FAILED")


def _search_docs_handler(params: dict) -> dict:
    query = params.get("query", "")
    if not query:
        return service_error("query is required", "INVALID_PARAMS")
    try:
        results = search_docs(query)
        return service_success(results)
    except Exception as exc:
        _log.exception("help.docs.search failed")
        return service_error(str(exc), "SEARCH_FAILED")


def _contract_handler() -> dict:
    return service_success(contract_descriptor())


def register_services(dispatcher: Any) -> None:
    """Register all Help plugin RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher instance.

    """
    dispatcher.register(METHOD_LIST_DOCS, lambda params: _list_docs_handler())
    dispatcher.register(METHOD_READ_DOC, _read_doc_handler)
    dispatcher.register(METHOD_SAVE_DOC, _save_doc_handler)
    dispatcher.register(METHOD_SEARCH_DOCS, _search_docs_handler)
    dispatcher.register(METHOD_CONTRACT, lambda params: _contract_handler())

    # Legacy aliases for backward compatibility
    dispatcher.register("help.list_docs", lambda params: _list_docs_handler())
    dispatcher.register("help.read_doc", _read_doc_handler)
    dispatcher.register("help.save_doc", _save_doc_handler)

    _log.info("Registered Help plugin services")
