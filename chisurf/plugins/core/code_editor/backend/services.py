from __future__ import annotations

from chisurf.plugins.core.code_editor.document_store import DocumentStore
from chisurf.plugins.core.code_editor.ruff_runner import RuffRunner
from chisurf.server.services import (
    INVALID_INPUT,
    NOT_FOUND,
    OPERATION_FAILED,
    ServiceResult,
    service_error,
)

STORE = DocumentStore()
RUNNER = RuffRunner()


def register_services(dispatcher: object) -> None:
    """Register editor document RPC methods with a dispatcher."""
    for name, handler in SERVICE_METHODS.items():
        dispatcher.register(name, lambda params, _handler=handler: _handler(**params))


def dispatch(
    method: str,
    params: dict | None = None,
    store: DocumentStore | None = None,
    runner: RuffRunner | None = None,
) -> ServiceResult:
    """Dispatch an editor document RPC method."""
    params = params or {}
    handler = SERVICE_METHODS.get(method)
    if handler is None:
        return service_error(
            f"method '{method}' not found",
            error_code=NOT_FOUND,
        )
    try:
        return handler(**params, _store=store, _runner=runner)
    except KeyError as exc:
        return service_error(str(exc), error_code=NOT_FOUND)
    except ValueError as exc:
        return service_error(str(exc), error_code=INVALID_INPUT)
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED)


def list_documents(
    _store: DocumentStore | None = None,
    _runner: RuffRunner | None = None,
) -> ServiceResult:
    """Return all open editor documents."""
    return {"ok": True, "documents": _resolve_store(_store).list_documents()}


def get_document(
    document_id: str | None = None,
    path: str | None = None,
    include_content: bool = True,
    _store: DocumentStore | None = None,
    _runner: RuffRunner | None = None,
) -> ServiceResult:
    """Return an open editor document."""
    store = _resolve_store(_store)
    doc = store.get(document_id=document_id, path=path, include_content=include_content)
    if doc is None:
        return service_error("document not found", error_code=NOT_FOUND)
    return {
        "ok": True,
        "document": doc.to_dict(
            include_content=include_content,
            active_id=store.active_document_id,
        ),
    }


def set_document(
    document_id: str | None = None,
    path: str | None = None,
    content: str | None = None,
    expected_revision: int | None = None,
    source: str = "agent",
    _store: DocumentStore | None = None,
    _runner: RuffRunner | None = None,
) -> ServiceResult:
    """Replace an open editor document."""
    if content is None:
        return service_error("content is required", error_code=INVALID_INPUT)
    store = _resolve_store(_store)
    resolved = _resolve_document(document_id, path, store)
    if "error" in resolved:
        return resolved
    try:
        snapshot = store.replace(resolved["document_id"], content, expected_revision)
        return _result_with_event(snapshot, "set", source)
    except ValueError as exc:
        return service_error(str(exc), error_code=INVALID_INPUT)


def apply_document_edits(
    document_id: str | None = None,
    path: str | None = None,
    edits: list[dict] | None = None,
    expected_revision: int | None = None,
    source: str = "agent",
    _store: DocumentStore | None = None,
    _runner: RuffRunner | None = None,
) -> ServiceResult:
    """Apply text edits to an open editor document."""
    if not isinstance(edits, list):
        return service_error("edits must be a list", error_code=INVALID_INPUT)
    store = _resolve_store(_store)
    resolved = _resolve_document(document_id, path, store)
    if "error" in resolved:
        return resolved
    try:
        snapshot = store.apply_edits(resolved["document_id"], edits, expected_revision)
        return _result_with_event(snapshot, "apply_edits", source)
    except ValueError as exc:
        return service_error(str(exc), error_code=INVALID_INPUT)


def ruff_check(
    document_id: str | None = None,
    path: str | None = None,
    content: str | None = None,
    extra_args: list[str] | None = None,
    timeout_ms: int | None = None,
    _store: DocumentStore | None = None,
    _runner: RuffRunner | None = None,
) -> ServiceResult:
    """Run Ruff on an open editor document."""
    resolved = _resolve_document(document_id, path, _resolve_store(_store))
    if "error" in resolved:
        return resolved
    doc = resolved["document"]
    check_path = path or doc["path"]
    check_content = content if content is not None else doc.get("content")
    result = _resolve_runner(_runner).check(check_path, check_content, extra_args, timeout_ms)
    result["document_id"] = doc["document_id"]
    result["revision"] = doc["revision"]
    return result


def ruff_fix(
    document_id: str | None = None,
    path: str | None = None,
    expected_revision: int | None = None,
    apply_to_document: bool = True,
    extra_args: list[str] | None = None,
    timeout_ms: int | None = None,
    _store: DocumentStore | None = None,
    _runner: RuffRunner | None = None,
) -> ServiceResult:
    """Run Ruff fixes on an open editor document."""
    store = _resolve_store(_store)
    resolved = _resolve_document(document_id, path, store)
    if "error" in resolved:
        return resolved
    doc = resolved["document"]
    result = _resolve_runner(_runner).fix(doc["path"], doc.get("content"), extra_args, timeout_ms)
    result["document_id"] = doc["document_id"]
    result["revision"] = doc["revision"]
    if apply_to_document and result.get("fixed_content") is not None:
        try:
            snapshot = store.replace(doc["document_id"], result["fixed_content"], expected_revision)
            result["revision"] = snapshot.revision
            result["event"] = _event(snapshot, "ruff_fix", "agent")
        except ValueError as exc:
            return service_error(str(exc), error_code=INVALID_INPUT)
    return result


def _resolve_store(store: DocumentStore | None) -> DocumentStore:
    return store if store is not None else STORE


def _resolve_runner(runner: RuffRunner | None) -> RuffRunner:
    return runner if runner is not None else RUNNER


def _resolve_document(
    document_id: str | None,
    path: str | None,
    store: DocumentStore,
) -> ServiceResult:
    doc = store.get(document_id=document_id, path=path, include_content=True)
    if doc is None:
        return service_error("document not found", error_code=NOT_FOUND)
    return {
        "document_id": doc.document_id,
        "document": doc.to_dict(
            include_content=True,
            active_id=store.active_document_id,
        ),
    }


def _result_with_event(snapshot: object, action: str, source: str) -> ServiceResult:
    return {
        "ok": True,
        "document_id": snapshot.document_id,
        "revision": snapshot.revision,
        "event": _event(snapshot, action, source),
    }


def _event(snapshot: object, action: str, source: str) -> dict:
    return {
        "topic": "editor.document.changed",
        "action": action,
        "source": source,
        "document_id": snapshot.document_id,
        "path": snapshot.path,
        "revision": snapshot.revision,
        "content": snapshot.content,
    }


SERVICE_METHODS = {
    "editor.document.list": list_documents,
    "editor.document.get": get_document,
    "editor.document.set": set_document,
    "editor.document.apply_edits": apply_document_edits,
    "editor.document.ruff_check": ruff_check,
    "editor.document.ruff_fix": ruff_fix,
}
