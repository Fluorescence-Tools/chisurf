from __future__ import annotations

from chisurf.plugins.core.code_editor.backend import services
from chisurf.plugins.core.code_editor.document_store import DocumentStore


class Dispatcher:
    """Tiny dispatcher stand-in for register_services tests."""

    def __init__(self) -> None:
        self.handlers = {}

    def register(self, name, handler) -> None:
        """Record a handler by RPC method name."""
        self.handlers[name] = handler


def _reset_store() -> None:
    services.STORE = DocumentStore()


def test_editor_services_list_get_set_and_event() -> None:
    """Editor RPC services expose document snapshots."""
    _reset_store()
    doc_id = "path:/tmp/sample.py"
    services.STORE.upsert(
        document_id=doc_id,
        path="/tmp/sample.py",
        name="sample.py",
        language="Python",
        content="x = 1\n",
    )

    listed = services.list_documents()
    assert listed["ok"] is True
    assert listed["documents"][0]["document_id"] == doc_id

    result = services.set_document(document_id=doc_id, content="x = 2\n", expected_revision=1)
    assert result["ok"] is True
    assert result["event"]["topic"] == "editor.document.changed"

    fetched = services.get_document(document_id=doc_id)
    assert fetched["document"]["content"] == "x = 2\n"


def test_editor_services_reject_stale_revision() -> None:
    """Editor RPC services preserve optimistic concurrency."""
    _reset_store()
    doc_id = "path:/tmp/sample.py"
    services.STORE.upsert(
        document_id=doc_id,
        path="/tmp/sample.py",
        name="sample.py",
        language="Python",
        content="x = 1\n",
    )

    result = services.set_document(document_id=doc_id, content="x = 2\n", expected_revision=0)

    assert result["ok"] is False
    assert result["error_code"] == "INVALID_INPUT"


def test_editor_services_apply_edits() -> None:
    """Editor RPC services can apply range edits."""
    _reset_store()
    doc_id = "path:/tmp/sample.py"
    services.STORE.upsert(
        document_id=doc_id,
        path="/tmp/sample.py",
        name="sample.py",
        language="Python",
        content="alpha = 1\n",
    )

    result = services.apply_document_edits(
        document_id=doc_id,
        edits=[
            {"start_line": 1, "start_column": 0, "end_line": 1, "end_column": 5, "text": "ALPHA"}
        ],
    )

    assert result["ok"] is True
    assert services.get_document(document_id=doc_id)["document"]["content"] == "ALPHA = 1\n"


def test_dispatch_can_use_explicit_document_store() -> None:
    """RPC dispatch can be bound to the GUI-owned document store."""
    store = DocumentStore()
    doc_id = "path:/tmp/gui-owned.py"
    store.upsert(
        document_id=doc_id,
        path="/tmp/gui-owned.py",
        name="gui-owned.py",
        language="Python",
        content="x = 1\n",
    )

    result = services.dispatch(
        "editor.document.set",
        {"document_id": doc_id, "content": "x = 2\n", "expected_revision": 1},
        store=store,
    )

    assert result["ok"] is True
    assert store.get(document_id=doc_id).content == "x = 2\n"
    assert services.STORE.get(document_id=doc_id) is None


def test_register_services_wraps_params_dict() -> None:
    """Registered editor services accept the dispatcher params dictionary."""
    _reset_store()
    doc_id = "path:/tmp/registered.py"
    services.STORE.upsert(
        document_id=doc_id,
        path="/tmp/registered.py",
        name="registered.py",
        language="Python",
        content="x = 1\n",
    )
    dispatcher = Dispatcher()

    services.register_services(dispatcher)
    result = dispatcher.handlers["editor.document.get"]({"document_id": doc_id})

    assert result["ok"] is True
    assert result["document"]["content"] == "x = 1\n"
