from __future__ import annotations

import pytest

from chisurf.plugins.core.code_editor.document_store import DocumentStore


def test_document_store_lists_gets_and_replaces() -> None:
    """The document store keeps JSON-safe editor snapshots."""
    store = DocumentStore()
    doc_id = "path:/tmp/sample.py"

    store.upsert(
        document_id=doc_id,
        path="/tmp/sample.py",
        name="sample.py",
        language="Python",
        content="x = 1\n",
    )

    doc = store.get(document_id=doc_id)
    assert doc is not None
    assert doc.revision == 1
    assert store.list_documents()[0]["name"] == "sample.py"

    replacement = store.replace(doc_id, "x = 2\n", expected_revision=1)
    assert replacement.revision == 2
    assert store.get(document_id=doc_id).content == "x = 2\n"


def test_document_store_rejects_stale_revision() -> None:
    """A stale expected_revision prevents overwriting editor content."""
    store = DocumentStore()
    doc_id = "path:/tmp/sample.py"
    store.upsert(
        document_id=doc_id,
        path="/tmp/sample.py",
        name="sample.py",
        language="Python",
        content="x = 1\n",
    )

    with pytest.raises(ValueError):
        store.replace(doc_id, "x = 2\n", expected_revision=0)


def test_document_store_applies_one_based_line_edits() -> None:
    """Range edits use one-based lines and zero-based columns."""
    store = DocumentStore()
    doc_id = "path:/tmp/sample.py"
    store.upsert(
        document_id=doc_id,
        path="/tmp/sample.py",
        name="sample.py",
        language="Python",
        content="alpha = 1\nbeta = 2\n",
    )

    snapshot = store.apply_edits(
        doc_id,
        [
            {
                "start_line": 1,
                "start_column": 0,
                "end_line": 1,
                "end_column": 5,
                "text": "ALPHA",
            }
        ],
    )

    assert snapshot.content == "ALPHA = 1\nbeta = 2\n"
    assert snapshot.revision == 2
