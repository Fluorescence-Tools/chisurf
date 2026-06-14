from __future__ import annotations

from typing import Any

from chisurf.plugins.core.code_editor.backend import services
from chisurf.server.session import SessionState


def list_documents(state: SessionState) -> Any:
    """List open editor documents."""
    return services.list_documents()


def get_document(
    state: SessionState,
    document_id: str | None = None,
    path: str | None = None,
    include_content: bool = True,
) -> Any:
    """Get an open editor document."""
    return services.get_document(
        document_id=document_id,
        path=path,
        include_content=include_content,
    )


def set_document(
    state: SessionState,
    document_id: str | None = None,
    path: str | None = None,
    content: str | None = None,
    expected_revision: int | None = None,
    source: str = "agent",
) -> Any:
    """Replace an open editor document."""
    return services.set_document(
        document_id=document_id,
        path=path,
        content=content,
        expected_revision=expected_revision,
        source=source,
    )


def apply_document_edits(
    state: SessionState,
    document_id: str | None = None,
    path: str | None = None,
    edits: list[dict] | None = None,
    expected_revision: int | None = None,
    source: str = "agent",
) -> Any:
    """Apply text edits to an open editor document."""
    return services.apply_document_edits(
        document_id=document_id,
        path=path,
        edits=edits,
        expected_revision=expected_revision,
        source=source,
    )


def ruff_check(
    state: SessionState,
    document_id: str | None = None,
    path: str | None = None,
    content: str | None = None,
    extra_args: list[str] | None = None,
    timeout_ms: int | None = None,
) -> Any:
    """Run Ruff on an open editor document."""
    return services.ruff_check(
        document_id=document_id,
        path=path,
        content=content,
        extra_args=extra_args,
        timeout_ms=timeout_ms,
    )


def ruff_fix(
    state: SessionState,
    document_id: str | None = None,
    path: str | None = None,
    expected_revision: int | None = None,
    apply_to_document: bool = True,
    extra_args: list[str] | None = None,
    timeout_ms: int | None = None,
) -> Any:
    """Run Ruff fixes on an open editor document."""
    return services.ruff_fix(
        document_id=document_id,
        path=path,
        expected_revision=expected_revision,
        apply_to_document=apply_to_document,
        extra_args=extra_args,
        timeout_ms=timeout_ms,
    )
