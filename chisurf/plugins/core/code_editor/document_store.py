from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentSnapshot:
    """JSON-safe snapshot of an open editor document."""

    document_id: str
    path: str
    name: str
    language: str
    content: str
    revision: int
    modified: bool

    def to_dict(self, include_content: bool = True, active_id: str | None = None) -> dict[str, Any]:
        """Return the snapshot as a JSON-safe dictionary."""
        data: dict[str, Any] = {
            "document_id": self.document_id,
            "path": self.path,
            "name": self.name,
            "language": self.language,
            "revision": self.revision,
            "modified": self.modified,
            "active": self.document_id == active_id if active_id is not None else False,
        }
        if include_content:
            data["content"] = self.content
        return data


class DocumentStore:
    """Thread-safe registry of open editor documents."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._documents: dict[str, DocumentSnapshot] = {}
        self.active_document_id: str | None = None

    @staticmethod
    def document_id_for_path(path: str | Path) -> str:
        """Return the stable document id for a filesystem path."""
        return f"path:{Path(path).resolve().as_posix()}"

    def upsert(
        self,
        *,
        document_id: str,
        path: str,
        name: str,
        language: str,
        content: str,
        modified: bool = False,
    ) -> DocumentSnapshot:
        """Insert or update a document snapshot."""
        with self._lock:
            current = self._documents.get(document_id)
            revision = 0 if current is None else current.revision
            if current is None or current.content != content or current.path != path:
                revision += 1
            snapshot = DocumentSnapshot(
                document_id=document_id,
                path=path,
                name=name,
                language=language,
                content=content,
                revision=revision,
                modified=modified,
            )
            self._documents[document_id] = snapshot
            return snapshot

    def remove(self, document_id: str) -> None:
        """Remove a document from the store."""
        with self._lock:
            self._documents.pop(document_id, None)

    def get(
        self,
        document_id: str | None = None,
        path: str | None = None,
        include_content: bool = True,
    ) -> DocumentSnapshot | None:
        """Return a document snapshot by id or path."""
        with self._lock:
            if document_id is not None:
                return self._documents.get(document_id)
            if path is not None:
                resolved_id = self.document_id_for_path(path)
                return self._documents.get(resolved_id)
        return None

    def list_documents(self) -> list[dict[str, Any]]:
        """Return summaries for all open documents."""
        with self._lock:
            return [
                doc.to_dict(include_content=False, active_id=self.active_document_id)
                for doc in sorted(
                    self._documents.values(),
                    key=lambda item: item.name.lower(),
                )
            ]

    def replace(
        self,
        document_id: str,
        content: str,
        expected_revision: int | None = None,
    ) -> DocumentSnapshot:
        """Replace a document's full text."""
        with self._lock:
            current = self._get_locked(document_id)
            self._check_revision(current, expected_revision)
            snapshot = DocumentSnapshot(
                document_id=current.document_id,
                path=current.path,
                name=current.name,
                language=current.language,
                content=content,
                revision=current.revision + 1,
                modified=True,
            )
            self._documents[document_id] = snapshot
            return snapshot

    def apply_edits(
        self,
        document_id: str,
        edits: list[dict[str, Any]],
        expected_revision: int | None = None,
    ) -> DocumentSnapshot:
        """Apply text-range edits to a document."""
        with self._lock:
            current = self._get_locked(document_id)
            self._check_revision(current, expected_revision)
            content = self._apply_edits_to_text(current.content, edits)
            snapshot = DocumentSnapshot(
                document_id=current.document_id,
                path=current.path,
                name=current.name,
                language=current.language,
                content=content,
                revision=current.revision + 1,
                modified=True,
            )
            self._documents[document_id] = snapshot
            return snapshot

    def _get_locked(self, document_id: str) -> DocumentSnapshot:
        current = self._documents.get(document_id)
        if current is None:
            raise KeyError(f"document '{document_id}' not found")
        return current

    @staticmethod
    def _check_revision(current: DocumentSnapshot, expected_revision: int | None) -> None:
        if expected_revision is not None and current.revision != expected_revision:
            raise ValueError(
                f"document revision mismatch: expected {expected_revision}, got {current.revision}"
            )

    @staticmethod
    def _apply_edits_to_text(content: str, edits: list[dict[str, Any]]) -> str:
        if not edits:
            return content

        offsets: list[tuple[int, int, str]] = []
        for edit in edits:
            start_line = int(edit.get("start_line", 1))
            start_column = int(edit.get("start_column", 0))
            end_line = int(edit.get("end_line", start_line))
            end_column = int(edit.get("end_column", start_column))
            text = str(edit.get("text", ""))
            offsets.append(
                (
                    DocumentStore._position_to_offset(content, start_line, start_column),
                    DocumentStore._position_to_offset(content, end_line, end_column),
                    text,
                )
            )

        for start, end, text in sorted(offsets, reverse=True):
            content = content[:start] + text + content[end:]
        return content

    @staticmethod
    def _position_to_offset(content: str, line: int, column: int) -> int:
        if line < 1:
            raise ValueError("line numbers are one-based")
        lines = content.splitlines(keepends=True)
        if line > len(lines):
            return len(content)
        block = lines[line - 1]
        newline_len = 1 if block.endswith("\r\n") else 0
        line_end = len(block) - newline_len
        return sum(len(item) for item in lines[: line - 1]) + max(0, min(column, line_end))
