"""Abstract MFDB client contract for repository and remote implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from os import PathLike
from typing import Any


class MFDBClientBase(ABC):
    """Minimal MFDB client interface used by plugins and registry helpers.

    Implementations may be local SQLite repositories, RPC clients, or test
    doubles. The interface intentionally exposes MFDB operations, not SQLite
    internals, so plugin code does not depend on table layout details.
    """

    @abstractmethod
    def transaction(self) -> AbstractContextManager[Any]:
        """Return a transaction context manager for grouped MFDB writes."""

    @abstractmethod
    def put_object(
        self,
        path: str | PathLike | None = None,
        data: bytes | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_by_user_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Store bytes or a file in the MFDB object store."""

    @abstractmethod
    def get_object(self, object_uuid: str) -> bytes:
        """Return object-store bytes for an object UUID."""

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Return an artifact by ID, or ``None`` when it does not exist."""

    @abstractmethod
    def register_artifact(self, artifact_id: str, **kwargs: Any) -> str:
        """Register or update an artifact row."""

    @abstractmethod
    def record_operation(self, operation_id: str, operation_type: str, **kwargs: Any) -> str:
        """Register or update an MFDB operation."""

    @abstractmethod
    def record_operation_link(
        self,
        operation_id: str,
        artifact_id: str,
        direction: str,
        **kwargs: Any,
    ) -> None:
        """Link an artifact to an operation as input or output."""

    @abstractmethod
    def record_parameter(self, parameter_uuid: str, operation_id: str, name: str, **kwargs: Any) -> str:
        """Register or update a scalar operation parameter."""

    @abstractmethod
    def add_edge(
        self,
        source_node_type: str,
        source_node_id: str,
        target_node_type: str,
        target_node_id: str,
        relationship_type: str,
        **kwargs: Any,
    ) -> None:
        """Add a canonical provenance edge between MFDB nodes."""

    @abstractmethod
    def sample_exists(self, sample_id: str) -> bool:
        """Return whether an active sample exists."""

    @abstractmethod
    def link_artifact_to_sample(self, artifact_id: str, sample_id: str) -> None:
        """Create an idempotent measured-sample edge for an artifact."""

    @abstractmethod
    def lookup_sample_by_md5(self, content_md5: str) -> str | None:
        """Return the sample associated with an object-store content MD5."""

    @abstractmethod
    def set_object_sample_id(self, object_uuid: str, sample_id: str | None) -> None:
        """Store or clear the sample associated with an object-store entry."""

    @abstractmethod
    def find_raw_artifact_by_md5(self, content_md5: str) -> str:
        """Return a raw-measurement artifact ID for content MD5, if present."""

    @abstractmethod
    def save_setup(
        self,
        setup_id: str,
        name: str,
        created_by_user_id: str | None = None,
        is_public: bool | int | None = None,
        **kwargs: Any,
    ) -> None:
        """Store or update an instrument/setup definition snapshot."""

    @abstractmethod
    def get_setup(self, setup_id: str) -> dict[str, Any] | None:
        """Return a setup definition by ID, or ``None`` when missing."""

    @abstractmethod
    def list_setups(self) -> list[dict[str, Any]]:
        """Return active setup definitions."""

    @abstractmethod
    def delete_setup(self, setup_id: str) -> None:
        """Mark a setup definition as deleted."""
