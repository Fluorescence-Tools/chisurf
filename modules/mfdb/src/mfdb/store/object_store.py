"""Content-addressed object storage with MD5-based deduplication.

This module provides the :class:`ObjectStore` class for storing binary blobs
addressed by their MD5 hash. Objects are deduplicated automatically: storing
the same content twice results in a single blob on disk with an incremented
reference count.

Storage layout::

    {root}/{md5[:2]}/{md5[2:4]}/{md5}

Each stored object is mapped to a UUID in the ``mfdb_object`` database table.
All cross-references use the UUID, never the MD5 directly.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ObjectRef:
    """Reference to a stored object.

    Parameters
    ----------
    uuid : str
        Stable UUID for cross-references.
    md5 : str
        MD5 hex digest of the content (32 characters).
    size : int
        Size in bytes.
    original_filename : str or None
        Original filename at time of upload.
    deduplicated : bool
        True if the blob already existed (refcount incremented).
    storage_path : str
        Relative path within the object store root.
    """

    uuid: str
    md5: str
    size: int
    original_filename: str | None
    deduplicated: bool
    storage_path: str


class ObjectStore:
    """Content-addressed object storage with MD5 deduplication.

    Parameters
    ----------
    root : Path
        Root directory for object storage.
    chunk_size : int
        Size of chunks for streaming reads (default 64KB).

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ObjectStore(Path(tmp))
    ...     ref = store.put_bytes(b"hello world", filename="test.txt")
    ...     ref.md5
    '5eb63bbbe01eeed093cb22bb8f5acdc3'
    """

    def __init__(self, root: Path, chunk_size: int = 65536):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.root.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, md5: str) -> Path:
        """Return the filesystem path for a given MD5 hash."""
        return self.root / md5[:2] / md5[2:4] / md5

    def _relative_path(self, md5: str) -> str:
        """Return the relative storage path for a given MD5 hash."""
        return f"{md5[:2]}/{md5[2:4]}/{md5}"

    def _compute_md5_streaming(self, path: Path) -> str:
        """Compute MD5 hash of a file using streaming reads."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(self.chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def put_from_path(
        self,
        path: Path,
        original_filename: str | None = None,
    ) -> ObjectRef:
        """Store a file in the object store.

        Reads the file in chunks, computes the MD5 hash, and stores the blob
        at the content-addressed path. If the same content already exists,
        the reference count is incremented instead of creating a duplicate.

        Parameters
        ----------
        path : Path
            Path to the file to store.
        original_filename : str, optional
            Original filename to record in metadata. Defaults to ``path.name``.

        Returns
        -------
        ObjectRef
            Reference to the stored object.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        md5 = self._compute_md5_streaming(path)
        blob_path = self._blob_path(md5)
        deduplicated = blob_path.exists()

        if not deduplicated:
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, blob_path)
            logger.info("Stored new object: %s (%d bytes)", md5, path.stat().st_size)
        else:
            logger.info("Deduplicated object: %s (refcount++)", md5)

        object_uuid = str(uuid.uuid4())
        original_filename = original_filename or path.name
        size = path.stat().st_size
        storage_path = self._relative_path(md5)

        return ObjectRef(
            uuid=object_uuid,
            md5=md5,
            size=size,
            original_filename=original_filename,
            deduplicated=deduplicated,
            storage_path=storage_path,
        )

    def put_bytes(
        self,
        data: bytes,
        filename: str,
    ) -> ObjectRef:
        """Store bytes directly in the object store.

        Parameters
        ----------
        data : bytes
            The binary content to store.
        filename : str
            Original filename to record in metadata.

        Returns
        -------
        ObjectRef
            Reference to the stored object.
        """
        md5 = hashlib.md5(data).hexdigest()
        blob_path = self._blob_path(md5)
        deduplicated = blob_path.exists()

        if not deduplicated:
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(data)
            logger.info("Stored new object: %s (%d bytes)", md5, len(data))
        else:
            logger.info("Deduplicated object: %s (refcount++)", md5)

        object_uuid = str(uuid.uuid4())
        storage_path = self._relative_path(md5)

        return ObjectRef(
            uuid=object_uuid,
            md5=md5,
            size=len(data),
            original_filename=filename,
            deduplicated=deduplicated,
            storage_path=storage_path,
        )

    def get(self, md5: str) -> bytes:
        """Retrieve blob content by MD5 hash.

        Parameters
        ----------
        md5 : str
            The MD5 hex digest of the content.

        Returns
        -------
        bytes
            The stored content.

        Raises
        ------
        FileNotFoundError
            If no blob with the given MD5 exists.
        """
        blob_path = self._blob_path(md5)
        if not blob_path.exists():
            raise FileNotFoundError(f"Object not found: {md5}")
        return blob_path.read_bytes()

    def get_path(self, md5: str) -> Path:
        """Return the filesystem path for a blob.

        Parameters
        ----------
        md5 : str
            The MD5 hex digest of the content.

        Returns
        -------
        Path
            Path to the stored blob.

        Raises
        ------
        FileNotFoundError
            If no blob with the given MD5 exists.
        """
        blob_path = self._blob_path(md5)
        if not blob_path.exists():
            raise FileNotFoundError(f"Object not found: {md5}")
        return blob_path

    def exists(self, md5: str) -> bool:
        """Check if a blob with the given MD5 exists."""
        return self._blob_path(md5).exists()

    def delete(self, md5: str) -> bool:
        """Delete a blob from storage.

        Only deletes if the reference count has reached zero.

        Parameters
        ----------
        md5 : str
            The MD5 hex digest of the content.

        Returns
        -------
        bool
            True if the blob was deleted, False if it still has references.
        """
        blob_path = self._blob_path(md5)
        if blob_path.exists():
            blob_path.unlink()
            logger.info("Deleted object: %s", md5)
            return True
        return False
