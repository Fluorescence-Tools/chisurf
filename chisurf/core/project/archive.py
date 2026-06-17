from __future__ import annotations

import io
import pathlib
import shutil
import tempfile
import time
import zipfile
from pathlib import PurePosixPath
from typing import Any

PathLike = str | pathlib.Path

PROJECT_JSON = "project.json"
HISTORY_FILENAME = "history.jsonl"
SESSION_FILENAME = "session.jsonl"
DATA_DIR = "data"
PROJECT_ARCHIVE_SUFFIX = ".csp"

# V5 MFDB artifact layer
MFDB_DIR = "mfdb"
MFDB_MANIFEST_JSON = "mfdb/manifest.json"
MFDB_PROVENANCE_JSONL = "mfdb/provenance.jsonl"
MFDB_OBJECTS_DIR = "mfdb/objects"
EXPORT_JSON = "export.json"


class ProjectArchive:
    """In-memory ZIP archive for ChiSurf project files.

    Parameters
    ----------
    compression : int, optional
        Compression method used for new entries.

    Notes
    -----
    The archive is backed by an in-memory ``BytesIO`` buffer. This lets the
    application read ``project.json`` from a ``.csp`` file without extracting
    the whole project to disk.
    """

    def __init__(self, compression: int = zipfile.ZIP_DEFLATED) -> None:
        self._buffer = io.BytesIO()
        self._compression = compression
        self._zip = zipfile.ZipFile(
            self._buffer,
            "w",
            compression=compression,
            allowZip64=True,
        )
        self._path: pathlib.Path | None = None
        self._written: set[str] = set()

    @classmethod
    def open_bytes(cls, data: bytes) -> ProjectArchive:
        """Open an existing ``.csp`` archive from raw bytes.

        Parameters
        ----------
        data : bytes
            Raw archive bytes.

        Returns
        -------
        ProjectArchive
            Archive opened in read mode.

        Raises
        ------
        ValueError
            If the data is not a valid ZIP archive or does not contain
            ``project.json``.
        """
        archive = cls.__new__(cls)
        archive._buffer = io.BytesIO(data)
        archive._compression = zipfile.ZIP_DEFLATED
        archive._zip = zipfile.ZipFile(archive._buffer, "r")
        archive._path = None
        archive._written = set()
        if not archive.has_entry(PROJECT_JSON):
            archive.close()
            raise ValueError(f"Project archive does not contain {PROJECT_JSON!r}")
        return archive

    @classmethod
    def open(cls, path: PathLike) -> ProjectArchive:
        """Open an existing ``.csp`` archive into memory.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the archive file.

        Returns
        -------
        ProjectArchive
            Archive opened in read mode.

        Raises
        ------
        ValueError
            If the file is not a valid ZIP archive or does not contain
            ``project.json``.
        """
        archive_path = pathlib.Path(path)
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Project archive is not a valid ZIP file: {archive_path}")

        archive = cls.__new__(cls)
        archive._buffer = io.BytesIO(archive_path.read_bytes())
        archive._compression = zipfile.ZIP_DEFLATED
        archive._zip = zipfile.ZipFile(archive._buffer, "r")
        archive._path = archive_path
        archive._written = set()
        if not archive.has_entry(PROJECT_JSON):
            archive.close()
            raise ValueError(f"Project archive does not contain {PROJECT_JSON!r}: {archive_path}")
        return archive

    @classmethod
    def is_project_archive(cls, path: PathLike) -> bool:
        """Return whether a path points to a ChiSurf project archive.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to test.

        Returns
        -------
        bool
            True if the path is a readable ``.csp`` archive with
            ``project.json``.
        """
        try:
            archive = cls.open(path)
        except Exception:
            return False
        archive.close()
        return True

    def write_text(self, name: str, text: str, overwrite: bool = False) -> None:
        """Write a UTF-8 text entry to the archive.

        Parameters
        ----------
        name : str
            Archive-internal entry name.
        text : str
            Text content.
        overwrite : bool
            If True, overwrite an existing entry with the same name.

        Raises
        ------
        ValueError
            If the entry already exists (and *overwrite* is False) or the name is unsafe.
        """
        self.write_bytes(name, text.encode("utf-8"), overwrite=overwrite)

    def write_bytes(self, name: str, data: bytes | bytearray | memoryview, overwrite: bool = False) -> None:
        """Write a binary entry to the archive.

        Parameters
        ----------
        name : str
            Archive-internal entry name.
        data : bytes-like
            Binary content.
        overwrite : bool
            If True, overwrite an existing entry with the same name.

        Raises
        ------
        ValueError
            If the entry already exists (and *overwrite* is False) or the name is unsafe.
        """
        archive_name = self._normalize_name(name)
        if archive_name in self._written:
            if not overwrite:
                raise ValueError(f"Archive entry already exists: {archive_name}")
            self._zip.writestr(archive_name, bytes(data))
        else:
            self._zip.writestr(archive_name, bytes(data))
            self._written.add(archive_name)

    def write_file(self, name: str, path: PathLike, overwrite: bool = False) -> None:
        """Write a file from disk into the archive.

        Parameters
        ----------
        name : str
            Archive-internal entry name.
        path : str or pathlib.Path
            Source file path.
        overwrite : bool
            If True, overwrite an existing entry with the same name.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        ValueError
            If the entry already exists (and *overwrite* is False) or the archive name is unsafe.
        """
        source_path = pathlib.Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(str(source_path))

        archive_name = self._normalize_name(name)
        if archive_name in self._written:
            if not overwrite:
                raise ValueError(f"Archive entry already exists: {archive_name}")
        else:
            self._written.add(archive_name)

        stat = source_path.stat()
        info = zipfile.ZipInfo(archive_name, date_time=time.localtime(stat.st_mtime)[:6])
        info.compress_type = self._compression
        info.external_attr = 0o644 << 16
        with self._zip.open(info, "w") as dst, source_path.open("rb") as src:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    def save(self, path: PathLike) -> pathlib.Path:
        """Finalize the archive and write it to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination ``.csp`` path.

        Returns
        -------
        pathlib.Path
            Destination path.
        """
        archive_path = pathlib.Path(path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        self._zip.close()
        archive_path.write_bytes(self._buffer.getvalue())
        self._path = archive_path
        return archive_path

    def to_bytes(self) -> bytes:
        """Finalize the archive and return its bytes.

        Returns
        -------
        bytes
            Complete ``.csp`` archive bytes.
        """
        self._zip.close()
        return self._buffer.getvalue()

    def read_text(self, name: str) -> str:
        """Read a UTF-8 text entry from the archive.

        Parameters
        ----------
        name : str
            Archive-internal entry name.

        Returns
        -------
        str
            Entry text.
        """
        return self.read_bytes(name).decode("utf-8")

    def read_bytes(self, name: str) -> bytes:
        """Read a binary entry from the archive.

        Parameters
        ----------
        name : str
            Archive-internal entry name.

        Returns
        -------
        bytes
            Entry bytes.
        """
        return self._zip.read(self._normalize_name(name))

    def has_entry(self, name: str) -> bool:
        """Return whether an entry exists in the archive.

        Parameters
        ----------
        name : str
            Archive-internal entry name.

        Returns
        -------
        bool
            True if the entry exists.
        """
        try:
            self._zip.getinfo(self._normalize_name(name))
        except KeyError:
            return False
        return True

    def getinfo(self, name: str) -> zipfile.ZipInfo:
        """Return metadata for an archive entry.

        Parameters
        ----------
        name : str
            Archive-internal entry name.

        Returns
        -------
        zipfile.ZipInfo
            Entry metadata.
        """
        return self._zip.getinfo(self._normalize_name(name))

    def list_entries(self) -> list[str]:
        """Return archive entry names.

        Returns
        -------
        list of str
            Entry names in archive order.
        """
        return list(self._zip.namelist())

    def extract_to(self, target_dir: PathLike) -> pathlib.Path:
        """Extract all entries to a directory.

        Parameters
        ----------
        target_dir : str or pathlib.Path
            Destination directory.

        Returns
        -------
        pathlib.Path
            Destination directory.
        """
        destination = pathlib.Path(target_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in self.list_entries():
            self.extract_entry_to(name, destination)
        return destination

    def extract_to_temp(self) -> tuple[pathlib.Path, tempfile.TemporaryDirectory]:
        """Extract all entries to a temporary directory.

        Returns
        -------
        tuple of pathlib.Path and tempfile.TemporaryDirectory
            Temporary directory path and its handle. Keep the handle alive until
            the extracted files are no longer needed.
        """
        temp_dir = tempfile.TemporaryDirectory(prefix="chisurf_project_")
        self.extract_to(temp_dir.name)
        return pathlib.Path(temp_dir.name), temp_dir

    def extract_entry_to(self, name: str, target_dir: PathLike) -> pathlib.Path:
        """Extract one entry to a directory.

        Parameters
        ----------
        name : str
            Archive-internal entry name.
        target_dir : str or pathlib.Path
            Destination directory.

        Returns
        -------
        pathlib.Path
            Extracted file path.
        """
        destination = self._safe_destination(target_dir, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.read_bytes(name))
        return destination

    def close(self) -> None:
        """Close the underlying ZIP file."""
        self._zip.close()

    def write_mfdb_layer(
        self,
        manifest: dict[str, Any],
        provenance_lines: list[str],
        object_blobs: dict[str, bytes] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Write the V5 MFDB artifact layer to the archive.

        Parameters
        ----------
        manifest : dict
            Artifact registry (``mfdb/manifest.json`` content).
        provenance_lines : list of str
            JSONL lines for ``mfdb/provenance.jsonl``.
        object_blobs : dict, optional
            Mapping of ``{object_uuid: blob_bytes}`` to write under
            ``mfdb/objects/``.
        overwrite : bool, default=False
            If True, overwrite existing entries.
        """
        import json as _json

        self.write_text(
            MFDB_MANIFEST_JSON,
            _json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2),
            overwrite=overwrite,
        )
        self.write_text(
            MFDB_PROVENANCE_JSONL,
            "\n".join(provenance_lines) + "\n" if provenance_lines else "",
            overwrite=overwrite,
        )
        if object_blobs:
            for obj_uuid, blob in object_blobs.items():
                entry_name = f"{MFDB_OBJECTS_DIR}/{obj_uuid}.blob"
                self.write_bytes(entry_name, blob, overwrite=overwrite)

    def __enter__(self) -> ProjectArchive:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _normalize_name(self, name: str) -> str:
        archive_name = name.replace("\\", "/")
        if archive_name.startswith("/"):
            archive_name = archive_name.lstrip("/")
        path = PurePosixPath(archive_name)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            raise ValueError(f"Unsafe archive entry name: {name!r}")
        if not path.parts:
            raise ValueError(f"Empty archive entry name: {name!r}")
        return path.as_posix()

    def _safe_destination(self, target_dir: PathLike, name: str) -> pathlib.Path:
        root = pathlib.Path(target_dir).resolve()
        destination = (root / self._normalize_name(name)).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Unsafe archive extraction path: {name!r}")
        return destination
