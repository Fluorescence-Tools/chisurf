from __future__ import annotations

import base64

from mfdb.config import configure_runtime, reset_runtime_config
from mfdb import database_resolver
from mfdb.object_store import ObjectStore


def test_object_store_root_defaults_to_settings_objects(tmp_path):
    """Object store root defaults to the settings directory objects folder."""
    configure_runtime(settings_dir=tmp_path)
    try:
        assert database_resolver.object_store_root() == tmp_path / "objects"
    finally:
        reset_runtime_config()


def test_object_store_root_uses_relative_mfdb_setting(tmp_path):
    """Relative MFDB object store settings are resolved below settings."""
    configure_runtime(settings_dir=tmp_path, object_store_root="custom_objects")
    try:
        assert database_resolver.object_store_root() == tmp_path / "custom_objects"
    finally:
        reset_runtime_config()


def test_object_store_root_uses_absolute_mfdb_setting(tmp_path):
    """Absolute MFDB object store settings are used as-is."""
    custom_root = tmp_path / "external" / "objects"
    configure_runtime(settings_dir=tmp_path, object_store_root=custom_root)
    try:
        assert database_resolver.object_store_root() == custom_root
    finally:
        reset_runtime_config()


def test_object_store_deduplicates_identical_files(tmp_path):
    """Identical files are stored once under the same content-addressed path."""
    store = ObjectStore(tmp_path)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    first_ref = store.put_from_path(first)
    second_ref = store.put_from_path(second)

    assert first_ref.md5 == second_ref.md5
    assert first_ref.storage_path == second_ref.storage_path
    assert second_ref.deduplicated is True
    assert (tmp_path / first_ref.storage_path).read_bytes() == b"same content"


def test_project_archive_handler_roundtrips_csp_object(tmp_path, monkeypatch):
    """MFDB project archival stores and restores the complete CSP object."""
    from mfdb.admin.backend import measurement_services

    db_path = tmp_path / "mfdb.sqlite"
    object_root = tmp_path / "objects"
    monkeypatch.setattr(measurement_services, "resolve_database_path", lambda: db_path)
    monkeypatch.setattr(database_resolver, "object_store_root", lambda: object_root)

    archive_bytes = b"PK\x03\x04fake-csp-archive"
    payload = {
        "project_format_version": 4,
        "meta": {"name": "ProteinMC project"},
        "datasets": {},
        "fits": [],
        "ui": {},
        "extra": {},
    }

    archived = measurement_services.archive_project_handler(
        project_id="proj-proteinmc",
        project_name="ProteinMC project",
        project_payload=payload,
        project_archive_data=base64.b64encode(archive_bytes).decode("ascii"),
        project_archive_filename="proteinmc.csp",
    )

    assert archived["ok"] is True
    assert archived["archive_object"]["object_uuid"]

    restored = measurement_services.restore_project_handler("proj-proteinmc")

    assert restored["ok"] is True
    assert base64.b64decode(restored["project_archive_data"]) == archive_bytes
    assert restored["project_archive"]["object_uuid"] == archived["archive_object"]["object_uuid"]
