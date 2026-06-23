"""PRD-28: ndXplorer ← MFDB launcher (the testable path-resolution core).

Runs under the hermetic harness, driving the real in-process MFDBClient so the
resolve-path step is exercised end to end (the ndXplorer launch itself is
interactive and not tested here).
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.mfdb.result_registry import register_raw_measurement, set_global_db
from chisurf.plugins.ndxplorer.mfdb_launcher import BURST_KINDS, resolve_dataset_path


def test_resolve_dataset_path_via_real_client(tmp_path):
    from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

    f = tmp_path / "m.ptu"
    f.write_bytes(b"\x00\x01\x02\x03")
    artifact_id = register_raw_measurement(str(f))
    assert artifact_id

    client = MFDBClient(inprocess=True)
    try:
        path = resolve_dataset_path(client, artifact_id)
        assert path, f"no local path resolved for {artifact_id}"
        assert Path(path).exists()
    finally:
        set_global_db(None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def test_resolve_external_reference_directory_via_metadata(tmp_path):
    """A burst output folder is registered as an external_reference (no object,
    no file_path) with its on-disk path in metadata — resolve_dataset_path must
    return that path (the .bur folder co-located with the TTTR files, preserving
    the photon-index linkage), not fail."""
    from chisurf.core.mfdb.result_registry import register_raw_measurement, register_result
    from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

    burst_dir = tmp_path / "burstwise"
    burst_dir.mkdir()
    (burst_dir / "f1.bur").write_text("0 100\n")
    raw_file = tmp_path / "m.ptu"
    raw_file.write_bytes(b"\x00\x01")

    # Register into the resolved (hermetic) DB the in-process client also opens.
    raw = register_raw_measurement(str(raw_file))
    ext = register_result(
        kind="external_reference",
        data=None,
        parent_artifact_id=raw,
        operation_type="burst_selection",
        data_format="directory",
        metadata={"plugin": "burst_selection", "output_role": "output_folder", "path": str(burst_dir)},
    )
    assert ext

    client = MFDBClient(inprocess=True)
    try:
        path = resolve_dataset_path(client, ext)
        assert path == str(burst_dir)
    finally:
        set_global_db(None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def test_resolve_dataset_path_empty_artifact_returns_none():
    class _NoCall:
        def call(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("should not call for empty artifact_id")

    assert resolve_dataset_path(_NoCall(), "") is None


def test_burst_kinds_target_the_on_disk_burst_folder():
    # The burst .bur files reference photons in the original TTTR file, so the
    # launcher opens the on-disk burst folder (a directory external_reference),
    # which is also the single group per multi-file run — not an object-store copy.
    from chisurf.plugins.ndxplorer.mfdb_launcher import BURST_FORMATS

    assert BURST_KINDS == ["external_reference"]
    assert "directory" in BURST_FORMATS
