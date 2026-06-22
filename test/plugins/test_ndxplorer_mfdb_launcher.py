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


def test_resolve_dataset_path_empty_artifact_returns_none():
    class _NoCall:
        def call(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("should not call for empty artifact_id")

    assert resolve_dataset_path(_NoCall(), "") is None


def test_burst_kinds_includes_burst_table():
    assert "burst_table" in BURST_KINDS
