"""PRD-18 Task 2: integration test against the REAL in-process MFDBClient.

Drives ``MFDBClient(inprocess=True)`` end to end through its public ``call``
method (not a mock), so interface drift like the historical ``.call`` vs
``_call`` bug fails loudly instead of silently returning zero datasets.

Runs under the hermetic harness (PRD-18 Task 1): registration and the client's
handlers both resolve the same temp database, so this exercises a real round trip
without touching ``~/.chisurf``.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.mfdb.result_registry import register_raw_measurement, set_global_db


def _register_one_raw(tmp_path: Path) -> str:
    """Register one raw measurement into the resolved (temp) user database."""
    f = tmp_path / "measurement.ptu"
    f.write_bytes(b"\x00\x01\x02\x03")
    # db=None -> falls back to resolve_database_path(), the same temp DB the
    # in-process client's handlers open under the hermetic harness.
    artifact_id = register_raw_measurement(str(f))
    assert artifact_id
    return artifact_id


def test_real_inprocess_client_browse_and_open(tmp_path):
    from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

    artifact_id = _register_one_raw(tmp_path)
    client = MFDBClient(inprocess=True)
    try:
        # Public contract: .call must exist and round-trip through the real
        # dispatcher (the bug this guards: .call missing -> AttributeError
        # swallowed -> browser shows 0).
        assert hasattr(client, "call")
        browse = client.call("mfdb.datasets.browse", {"scope": "all", "limit": 50})
        ids = {d.get("artifact_id") or d.get("id") for d in browse.get("datasets", [])}
        assert artifact_id in ids, f"registered {artifact_id} not in browse {ids}"

        opened = client.call("mfdb.datasets.open", {"artifact_id": artifact_id})
        # The open handler returns a readable local path for the artifact.
        path = opened.get("path") or opened.get("local_path")
        assert path, f"datasets.open returned no path: {opened}"
    finally:
        set_global_db(None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
