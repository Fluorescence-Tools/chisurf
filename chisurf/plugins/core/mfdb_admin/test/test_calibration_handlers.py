"""PRD-05: mfdb-admin calibration RPC handlers.

Calibrations are reachable through the admin backend (and thus MFDBClient): register a
value, list records, and surface stale uses (a calibration superseded by a newer one of
the same type). A non-numeric value comes back as an ``error`` field.
"""

from __future__ import annotations

from mfdb.lifecycle.staleness import record_calibration_use
from mfdb.admin.backend.services import (
    create_calibration_handler,
    list_calibrations_handler,
    stale_calibrations_handler,
)

from .conftest import patch_db


def test_create_and_list_calibration(db):
    with patch_db(db):
        create_calibration_handler("forster_radius", 54.0, notes="Hellenkamp 2018")
        cals = list_calibrations_handler()["calibrations"]
    assert len(cals) == 1
    assert cals[0]["calibration_type"] == "forster_radius"
    assert cals[0]["method"] == "user_provided"
    assert cals[0]["notes"] == "Hellenkamp 2018"
    assert float(cals[0]["value"]) == 54.0


def test_non_numeric_value_returns_error(db):
    with patch_db(db):
        res = create_calibration_handler("g_factor", "abc")
    assert "error" in res


def test_stale_handler_flags_superseded_use(db):
    with patch_db(db):
        old = create_calibration_handler("g_factor", 1.02)["artifact_id"]
        # a fit (stand-in artifact) used the old calibration
        record_calibration_use(db, used_by_id="fit-1", calibration_artifact_id=old,
                               used_by_type="artifact")
        # a newer g_factor calibration supersedes it
        create_calibration_handler("g_factor", 1.05)
        stale = stale_calibrations_handler()["stale"]
    assert len(stale) == 1
    assert stale[0]["used_by_id"] == "fit-1"
    assert stale[0]["calibration_type"] == "g_factor"
    assert stale[0]["used_artifact_id"] == old


def test_via_inprocess_client(db):
    with patch_db(db):
        from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

        client = MFDBClient(inprocess=True)
        client.create_calibration("gamma", 0.9)
        cals = client.list_calibrations()
        stale = client.stale_calibrations()
    assert any(c["calibration_type"] == "gamma" for c in cals)
    assert stale == []
