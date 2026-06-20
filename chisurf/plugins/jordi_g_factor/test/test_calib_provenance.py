from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import os
import sqlite3
import json

from chisurf.plugins.jordi_g_factor.gui.client import JordiGFactorClient
from chisurf.plugins.jordi_g_factor.backend.services import archive_g_factor_handler
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb import schema

def test_archive_g_factor_provenance(tmp_path, monkeypatch):
    """Verify archive_g_factor registers the reference decay and parented calibration."""
    db_path = tmp_path / "test_mfdb.sqlite"
    object_root = tmp_path / "objects"
    object_root.mkdir()
    
    # Create and migrate empty test DB
    conn = sqlite3.connect(str(db_path))
    schema.migrate_schema(conn)
    conn.commit()
    conn.close()
    
    # Monkeypatch the database resolver
    monkeypatch.setattr(
        "chisurf.core.mfdb.database_resolver.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.database_resolver.object_store_root",
        lambda: object_root,
    )
    
    # Create dummy Jordi file (parallel and perpendicular decays)
    jordi_file = tmp_path / "dummy_jordi.txt"
    vv = np.exp(-np.linspace(0, 10, 1000) / 2.0)
    vh = vv / 1.5
    merged = np.concatenate([vv, vh])
    np.savetxt(jordi_file, merged)
    
    # Run service archival via client
    params = {
        "g_factor": 1.5,
        "region_min": 700.0,
        "region_max": 900.0,
        "decay_shift": 0.0,
        "flip": False,
        "use_bg": False,
        "l1": 0.0308,
        "l2": 0.0308,
        "micro_time_resolution": 0.032,
    }
    
    client = JordiGFactorClient()
    res = client.archive_g_factor(
        file_path=str(jordi_file),
        parameters=params,
        active_user="test_user",
    )
    
    assert res.get("ok") is True
    calib_id = res.get("calibration_id")
    ref_decay_id = res.get("reference_decay_id")
    assert calib_id != ""
    assert ref_decay_id != ""
    
    # Verify DB records
    with MFDatabase(str(db_path)) as db:
        # Check raw measurement
        artifact = db.get_artifact(ref_decay_id)
        assert artifact is not None
        assert artifact["artifact_kind"] == "raw_measurement"
        meta = json.loads(artifact["metadata_json"])
        assert meta["filename"] == "dummy_jordi.txt"
        assert meta["micro_time_resolution"] == 0.032
        
        # Check calibration
        calib = db.get_artifact(calib_id)
        assert calib is not None
        assert calib["artifact_kind"] == "calibration_data"
        calib_meta = json.loads(calib["metadata_json"])
        assert calib_meta["calibration_type"] == "g_factor"
        
        # Check calibration payload
        from chisurf.core.mfdb.result_registry import read_result
        payload = read_result(db, calib_id)
        assert payload is not None
        assert np.allclose(payload.data["g_factor"], 1.5)
        assert np.allclose(payload.data["l1"], 0.0308)
        assert np.allclose(payload.data["l2"], 0.0308)
        # r_inf is automatically computed
        assert "r_inf" in payload.data
        r_inf_val = payload.data["r_inf"][0]
        assert np.isnan(r_inf_val) or isinstance(r_inf_val, float) or isinstance(r_inf_val, np.floating)

        # The whole point: the calibration must be parented to the reference
        # decay via a derived_from provenance edge.
        edge = db.conn.execute(
            "SELECT relationship_type FROM mfdb_edge "
            "WHERE source_node_id = ? AND target_node_id = ? AND deleted_at IS NULL",
            (calib_id, ref_decay_id),
        ).fetchone()
        assert edge is not None, "calibration must be parented to the reference decay"
        assert edge[0] == "derived_from"


def test_archive_g_factor_graceful_failure(tmp_path, monkeypatch):
    """Verify that archive_g_factor fails gracefully without raising when database is missing."""
    # Resolve to a db path in a non-existent subdirectory
    monkeypatch.setattr(
        "chisurf.core.mfdb.database_resolver.resolve_database_path",
        lambda: tmp_path / "nonexistent_dir" / "db.sqlite",
    )
    
    jordi_file = tmp_path / "dummy_jordi.txt"
    np.savetxt(jordi_file, np.ones(100))
    
    client = JordiGFactorClient()
    res = client.archive_g_factor(
        file_path=str(jordi_file),
        parameters={"g_factor": 1.5, "region_min": 10, "region_max": 20},
        active_user="test_user",
    )
    
    assert res.get("ok") is False
    assert res.get("calibration_id") == ""
    assert "unavailable" in res.get("error", "").lower()
