"""Tests for probe verification/approval workflow (PRD-06 Task 8)."""
from __future__ import annotations

import pytest

from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def db():
    db = MFDatabase(":memory:")
    yield db
    db.close()


def test_approve_probe(db):
    db.conn.execute(
        "INSERT INTO probes (chromophore_name, verification_status, is_curated) VALUES ('TestProbe', 'unverified', 0)"
    )
    row = db.conn.execute("SELECT probe_id FROM probes WHERE chromophore_name = 'TestProbe'").fetchone()
    probe_id = int(row["probe_id"])

    db.approve_probe(probe_id, verified_by="test_user")
    result = db.conn.execute(
        "SELECT verification_status, is_curated, verified_by FROM probes WHERE probe_id = ?",
        (probe_id,),
    ).fetchone()
    assert result["verification_status"] == "approved"
    assert result["is_curated"] == 1
    assert result["verified_by"] == "test_user"


def test_reject_probe(db):
    db.conn.execute(
        "INSERT INTO probes (chromophore_name, verification_status, is_curated) VALUES ('BadProbe', 'unverified', 0)"
    )
    row = db.conn.execute("SELECT probe_id FROM probes WHERE chromophore_name = 'BadProbe'").fetchone()
    probe_id = int(row["probe_id"])

    db.reject_probe(probe_id, verified_by="test_user")
    result = db.conn.execute(
        "SELECT verification_status FROM probes WHERE probe_id = ?",
        (probe_id,),
    ).fetchone()
    assert result["verification_status"] == "rejected"


def test_set_probe_quality(db):
    db.conn.execute(
        "INSERT INTO probes (chromophore_name, verification_status) VALUES ('QProbe', 'unverified')"
    )
    row = db.conn.execute("SELECT probe_id FROM probes WHERE chromophore_name = 'QProbe'").fetchone()
    probe_id = int(row["probe_id"])

    db.set_probe_quality(probe_id, "high")
    result = db.conn.execute(
        "SELECT quality FROM probes WHERE probe_id = ?",
        (probe_id,),
    ).fetchone()
    assert result["quality"] == "high"

    with pytest.raises(ValueError, match="Invalid quality"):
        db.set_probe_quality(probe_id, "superb")


def test_get_probes_approved_only(db):
    for name, status in [("A", "approved"), ("B", "unverified"), ("C", "approved")]:
        db.conn.execute(
            "INSERT INTO probes (chromophore_name, verification_status) VALUES (?, ?)",
            (name, status),
        )

    approved = db.get_probes_approved_only()
    names = [r["chromophore_name"] for r in approved]
    assert "A" in names
    assert "B" not in names
    assert "C" in names


def test_lookup_forster_radius(db):
    """Test lookup through the repository method."""
    db.conn.execute(
        "INSERT INTO probes (probe_id, chromophore_name, verification_status) VALUES (1, 'Donor', 'approved')"
    )
    db.conn.execute(
        "INSERT INTO probes (probe_id, chromophore_name, verification_status) VALUES (2, 'Acceptor', 'approved')"
    )
    # Create a dummy sample to satisfy the FK
    db.conn.execute(
        """INSERT INTO flr_sample (sample_id, description, created_at, updated_at)
           VALUES ('test_sample', 'Test for Forster radius lookup', '2026-01-01', '2026-01-01')"""
    )
    db.conn.execute(
        """INSERT INTO flr_fret_forster_radius
           (forster_radius_id, sample_id, donor_probe_id, acceptor_probe_id,
            forster_radius, kappa_squared, index_of_refraction, overlap_integral)
           VALUES ('test_r0', 'test_sample', 1, 2, 54.0, 0.667, 1.33, 1e15)"""
    )

    r0 = db.lookup_forster_radius("Donor", "Acceptor")
    assert r0 is not None
    assert r0 == pytest.approx(54.0, abs=0.1)

    # Unverified probes should not be found
    db.conn.execute("UPDATE probes SET verification_status = 'unverified' WHERE probe_id = 2")
    r0 = db.lookup_forster_radius("Donor", "Acceptor")
    assert r0 is None
