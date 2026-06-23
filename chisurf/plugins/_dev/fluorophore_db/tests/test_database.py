import os
import pytest
import sqlite3
import numpy as np
from chisurf.fio.mmcif.db import FluorophoreDatabase

@pytest.fixture
def db():
    # Use an in-memory database for testing
    return FluorophoreDatabase(":memory:")

def test_connection_and_version(db):
    assert db.conn is not None
    # Fresh DB should be at the latest version (v4)
    assert db._get_schema_version() == 4
    
    # Check if WAL mode is enabled (journal_mode might be 'memory' for :memory: but we check pragma anyway)
    # Actually for :memory: it might stay as 'memory' or 'delete'.
    pass

def test_probe_types(db):
    tid = db.add_probe_type("organic", "Organic Dye")
    assert isinstance(tid, int)
    types = db.get_probe_types()
    assert any(t["type_name"] == "organic" for t in types)

def test_probe_crud_and_validation(db):
    tid = db.add_probe_type("organic", "Organic Dye")
    
    # Valid probe
    pid = db.add_probe("Alexa 488", tid, category="organic_dye", probe_origin="extrinsic")
    assert pid > 0
    
    # Invalid category
    with pytest.raises(ValueError, match="Invalid category"):
        db.add_probe("Bad Dye", tid, category="unknown")
        
    # Invalid origin
    with pytest.raises(ValueError, match="Invalid probe_origin"):
        db.update_probe(pid, probe_origin="alien")

def test_optical_property_normalization(db):
    tid = db.add_probe_type("organic", "Organic Dye")
    pid = db.add_probe("Normal Dye", tid)
    
    # Add property with legacy key
    db.add_optical_property(pid, "ηfl", "0.92") # ηfl should map to qy
    
    # Trigger normalization (usually happens in migration, but we can call it manually)
    db._normalize_optical_property_keys()
    
    props = db.get_optical_properties(pid)
    assert "qy" in props
    assert props["qy"] == "0.92"
    assert "ηfl" not in props

def test_data_validation(db):
    tid = db.add_probe_type("organic", "Organic Dye")
    pid = db.add_probe("Valid Dye", tid)
    
    # Good data
    db.add_optical_property(pid, "qy", "0.8")
    db.add_optical_property(pid, "abs_max", "488")
    db.add_spectrum(pid, "absorption", np.array([400, 500]), np.array([0, 1]))
    assert len(db.validate_probe(pid)) == 0
    
    # Bad QY
    db.add_optical_property(pid, "qy", "1.5")
    errors = db.validate_probe(pid)
    assert any("QY 1.5 out of range [0, 1]" in e for e in errors)
    
    # Bad peak
    db.add_optical_property(pid, "abs_max", "10")
    errors = db.validate_probe(pid)
    assert any("abs_max 10.0 out of range [200, 1000] nm" in e for e in errors)

def test_sample_management_crud(db):
    # Entities
    db.add_entity("PROT1", common_name="Protein A")
    entities = db.get_entities()
    assert len(entities) == 1
    assert entities[0]["entity_id"] == "PROT1"
    
    # Sequences
    db.set_sequence("PROT1", ["ALA", "CYS", "GLY"])
    seq = db.get_sequence("PROT1")
    assert len(seq) == 3
    assert seq[1]["mon_id"] == "CYS"
    
    # Labeling
    tid = db.add_probe_type("organic", "Organic Dye")
    pid = db.add_probe("Dye1", tid)
    db.add_poly_probe_position(pid, "PROT1", 2, residue_name="CYS")
    pos = db.get_poly_probe_positions(entity_id="PROT1")
    assert len(pos) == 1
    assert pos[0]["residue_number"] == 2
    
    # Cleanup entity (should cascade if foreign keys are enabled)
    # Note: In-memory SQLite with PRAGMA foreign_keys = ON should work
    db.delete_entity("PROT1")
    assert len(db.get_entities()) == 0
    assert len(db.get_sequence("PROT1")) == 0
    assert len(db.get_poly_probe_positions(entity_id="PROT1")) == 0

def test_search_and_rich_lookup(db):
    tid = db.add_probe_type("organic", "Organic Dye")
    pid = db.add_probe("Cy3", tid, category="organic_dye")
    db.add_optical_property(pid, "qy", "0.15")
    
    # Search
    results = db.search_probes("Cy")
    assert len(results) >= 1
    
    # Full lookup
    full = db.get_probe_full(pid)
    assert full["chromophore_name"] == "Cy3"
    assert full["qy"] == 0.15
    assert "has_abs" in full
