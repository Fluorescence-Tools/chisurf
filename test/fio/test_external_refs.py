"""Headless tests for PRD-39 sequence provenance & external references.

Covers the dependency-free foundation: the extended ``EntityDefinition`` /
``MutationDefinition`` data model, the ``struct_ref*`` schema tables, and the
pure construct-vs-reference auto-diff. No network access.
"""

import sqlite3

import pytest

from chisurf.core.mfdb.models import EntityDefinition, MutationDefinition
from chisurf.core.mfdb.schema import CREATE_TABLES_SQL
from chisurf.core.mfdb.external_refs import (
    diff_sequences, fetch_uniprot, fetch_sifts_uniprot_mapping,
)


# ── Data model ───────────────────────────────────────────────────────────────

def test_entity_definition_external_ref_fields():
    """EntityDefinition carries UniProt/PDB refs + mutations (all optional)."""
    e = EntityDefinition(name="T4 Lysozyme", entity_type="protein")
    # Backward compatible: new fields default to None / empty.
    assert e.uniprot_accession is None
    assert e.pdb_id is None
    assert e.organism is None
    assert e.reference_sequence is None
    assert e.mutations == []

    e2 = EntityDefinition(
        name="T4 Lysozyme",
        entity_type="protein",
        sequence="MNIFEMLR",
        uniprot_accession="P00720",
        pdb_id="2LZM",
        pdb_chain_id="A",
        organism="Enterobacteria phage T4",
        mutations=[MutationDefinition(seq_id=48, mut_comp_id="CYS",
                                      wt_comp_id="SER", auth_name="S48C")],
    )
    assert e2.uniprot_accession == "P00720"
    assert e2.mutations[0].kind == "engineered_mutation"


# ── Schema ───────────────────────────────────────────────────────────────────

def test_struct_ref_tables_create_and_accept_rows():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    for stmt in CREATE_TABLES_SQL:
        con.execute(stmt)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"struct_ref", "struct_ref_seq", "struct_ref_seq_dif"} <= tables

    con.execute("INSERT INTO struct_ref(ref_id,entity_id,db_name,"
                "pdbx_db_accession) VALUES('r1','e1','UNP','P00720')")
    con.execute("INSERT INTO struct_ref_seq(align_id,ref_id,seq_align_beg,"
                "seq_align_end,db_align_beg,db_align_end) "
                "VALUES('a1','r1',1,164,1,164)")
    con.execute("INSERT INTO struct_ref_seq_dif(align_id,seq_num,mon_id,"
                "db_mon_id,details) VALUES('a1',48,'CYS','SER',"
                "'ENGINEERED MUTATION')")
    dif = con.execute("SELECT * FROM struct_ref_seq_dif").fetchone()
    assert dif["mon_id"] == "CYS" and dif["db_mon_id"] == "SER"
    assert dif["details"] == "ENGINEERED MUTATION"


# ── Auto-diff ────────────────────────────────────────────────────────────────

def test_diff_detects_engineered_cysteines():
    """T4L double-cysteine construct vs WT yields S48C and S131C."""
    wt = list("M" + "A" * 163)          # 164-residue stand-in reference
    wt[47] = "S"                         # residue 48 (1-based)
    wt[130] = "S"                        # residue 131
    construct = list(wt)
    construct[47] = "C"                  # S48C
    construct[130] = "C"                 # S131C

    muts = diff_sequences("".join(construct), "".join(wt))
    by_pos = {m.seq_id: m for m in muts}
    assert set(by_pos) == {48, 131}
    assert by_pos[48].auth_name == "S48C"
    assert by_pos[48].mut_comp_id == "CYS" and by_pos[48].wt_comp_id == "SER"
    assert by_pos[131].auth_name == "S131C"
    assert all(m.kind == "engineered_mutation" for m in muts)


def test_diff_identical_sequences_no_mutations():
    assert diff_sequences("MNIFEMLR", "MNIFEMLR") == []


def test_diff_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal-length"):
        diff_sequences("MNIFEM", "MNIFEMLR")


def test_diff_numbering_start_offset():
    muts = diff_sequences("AC", "AS", numbering_start=47)
    assert muts[0].seq_id == 48
    assert muts[0].auth_name == "S48C"


# ── Fetch (offline-safe) ─────────────────────────────────────────────────────

def test_fetch_uniprot_uses_disk_cache(tmp_path):
    """A cached JSON response is read without touching the network."""
    import json
    (tmp_path / "uniprot_P00720.json").write_text(json.dumps({
        "uniProtkbId": "ENLYS_BPT4",
        "organism": {"scientificName": "Enterobacteria phage T4"},
        "sequence": {"value": "MNIFEMLR"},
    }))
    res = fetch_uniprot("P00720", cache_dir=tmp_path)
    assert res is not None
    assert res["organism"] == "Enterobacteria phage T4"
    assert res["sequence"] == "MNIFEMLR"


def test_fetch_uniprot_empty_accession_returns_none():
    assert fetch_uniprot("") is None


# ── Persistence round-trip (create_sample → get_sample_full_description) ──────

@pytest.fixture
def db():
    import os
    import tempfile
    from chisurf.core.mfdb.repository import MFDatabase

    with tempfile.TemporaryDirectory() as tmpdir:
        database = MFDatabase(os.path.join(tmpdir, "test.db"))
        try:
            yield database
        finally:
            database.close()


def _t4l_definition():
    from chisurf.core.mfdb.models import (
        SampleDefinition, EntityDefinition, ProbeDefinition, MutationDefinition,
    )
    return SampleDefinition(
        name="T4L-extref",
        entities=[EntityDefinition(
            name="T4 Lysozyme", entity_type="protein", sequence="MNIFEMLR",
            uniprot_accession="P00720", pdb_id="2LZM", pdb_chain_id="A",
            organism="Enterobacteria phage T4", reference_sequence="MNIFEMLS",
            mutations=[MutationDefinition(seq_id=48, mut_comp_id="CYS",
                                          wt_comp_id="SER", auth_name="S48C")],
        )],
        probes=[ProbeDefinition(name="Cy3B", entity_index=0, seq_id=48,
                                comp_id="CYS", asym_id="A", mutation_flag="yes",
                                auth_name="S48C")],
    )


def test_create_sample_persists_struct_ref_tables(db):
    from chisurf.core.mfdb.sample_manager import create_sample

    create_sample(db, _t4l_definition())

    refs = [dict(r) for r in db.conn.execute(
        "SELECT * FROM struct_ref WHERE deleted_at IS NULL")]
    by_db = {r["db_name"]: r for r in refs}
    assert set(by_db) == {"UNP", "PDB"}
    assert by_db["UNP"]["pdbx_db_accession"] == "P00720"
    assert by_db["UNP"]["organism"] == "Enterobacteria phage T4"
    assert by_db["PDB"]["pdbx_db_accession"] == "2LZM"

    seqs = [dict(r) for r in db.conn.execute(
        "SELECT * FROM struct_ref_seq WHERE deleted_at IS NULL")]
    assert len(seqs) == 1 and seqs[0]["seq_align_end"] == 8

    difs = [dict(r) for r in db.conn.execute(
        "SELECT * FROM struct_ref_seq_dif WHERE deleted_at IS NULL")]
    assert len(difs) == 1
    assert difs[0]["seq_num"] == 48
    assert difs[0]["mon_id"] == "CYS" and difs[0]["db_mon_id"] == "SER"
    assert difs[0]["details"] == "ENGINEERED MUTATION"


def test_full_description_surfaces_external_refs_and_mutations(db):
    from chisurf.core.mfdb.sample_manager import (
        create_sample, get_sample_full_description)

    sample_id = create_sample(db, _t4l_definition())
    entity = get_sample_full_description(db, sample_id)["entities"][0]

    accessions = {r["db_name"]: r["accession"] for r in entity["external_refs"]}
    assert accessions == {"UNP": "P00720", "PDB": "2LZM"}
    assert len(entity["mutations"]) == 1
    assert entity["mutations"][0]["seq_id"] == 48
    assert entity["mutations"][0]["details"] == "ENGINEERED MUTATION"


def test_sample_without_external_refs_has_empty_lists(db):
    from chisurf.core.mfdb.models import (
        SampleDefinition, EntityDefinition, ProbeDefinition)
    from chisurf.core.mfdb.sample_manager import (
        create_sample, get_sample_full_description)

    defn = SampleDefinition(
        name="plain-GFP",
        entities=[EntityDefinition(name="GFP", entity_type="protein",
                                   sequence="MVSK")],
        probes=[ProbeDefinition(name="Alexa Fluor 488", entity_index=0,
                                seq_id=1, asym_id="A")],
    )
    sample_id = create_sample(db, defn)
    # No struct_ref rows are written for an entity without refs/mutations.
    assert db.conn.execute(
        "SELECT COUNT(*) FROM struct_ref").fetchone()[0] == 0
    entity = get_sample_full_description(db, sample_id)["entities"][0]
    assert entity["external_refs"] == []
    assert entity["mutations"] == []


# ── Probe <-> mutation consistency validator (Task 6) ────────────────────────

def test_validate_consistent_probe_mutation_no_warning(db):
    from chisurf.core.mfdb.sample_manager import (
        create_sample, validate_sample_for_export)

    sample_id = create_sample(db, _t4l_definition())
    warnings = validate_sample_for_export(db, sample_id)
    assert not any("mutation" in w.lower() for w in warnings)


def test_validate_probe_on_mutation_without_record_warns(db):
    from chisurf.core.mfdb.models import (
        SampleDefinition, EntityDefinition, ProbeDefinition)
    from chisurf.core.mfdb.sample_manager import (
        create_sample, validate_sample_for_export)

    # Probe sits on a mutated residue, but the entity records no mutation.
    defn = SampleDefinition(
        name="orphan-mutation",
        entities=[EntityDefinition(name="P", entity_type="protein",
                                   sequence="MNIFEMLR")],
        probes=[ProbeDefinition(name="Cy3B", entity_index=0, seq_id=48,
                                comp_id="CYS", asym_id="A",
                                mutation_flag="yes", auth_name="S48C")],
    )
    sample_id = create_sample(db, defn)
    warnings = validate_sample_for_export(db, sample_id)
    assert any("no matching mutation record" in w for w in warnings)


# ── SIFTS PDB↔UniProt mapping (Task 4, offline-safe) ─────────────────────────

def test_fetch_sifts_uses_disk_cache(tmp_path):
    import json
    (tmp_path / "sifts_2lzm.json").write_text(json.dumps({
        "2lzm": {"UniProt": {"P00720": {"mappings": [{
            "chain_id": "A",
            "start": {"author_residue_number": 1},
            "end": {"author_residue_number": 164},
            "unp_start": 1, "unp_end": 164,
        }]}}}
    }))
    segs = fetch_sifts_uniprot_mapping("2LZM", cache_dir=tmp_path)
    assert segs == [{
        "accession": "P00720", "chain_id": "A",
        "pdb_start": 1, "pdb_end": 164, "unp_start": 1, "unp_end": 164,
    }]


def test_fetch_sifts_chain_filter(tmp_path):
    import json
    (tmp_path / "sifts_2lzm.json").write_text(json.dumps({
        "2lzm": {"UniProt": {"P00720": {"mappings": [
            {"chain_id": "A", "start": {}, "end": {}, "unp_start": 1, "unp_end": 10},
            {"chain_id": "B", "start": {}, "end": {}, "unp_start": 1, "unp_end": 10},
        ]}}}
    }))
    segs = fetch_sifts_uniprot_mapping("2lzm", chain_id="B", cache_dir=tmp_path)
    assert len(segs) == 1 and segs[0]["chain_id"] == "B"


def test_fetch_sifts_empty_pdb_returns_none():
    assert fetch_sifts_uniprot_mapping("") is None


# ── Auto-diff wiring (Task 5) ────────────────────────────────────────────────

def test_auto_diff_wired_in_create_sample(db):
    """Creating a sample with reference_sequence but no mutations auto-populates them."""
    from chisurf.core.mfdb.models import (
        SampleDefinition, EntityDefinition, ProbeDefinition)
    from chisurf.core.mfdb.sample_manager import (
        create_sample, get_sample_full_description)

    # Simple 8-residue test: reference has SER at positions 2 & 5,
    # construct has CYS at those positions -> S2C, S5C.
    reference = "MSKLSKLM"
    construct = "MCKLCKLM"

    defn = SampleDefinition(
        name="auto-diff-test",
        entities=[EntityDefinition(
            name="test_protein", entity_type="protein",
            sequence=construct, uniprot_accession="P00000",
            reference_sequence=reference, organism="Test",
            # mutations list intentionally empty -> auto-diff should fire
        )],
        probes=[ProbeDefinition(name="Cy3B", entity_index=0, seq_id=2,
                                comp_id="CYS", asym_id="A",
                                mutation_flag="yes", auth_name="S2C")],
    )
    sample_id = create_sample(db, defn)
    entity = get_sample_full_description(db, sample_id)["entities"][0]

    # Auto-diff should have produced mutations for S2C and S5C
    muts = entity.get("mutations", [])
    by_seq = {m["seq_id"]: m for m in muts}
    assert 2 in by_seq, f"S2C missing from auto-diff: {muts}"
    assert by_seq[2]["mut_comp_id"] == "CYS"
    assert by_seq[2]["wt_comp_id"] == "SER"
    assert 5 in by_seq, f"S5C missing from auto-diff: {muts}"
    assert by_seq[5]["mut_comp_id"] == "CYS"
    assert by_seq[5]["wt_comp_id"] == "SER"

    # Manual mutations are preserved (not clobbered) when explicitly supplied
    defn2 = SampleDefinition(
        name="auto-diff-manual",
        entities=[EntityDefinition(
            name="test_protein_manual", entity_type="protein",
            sequence=construct, uniprot_accession="P00000",
            reference_sequence=reference,
            mutations=[MutationDefinition(seq_id=9, mut_comp_id="CYS",
                                          wt_comp_id="SER", auth_name="L9C")],
        )],
        probes=[ProbeDefinition(name="Cy3B", entity_index=0, seq_id=9,
                                comp_id="CYS", asym_id="A")],
    )
    sample_id2 = create_sample(db, defn2)
    # Query struct_ref_seq_dif for the manual entity only
    difs = [
        dict(r) for r in db.conn.execute(
            "SELECT d.seq_num, d.mon_id, d.db_mon_id "
            "FROM struct_ref_seq_dif d "
            "JOIN struct_ref_seq s ON d.align_id = s.align_id "
            "JOIN struct_ref r ON s.ref_id = r.ref_id "
            "WHERE r.entity_id = 'test_protein_manual' AND d.deleted_at IS NULL"
        ).fetchall()
    ]
    by_seq2 = {d["seq_num"]: d for d in difs}
    # Only the manually-supplied L9C (seq_id=9) should appear, not auto-diff for 2/5
    assert set(by_seq2) == {9}, f"Manual mutations clobbered: {difs}"
    assert by_seq2[9]["mon_id"] == "CYS"
    assert by_seq2[9]["db_mon_id"] == "SER"


# ── flrCIF round-trip (Task 8) ───────────────────────────────────────────────

def test_flr_cif_round_trip_preserves_struct_ref(db, tmp_path):
    """Export a sample with external refs to FLR CIF, re-import via pdbx reader, and verify."""
    from chisurf.core.mfdb.models import (
        SampleDefinition, EntityDefinition, ProbeDefinition, MutationDefinition)
    from chisurf.core.mfdb.sample_manager import (
        create_sample)

    defn = SampleDefinition(
        name="T4L-rtrip",
        entities=[EntityDefinition(
            name="T4 Lysozyme", entity_type="protein", sequence="MNIFEMLR",
            uniprot_accession="P00720", pdb_id="2LZM", pdb_chain_id="A",
            organism="Enterobacteria phage T4", reference_sequence="MNIFEMLS",
            mutations=[MutationDefinition(seq_id=48, mut_comp_id="CYS",
                                          wt_comp_id="SER", auth_name="S48C")],
        )],
        probes=[ProbeDefinition(name="Cy3B", entity_index=0, seq_id=48,
                                comp_id="CYS", asym_id="A",
                                mutation_flag="yes", auth_name="S48C")],
    )
    create_sample(db, defn)

    cif_text = db.export_flr_cif_to_text(include_extension=True)

    # Verify the CIF text contains struct_ref categories
    assert "_struct_ref.ref_id" in cif_text
    assert "_struct_ref.db_name" in cif_text
    assert "_struct_ref.pdbx_db_accession" in cif_text
    assert "UNP" in cif_text
    assert "P00720" in cif_text
    assert "2LZM" in cif_text

    assert "_struct_ref_seq.align_id" in cif_text
    assert "_struct_ref_seq.ref_id" in cif_text

    assert "_struct_ref_seq_dif.align_id" in cif_text
    assert "_struct_ref_seq_dif.seq_num" in cif_text
    assert "_struct_ref_seq_dif.mon_id" in cif_text
    assert "_struct_ref_seq_dif.db_mon_id" in cif_text

    # Parse the CIF text with pdbx reader and verify data round-trips
    from pdbx.reader import PdbxReader
    from pdbx import containers
    import io

    data: list[containers.DataContainer] = []
    PdbxReader(io.StringIO(cif_text)).read(data)
    assert len(data) > 0

    # Check struct_ref data
    struct_ref_found = False
    struct_ref_seq_dif_found = False
    for block in data:
        for name in block.get_object_name_list():
            obj = block.get_object(name)
            clean = name.lstrip("_")
            if clean == "struct_ref":
                for row_idx in range(obj.row_count):
                    row = {attr: obj.get_value(attr, row_idx) for attr in obj.attribute_list}
                    if row.get("db_name") == "UNP":
                        assert row["pdbx_db_accession"] == "P00720"
                        struct_ref_found = True
            elif clean == "struct_ref_seq_dif":
                for row_idx in range(obj.row_count):
                    row = {attr: obj.get_value(attr, row_idx) for attr in obj.attribute_list}
                    if row.get("mon_id") == "CYS" and row.get("db_mon_id") == "SER":
                        struct_ref_seq_dif_found = True
    assert struct_ref_found, "struct_ref with UNP P00720 not found in CIF export"
    assert struct_ref_seq_dif_found, "struct_ref_seq_dif CYS/SER not found in CIF export"
