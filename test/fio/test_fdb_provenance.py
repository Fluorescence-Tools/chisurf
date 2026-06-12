"""Tests for fdb Phase 1 provenance storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from chisurf.core.fio.mmcif.db import FluorophoreDatabase, schema


def test_v12_database_migrates_to_v13_without_losing_existing_rows(tmp_path: Path) -> None:
    """Verify existing experiment rows survive the Phase 1 provenance migration."""
    db_path = tmp_path / "legacy_v12.db"
    conn = sqlite3.connect(db_path)
    try:
        for sql in schema.CREATE_TABLES_SQL:
            if "CREATE TABLE IF NOT EXISTS fdb_" not in sql:
                conn.execute(sql)
        conn.execute("DELETE FROM _schema_version")
        conn.execute("INSERT INTO _schema_version (version) VALUES (12)")
        conn.execute(
            "INSERT INTO flr_sample (sample_id, description) VALUES (?, ?)",
            ("sample_1", "legacy sample"),
        )
        conn.execute(
            "INSERT INTO flr_experiment (experiment_id, sample_id, status) VALUES (?, ?, ?)",
            ("exp_1", "sample_1", "complete"),
        )
        conn.execute(
            """INSERT INTO flr_experiment_data
               (experiment_id, data_type, storage_mode, file_path)
               VALUES (?, ?, ?, ?)""",
            ("exp_1", "TTTR", "local_file", "legacy.ptu"),
        )
        conn.commit()
    finally:
        conn.close()

    with FluorophoreDatabase(db_path) as db:
        assert db._get_schema_version() == schema.SCHEMA_VERSION

        assert db.get_experiment("exp_1")["sample_id"] == "sample_1"
        assert len(db.get_experiment_data("exp_1")) == 1
        tables = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "fdb_raw_data" in tables
        assert "fdb_processing_run" in tables
        assert "fdb_processed_data" in tables
        assert "fdb_provenance_edge" in tables


def test_burst_provenance_chain_and_manifest(tmp_path: Path) -> None:
    """Register raw data, processing, products, edges, and an archive manifest."""
    raw_path = tmp_path / "input.ptu"
    raw_path.write_bytes(b"fake tttr")
    bur_path = tmp_path / "output.bur"
    bur_path.write_text("n_ph\tduration\n10\t0.1\n", encoding="utf-8")

    with FluorophoreDatabase(tmp_path / "test.db") as db:
        db.add_sample("sample_1")
        db.add_experiment("exp_1", sample_id="sample_1", status="complete")
        raw_id = db.add_raw_data_reference(
            experiment_id="exp_1",
            data_type="PTU",
            storage_mode="local_file",
            file_path=str(raw_path),
            size_bytes=raw_path.stat().st_size,
            checksum="raw-sha",
            header_metadata={"CreatorSW_Name": "test"},
            detector_mapping={"green": [0, 1]},
            validation_status="valid",
        )
        processing_id = db.add_processing_run(
            experiment_id="exp_1",
            input_raw_data_ids=[raw_id],
            settings={"burst_detection": {"min_photons": 10}},
            selected_setup_name="MFD",
            detector_definitions={"green": {"detectors": [0, 1]}},
            pie_window_definitions={"green": [0, 100]},
            status="succeeded",
            file_count=1,
            photon_count=100,
            selected_photon_count=50,
            burst_count=1,
        )
        product_id = db.add_processed_data_product(
            processing_id,
            "bur",
            "local_file",
            file_path=str(bur_path),
            size_bytes=bur_path.stat().st_size,
            checksum="bur-sha",
            row_count=1,
            validation_status="valid",
        )
        trace = db.trace_processed_data(product_id)
        assert trace is not None
        assert trace["processing_run"]["settings"]["burst_detection"]["min_photons"] == 10
        assert trace["processing_run"]["input_raw_data"][0]["raw_data_id"] == raw_id

        manifest = db.export_burst_processing_manifest(processing_id)
        assert manifest["schema"] == "fdb.burst_processing_manifest.v1"
        assert manifest["raw_data"][0]["checksum"] == "raw-sha"
        assert manifest["processed_data"][0]["checksum"] == "bur-sha"

        archive_id = db.register_archive_manifest(processing_id, manifest)
        included_edges = db.get_provenance_edges(
            source_node_id=product_id,
            target_node_id=archive_id,
            relationship_type="included_in",
        )
        assert len(included_edges) == 1
