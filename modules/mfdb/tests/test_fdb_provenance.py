"""Tests for fdb Phase 1 provenance storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mfdb import schema
from mfdb.repository import MFDatabase


# Note: the former test_v12_database_migrates_to_v13_without_losing_existing_rows test
# asserted the legacy fdb_* provenance tables (fdb_raw_data/fdb_processing_run/…) exist
# after a version migration. PRD-19 removed all fdb_* tables (legacy-free) and the
# version-chain migration; _drop_legacy_tables drops any leftover on open. The current
# provenance path (register_artifact / mfdb_* via the add_*_reference shims) is exercised
# by test_burst_provenance_chain_and_manifest below.


def test_burst_provenance_chain_and_manifest(tmp_path: Path) -> None:
    """Register raw data, processing, products, edges, and an archive manifest."""
    raw_path = tmp_path / "input.ptu"
    raw_path.write_bytes(b"fake tttr")
    bur_path = tmp_path / "output.bur"
    bur_path.write_text("n_ph\tduration\n10\t0.1\n", encoding="utf-8")

    with MFDatabase(tmp_path / "test.db") as db:
        db.add_sample("sample_1")
        db.add_experiment("exp_1", sample_id="sample_1", status="complete")
        raw_id = db.add_raw_data_reference(
            experiment_id="exp_1",
            data_type="PTU",
            storage_mode="local_file",
            file_path=str(raw_path),
            size_bytes=raw_path.stat().st_size,
            checksum="0" * 64,
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
            checksum="1" * 64,
            row_count=1,
            validation_status="valid",
        )
        trace = db.trace_processed_data(product_id)
        assert trace is not None
        assert trace["processing_run"]["settings"]["burst_detection"]["min_photons"] == 10
        assert trace["processing_run"]["input_raw_data"][0]["raw_data_id"] == raw_id

        manifest = db.export_burst_processing_manifest(processing_id)
        assert manifest["schema"] == "mfdb.burst_processing_manifest.v1"
        assert manifest["raw_data"][0]["checksum"] == "0" * 64
        assert manifest["processed_data"][0]["checksum"] == "1" * 64

        archive_id = db.register_archive_manifest(processing_id, manifest)
        included_edges = db.get_provenance_edges(
            source_node_id=product_id,
            target_node_id=archive_id,
            relationship_type="included_in",
        )
        assert len(included_edges) == 1
