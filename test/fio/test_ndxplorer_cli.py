#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for chisurf ndxplorer plugin CLI."""

import json
import pathlib
import sys
import numpy as np
import pytest
from click.testing import CliRunner

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.plugins.ndxplorer.cli import filter_cmd, image_cmd


@pytest.fixture
def temp_mfdb(tmp_path):
    """Setup a temporary MFDatabase with a dummy raw and burst selection product."""
    db_path = tmp_path / "test_ndx_cli.db"
    
    # Create source burst directory
    bur_dir = tmp_path / "bur_source"
    bur_subdir = bur_dir / "bi4_bur"
    bur_subdir.mkdir(parents=True, exist_ok=True)
    
    dummy_bur = bur_subdir / "measurement_1.bur"
    dummy_bur.write_text(
        "First Photon\tLast Photon\tFirst File\tLast File\tproximity_ratio\tn_photons\tMean Macro Time (ms)\n"
        "100\t200\tfile1.ptu\tfile1.ptu\t0.5\t100\t1000\n"
        "300\t400\tfile1.ptu\tfile1.ptu\t0.2\t40\t2000\n"
        "500\t600\tfile1.ptu\tfile1.ptu\t0.8\t150\t3000\n"
        "700\t800\tfile1.ptu\tfile1.ptu\t0.4\t60\t4000\n",
        encoding="utf-8",
    )
    
    with MFDatabase(db_path) as db:
        db.add_sample("sample_1")
        db.add_experiment("exp_1", sample_id="sample_1", status="complete")
        
        # Add raw reference
        raw_id = db.add_raw_data_reference(
            experiment_id="exp_1",
            data_type="PTU",
            storage_mode="local_file",
            file_path=str(tmp_path / "dummy.ptu"),
            checksum="0" * 64,
        )
        
        # Add processing run
        run_id = db.add_processing_run(
            experiment_id="exp_1",
            input_raw_data_ids=[raw_id],
            settings={"burst_detection": {"min_photons": 10}},
            status="succeeded",
        )
        
        # Register the burst folder
        prod_id = db.add_processed_data_product(
            processing_id=run_id,
            product_type="derived_product",
            storage_mode="folder",
            folder_path=str(bur_dir),
            checksum="1" * 64,
            row_count=4,
            validation_status="valid",
        )
        
        # Add registration to the artifacts table
        db.register_artifact(
            artifact_id="art_burst_src",
            artifact_kind="external_reference",
            data_format="directory",
            storage_mode="folder",
            metadata={"path": str(bur_dir), "folder_path": str(bur_dir)},
        )
        db.link_artifact_to_sample("art_burst_src", "sample_1")
        
    return db_path, "art_burst_src", "sample_1"


def test_csc_ndxplorer_filter(temp_mfdb, tmp_path):
    """Test csc ndxplorer filter command registers filtered bursts in MFDB."""
    db_path, art_id, sample_id = temp_mfdb
    runner = CliRunner()
    
    out_dir = tmp_path / "filtered_output"
    
    result = runner.invoke(filter_cmd, [
        "--from-mfdb", art_id,
        "--select", "proximity_ratio:0.3-0.6",
        "--out", str(out_dir),
        "--to-mfdb",
        "--sample-id", sample_id,
        "--db", str(db_path),
        "--skip-nth-row", "1",
    ])
    
    assert result.exit_code == 0
    res_data = json.loads(result.output)
    assert res_data["ok"] is True
    assert res_data["n_out"] == 2
    
    new_artifact_id = res_data["artifact_id"]
    assert new_artifact_id is not None
    
    # Check that artifact is registered in the database
    with MFDatabase(db_path) as db:
        artifact = db.get_artifact(new_artifact_id)
        assert artifact is not None
        assert artifact["artifact_kind"] == "external_reference"
        metadata = artifact.get("metadata") or json.loads(artifact.get("metadata_json") or "{}")
        assert metadata["query"] is None
