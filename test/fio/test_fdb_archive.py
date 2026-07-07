import json
import pathlib
import zipfile
from unittest.mock import patch

import pytest

from mfdb.repository import MFDatabase
from mfdb.admin.backend.measurement_services import (
    database_backup_handler,
    export_provenance_graph_handler,
    export_zip_archive_handler,
)


@pytest.fixture
def temp_db_setup(tmp_path):
    """Fixture to setup a temporary database with sample, raw, processed, and analysis run data."""
    db_path = tmp_path / "archive_test.db"
    
    with MFDatabase(db_path) as db:
        # 1. Add sample
        db.add_sample("sample_1")
        
        # 2. Add experiment
        db.add_experiment("exp_1", sample_id="sample_1", status="complete")
        
        # 3. Add raw data
        raw_id = db.add_raw_data_reference(
            experiment_id="exp_1",
            data_type="ptu",
            storage_mode="local_file",
            file_path=str(tmp_path / "raw.ptu"),
            checksum="0" * 64,
        )
        
        # 4. Add processing run
        proc_id = db.add_processing_run(
            experiment_id="exp_1",
            status="succeeded",
            processing_type="burst_selection",
        )
        
        # Link raw_data -> processing_run
        db.add_provenance_edge(
            source_node_type="raw_data",
            source_node_id=raw_id,
            target_node_type="processing_run",
            target_node_id=proc_id,
            relationship_type="input_to",
            processing_id=proc_id,
        )
        
        # 5. Add processed data product
        real_file_path = pathlib.Path("./test/data/sample_anisotropy.csv").resolve()
        prod_id = db.add_processed_data_product(
            processing_id=proc_id,
            product_type="bur",
            storage_mode="local_file",
            file_path=str(real_file_path),
            checksum="1" * 64,
        )
        
        # Link processing_run -> processed_data
        db.add_provenance_edge(
            source_node_type="processing_run",
            source_node_id=proc_id,
            target_node_type="processed_data",
            target_node_id=prod_id,
            relationship_type="produced",
            processing_id=proc_id,
        )
        
        # 6. Add analysis run
        analysis_id = db.add_analysis_run(
            analysis_type="tcspc_fitting",
            experiment_id="exp_1",
            model_name="TCSPC model",
            convergence_status="converged",
        )
        
        # Link processed_data -> analysis_run
        db.add_provenance_edge(
            source_node_type="processed_data",
            source_node_id=prod_id,
            target_node_type="analysis_run",
            target_node_id=analysis_id,
            relationship_type="input_to",
            processing_id=analysis_id,
        )
        
    return db_path, raw_id, prod_id, analysis_id


def test_export_provenance_graph(temp_db_setup, tmp_path):
    """Test exporting provenance subgraph to JSON and JSONL formats."""
    db_path, raw_id, prod_id, analysis_id = temp_db_setup

    with patch("mfdb.admin.backend.measurement_services.resolve_database_path", return_value=db_path):
        # Export as standard JSON
        json_path = tmp_path / "graph.json"
        res = export_provenance_graph_handler(
            seed_node_type="analysis_run",
            seed_node_id=analysis_id,
            output_path=str(json_path),
        )
        assert res.get("ok") is True
        assert json_path.exists()
        
        # Verify JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data
        
        node_ids = {n["node_id"] for n in data["nodes"]}
        assert raw_id in node_ids
        assert prod_id in node_ids
        assert analysis_id in node_ids

        # Export as JSONL (JSON Lines)
        jsonl_path = tmp_path / "graph.jsonl"
        res_jsonl = export_provenance_graph_handler(
            seed_node_type="analysis_run",
            seed_node_id=analysis_id,
            output_path=str(jsonl_path),
        )
        assert res_jsonl.get("ok") is True
        assert jsonl_path.exists()
        
        # Verify JSONL lines
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) > 0
        assert all("type" in item and "data" in item for item in lines)


def test_database_backup(temp_db_setup, tmp_path):
    """Test safe hot backup of active SQLite database."""
    db_path, _, _, _ = temp_db_setup
    backup_path = tmp_path / "backup_snapshot.db"

    with patch("mfdb.admin.backend.measurement_services.resolve_database_path", return_value=db_path):
        res = database_backup_handler(str(backup_path))
        assert res.get("ok") is True
        assert backup_path.exists()
        
        # Verify we can open the backup and read data
        with MFDatabase(backup_path) as db:
            samples = db.conn.execute("SELECT sample_id FROM flr_sample").fetchall()
            assert len(samples) == 1
            assert samples[0][0] == "sample_1"


def test_export_zip_archive_without_data(temp_db_setup, tmp_path):
    """Test exporting ZIP archive packaging manifest, snapshot, and graph only (no large files)."""
    db_path, _, _, analysis_id = temp_db_setup
    zip_path = tmp_path / "archive_metadata.zip"

    with patch("mfdb.admin.backend.measurement_services.resolve_database_path", return_value=db_path):
        res = export_zip_archive_handler(
            target_zip_path=str(zip_path),
            seed_node_type="analysis_run",
            seed_node_id=analysis_id,
            include_external_data=False,
        )
        assert res.get("ok") is True
        assert zip_path.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            assert "manifest.json" in namelist
            assert "database_snapshot.db" in namelist
            assert "provenance_graph.json" in namelist
            
            # Ensure no external data folder/files inside
            assert not any(name.startswith("external_data/") for name in namelist)


def test_export_zip_archive_with_data_and_remapping(temp_db_setup, tmp_path):
    """Test ZIP export bundling actual files and remapping/relocating path prefixes."""
    db_path, raw_id, prod_id, analysis_id = temp_db_setup
    
    import shutil
    original_file_path = pathlib.Path("./test/data/sample_anisotropy.csv").resolve()
    
    # Now simulate relocation: Copy processed file to a new path
    relocated_dir = tmp_path / "new_storage_drive"
    relocated_dir.mkdir(exist_ok=True)
    relocated_file_path = relocated_dir / "sample_anisotropy.csv"
    shutil.copy2(original_file_path, relocated_file_path)
    
    assert relocated_file_path.exists()
    
    zip_path = tmp_path / "archive_full.zip"
    
    # We specify remapping base path map to resolve the missing file
    base_path_map = {
        str(original_file_path.parent): str(relocated_dir)
    }

    with patch("mfdb.admin.backend.measurement_services.resolve_database_path", return_value=db_path):
        res = export_zip_archive_handler(
            target_zip_path=str(zip_path),
            seed_node_type="analysis_run",
            seed_node_id=analysis_id,
            include_external_data=True,
            base_path_map=base_path_map,
        )
        assert res.get("ok") is True
        assert zip_path.exists()
        
        # Verify ZIP contains the copied file from remapped location
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            assert "manifest.json" in namelist
            assert "external_data/sample_anisotropy.csv" in namelist
            
            # Read manifest and check file status
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            proc_file_info = next(
                f for f in manifest["files"]
                if f.get("node_id") == prod_id
            )
            assert proc_file_info["copied"] is True
            assert proc_file_info["resolved_path"] == str(relocated_file_path)
