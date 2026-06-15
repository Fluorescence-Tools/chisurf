"""Tests for fdb Phase 2 ndxplorer integration."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import numpy as np

from chisurf.core.fio.mmcif.db import FluorophoreDatabase
from chisurf.plugins.sample_database.backend.ndxplorer_services import (
    load_burst_product_handler,
    record_analysis_handler,
)


def test_ndxplorer_load_and_record(tmp_path: pathlib.Path) -> None:
    """Verify that we can load registered products into ndxplorer and record analyses."""
    db_path = tmp_path / "test_ndx.db"

    # Setup dummy burst folder structure
    bur_dir = tmp_path / "bur_output"
    bur_subdir = bur_dir / "bi4_bur"
    bur_subdir.mkdir(parents=True, exist_ok=True)
    
    # Write a dummy .bur file
    # columns: Mean Macro Time (ms), N_ph, duration, dummy (which will be dropped)
    dummy_bur = bur_subdir / "measurement_1.bur"
    dummy_bur.write_text(
        "Mean Macro Time (ms)\tN_ph\tduration\tdummy_col\n"
        "1000\t10\t0.1\t0\n"
        "2000\t20\t0.2\t0\n"
        "3000\t30\t0.3\t0\n"
        "4000\t40\t0.4\t0\n",
        encoding="utf-8",
    )

    with FluorophoreDatabase(db_path) as db:
        db.add_sample("sample_1")
        db.add_experiment("exp_1", sample_id="sample_1", status="complete")
        
        # Add dummy raw reference
        raw_id = db.add_raw_data_reference(
            experiment_id="exp_1",
            data_type="PTU",
            storage_mode="local_file",
            file_path=str(tmp_path / "dummy.ptu"),
            checksum="0" * 64,
        )
        
        # Add dummy burst selection run
        run_id = db.add_processing_run(
            experiment_id="exp_1",
            input_raw_data_ids=[raw_id],
            settings={"burst_detection": {"min_photons": 10}},
            status="succeeded",
        )
        
        # Register the generated burst folder as product
        prod_id = db.add_processed_data_product(
            processing_id=run_id,
            product_type="derived_product",
            storage_mode="folder",
            folder_path=str(bur_dir),
            checksum="1" * 64,
            row_count=4,
            validation_status="valid",
        )

    # Patch database resolver to use our temporary test database
    patcher = patch(
        "chisurf.plugins.sample_database.backend.ndxplorer_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()
    
    try:
        # 1. Test load_burst_product_handler
        result = load_burst_product_handler(processed_data_id=prod_id)
        assert result.get("ok") is True, f"Failed with: {result}"
        assert result["processed_data_id"] == prod_id
        
        # Mean Macro Time (ms) is converted to Mean Macro Time (s), and dummy_col is dropped
        expected_params = ["Mean Macro Time (s)", "N_ph", "duration"]
        assert result["parameter_names"] == expected_params
        
        # The values should be a 2D float32 array of shape (3, 2)
        values = result["values"]
        assert isinstance(values, np.ndarray)
        assert values.shape == (3, 2)
        # Check conversion of macro time from ms to seconds for kept rows (indices 1 and 3)
        assert np.allclose(values[0], [2.0, 4.0])
        assert np.allclose(values[1], [20.0, 40.0])
        assert np.allclose(values[2], [0.2, 0.4])

        # 2. Test record_analysis_handler
        analysis_settings = {"gate": {"min_N_ph": 15}}
        selection_mask_product = {
            "product_type": "selection_mask",
            "storage_mode": "embedded_json",
            "data": {"mask": [False, True]},
            "validation_status": "valid",
        }
        
        rec_result = record_analysis_handler(
            experiment_id="exp_1",
            input_processed_data_ids=[prod_id],
            analysis_type="selection",
            settings=analysis_settings,
            products=[selection_mask_product],
            software_version="1.0.0",
        )
        
        assert rec_result["ok"] is True
        ndx_run = rec_result["processing_run"]
        assert ndx_run["processing_type"] == "ndxplorer_selection"
        assert ndx_run["settings"] == analysis_settings
        
        # Verify registered products list in response
        products = rec_result["products"]
        assert len(products) == 1
        mask_prod = products[0]
        assert mask_prod["product_type"] == "selection_mask"
        assert mask_prod["storage_mode"] == "embedded_json"
        
        # Verify provenance in the database
        with FluorophoreDatabase(db_path) as db:
            # Check input_to edge: original burst product -> input_to -> ndxplorer run
            input_edges = db.get_provenance_edges(
                source_node_type="processed_data",
                source_node_id=prod_id,
                relationship_type="input_to",
            )
            assert len(input_edges) == 1
            assert input_edges[0]["target_node_id"] == ndx_run["processing_id"]
            
            # Check produced edge: ndxplorer run -> produced -> selection mask
            produced_edges = db.get_provenance_edges(
                source_node_type="processing_run",
                source_node_id=ndx_run["processing_id"],
                relationship_type="produced",
            )
            assert len(produced_edges) == 1
            assert produced_edges[0]["target_node_id"] == mask_prod["processed_data_id"]

    finally:
        patcher.stop()
