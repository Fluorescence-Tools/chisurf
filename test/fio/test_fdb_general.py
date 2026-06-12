"""Tests for fdb Phase 4 general processing and dependency queries."""

from __future__ import annotations

import pathlib
from unittest.mock import patch
import pytest

from chisurf.core.fio.mmcif.db import FluorophoreDatabase
from chisurf.plugins.sample_database.backend.measurement_services import (
    record_general_processing_run_handler,
    get_upstream_dependencies_handler,
    get_downstream_dependencies_handler,
)


def test_dependency_trace_and_general_processing(tmp_path: pathlib.Path) -> None:
    """Verify recursive dependency traces and general processing run registration."""
    db_path = tmp_path / "test_general.db"

    # Patch database resolver to use our temporary test database
    patcher = patch(
        "chisurf.plugins.sample_database.backend.measurement_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()

    try:
        # 1. Setup sample and experiment
        with FluorophoreDatabase(db_path) as db:
            db.add_sample("sample_1")
            db.add_experiment("exp_1", sample_id="sample_1", status="complete")

            # Add dummy raw reference
            raw_id = db.add_raw_data_reference(
                experiment_id="exp_1",
                data_type="PTU",
                storage_mode="local_file",
                file_path=str(tmp_path / "dummy.ptu"),
                checksum="raw-sha",
            )

        # 2. Record Step 1: Raw data -> burst selection -> burst table product.
        step1_res = record_general_processing_run_handler(
            experiment_id="exp_1",
            processing_type="burst_selection",
            processing_id="step1_run",
            input_raw_data_ids=[raw_id],
            settings={"min_photons": 10},
            products=[
                {
                    "processed_data_id": "step1_product",
                    "file_path": str(tmp_path / "measurement_1.bur"),
                    "storage_mode": "local_file",
                    "row_count": 100,
                    "validation_status": "valid",
                }
            ],
            status="succeeded",
        )
        assert step1_res.get("ok") is True
        assert step1_res["products"][0]["product_type"] == "bur"

        # 3. Record Step 2: Burst table product -> ndxplorer filtering -> selection mask.
        step2_res = record_general_processing_run_handler(
            experiment_id="exp_1",
            processing_type="burst_filtering",
            processing_id="step2_run",
            input_processed_data_ids=["step1_product"],
            settings={"filter_threshold": 1.5},
            products=[
                {
                    "processed_data_id": "step2_product",
                    "file_path": str(tmp_path / "mask.json"),
                    "storage_mode": "embedded_json",
                    "data": {"mask": [True, False, True]},
                    "validation_status": "valid",
                }
            ],
            status="succeeded",
        )
        assert step2_res.get("ok") is True
        assert step2_res["products"][0]["product_type"] == "json_summary"

        # 4. Record Step 3: Selection mask -> GMM fitting -> model parameters product.
        step3_res = record_general_processing_run_handler(
            experiment_id="exp_1",
            processing_type="gmm_fitting",
            processing_id="step3_run",
            input_processed_data_ids=["step2_product"],
            settings={"n_components": 2},
            products=[
                {
                    "processed_data_id": "step3_product",
                    "file_path": str(tmp_path / "fit.json"),
                    "storage_mode": "embedded_json",
                    "data": {"mu": [0.1, 0.9]},
                    "validation_status": "valid",
                }
            ],
            status="succeeded",
        )
        assert step3_res.get("ok") is True

        # 5. Test Upstream Dependency Query starting from step3_product
        upstream_res = get_upstream_dependencies_handler(
            node_type="processed_data",
            node_id="step3_product",
        )
        assert upstream_res.get("ok") is True
        edges = upstream_res["edges"]
        
        # Verify the dependency path back to raw data
        # We expect:
        # - step3_product (target) <- produced <- step3_run (source)
        # - step3_run (target) <- input_to <- step2_product (source)
        # - step2_product (target) <- produced <- step2_run (source)
        # - step2_run (target) <- input_to <- step1_product (source)
        # - step1_product (target) <- produced <- step1_run (source)
        # - step1_run (target) <- input_to <- raw_id (source)
        
        expected_upstream = [
            ("processing_run", "step3_run", "processed_data", "step3_product", "produced"),
            ("processed_data", "step2_product", "processing_run", "step3_run", "input_to"),
            ("processing_run", "step2_run", "processed_data", "step2_product", "produced"),
            ("processed_data", "step1_product", "processing_run", "step2_run", "input_to"),
            ("processing_run", "step1_run", "processed_data", "step1_product", "produced"),
            ("raw_data", raw_id, "processing_run", "step1_run", "input_to"),
        ]
        
        actual_upstream = [
            (e["source_node_type"], e["source_node_id"], e["target_node_type"], e["target_node_id"], e["relationship_type"])
            for e in edges
        ]
        
        for exp in expected_upstream:
            assert exp in actual_upstream

        # 6. Test Downstream Dependency Query starting from raw_data
        downstream_res = get_downstream_dependencies_handler(
            node_type="raw_data",
            node_id=raw_id,
        )
        assert downstream_res.get("ok") is True
        edges_down = downstream_res["edges"]
        
        actual_downstream = [
            (e["source_node_type"], e["source_node_id"], e["target_node_type"], e["target_node_id"], e["relationship_type"])
            for e in edges_down
        ]
        
        for exp in expected_upstream:
            assert exp in actual_downstream

        # 7. Verify _infer_product_type suffix mapping via record_general_processing_run_handler
        suffixes_res = record_general_processing_run_handler(
            experiment_id="exp_1",
            processing_type="fcs_correlation",
            processing_id="fcs_run",
            input_raw_data_ids=[raw_id],
            products=[
                {
                    "processed_data_id": "fcs_prod",
                    "file_path": "test.fcs",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "dec_prod",
                    "file_path": "test.dec",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "tcspc_prod",
                    "file_path": "test.tcspc",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "irf_prod",
                    "file_path": "test.irf",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "spc_prod",
                    "file_path": "test.spc",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "spectra_prod",
                    "file_path": "test.spectra",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "aniso_prod",
                    "file_path": "test.aniso",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "pda_prod",
                    "file_path": "test.pda",
                    "storage_mode": "embedded_json",
                },
                {
                    "processed_data_id": "fit_prod",
                    "file_path": "test.fit",
                    "storage_mode": "embedded_json",
                },
            ],
            status="succeeded",
        )
        assert suffixes_res.get("ok") is True
        prods = {p["processed_data_id"]: p["product_type"] for p in suffixes_res["products"]}
        assert prods["fcs_prod"] == "fcs_correlation"
        assert prods["dec_prod"] == "tcspc_decay"
        assert prods["tcspc_prod"] == "tcspc_decay"
        assert prods["irf_prod"] == "irf_curve"
        assert prods["spc_prod"] == "spectra"
        assert prods["spectra_prod"] == "spectra"
        assert prods["aniso_prod"] == "anisotropy_curve"
        assert prods["pda_prod"] == "pda_histogram"
        assert prods["fit_prod"] == "fit_results"

    finally:
        patcher.stop()
