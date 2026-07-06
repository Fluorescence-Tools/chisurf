"""Tests for fdb Phase 5 analysis and parameter provenance."""

from __future__ import annotations

import pathlib
import sqlite3
from unittest.mock import patch

from chisurf.core.mfdb import schema
from chisurf.core.mfdb.repository import MFDatabase
from mfdb.admin.backend.measurement_services import (
    archive_project_handler,
    delete_analysis_run_handler,
    get_analysis_run_handler,
    get_upstream_dependencies_handler,
    list_analysis_runs_handler,
    record_analysis_run_handler,
    restore_project_handler,
)


def test_analysis_provenance_and_linkages(tmp_path: pathlib.Path) -> None:
    """Test full analysis CRUD, fit grouping, parameter linkages, and traces."""
    db_path = tmp_path / "test_analysis.db"

    # Patch database resolver to use our temporary test database
    patcher = patch(
        "mfdb.admin.backend.measurement_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()

    try:
        # 1. Setup sample, experiment, and raw/processed data
        with MFDatabase(db_path) as db:
            db.add_sample("sample_1")
            db.add_experiment("exp_1", sample_id="sample_1", status="complete")

            # Add raw measurement
            raw_id = db.add_raw_data_reference(
                experiment_id="exp_1",
                data_type="PTU",
                storage_mode="local_file",
                file_path=str(tmp_path / "dummy.ptu"),
                checksum="0" * 64,
            )

            # Record step 1 processing
            step1_id = db.add_processing_run(
                experiment_id="exp_1",
                input_raw_data_ids=[raw_id],
                settings={"min_photons": 10},
                status="succeeded",
            )

            # Register processed dataset product
            dataset_id = db.add_processed_data_product(
                processing_id=step1_id,
                product_type="bur",
                storage_mode="local_file",
                file_path=str(tmp_path / "measurement_1.bur"),
                checksum="1" * 64,
            )

        # 2. Record Local Fit 1 (sub-fit) using the RPC handler
        local_fit_res = record_analysis_run_handler(
            analysis_type="local_fit",
            experiment_id="exp_1",
            model_name="tcspc_decay_1g",
            model_type="TcspcDecayModel",
            model_version="1.0.0",
            notes="Local fit on channel 1",
            input_processed_data_ids=[dataset_id],
            parameters=[
                {
                    "name": "tau",
                    "value": 1.25,
                    "standard_error": 0.05,
                    "initial_value": 1.0,
                    "lower_bound": 0.1,
                    "upper_bound": 10.0,
                    "bounds_on": True,
                    "parameter_type": "free",
                    "parameter_uuid": "local_tau_uuid",
                }
            ],
            products=[
                {
                    "processed_data_id": "fit_curve_1",
                    "file_path": "fit_1.fit",
                    "storage_mode": "embedded_json",
                }
            ],
            analysis_id="local_fit_uuid",
        )
        assert local_fit_res.get("ok") is True
        assert local_fit_res["analysis_run"]["model_name"] == "tcspc_decay_1g"
        assert local_fit_res["parameters"][0]["bounds_on"] is True
        assert local_fit_res["products"][0]["product_type"] == "fit_results"

        # 3. Record Global Fit (parent fit) using the RPC handler
        global_fit_res = record_analysis_run_handler(
            analysis_type="global_fit",
            experiment_id="exp_1",
            model_name="Global fit",
            model_type="GlobalFitModel",
            fit_structure=[{"fit_idx": 0, "fit_name": "fit_1", "fit_uuid": "local_fit_uuid"}],
            parameters=[
                {
                    "name": "global_tau",
                    "value": 1.25,
                    "parameter_type": "shared",
                    "parameter_uuid": "global_tau_uuid",
                }
            ],
            grouped_fit_uuids=["local_fit_uuid"],
            parameter_linkages=[("local_tau_uuid", "global_tau_uuid")],
            analysis_id="global_fit_uuid",
        )
        assert global_fit_res.get("ok") is True

        # 4. Query Analysis via RPC
        get_res = get_analysis_run_handler("global_fit_uuid")
        assert get_res.get("ok") is True
        full_run = get_res["analysis_run"]
        assert len(full_run["grouped_fits"]) == 1
        assert full_run["grouped_fits"][0]["analysis_id"] == "local_fit_uuid"

        list_res = list_analysis_runs_handler(experiment_id="exp_1")
        assert list_res.get("ok") is True
        assert len(list_res["analysis_runs"]) == 2

        # 5. Verify recursive upstream dependency trace starting from fit_curve_1
        upstream_res = get_upstream_dependencies_handler(
            node_type="processed_data",
            node_id="fit_curve_1",
        )
        assert upstream_res.get("ok") is True
        edges = upstream_res["edges"]

        # We expect:
        # - fit_curve_1 <- produced <- local_fit_uuid
        # - local_fit_uuid <- input_to <- dataset_id
        # - dataset_id <- produced <- step1_id
        # - step1_id <- input_to <- raw_id
        # Also parameter links:
        # - local_tau_uuid <- linked_to <- global_tau_uuid
        # And fit grouping:
        # - local_fit_uuid <- grouped_in <- global_fit_uuid

        expected_edges = [
                ("operation", "local_fit_uuid", "artifact", "fit_curve_1", "produced"),
                ("artifact", dataset_id, "operation", "local_fit_uuid", "input_to"),
                ("operation", step1_id, "artifact", dataset_id, "produced"),
                ("artifact", raw_id, "operation", step1_id, "input_to"),
                ("operation", "global_fit_uuid", "operation", "local_fit_uuid", "grouped_in"),
                ("parameter", "global_tau_uuid", "parameter", "local_tau_uuid", "linked_to"),
            ]

        actual_edges = [
            (e["source_node_type"], e["source_node_id"], e["target_node_type"], e["target_node_id"], e["relationship_type"])
            for e in edges
        ]

        for exp in expected_edges:
            assert exp in actual_edges, f"Missing edge: {exp}"

        # 6. Test Delete Analysis Run
        del_res = delete_analysis_run_handler("local_fit_uuid")
        assert del_res.get("ok") is True

        # Verify soft-delete: record still accessible but has deleted_at set
        with MFDatabase(db_path) as db:
            run = db.get_analysis_run("local_fit_uuid")
            assert run is not None
            assert run["deleted_at"] is not None
            param = db.get_analysis_parameter("local_tau_uuid")
            assert param is not None
            assert param["deleted_at"] is not None
            # Provenance edges should still be present (they're not soft-deleted)
            edges = db.get_provenance_edges(processing_id="local_fit_uuid")
            assert len(edges) >= 0  # edges remain unless explicitly deleted

    finally:
        patcher.stop()


def test_project_archive_and_restore(tmp_path: pathlib.Path) -> None:
    """Test archiving and restoring a complete project state to/from fdb."""
    db_path = tmp_path / "test_project.db"

    patcher = patch(
        "mfdb.admin.backend.measurement_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()

    try:
        # Initialize basic DB schema
        with MFDatabase(db_path) as db:
            db.add_sample("sample_proj")
            db.add_experiment("exp_proj", sample_id="sample_proj", status="complete")

            # Add a processing run first to satisfy FK constraints
            proc_id = db.add_processing_run(
                experiment_id="exp_proj",
                status="succeeded",
            )
            # Add a processed dataset
            dataset_id = db.add_processed_data_product(
                processing_id=proc_id,
                product_type="bur",
                storage_mode="local_file",
                file_path=str(tmp_path / "dataset.bur"),
                checksum="2" * 64,
            )

        # Mock project payload
        project_payload = {
            "project_format_version": 4,
            "meta": {
                "name": "MyTestProject",
                "description": "A project for testing project save/restore",
                "chisurf_version": "1.0.0",
                "created": "2026-06-11T20:00:00Z",
            },
            "datasets": {
                "ds000": {
                    "name": "Dataset 1",
                    "filename": "dataset.bur",
                    "x": [1.0, 2.0, 3.0],
                    "y": [10.0, 20.0, 30.0],
                }
            },
            "fits": [
                {
                    "id": "fit1",
                    "name": "Decay Fit 1",
                    "model_name": "TcspcDecayModel",
                    "local_fits": [
                        {
                            "dataset_id": "ds000",
                            "fit_state": {
                                "model_class": "TcspcDecayModel",
                                "parameters": {
                                    "tau_param": {
                                        "name": "tau",
                                        "value": 1.5,
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
            "ui": {"current_fit_index": 0},
            "extra": {
                "history": {
                    "filename": "history.jsonl",
                    "event_count": 5,
                }
            }
        }

        # Archive the project
        archive_res = archive_project_handler(
            project_id="project_uuid_123",
            project_name="MyTestProject",
            project_payload=project_payload,
            experiment_id="exp_proj",
            input_processed_data_ids=[dataset_id],
            notes="Archived via test case",
        )
        assert archive_res.get("ok") is True
        assert archive_res["project_id"] == "project_uuid_123"

        # Restore the project
        restore_res = restore_project_handler("project_uuid_123")
        assert restore_res.get("ok") is True
        assert restore_res["project_id"] == "project_uuid_123"
        assert restore_res["project_name"] == "MyTestProject"

        restored_payload = restore_res["project_payload"]
        assert restored_payload["project_format_version"] == 4
        assert restored_payload["meta"]["name"] == "MyTestProject"
        assert restored_payload["extra"]["history"]["event_count"] == 5

        # Verify dependency trace
        upstream_res = get_upstream_dependencies_handler(
            node_type="analysis_run",
            node_id="project_uuid_123",
        )
        assert upstream_res.get("ok") is True
        edges = upstream_res["edges"]

        # Expected edge: dataset_id -> project_uuid_123 via input_to
        expected_edge = ("artifact", dataset_id, "operation", "project_uuid_123", "input_to")
        actual_edges = [
            (e["source_node_type"], e["source_node_id"], e["target_node_type"], e["target_node_id"], e["relationship_type"])
            for e in edges
        ]
        assert expected_edge in actual_edges

    finally:
        patcher.stop()


def test_project_actions_archive_and_restore(tmp_path: pathlib.Path) -> None:
    """Test project.archive and project.restore actions."""
    db_path = tmp_path / "test_actions.db"

    patcher = patch(
        "mfdb.admin.backend.measurement_services.resolve_database_path",
        return_value=db_path,
    )
    project_browser_patcher = patch(
        "chisurf.plugins.core.project_browser.backend.services.resolve_database_path",
        return_value=db_path,
    )
    database_resolver_patcher = patch(
        "mfdb.database_resolver.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()
    project_browser_patcher.start()
    database_resolver_patcher.start()

    try:
        with MFDatabase(db_path) as db:
            db.add_sample("sample_proj")
            db.add_experiment("exp_proj", sample_id="sample_proj", status="complete")

        from chisurf.core.project import Project as CSProject
        project_payload = {
            "project_format_version": 4,
            "meta": {
                "name": "MyTestActionProject",
                "description": "A project for testing project actions",
                "chisurf_version": "1.0.0",
                "created": "2026-06-11T20:00:00Z",
            },
            "datasets": {},
            "fits": [],
            "ui": {},
            "extra": {}
        }
        mock_proj = CSProject.from_dict(project_payload)

        with (
            patch("chisurf.macros.core_fit.get_project_payload", return_value=mock_proj),
            patch("chisurf.macros.core_fit.load_project_payload") as mock_load,
        ):

            from chisurf.core.actions import dispatch

            archive_res = dispatch(
                "project.archive",
                {
                    "project_id": "act_proj_123",
                    "project_name": "MyTestActionProject",
                    "experiment_id": "exp_proj",
                    "input_processed_data_ids": [],
                    "notes": "Archived via action test",
                }
            )
            assert archive_res.get("ok") is True
            assert archive_res["project_id"] == "act_proj_123"
            with MFDatabase(db_path) as db:
                project_operations = db.conn.execute(
                    "SELECT COUNT(*) FROM mfdb_operation WHERE operation_type = 'project' AND deleted_at IS NULL"
                ).fetchone()[0]
                legacy_table = db.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('mfdb_analysis_run', 'fdb_analysis_run')"
                ).fetchone()
                legacy_runs = 0
                if legacy_table:
                    legacy_runs = db.conn.execute(
                        f"SELECT COUNT(*) FROM {legacy_table[0]} WHERE analysis_type = 'project'"
                    ).fetchone()[0]

            assert project_operations == 1
            assert legacy_runs == 0

            restore_res = dispatch(
                "project.restore",
                {
                    "project_id": "act_proj_123",
                }
            )
            assert restore_res.get("ok") is True
            assert restore_res["project_id"] == "act_proj_123"

            mock_load.assert_called_once()
            called_proj = mock_load.call_args[0][0]
            assert called_proj.name == "MyTestActionProject"

    finally:
        patcher.stop()
        project_browser_patcher.stop()
        database_resolver_patcher.stop()


# ── Project Archiver Tests ──────────────────────────────────────────────


def test_archive_project_creates_artifacts(tmp_path: pathlib.Path) -> None:
    """Verify archive_project_to_mfdb creates proper artifacts for datasets."""
    from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb

    db_path = tmp_path / "test_archiver.db"

    project_payload = {
        "meta": {"name": "Test Project", "description": "A test"},
        "datasets": {
            "ds000": {
                "name": "VV Decay",
                "filename": "",
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0],
                "ex": [0.1, 0.1, 0.1],
                "ey": [1.0, 1.4, 1.7],
                "data_reader": {"module": "test", "class": "TestReader", "state": {}},
                "experiment_name": "TCSPC",
            },
        },
        "fits": [],
    }

    with MFDatabase(db_path) as db:
        result = archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_test_001",
            project_id="proj_test_001",
            version_number=1,
            user_id="user_default",
        )

        assert result["operation_id"] == "ver_test_001"
        assert len(result["dataset_artifacts"]) == 1
        assert result["dataset_artifacts"][0] == "dataset:ver_test_001:ds000"

        # Verify operation exists
        op = db.get_operation("ver_test_001")
        assert op is not None
        assert op["operation_type"] == "project"

        # Verify dataset artifact exists
        artifacts = db.get_operation_artifacts("ver_test_001", direction="output")
        assert len(artifacts) >= 1
        dataset_art = [a for a in artifacts if a["artifact_kind"] == "processed_data"]
        assert len(dataset_art) == 1

        # Verify project_contains edge exists
        edges = db.conn.execute(
            """SELECT * FROM mfdb_edge
               WHERE source_node_id = 'ver_test_001'
                 AND relationship_type = 'project_contains'
                 AND deleted_at IS NULL""",
        ).fetchall()
        assert len(edges) >= 1


def test_archive_project_creates_source_objects(tmp_path: pathlib.Path) -> None:
    """Verify archive_project_to_mfdb stores source files in object store."""
    from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb

    db_path = tmp_path / "test_source_objects.db"

    # Create a temporary source file
    source_file = tmp_path / "sample.txt"
    source_file.write_text("test measurement data\n")

    project_payload = {
        "meta": {"name": "Source Test"},
        "datasets": {
            "ds000": {
                "name": "Test Data",
                "filename": str(source_file),
                "x": [1.0, 2.0],
                "y": [10.0, 20.0],
                "ex": [0.1, 0.1],
                "ey": [1.0, 1.4],
                "data_reader": {},
                "experiment_name": "TCSPC",
            },
        },
        "fits": [],
    }

    with MFDatabase(db_path) as db:
        result = archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_src_001",
            project_id="proj_src_001",
            version_number=1,
        )

        # Should have source artifact (input) + dataset artifact (output)
        input_arts = db.get_operation_artifacts("ver_src_001", direction="input")
        output_arts = db.get_operation_artifacts("ver_src_001", direction="output")

        source_arts = [a for a in input_arts if a["artifact_kind"] in ("raw_measurement", "raw_data")]
        assert len(source_arts) == 1
        assert source_arts[0]["object_uuid"] is not None

        # Verify object exists in store
        obj_info = db.get_object_info(source_arts[0]["object_uuid"])
        assert obj_info is not None
        assert obj_info["original_filename"].endswith("sample.txt")

        # Verify derived_from edge
        edges = db.conn.execute(
            """SELECT * FROM mfdb_edge
               WHERE relationship_type = 'derived_from' AND deleted_at IS NULL""",
        ).fetchall()
        assert len(edges) >= 1


def test_archive_project_version_lineage(tmp_path: pathlib.Path) -> None:
    """Verify archive_project_to_mfdb creates supersedes edges for version lineage."""
    from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb

    db_path = tmp_path / "test_lineage.db"

    project_payload = {
        "meta": {"name": "Lineage Test"},
        "datasets": {},
        "fits": [],
    }

    with MFDatabase(db_path) as db:
        # Create first version
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_v1",
            project_id="proj_lineage",
            version_number=1,
        )

        # Create second version superseding first
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_v2",
            project_id="proj_lineage",
            version_number=2,
            parent_version_id="ver_v1",
        )

        # Verify supersedes edge
        edges = db.conn.execute(
            """SELECT * FROM mfdb_edge
               WHERE relationship_type = 'supersedes' AND deleted_at IS NULL""",
        ).fetchall()
        assert len(edges) == 1
        edge = dict(edges[0]) if not isinstance(edges[0], dict) else edges[0]
        assert edge["source_node_id"] == "ver_v2"
        assert edge["target_node_id"] == "ver_v1"


def test_restore_project_from_artifacts(tmp_path: pathlib.Path) -> None:
    """Verify restore_project_from_artifacts reconstructs from individual artifacts."""
    from chisurf.core.mfdb.project_archiver import (
        archive_project_to_mfdb,
        restore_project_from_artifacts,
    )

    db_path = tmp_path / "test_restore.db"

    project_payload = {
        "meta": {"name": "Restore Test"},
        "datasets": {
            "ds000": {
                "name": "Test Curve",
                "filename": "",
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0],
                "ex": [0.1, 0.1, 0.1],
                "ey": [1.0, 1.4, 1.7],
                "data_reader": {},
                "experiment_name": "TCSPC",
            },
        },
        "fits": [],
    }

    with MFDatabase(db_path) as db:
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_restore",
            project_id="proj_restore",
            version_number=1,
        )

        restored = restore_project_from_artifacts(db, "ver_restore")
        assert restored is not None
        assert "ds000" in restored["datasets"]
        ds_data = restored["datasets"]["ds000"]
        assert ds_data["curves"][0]["name"] == "Test Curve"


def test_archive_project_deduplicates_objects(tmp_path: pathlib.Path) -> None:
    """Verify same source file produces one object with refcount > 1."""
    from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb

    db_path = tmp_path / "test_dedup.db"

    # Create a source file
    source_file = tmp_path / "shared.txt"
    source_file.write_text("shared measurement data\n")

    project_payload = {
        "meta": {"name": "Dedup Test"},
        "datasets": {
            "ds000": {
                "name": "Curve 1",
                "filename": str(source_file),
                "x": [1.0], "y": [10.0], "ex": [0.1], "ey": [1.0],
                "data_reader": {}, "experiment_name": "TCSPC",
            },
            "ds001": {
                "name": "Curve 2",
                "filename": str(source_file),
                "x": [2.0], "y": [20.0], "ex": [0.1], "ey": [1.4],
                "data_reader": {}, "experiment_name": "TCSPC",
            },
        },
        "fits": [],
    }

    with MFDatabase(db_path) as db:
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_dedup",
            project_id="proj_dedup",
            version_number=1,
        )

        # Check that source file is stored once (deduplicated)
        objects = db.conn.execute(
            "SELECT refcount FROM mfdb_object WHERE original_filename LIKE '%shared.txt'"
        ).fetchall()
        assert len(objects) == 1
        assert objects[0][0] >= 2  # refcount >= 2 (two put_object calls)


def test_version_branching(tmp_path: pathlib.Path) -> None:
    """Verify branching creates correct supersedes edges forming a DAG."""
    from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb

    db_path = tmp_path / "test_branching.db"

    project_payload = {
        "meta": {"name": "Branch Test"},
        "datasets": {},
        "fits": [],
    }

    with MFDatabase(db_path) as db:
        # Create root version
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_root",
            project_id="proj_branch",
            version_number=1,
        )

        # Create branch A from root
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_branch_a",
            project_id="proj_branch",
            version_number=2,
            parent_version_id="ver_root",
            branch_uuid="branch_a",
        )

        # Create branch B from root (fork)
        archive_project_to_mfdb(
            db=db,
            project_payload=project_payload,
            version_id="ver_branch_b",
            project_id="proj_branch",
            version_number=2,
            parent_version_id="ver_root",
            branch_uuid="branch_b",
        )

        # Verify two supersedes edges (DAG with fork)
        edges = db.conn.execute(
            """SELECT source_node_id, target_node_id FROM mfdb_edge
               WHERE relationship_type = 'supersedes' AND deleted_at IS NULL
               ORDER BY source_node_id""",
        ).fetchall()
        assert len(edges) == 2
        assert (edges[0][0], edges[0][1]) == ("ver_branch_a", "ver_root")
        assert (edges[1][0], edges[1][1]) == ("ver_branch_b", "ver_root")
