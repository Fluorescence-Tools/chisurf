import pathlib
from unittest.mock import patch

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.plugins.sample_database.backend.measurement_services import (
    archive_project_handler,
    database_backup_handler,
    restore_project_handler,
)
from chisurf.plugins.sample_database.gui.client import SampleDatabaseClient


def test_automatic_repository_audit_logging(tmp_path: pathlib.Path) -> None:
    """Verify that core repository operations automatically write audit log entries."""
    db_path = tmp_path / "audit_test.db"

    with MFDatabase(db_path) as db:
        # Initial logs must be empty
        assert len(db.get_audit_logs()) == 0

        db.add_sample("sample_1")
        db.add_experiment("exp_1", sample_id="sample_1", status="complete")

        # 1. Register raw reference -> audit create raw_data
        raw_id = db.add_raw_data_reference(
            experiment_id="exp_1",
            data_type="ptu",
            storage_mode="local_file",
            file_path=str(tmp_path / "raw.ptu"),
            checksum="0" * 64,
        )

        logs = db.get_audit_logs(target_type="raw_data")
        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["target_id"] == raw_id

        # 2. Add processing run -> audit create processing_run
        proc_id = db.add_processing_run(
            experiment_id="exp_1",
            status="succeeded",
            processing_type="burst_selection",
        )

        logs = db.get_audit_logs(target_type="processing_run")
        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["target_id"] == proc_id

        # 3. Update processing run status -> audit update processing_run
        db.update_processing_run_status(
            processing_id=proc_id,
            status="failed",
            error_message="Test failure",
        )

        logs = db.get_audit_logs(target_type="processing_run")
        assert len(logs) == 2
        # Order is DESC, so first log is the update
        assert logs[0]["action"] == "update"
        assert logs[0]["details"]["status"] == "failed"

        # 4. Add processed product -> audit create processed_data
        prod_id = db.add_processed_data_product(
            processing_id=proc_id,
            product_type="bur",
            storage_mode="local_file",
            file_path=str(tmp_path / "output.bur"),
            checksum="1" * 64,
        )

        logs = db.get_audit_logs(target_type="processed_data")
        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["target_id"] == prod_id

        # 5. Add setup definition -> audit create setup_definition
        db.add_setup_definition(
            setup_id="setup_mfd",
            name="MFD ALEX Setup",
        )

        logs = db.get_audit_logs(target_type="setup_definition")
        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["target_id"] == "setup_mfd"

        # 6. Add analysis run -> audit create analysis_run
        anal_id = db.add_analysis_run(
            analysis_type="local_fit",
            experiment_id="exp_1",
            model_name="FRET Fit Model",
            model_type="FretFit",
        )

        logs = db.get_audit_logs(target_type="analysis_run")
        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["target_id"] == anal_id

        # 7. Delete analysis run -> audit delete analysis_run
        db.delete_analysis_run(anal_id)

        logs = db.get_audit_logs(target_type="analysis_run")
        assert len(logs) == 2
        assert logs[0]["action"] == "delete"
        assert logs[0]["target_id"] == anal_id


def test_service_level_audit_logging(tmp_path: pathlib.Path) -> None:
    """Verify that high-level service handlers correctly log backup, zip and project events."""
    db_path = tmp_path / "service_audit.db"

    # Patch database resolver to use our temporary test database
    patcher = patch(
        "chisurf.plugins.sample_database.backend.measurement_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()

    try:
        # Init db tables
        with MFDatabase(db_path) as db:
            db.add_sample("sample_1")
            db.add_experiment("exp_1", sample_id="sample_1", status="complete")

        # 1. Test database backup logs backup event
        backup_path = str(tmp_path / "backup.db")
        res = database_backup_handler(target_path=backup_path)
        assert res["ok"] is True

        with MFDatabase(db_path) as db:
            logs = db.get_audit_logs(action="backup")
            assert len(logs) == 1
            assert logs[0]["target_type"] == "database"
            assert logs[0]["target_id"] == backup_path

        # 2. Test archive project logs archive event
        project_payload = {"some": "state"}
        res = archive_project_handler(
            project_id="proj_1",
            project_name="DNA smFRET",
            project_payload=project_payload,
            experiment_id="exp_1",
        )
        assert res["ok"] is True

        with MFDatabase(db_path) as db:
            logs = db.get_audit_logs(action="archive", target_type="project")
            assert len(logs) == 1
            assert logs[0]["target_id"] == "proj_1"
            assert logs[0]["details"]["project_name"] == "DNA smFRET"

        # 3. Test restore project logs restore event
        res = restore_project_handler(project_id="proj_1")
        assert res["ok"] is True

        with MFDatabase(db_path) as db:
            logs = db.get_audit_logs(action="restore", target_type="project")
            assert len(logs) == 1
            assert logs[0]["target_id"] == "proj_1"
            assert logs[0]["details"]["project_name"] == "DNA smFRET"

    finally:
        patcher.stop()


def test_client_list_audit_logs(tmp_path: pathlib.Path) -> None:
    """Verify that we can retrieve audit logs via client wrapper method."""
    db_path = tmp_path / "client_audit.db"

    # Patch database resolver to use our temporary test database
    patcher = patch(
        "chisurf.plugins.sample_database.backend.measurement_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()


    try:
        # Create some logs
        with MFDatabase(db_path) as db:
            db.add_sample("sample_1")
            db.add_experiment("exp_1", sample_id="sample_1", status="complete")
            db.add_raw_data_reference(
                experiment_id="exp_1",
                data_type="ptu",
                storage_mode="local_file",
                file_path=str(tmp_path / "raw.ptu"),
                checksum="0" * 64,
            )

        client = SampleDatabaseClient()
        logs = client.list_audit_logs()
        assert len(logs) == 1
        assert logs[0]["target_type"] == "raw_data"
        assert logs[0]["action"] == "create"

        # Filter by target_type
        logs_filtered = client.list_audit_logs(target_type="nonexistent")
        assert len(logs_filtered) == 0

    finally:
        patcher.stop()
