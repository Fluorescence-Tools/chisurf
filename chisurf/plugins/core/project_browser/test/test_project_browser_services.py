from __future__ import annotations

import base64
from pathlib import Path

import pytest


@pytest.fixture
def project_db(tmp_path, monkeypatch):
    from chisurf.core.mfdb.auth import create_session
    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.plugins.core.project_browser.backend import services

    db_path = tmp_path / "project_browser.db"
    object_root = tmp_path / "objects"
    with MFDatabase(db_path) as db:
        db.add_user("user_default", "Default User")
        db.add_user("admin_user", "Admin User", is_admin=1)
        session = create_session(db.conn, "admin_user", client_name="pytest")
        db.conn.commit()

    monkeypatch.setattr(services, "resolve_database_path", lambda: db_path)
    monkeypatch.setattr(
        "chisurf.core.mfdb.database_resolver.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.database_resolver.object_store_root",
        lambda: object_root,
    )
    return {"path": db_path, "auth": {"token": session["token"]}, "tmp_path": tmp_path}


@pytest.fixture
def sample_project_payload(project_db):
    data_path = project_db["tmp_path"] / "sample_curve.csv"
    data_path.write_text("time,intensity\n0,10\n1,12\n2,11\n", encoding="utf-8")
    return {
        "project_format_version": 5,
        "meta": {
            "name": "Sample Project",
            "description": "sample-data-backed service test",
            "chisurf_version": "test",
        },
        "datasets": {
            "ds_sample": {
                "uid": "ds_sample",
                "name": "Sample Curve",
                "filename": str(data_path),
                "experiment_name": "TCSPC",
                "x": [0.0, 1.0, 2.0],
                "y": [10.0, 12.0, 11.0],
                "data_reader": {
                    "module": "chisurf.test",
                    "class": "SampleReader",
                    "state": {"source": "pytest"},
                },
            },
        },
        "fits": [],
        "experiments": {},
        "ui": {},
        "extra": {},
    }


def test_save_list_restore_and_export_sample_project(project_db, sample_project_payload):
    from chisurf.plugins.core.project_browser.backend.services import (
        export_csp_handler,
        list_projects_handler,
        restore_project_handler,
        save_project_handler,
    )

    auth = project_db["auth"]
    saved = save_project_handler(
        auth=auth,
        project_name="Sample Project",
        project_payload=sample_project_payload,
        visibility="private",
        notes="sample payload",
    )

    assert saved["ok"] is True
    assert saved["artifact_count"] == 1

    listed = list_projects_handler(auth=auth)
    assert listed["ok"] is True
    assert len(listed["projects"]) == 1
    project = listed["projects"][0]
    assert project["project_name"] == "Sample Project"
    assert project["latest_version_id"] == saved["version_id"]
    assert project["versions"][0]["dataset_count"] == 1

    restored = restore_project_handler(auth=auth, version_id=saved["version_id"])
    assert restored["ok"] is True
    restored_payload = restored["project_payload"]
    assert "ds_sample" in restored_payload["datasets"]
    assert restored_payload["datasets"]["ds_sample"]["curves"][0]["name"] == "Sample Curve"

    exported = export_csp_handler(auth=auth, version_id=saved["version_id"])
    assert exported["ok"] is True
    assert base64.b64decode(exported["archive_bytes"])


def test_project_browser_services_register_with_dispatcher(project_db):
    from chisurf.core.plugin.client import InProcessClient
    from chisurf.plugins.core.project_browser.backend.services import register_services
    from chisurf.server.dispatcher import ServiceDispatcher
    from chisurf.server.session import SessionState

    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)
    client = InProcessClient(dispatcher)

    result = client.call("project_browser.list", {"auth": project_db["auth"]})

    assert result["ok"] is True
    assert result["projects"] == []
    assert "project_browser.create_branch" in dispatcher.list_methods()
    assert "project_browser.parameters" in dispatcher.list_methods()
