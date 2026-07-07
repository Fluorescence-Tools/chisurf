from __future__ import annotations

import base64
import copy
from pathlib import Path

import pytest


@pytest.fixture
def project_db(tmp_path, monkeypatch):
    from mfdb.security.auth import create_session
    from mfdb.repository import MFDatabase
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
        "mfdb.store.database_resolver.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "mfdb.store.database_resolver.object_store_root",
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


def _payload_with_fit(sample_project_payload):
    payload = copy.deepcopy(sample_project_payload)
    payload["fits"] = [
        {
            "id": "fit_sample",
            "name": "Sample Fit",
            "model_name": "Linear",
            "local_fits": [
                {
                    "id": "local_0",
                    "dataset_id": "ds_sample",
                    "fit_state": {
                        "model_module": "chisurf.test",
                        "model_class": "SampleModel",
                        "parameters": {
                            "amp": {
                                "uid": "amp",
                                "name": "amp",
                                "value": 2.0,
                                "fixed": False,
                                "bounds": [0.0, 10.0],
                                "bounds_on": True,
                                "link_target": None,
                            },
                            "tau": {
                                "uid": "tau",
                                "name": "tau",
                                "value": 4.0,
                                "fixed": True,
                                "bounds": [0.0, 20.0],
                                "bounds_on": False,
                                "link_target": None,
                            },
                        },
                    },
                },
            ],
        },
    ]
    return payload


def test_artifact_and_parameter_browsing_for_project_version(
    project_db,
    sample_project_payload,
):
    from chisurf.plugins.core.project_browser.backend.services import (
        list_project_artifacts_handler,
        list_project_parameters_handler,
        save_project_handler,
    )

    auth = project_db["auth"]
    saved = save_project_handler(
        auth=auth,
        project_name="Sample Project",
        project_payload=_payload_with_fit(sample_project_payload),
        visibility="private",
        notes="artifact and parameter coverage",
    )
    assert saved["ok"] is True
    assert saved["artifact_count"] >= 2
    assert saved["parameter_count"] == 2

    artifacts = list_project_artifacts_handler(auth=auth, version_id=saved["version_id"])
    assert artifacts["ok"] is True
    artifact_kinds = {artifact["artifact_kind"] for artifact in artifacts["artifacts"]}
    assert {"processed_data", "fit_result"}.issubset(artifact_kinds)

    parameters = list_project_parameters_handler(auth=auth, version_id=saved["version_id"])
    assert parameters["ok"] is True
    by_name = {parameter["name"]: parameter for parameter in parameters["parameters"]}
    assert by_name["amp"]["value"] == 2.0
    assert by_name["amp"]["bounds_on"] == 1
    assert by_name["tau"]["parameter_type"] == "fixed"


def test_branch_dag_and_version_graph_roots_and_leaves(
    project_db,
    sample_project_payload,
):
    from chisurf.plugins.core.project_browser.backend.services import (
        create_branch_handler,
        get_version_graph_handler,
        list_branches_handler,
        save_project_handler,
    )

    auth = project_db["auth"]
    root = save_project_handler(
        auth=auth,
        project_name="Branch Project",
        project_payload=sample_project_payload,
        visibility="private",
        notes="root",
    )
    assert root["ok"] is True

    main_child = save_project_handler(
        auth=auth,
        project_name="Branch Project",
        project_payload=sample_project_payload,
        project_id=root["project_id"],
        parent_version_id=root["version_id"],
        visibility="private",
        notes="main child",
    )
    assert main_child["ok"] is True

    branch = create_branch_handler(
        auth=auth,
        project_id=root["project_id"],
        from_version_id=root["version_id"],
        branch_name="analysis fork",
    )
    assert branch["ok"] is True
    fork_child = save_project_handler(
        auth=auth,
        project_name="Branch Project",
        project_payload=sample_project_payload,
        project_id=root["project_id"],
        parent_version_id=root["version_id"],
        branch_uuid=branch["branch_uuid"],
        visibility="private",
        notes="fork child",
    )
    assert fork_child["ok"] is True

    branches = list_branches_handler(auth=auth, project_id=root["project_id"])
    assert branches["ok"] is True
    branch_counts = {item["name"]: item["version_count"] for item in branches["branches"]}
    assert branch_counts[f"project_{root['project_id']}"] == 2
    assert branch_counts["analysis fork"] == 1

    graph = get_version_graph_handler(auth=auth, project_id=root["project_id"])
    assert graph["ok"] is True
    assert set(graph["graph"]["roots"]) == {root["version_id"]}
    assert set(graph["graph"]["leaves"]) == {
        main_child["version_id"],
        fork_child["version_id"],
    }
    assert {
        (edge["source"], edge["target"]) for edge in graph["graph"]["edges"]
    } == {
        (main_child["version_id"], root["version_id"]),
        (fork_child["version_id"], root["version_id"]),
    }


def test_import_collision_preview_requires_and_applies_remap(
    project_db,
    sample_project_payload,
):
    from chisurf.plugins.core.project_browser.backend.services import (
        export_csp_handler,
        import_csp_handler,
        import_preview_handler,
        list_projects_handler,
        save_project_handler,
    )

    auth = project_db["auth"]
    saved = save_project_handler(
        auth=auth,
        project_name="Collision Project",
        project_payload=_payload_with_fit(sample_project_payload),
        visibility="private",
        notes="collision source",
    )
    assert saved["ok"] is True
    exported = export_csp_handler(auth=auth, version_id=saved["version_id"])
    assert exported["ok"] is True

    preview = import_preview_handler(auth=auth, archive_base64=exported["archive_bytes"])
    assert preview["ok"] is True
    assert preview["has_collisions"] is True
    assert saved["version_id"] in preview["collisions"]["operations"]

    rejected = import_csp_handler(auth=auth, archive_base64=exported["archive_bytes"])
    assert rejected["ok"] is False
    accepted = import_csp_handler(
        auth=auth,
        archive_base64=exported["archive_bytes"],
        resolve_collisions=True,
    )
    assert accepted["ok"] is True
    assert accepted["id_remap"]["operations"][saved["version_id"]] == accepted["version_id"]

    listed = list_projects_handler(auth=auth)
    assert listed["ok"] is True
    projects = [project for project in listed["projects"] if project["project_id"] == saved["project_id"]]
    assert projects[0]["version_count"] == 2


def test_delete_version_requires_manage_permission(project_db, sample_project_payload):
    from mfdb.security.auth import create_session
    from mfdb.repository import MFDatabase
    from chisurf.plugins.core.project_browser.backend.services import (
        delete_version_handler,
        list_projects_handler,
        save_project_handler,
    )

    auth = project_db["auth"]
    saved = save_project_handler(
        auth=auth,
        project_name="Delete Project",
        project_payload=sample_project_payload,
        visibility="public",
        notes="delete permission coverage",
    )
    assert saved["ok"] is True

    with MFDatabase(project_db["path"]) as db:
        db.add_user("viewer_user", "Viewer User")
        viewer_session = create_session(db.conn, "viewer_user", client_name="pytest-viewer")
        db.conn.commit()

    viewer_auth = {"token": viewer_session["token"]}
    denied = delete_version_handler(auth=viewer_auth, version_id=saved["version_id"])
    assert denied["ok"] is False

    deleted = delete_version_handler(auth=auth, version_id=saved["version_id"])
    assert deleted["ok"] is True
    listed = list_projects_handler(auth=auth)
    assert listed["ok"] is True
    assert listed["projects"] == []
