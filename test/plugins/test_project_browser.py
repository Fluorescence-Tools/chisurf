from __future__ import annotations

import json
import uuid
import base64
import pathlib
from unittest.mock import patch, MagicMock

import pytest

from chisurf.core.mfdb.security.auth import PERM_READ, PERM_WRITE, PERM_MANAGE
from chisurf.core.project.archive import ProjectArchive
from chisurf.plugins.core.project_browser.backend.services import (
    list_projects_handler,
    save_project_handler,
    restore_project_handler,
    export_csp_handler,
    import_preview_handler,
    import_csp_handler,
    delete_version_handler,
    _find_collisions,
    _generate_id_remap,
    _apply_remap_to_export,
)


@pytest.fixture
def mock_auth():
    return {"token": "test-session-token"}


@pytest.fixture
def admin_auth():
    return {"token": "admin-session-token"}


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Create a temporary MFDB and patch resolve_database_path to use it."""
    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.security.auth import _hash_token

    db_path = tmp_path / "test_mfdb.db"
    db = MFDatabase(db_path)

    session_token = "test-session-token"
    token_hash = _hash_token(session_token)
    admin_token = "admin-session-token"
    admin_hash = _hash_token(admin_token)
    db.conn.execute(
        """INSERT INTO flr_sample_users (user_id, display_name, is_admin)
           VALUES (?, ?, ?)""",
        ("user_test", "Test User", 0),
    )
    db.conn.execute(
        """INSERT INTO flr_sample_users (user_id, display_name, is_admin)
           VALUES (?, ?, ?)""",
        ("admin_user", "Admin User", 1),
    )
    db.conn.execute(
        """INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at)
           VALUES (?, ?, ?, ?)""",
        ("sess_test", "user_test", token_hash, "2099-12-31T23:59:59"),
    )
    db.conn.execute(
        """INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at)
           VALUES (?, ?, ?, ?)""",
        ("sess_admin", "admin_user", admin_hash, "2099-12-31T23:59:59"),
    )
    db.conn.commit()

    def _mock_resolve():
        return db_path

    monkeypatch.setattr(
        "chisurf.plugins.core.project_browser.backend.services.resolve_database_path",
        _mock_resolve,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.resolve_database_path",
        _mock_resolve,
    )

    yield db.conn, db_path

    db.close()


@pytest.fixture
def sample_payload():
    return {
        "name": "test_project",
        "project_format_version": 4,
        "meta": {
            "chisurf_version": "1.0.0",
            "created": "2026-01-01T00:00:00",
        },
        "datasets": {},
        "experiments": {},
        "fits": [],
        "ui": {},
        "extra": {},
    }


class TestListProjects:
    def test_empty(self, temp_db, mock_auth):
        result = list_projects_handler(auth=mock_auth)
        assert result["ok"]
        assert result["projects"] == []

    def test_after_save(self, temp_db, mock_auth):
        save_project_handler(
            auth=mock_auth,
            project_name="TestProject",
            project_payload={"test": "data"},
        )
        result = list_projects_handler(auth=mock_auth)
        assert result["ok"]
        assert len(result["projects"]) == 1
        proj = result["projects"][0]
        assert proj["project_name"] == "TestProject"
        assert proj["project_id"] is not None
        assert proj["version_count"] == 1


class TestSaveProject:
    def test_version_isolation(self, temp_db, mock_auth):
        r1 = save_project_handler(
            auth=mock_auth,
            project_name="Proj",
            project_payload={"v": 1},
        )
        assert r1["ok"]
        v1_id = r1["version_id"]

        r2 = save_project_handler(
            auth=mock_auth,
            project_name="Proj",
            project_payload={"v": 2},
            project_id=r1["project_id"],
            parent_version_id=r1["version_id"],
        )
        assert r2["ok"]
        v2_id = r2["version_id"]

        # Versions must be different
        assert v1_id != v2_id
        # Version numbers must be sequential
        assert r1["version_number"] == 1
        assert r2["version_number"] == 2

        # List should show 2 versions grouped under 1 project
        result = list_projects_handler(auth=mock_auth)
        assert len(result["projects"]) == 1
        proj = result["projects"][0]
        assert proj["version_count"] == 2

    def test_no_overwrite(self, temp_db, mock_auth):
        r1 = save_project_handler(
            auth=mock_auth,
            project_name="NoOverwrite",
            project_payload={"data": "a"},
        )
        r2 = save_project_handler(
            auth=mock_auth,
            project_name="NoOverwrite",
            project_payload={"data": "b"},
            project_id=r1["project_id"],
        )
        assert r2["version_number"] > r1["version_number"]

        # Both versions should exist
        list_result = list_projects_handler(auth=mock_auth)
        proj = list_result["projects"][0]
        assert proj["version_count"] == 2

    def test_ownership(self, temp_db, admin_auth):
        result = save_project_handler(
            auth=admin_auth,
            project_name="OwnedProject",
            project_payload={},
        )
        assert result["ok"]
        list_result = list_projects_handler(auth=admin_auth)
        proj = list_result["projects"][0]
        assert proj["owner_user_id"] == "admin_user"

    def test_owner_uses_authenticated_user(self, temp_db, mock_auth):
        result = save_project_handler(
            auth=mock_auth,
            project_name="OwnedProject",
            project_payload={},
        )
        assert result["ok"]
        list_result = list_projects_handler(auth=mock_auth)
        proj = list_result["projects"][0]
        assert proj["owner_user_id"] == "user_test"

    def test_counts_from_payload(self, temp_db, mock_auth):
        result = save_project_handler(
            auth=mock_auth,
            project_name="CountProject",
            project_payload={"datasets": {"ds1": {}, "ds2": {}}, "fits": [{"uid": "f1"}]},
        )
        assert result["ok"]
        list_result = list_projects_handler(auth=mock_auth)
        proj = list_result["projects"][0]
        assert proj["versions"][0]["dataset_count"] == 2
        assert proj["versions"][0]["fit_count"] == 1

    def test_visibility_private(self, temp_db, mock_auth):
        result = save_project_handler(
            auth=mock_auth,
            project_name="PrivateProj",
            project_payload={},
            visibility="private",
        )
        assert result["ok"]
        list_result = list_projects_handler(auth=mock_auth)
        proj = list_result["projects"][0]
        assert proj["visibility"] in ("private", "shared")

    def test_visibility_public(self, temp_db, mock_auth):
        result = save_project_handler(
            auth=mock_auth,
            project_name="PublicProj",
            project_payload={},
            visibility="public",
        )
        assert result["ok"]
        list_result = list_projects_handler(auth=mock_auth)
        proj = list_result["projects"][0]
        assert proj["visibility"] == "public"


class TestRestoreProject:
    def test_restore_version(self, temp_db, mock_auth):
        save = save_project_handler(
            auth=mock_auth,
            project_name="RestoreTest",
            project_payload={
                "name": "RestoreTest",
                "project_format_version": 4,
                "meta": {
                    "chisurf_version": "1.0.0",
                    "created": "2026-01-01T00:00:00",
                    "name": "RestoreTest",
                },
                "datasets": {},
                "experiments": {},
                "fits": [],
                "ui": {},
                "extra": {},
            },
        )
        version_id = save["version_id"]
        result = restore_project_handler(auth=mock_auth, version_id=version_id)
        assert result["ok"]
        assert result["project_name"] == "RestoreTest"
        payload = result["project_payload"]
        assert isinstance(payload, dict)
        assert payload.get("project_format_version") == 5
        assert payload.get("datasets") == {}
        assert payload.get("fits") == []
        assert payload.get("experiments") == {}


class TestExportImport:
    def test_export_roundtrip(self, temp_db, mock_auth):
        save = save_project_handler(
            auth=mock_auth,
            project_name="ExportTest",
            project_payload={"data": "hello"},
        )
        version_id = save["version_id"]

        export = export_csp_handler(auth=mock_auth, version_id=version_id)
        assert export["ok"]
        archive_b64 = export.get("archive_bytes")
        assert archive_b64 is not None

        preview = import_preview_handler(
            auth=mock_auth,
            archive_base64=archive_b64,
        )
        assert preview["ok"]
        assert preview["origin"]["project_id"] == save["project_id"]

        imported = import_csp_handler(
            auth=mock_auth,
            archive_base64=archive_b64,
            resolve_collisions=True,
        )
        assert imported["ok"]
        assert imported["project_id"] is not None
        assert imported["version_id"] is not None

    def test_collision_detection(self, temp_db, mock_auth):
        save = save_project_handler(
            auth=mock_auth,
            project_name="CollisionTest",
            project_payload={"x": 1},
        )
        version_id = save["version_id"]

        export = export_csp_handler(auth=mock_auth, version_id=version_id)
        archive_b64 = export["archive_bytes"]

        import_first = import_csp_handler(
            auth=mock_auth,
            archive_base64=archive_b64,
            resolve_collisions=True,
        )
        assert import_first["ok"]

        # Second import should have collisions (same IDs after remap? no - the remapped IDs are new)
        # Actually the export archive still has the original IDs, so re-importing same archive
        # into the same DB with resolve_collisions=False should detect collisions.
        preview = import_preview_handler(
            auth=mock_auth, archive_base64=archive_b64,
        )
        collisions = preview.get("collisions", {})
        has_collisions = any(v for v in collisions.values())
        assert has_collisions, "Expected collision on duplicate import"

    def test_collision_resolution(self, temp_db, mock_auth):
        save = save_project_handler(
            auth=mock_auth,
            project_name="CollisionResolve",
            project_payload={"z": 2},
        )
        version_id = save["version_id"]

        export = export_csp_handler(auth=mock_auth, version_id=version_id)
        archive_b64 = export["archive_bytes"]

        import_csp_handler(
            auth=mock_auth, archive_base64=archive_b64, resolve_collisions=False,
        )

        preview = import_preview_handler(
            auth=mock_auth, archive_base64=archive_b64,
        )
        collisions = preview.get("collisions", {})
        has_collisions = any(v for v in collisions.values())
        assert has_collisions

        imported2 = import_csp_handler(
            auth=mock_auth, archive_base64=archive_b64, resolve_collisions=True,
        )
        assert imported2["ok"]
        # New project should have a different version_id
        assert imported2["version_id"] != version_id


class TestDeleteVersion:
    def test_delete_version(self, temp_db, mock_auth):
        save = save_project_handler(
            auth=mock_auth,
            project_name="DeleteTest",
            project_payload={},
        )
        version_id = save["version_id"]

        result = delete_version_handler(auth=mock_auth, version_id=version_id)
        assert result["ok"]
        assert result["deleted_version_id"] == version_id


class TestFiltering:
    def test_show_public_no_collapse(self, temp_db, mock_auth):
        save_project_handler(
            auth=mock_auth, project_name="PublicProj", project_payload={},
            visibility="public",
        )
        list_all = list_projects_handler(auth=mock_auth, show_public=True)
        list_no_public = list_projects_handler(auth=mock_auth, show_public=False)
        assert len(list_all["projects"]) >= 1
        # Public projects should be excluded when show_public=False
        assert len(list_no_public["projects"]) == 0

    def test_search_filter(self, temp_db, mock_auth):
        save_project_handler(
            auth=mock_auth, project_name="AlphaProject", project_payload={},
        )
        save_project_handler(
            auth=mock_auth, project_name="BetaProject", project_payload={},
        )
        result = list_projects_handler(auth=mock_auth, search="Alpha")
        assert len(result["projects"]) == 1
        assert result["projects"][0]["project_name"] == "AlphaProject"


class TestCollisionHelpers:
    def test_generate_id_remap(self):
        collisions = {
            "operations": ["op_1", "op_2"],
            "artifacts": ["art_1"],
            "objects": [],
            "parameters": [],
        }
        remap = _generate_id_remap(collisions)
        assert len(remap["operations"]) == 2
        assert len(remap["artifacts"]) == 1
        assert len(remap["objects"]) == 0
        assert "op_1" in remap["operations"]
        assert remap["operations"]["op_1"] != "op_1"

    def test_apply_remap_to_export(self):
        export_meta = {
            "dependencies": {
                "operations": [{"operation_id": "op_1"}, {"operation_id": "op_3"}],
                "artifacts": [{"artifact_id": "art_1"}],
                "objects": [],
                "parameters": [],
                "provenance_edges": [
                    {
                        "source_node_id": "op_1",
                        "target_node_id": "art_1",
                        "processing_id": "op_1",
                        "source_node_type": "operation",
                        "target_node_type": "artifact",
                        "relationship_type": "produced",
                    },
                ],
            },
        }
        remap = {
            "operations": {"op_1": "op_1_remapped"},
            "artifacts": {"art_1": "art_1_remapped"},
            "objects": {},
            "parameters": {},
        }
        result = _apply_remap_to_export(export_meta, remap)
        deps = result["dependencies"]
        assert deps["operations"][0]["operation_id"] == "op_1_remapped"
        assert deps["operations"][1]["operation_id"] == "op_3"
        assert deps["artifacts"][0]["artifact_id"] == "art_1_remapped"
        assert deps["provenance_edges"][0]["source_node_id"] == "op_1_remapped"
        assert deps["provenance_edges"][0]["target_node_id"] == "art_1_remapped"
