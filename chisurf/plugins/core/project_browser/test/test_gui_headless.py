from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _seed_project_browser_db(tmp_path, monkeypatch):
    from mfdb.auth import create_session
    from mfdb.repository import MFDatabase
    from chisurf.plugins.core.project_browser.backend import services

    db_path = tmp_path / "project_browser_gui.db"
    object_root = tmp_path / "objects"
    with MFDatabase(db_path) as db:
        db.add_user("user_default", "Default User")
        db.add_user("admin_user", "Admin User", is_admin=1)
        session = create_session(db.conn, "admin_user", client_name="pytest-gui")
        db.conn.commit()

    monkeypatch.setattr(services, "resolve_database_path", lambda: db_path)
    monkeypatch.setattr(
        "mfdb.database_resolver.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "mfdb.database_resolver.object_store_root",
        lambda: object_root,
    )
    return db_path, {"token": session["token"]}


def _sample_payload(tmp_path):
    data_path = tmp_path / "sample_curve.csv"
    data_path.write_text("time,intensity\n0,10\n1,12\n2,11\n", encoding="utf-8")
    return {
        "project_format_version": 5,
        "meta": {"name": "Sample Project", "description": "headless GUI test"},
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
                    "state": {"source": "pytest-gui"},
                },
            },
        },
        "fits": [],
        "experiments": {},
        "ui": {},
        "extra": {},
    }


class FakeProjectBrowserClient:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def list_projects(self, show_public: bool = True, search: str | None = None):
        self.calls.append({"show_public": show_public, "search": search})
        return [
            {
                "project_id": "proj_sample",
                "project_name": "Sample Project",
                "owner_user_id": "admin_user",
                "visibility": "private",
                "version_count": 2,
                "latest_version_number": 2,
                "latest_version_id": "ver_2",
                "created_at": "2026-07-05T10:00:00",
                "updated_at": "2026-07-05T11:00:00",
                "versions": [
                    {
                        "version_id": "ver_2",
                        "project_id": "proj_sample",
                        "project_name": "Sample Project",
                        "version_number": 2,
                        "owner_user_id": "admin_user",
                        "status": "succeeded",
                        "dataset_count": 1,
                        "fit_count": 0,
                        "created_at": "2026-07-05T11:00:00",
                        "notes": "sample data",
                    },
                    {
                        "version_id": "ver_1",
                        "project_id": "proj_sample",
                        "project_name": "Sample Project",
                        "version_number": 1,
                        "owner_user_id": "admin_user",
                        "status": "succeeded",
                        "dataset_count": 1,
                        "fit_count": 0,
                        "created_at": "2026-07-05T10:00:00",
                        "notes": "first sample",
                    },
                ],
            },
        ]


def test_project_browser_renders_sample_project_tree_headless(qapp, monkeypatch):
    from chisurf.plugins.core.project_browser.gui.tool import ProjectBrowserTool

    fake_client = FakeProjectBrowserClient()
    monkeypatch.setattr(ProjectBrowserTool, "_make_client", lambda self: fake_client)

    widget = ProjectBrowserTool()
    try:
        qapp.processEvents()

        assert widget._tree.topLevelItemCount() == 1
        project_item = widget._tree.topLevelItem(0)
        assert project_item.text(0) == "Sample Project  (2 versions)"
        assert project_item.childCount() == 2
        assert project_item.child(0).text(1) == "ver_2"
        assert project_item.child(0).text(5) == "1"

        widget._tree.setCurrentItem(project_item)
        selected = widget._selected_restore_version()
        assert selected is not None
        assert selected["version_id"] == "ver_2"

        widget._search_edit.setText("sample")
        qapp.processEvents()
        assert fake_client.calls[-1]["search"] == "sample"
    finally:
        widget.close()


def test_project_browser_gui_uses_inprocess_chisurf_services_with_sample_data(
    qapp,
    tmp_path,
    monkeypatch,
):
    from chisurf.plugins.core.project_browser.backend.services import save_project_handler
    from chisurf.plugins.core.project_browser.gui.tool import ProjectBrowserTool

    _, auth = _seed_project_browser_db(tmp_path, monkeypatch)
    saved = save_project_handler(
        auth=auth,
        project_name="Sample Project",
        project_payload=_sample_payload(tmp_path),
        visibility="public",
        notes="headless in-process GUI test",
    )
    assert saved["ok"] is True

    widget = ProjectBrowserTool()
    try:
        qapp.processEvents()

        assert widget._tree.topLevelItemCount() == 1
        project_item = widget._tree.topLevelItem(0)
        assert project_item.text(1) == saved["project_id"]
        assert project_item.childCount() == 1
        version_item = project_item.child(0)
        assert version_item.text(1) == saved["version_id"]
        assert version_item.text(5) == "1"
    finally:
        widget.close()


def test_project_browser_gui_deletes_selected_version_with_confirmation(
    qapp,
    tmp_path,
    monkeypatch,
):
    from qtpy import QtWidgets
    from mfdb.credentials import _RUNTIME_SESSION_TOKENS
    from chisurf.plugins.core.project_browser.backend.services import (
        save_project_handler,
    )
    from chisurf.plugins.core.project_browser.gui.tool import ProjectBrowserTool

    _, auth = _seed_project_browser_db(tmp_path, monkeypatch)
    saved = save_project_handler(
        auth=auth,
        project_name="Delete Project",
        project_payload=_sample_payload(tmp_path),
        visibility="public",
        notes="headless delete GUI test",
    )
    assert saved["ok"] is True
    monkeypatch.setitem(_RUNTIME_SESSION_TOKENS, "pytest-gui", auth["token"])

    widget = ProjectBrowserTool()
    try:
        qapp.processEvents()
        version_item = widget._tree.topLevelItem(0).child(0)
        widget._tree.setCurrentItem(version_item)
        version_item.setSelected(True)
        widget._tree.selectionModel().select(
            widget._tree.indexFromItem(version_item),
            widget._tree.selectionModel().ClearAndSelect | widget._tree.selectionModel().Rows,
        )
        assert widget._selected_version()["version_id"] == saved["version_id"]

        with (
            mock.patch.object(
                QtWidgets.QMessageBox,
                "question",
                return_value=QtWidgets.QMessageBox.Yes,
            ),
            mock.patch.object(QtWidgets.QMessageBox, "information"),
            mock.patch.object(QtWidgets.QMessageBox, "warning"),
        ):
            widget._on_delete()
        qapp.processEvents()

        assert widget._tree.topLevelItemCount() == 0
    finally:
        widget.close()


def test_project_browser_gui_imports_archive_with_collision_remap(
    qapp,
    tmp_path,
    monkeypatch,
):
    from qtpy import QtWidgets
    from chisurf.plugins.core.project_browser.backend.services import (
        export_csp_handler,
        save_project_handler,
    )
    from chisurf.plugins.core.project_browser.gui.tool import (
        CollisionDialog,
        ProjectBrowserTool,
    )

    _, auth = _seed_project_browser_db(tmp_path, monkeypatch)
    saved = save_project_handler(
        auth=auth,
        project_name="Import Project",
        project_payload=_sample_payload(tmp_path),
        visibility="public",
        notes="headless import GUI test",
    )
    assert saved["ok"] is True
    archive_path = tmp_path / "import-project.csp"
    exported = export_csp_handler(
        auth=auth,
        version_id=saved["version_id"],
        target_path=str(archive_path),
    )
    assert exported["ok"] is True

    widget = ProjectBrowserTool()
    try:
        qapp.processEvents()
        with (
            mock.patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileName",
                return_value=(str(archive_path), "Chisurf Project (*.csp)"),
            ),
            mock.patch.object(
                CollisionDialog,
                "exec",
                return_value=QtWidgets.QDialog.Accepted,
            ),
            mock.patch.object(QtWidgets.QMessageBox, "information"),
        ):
            widget._on_import()
        qapp.processEvents()

        assert widget._tree.topLevelItemCount() == 1
        project_item = widget._tree.topLevelItem(0)
        assert project_item.text(1) == saved["project_id"]
        assert project_item.childCount() == 2
    finally:
        widget.close()


def test_project_browser_gui_restores_selected_version_into_chisurf_context(
    qapp,
    tmp_path,
    monkeypatch,
):
    import chisurf as cs
    import chisurf.macros.core_fit as core_fit
    from mfdb.credentials import _RUNTIME_SESSION_TOKENS
    from qtpy import QtCore, QtWidgets
    from chisurf.plugins.core.project_browser.backend.services import (
        save_project_handler,
    )
    from chisurf.plugins.core.project_browser.gui.tool import ProjectBrowserTool

    _, auth = _seed_project_browser_db(tmp_path, monkeypatch)
    payload = _sample_payload(tmp_path)
    payload["meta"]["name"] = "Restore Project"
    saved = save_project_handler(
        auth=auth,
        project_name="Restore Project",
        project_payload=payload,
        visibility="private",
        notes="headless restore GUI test",
    )
    assert saved["ok"] is True
    monkeypatch.setitem(_RUNTIME_SESSION_TOKENS, "pytest-gui", auth["token"])

    chisurf_context = SimpleNamespace(
        _current_project_id=None,
        _current_project_version_id=None,
        _current_project_name=None,
        _current_project_visibility=None,
    )
    loaded_projects = []
    gui_restore_projects = []

    monkeypatch.setattr(cs, "cs", chisurf_context, raising=False)
    monkeypatch.setattr(
        core_fit,
        "load_project_payload",
        lambda proj, project_path=None: loaded_projects.append((proj, project_path)),
    )
    monkeypatch.setattr(
        ProjectBrowserTool,
        "_restore_gui_from_fits",
        staticmethod(lambda proj: gui_restore_projects.append(proj)),
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, callback: callback())

    widget = ProjectBrowserTool()
    try:
        qapp.processEvents()
        version_item = widget._tree.topLevelItem(0).child(0)
        widget._tree.setCurrentItem(version_item)

        with (
            mock.patch.object(QtWidgets.QMessageBox, "information"),
            mock.patch.object(QtWidgets.QMessageBox, "warning"),
        ):
            widget._on_open()
        qapp.processEvents()

        assert len(loaded_projects) == 1
        loaded_project, project_path = loaded_projects[0]
        assert project_path is None
        assert loaded_project.name == "Restore Project"
        assert "ds_sample" in loaded_project.datasets
        assert gui_restore_projects == [loaded_project]
        assert chisurf_context._current_project_id == saved["project_id"]
        assert chisurf_context._current_project_version_id == saved["version_id"]
        assert chisurf_context._current_project_name == "Restore Project"
        assert chisurf_context._current_project_visibility == "private"
    finally:
        widget.close()
