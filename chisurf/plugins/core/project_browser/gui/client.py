from __future__ import annotations

from typing import Any

from chisurf import logging
from chisurf.core.plugin.client import InProcessClient
from mfdb.admin.gui.client import MFDBClient


class ProjectBrowserClient(MFDBClient):
    """Client for project_browser RPC handlers."""

    def __init__(self, mfdb_client: MFDBClient | None = None, **kwargs: Any):
        if mfdb_client is not None:
            self._client = mfdb_client._client
            self._token = mfdb_client.token
        else:
            kwargs.setdefault("inprocess", True)
            super().__init__(**kwargs)
            if self._token is None:
                self._auto_auth()

    def _make_inprocess_client(self) -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState
        from mfdb.admin.backend.services import register_services as register_mfdb_services
        from chisurf.plugins.core.project_browser.backend.services import register_services as register_project_browser_services

        dispatcher = ServiceDispatcher(SessionState())
        register_mfdb_services(dispatcher)
        register_project_browser_services(dispatcher)
        return InProcessClient(dispatcher)

    def _auto_auth(self) -> None:
        """Use the active MFDB login token for in-process project-browser RPC calls."""
        from mfdb.database_resolver import resolve_database_path
        from mfdb.repository import MFDatabase
        from mfdb.credentials import _RUNTIME_SESSION_TOKENS
        from mfdb.auth import _hash_token
        import chisurf.core.settings as cs_settings

        try:
            mfdb_settings = getattr(cs_settings, "cs_settings", {}).get("mfdb", {})
            server_host = mfdb_settings.get("last_server", "127.0.0.1")
            server_port = int(mfdb_settings.get("last_port", 8765))
            prefix = f"{server_host}:{server_port}:"
            token = None
            for key, value in _RUNTIME_SESSION_TOKENS.items():
                if key.startswith(prefix):
                    token = value
                    break
            if token is None and _RUNTIME_SESSION_TOKENS:
                token = next(iter(_RUNTIME_SESSION_TOKENS.values()))

            db = MFDatabase(resolve_database_path())
            if token:
                token_hash = _hash_token(token)
                row = db.conn.execute(
                    """SELECT u.user_id
                       FROM mfdb_session AS s
                       JOIN flr_sample_users AS u ON u.user_id = s.user_id
                       WHERE s.token_hash = ?""",
                    (token_hash,),
                ).fetchone()
                if row:
                    user_id = row[0]
                    self._token = token
                    logging.info("In-process auto-auth: using active token for user=%s", user_id)
                    return

            import uuid

            token = f"inproc_{uuid.uuid4().hex}"
            token_hash = _hash_token(token)
            session_id = f"inproc_{uuid.uuid4().hex[:12]}"
            requested_user_id = mfdb_settings.get("default_user_id", "user_default")
            row = db.conn.execute(
                "SELECT user_id FROM flr_sample_users WHERE user_id = ? LIMIT 1",
                (requested_user_id,),
            ).fetchone()
            if row is None:
                row = db.conn.execute(
                    """SELECT user_id FROM flr_sample_users
                       ORDER BY is_admin ASC, user_id ASC LIMIT 1"""
                ).fetchone()
            user_id = row[0] if row else "user_default"
            db.conn.execute(
                "INSERT OR IGNORE INTO mfdb_session (session_id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
                (session_id, user_id, token_hash, "2099-12-31T23:59:59"),
            )
            db.conn.commit()
            self._token = token
            logging.info("In-process auto-auth: token created for user=%s", user_id)
        except Exception as exc:
            logging.warning("In-process auto-auth failed: %s", exc)

    def list_projects(
        self,
        show_public: bool = True,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._call("project_browser.list", {
            "show_public": show_public,
            "search": search,
        }).get("projects", [])

    def save_project(
        self,
        project_name: str,
        project_payload: dict[str, Any] | None = None,
        project_id: str | None = None,
        parent_version_id: str | None = None,
        notes: str | None = None,
        visibility: str = "private",
        fit_count: int = 0,
        dataset_count: int = 0,
    ) -> dict[str, Any]:
        return self._call("project_browser.save", {
            "project_name": project_name,
            "project_payload": project_payload,
            "project_id": project_id,
            "parent_version_id": parent_version_id,
            "notes": notes,
            "visibility": visibility,
            "fit_count": fit_count,
            "dataset_count": dataset_count,
        })

    def restore_project(self, version_id: str) -> dict[str, Any]:
        return self._call("project_browser.restore", {
            "version_id": version_id,
        })

    def export_csp(
        self,
        version_id: str,
        target_path: str | None = None,
    ) -> dict[str, Any]:
        return self._call("project_browser.export_csp", {
            "version_id": version_id,
            "target_path": target_path,
        })

    def import_preview(
        self,
        archive_base64: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        return self._call("project_browser.import_preview", {
            "archive_base64": archive_base64,
            "file_path": file_path,
        })

    def import_csp(
        self,
        archive_base64: str | None = None,
        file_path: str | None = None,
        resolve_collisions: bool = False,
    ) -> dict[str, Any]:
        return self._call("project_browser.import_csp", {
            "archive_base64": archive_base64,
            "file_path": file_path,
            "resolve_collisions": resolve_collisions,
        })

    def delete_version(self, version_id: str) -> dict[str, Any]:
        return self._call("project_browser.delete_version", {
            "version_id": version_id,
        })

    def create_branch(
        self,
        project_id: str,
        from_version_id: str,
        branch_name: str,
    ) -> dict[str, Any]:
        return self._call("project_browser.create_branch", {
            "project_id": project_id,
            "from_version_id": from_version_id,
            "branch_name": branch_name,
        })

    def list_branches(self, project_id: str) -> list[dict[str, Any]]:
        return self._call("project_browser.list_branches", {
            "project_id": project_id,
        }).get("branches", [])

    def version_graph(self, project_id: str) -> dict[str, Any]:
        return self._call("project_browser.version_graph", {
            "project_id": project_id,
        }).get("graph", {})

    def list_artifacts(self, version_id: str) -> list[dict[str, Any]]:
        return self._call("project_browser.artifacts", {
            "version_id": version_id,
        }).get("artifacts", [])

    def list_parameters(self, version_id: str) -> list[dict[str, Any]]:
        return self._call("project_browser.parameters", {
            "version_id": version_id,
        }).get("parameters", [])
