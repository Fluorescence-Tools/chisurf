"""Client wrapper for MFDB JSON-RPC services."""

from __future__ import annotations

from typing import Any

from chisurf import logging
from chisurf.core.plugin.client import InProcessClient


class MFDBClient:
    """Client for MFDB RPC handlers."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        host: str = "127.0.0.1",
        cmd_port: int = 8765,
        pub_port: int = 8766,
        timeout_ms: int = 5000,
        inprocess: bool = False,
    ):
        if client is None:
            client = self._make_inprocess_client() if inprocess else self._make_zmq_client(
                host=host,
                cmd_port=cmd_port,
                pub_port=pub_port,
                timeout_ms=timeout_ms,
            )
        self._client = client
        self._token: str | None = None

    def _make_zmq_client(
        self,
        host: str,
        cmd_port: int,
        pub_port: int,
        timeout_ms: int,
    ) -> Any:
        from chisurf.server.transport.zmq import ZmqClient

        return ZmqClient(
            cmd_port=cmd_port,
            pub_port=pub_port,
            host=host,
            timeout_ms=timeout_ms,
        )

    def _make_inprocess_client(self) -> InProcessClient:
        from chisurf.plugins.core.project_browser.backend.services import (
            register_services as register_project_browser_services,
        )
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from ..backend.services import register_services

        dispatcher = ServiceDispatcher(SessionState())
        register_services(dispatcher)
        register_project_browser_services(dispatcher)
        return InProcessClient(dispatcher)

    def status(self) -> dict[str, Any]:
        return self._call("mfdb.status")

    def list_samples(self) -> list[dict[str, Any]]:
        return self._call("mfdb.samples.list").get("samples", [])

    def get_sample(self, sample_id: str) -> dict[str, Any]:
        return self._call("mfdb.samples.get", {"sample_id": sample_id}).get("sample")

    def save_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._call("mfdb.samples.save", {"sample": sample}).get("sample")

    def delete_sample(self, sample_id: str) -> dict[str, Any]:
        return self._call("mfdb.samples.delete", {"sample_id": sample_id})

    def search_samples(self, query: str | None = None) -> list[dict[str, Any]]:
        return self._call("mfdb.samples.search", {"query": query or ""}).get("samples", [])

    def save_sample_key_values(
        self,
        sample_id: str,
        key_values: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._call(
            "mfdb.samples.key_values.save",
            {"sample_id": sample_id, "key_values": key_values},
        ).get("sample")

    def lifecycle_state(self, entity_type: str, entity_id: str) -> str | None:
        return self._call(
            "mfdb.lifecycle.state",
            {"entity_type": entity_type, "entity_id": entity_id},
        ).get("state")

    def lifecycle_history(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.lifecycle.history",
            {"entity_type": entity_type, "entity_id": entity_id},
        ).get("history", [])

    def lifecycle_transition(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        reason: str = "",
        operator_user_id: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "mfdb.lifecycle.transition",
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "to_state": to_state,
                "reason": reason,
                "operator_user_id": operator_user_id,
            },
        )

    def lifecycle_definitions(self) -> dict[str, Any]:
        return self._call("mfdb.lifecycle.definitions").get("definitions", {})

    def list_protocols(self, scope: str = "all") -> list[dict[str, Any]]:
        return self._call("mfdb.protocols.list", {"scope": scope}).get("protocols", [])

    def get_protocol(self, name: str, version: Any = "latest") -> dict[str, Any]:
        return self._call(
            "mfdb.protocols.get", {"name": name, "version": version}
        )

    def list_protocol_versions(self, name: str) -> list[dict[str, Any]]:
        return self._call("mfdb.protocols.versions", {"name": name}).get("versions", [])

    def create_protocol(
        self,
        name: str,
        category: str,
        operation_type: str | None = None,
        setup_id: str | None = None,
        description: str = "",
        is_public: bool = False,
    ) -> dict[str, Any]:
        return self._call(
            "mfdb.protocols.create",
            {
                "name": name,
                "category": category,
                "operation_type": operation_type,
                "setup_id": setup_id,
                "description": description,
                "is_public": is_public,
            },
        )

    def protocol_for_operation(self, operation_id: str) -> dict[str, Any]:
        return self._call(
            "mfdb.protocols.for_operation", {"operation_id": operation_id}
        )

    def get_sample_condition(self, condition_id: str) -> dict[str, Any]:
        return self._call("mfdb.sample_conditions.get", {"condition_id": condition_id}).get("condition", {})

    def save_sample_condition(self, condition: dict[str, Any]) -> dict[str, Any]:
        return self._call("mfdb.sample_conditions.save", {"condition": condition}).get("condition", {})

    def list_probes(self) -> list[dict[str, Any]]:
        return self._call("mfdb.probes.list").get("probes", [])

    def get_probe(self, probe_id: int) -> dict[str, Any]:
        return self._call("mfdb.probes.get", {"probe_id": probe_id}).get("probe", {})

    def get_probe_optical_properties(self, probe_id: int) -> dict[str, Any]:
        return self._call(
            "mfdb.probes.optical_properties.get",
            {"probe_id": probe_id},
        ).get("optical_properties", {})

    def get_sample_full_description(self, sample_id: str) -> dict[str, Any]:
        try:
            return self._call(
                "mfdb.samples.full_description",
                {"sample_id": sample_id},
            ).get("description", {})
        except Exception:
            return {}

    def validate_sample_export(self, sample_id: str) -> dict[str, Any]:
        return self._call(
            "mfdb.samples.validate_export",
            {"sample_id": sample_id},
        )

    def create_structured_sample(self, sample_data: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "mfdb.samples.create_structured",
            {"sample_data": sample_data},
        )

    def list_entities(self, sample_id: str | None = None) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.entities.list",
            {"sample_id": sample_id},
        ).get("entities", [])

    def save_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "mfdb.entities.save",
            {"entity": entity},
        ).get("entity", {})

    def delete_entity(self, entity_id: str) -> dict[str, Any]:
        return self._call(
            "mfdb.entities.delete",
            {"entity_id": entity_id},
        )

    def save_probe(self, probe: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "mfdb.probes.save",
            {"probe": probe},
        ).get("probe", {})

    def save_probe_optical_properties(
        self,
        probe_id: int,
        properties: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.probes.optical_properties.save",
            {"probe_id": probe_id, "properties": properties},
        ).get("optical_properties", [])

    def list_probe_positions(
        self,
        sample_id: str | None = None,
        probe_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.probes.positions.list",
            {"sample_id": sample_id, "probe_id": probe_id},
        ).get("positions", [])

    def list_fret_pairs(self, sample_id: str) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.fret_pairs.list",
            {"sample_id": sample_id},
        ).get("fret_pairs", [])

    def save_fret_pair(self, pair: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "mfdb.fret_pairs.save",
            {"pair": pair},
        ).get("fret_pair", {})

    def delete_fret_pair(self, forster_radius_id: str) -> dict[str, Any]:
        return self._call(
            "mfdb.fret_pairs.delete",
            {"forster_radius_id": forster_radius_id},
        )

    def suggest_pdbx_keys(self, prefix: str) -> list[dict[str, str]]:
        return self._call(
            "mfdb.pdbx.suggest_keys",
            {"prefix": prefix},
        ).get("keys", [])

    def validate_pdbx_value(self, key: str, value: str) -> dict[str, Any]:
        return self._call(
            "mfdb.pdbx.validate_value",
            {"key": key, "value": value},
        )

    def populate_mock_data(self) -> dict[str, Any]:
        return self._call("mfdb.mock_data.populate").get("summary", {})

    def list_users(self) -> list[dict[str, Any]]:
        return self._call("mfdb.users.list").get("users", [])

    def save_user(self, user: dict[str, Any]) -> list[dict[str, Any]]:
        return self._call("mfdb.users.save", {"user": user}).get("users", [])

    def delete_user(self, user_id: str, force: bool = False, requester_id: str = None) -> list[dict[str, Any]]:
        return self._call("mfdb.users.delete", {"user_id": user_id, "force": force, "requester_id": requester_id}).get("users", [])

    def change_password(self, user_id: str, password: str, requester_id: str = None) -> dict[str, Any]:
        return self._call("mfdb.auth.change_password", {"password": password})

    def list_devices(self) -> list[dict[str, Any]]:
        return self._call("mfdb.devices.list").get("devices", [])

    def save_device(self, device: dict[str, Any]) -> list[dict[str, Any]]:
        return self._call("mfdb.devices.save", {"device": device}).get("devices", [])

    def delete_device(self, device_id: str) -> list[dict[str, Any]]:
        return self._call("mfdb.devices.delete", {"device_id": device_id}).get("devices", [])

    def list_experiment_types(self) -> list[dict[str, Any]]:
        return self._call("mfdb.experiment_types.list").get("experiment_types", [])

    def save_experiment_type(self, experiment_type: dict[str, Any]) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.experiment_types.save",
            {"experiment_type": experiment_type},
        ).get("experiment_types", [])

    def delete_experiment_type(self, type_id: int) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.experiment_types.delete",
            {"type_id": type_id},
        ).get("experiment_types", [])

    def list_experiments(
        self,
        sample_id: str | None = None,
        project_id: str | None = None,
        type_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._call(
            "mfdb.experiments.list",
            {
                "sample_id": sample_id,
                "project_id": project_id,
                "type_id": type_id,
            },
        ).get("experiments", [])

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._call(
            "mfdb.experiments.get",
            {"experiment_id": experiment_id},
        ).get("experiment")

    def save_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "mfdb.experiments.save",
            {"experiment": experiment},
        ).get("experiment")

    def delete_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._call(
            "mfdb.experiments.delete",
            {"experiment_id": experiment_id},
        )

    def save_experiment_key_values(
        self,
        experiment_id: str,
        key_values: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._call(
            "mfdb.experiments.key_values.save",
            {"experiment_id": experiment_id, "key_values": key_values},
        ).get("experiment")

    def save_experiment_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "mfdb.experiments.data.save",
            {"data": data},
        ).get("experiment")

    def delete_experiment_data(self, data_id: int) -> dict[str, Any]:
        return self._call(
            "mfdb.experiments.data.delete",
            {"data_id": data_id},
        )

    def import_file(self, path: str) -> dict[str, Any]:
        return self._call("mfdb.import_file", {"path": path}).get("summary", {})

    def export_sample(
        self,
        sample_id: str,
        output_path: str | None = None,
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "mfdb.export_sample",
            {
                "sample_id": sample_id,
                "output_path": output_path,
                "analysis_id": analysis_id,
            },
        )

    def export_table(self, output_path: str, sample_id: str | None = None) -> dict[str, Any]:
        return self._call(
            "mfdb.export_table",
            {"output_path": output_path, "sample_id": sample_id},
        )

    def backup(self) -> str:
        return self._call("mfdb.backup").get("backup_path", "")

    def reset_from_source(self) -> dict[str, Any]:
        return self._call("mfdb.reset_from_source")

    def archive_project(
        self,
        project_id: str,
        project_name: str,
        project_payload: dict[str, Any],
        project_archive_data: str | None = None,
        project_archive_filename: str | None = None,
        experiment_id: str | None = None,
        input_processed_data_ids: list[str] | None = None,
        notes: str | None = None,
        visibility: str = "private",
    ) -> dict[str, Any]:
        from chisurf.plugins.core.project_browser.gui.client import ProjectBrowserClient

        client = ProjectBrowserClient(mfdb_client=self)
        return client.save_project(
            project_name=project_name,
            project_payload=project_payload,
            project_id=project_id,
            notes=notes,
            visibility=visibility,
        )

    def restore_project(self, project_id: str) -> dict[str, Any]:
        from chisurf.plugins.core.project_browser.gui.client import ProjectBrowserClient

        client = ProjectBrowserClient(mfdb_client=self)
        if project_id.startswith("ver_"):
            return client.restore_project(version_id=project_id)

        for project in client.list_projects(show_public=True):
            versions = project.get("versions", [])
            if project.get("project_id") == project_id:
                latest = versions[0] if versions else None
                if latest:
                    return client.restore_project(version_id=latest["version_id"])
            for version in versions:
                if version.get("version_id") == project_id:
                    return client.restore_project(version_id=version["version_id"])
        return client.restore_project(version_id=project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        from chisurf.plugins.core.project_browser.gui.client import ProjectBrowserClient

        client = ProjectBrowserClient(mfdb_client=self)
        grouped = client.list_projects(show_public=True)
        rows: list[dict[str, Any]] = []
        for project in grouped:
            versions = project.get("versions", []) or []
            latest = versions[0] if versions else project
            row = dict(project)
            row.update({
                "analysis_id": latest.get("version_id") or project.get("latest_version_id"),
                "analysis_run_id": latest.get("version_id") or project.get("latest_version_id"),
                "analysis_type": "project",
                "model_name": latest.get("project_name") or project.get("project_name", ""),
                "project_name": project.get("project_name", ""),
                "project_id": project.get("project_id", ""),
                "version_id": latest.get("version_id", ""),
                "version_number": latest.get("version_number", 1),
                "experiment_id": "",
                "created_at": latest.get("created_at") or project.get("created_at", ""),
                "updated_at": project.get("updated_at", latest.get("created_at", "")),
                "notes": latest.get("notes") or project.get("notes", ""),
                "owner_user_id": latest.get("owner_user_id") or project.get("owner_user_id", ""),
                "visibility": project.get("visibility", latest.get("visibility", "private")),
            })
            rows.append(row)
        return rows

    def delete_project(self, project_id: str) -> dict[str, Any]:
        from chisurf.plugins.core.project_browser.gui.client import ProjectBrowserClient

        return ProjectBrowserClient(mfdb_client=self).delete_version(version_id=project_id)

    def list_raw_data(self, experiment_id: str | None = None, data_type: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if experiment_id is not None:
            params["experiment_id"] = experiment_id
        if data_type is not None:
            params["data_type"] = data_type
        return self._call("mfdb.raw_data.list", params or None).get("raw_data", [])

    def get_raw_data(self, raw_data_id: str) -> dict[str, Any]:
        return self._call("mfdb.raw_data.get", {"raw_data_id": raw_data_id}).get("raw_data", {})

    def list_processing_runs(self, experiment_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        return self._call("processing.burst_selection.list", {"experiment_id": experiment_id, "status": status}).get("processing_runs", [])

    def get_processing_run(self, processing_id: str) -> dict[str, Any]:
        return self._call("processing.burst_selection.get", {"processing_id": processing_id}).get("processing_run", {})

    def list_processed_data(self, processing_id: str | None = None, product_type: str | None = None) -> list[dict[str, Any]]:
        return self._call("processed_data.list", {"processing_id": processing_id, "product_type": product_type}).get("processed_data", [])

    def get_processed_data(self, processed_data_id: str) -> dict[str, Any]:
        return self._call("processed_data.get", {"processed_data_id": processed_data_id}).get("processed_data", {})

    def list_analysis_runs(self, experiment_id: str | None = None, analysis_type: str | None = None) -> list[dict[str, Any]]:
        return self._call("analysis.run.list", {"experiment_id": experiment_id, "analysis_type": analysis_type}).get("analysis_runs", [])

    def get_analysis_run(self, analysis_id: str) -> dict[str, Any]:
        return self._call("analysis.run.get", {"analysis_id": analysis_id}).get("analysis_run", {})

    def dependencies_upstream(self, node_type: str, node_id: str) -> dict[str, Any]:
        return self._call("provenance.dependencies.upstream", {"node_type": node_type, "node_id": node_id})

    def dependencies_downstream(self, node_type: str, node_id: str) -> dict[str, Any]:
        return self._call("provenance.dependencies.downstream", {"node_type": node_type, "node_id": node_id})

    def list_provenance_edges(self, **filters: Any) -> list[dict[str, Any]]:
        return self._call("provenance.edges.list", filters).get("provenance_edges", [])

    def export_provenance_graph(
        self,
        seed_node_type: str,
        seed_node_id: str,
        target_path: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "provenance.graph.export",
            {
                "seed_node_type": seed_node_type,
                "seed_node_id": seed_node_id,
                "output_path": target_path,
            },
        )

    def backup_to_path(self, target_path: str) -> dict[str, Any]:
        return self._call("database.backup", {"target_path": target_path})

    def export_zip_archive(
        self,
        target_zip_path: str,
        seed_node_type: str,
        seed_node_id: str,
        include_external_data: bool = False,
        base_path_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "archive.zip.export",
            {
                "target_zip_path": target_zip_path,
                "seed_node_type": seed_node_type,
                "seed_node_id": seed_node_id,
                "include_external_data": include_external_data,
                "base_path_map": base_path_map,
            },
        )

    def list_audit_logs(
        self,
        action: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._call(
            "audit_log.list",
            {
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "limit": limit,
            },
        ).get("audit_logs", [])

    def create_branch(
        self,
        branch_uuid: str | None = None,
        name: str | None = None,
        parent_branch_uuid: str | None = None,
        head_operation_id: str | None = None,
        created_by_user_id: str | None = None,
        description: str | None = None,
    ) -> str:
        return self._call(
            "mfdb.v1.branches.create",
            {
                "branch_uuid": branch_uuid,
                "name": name,
                "parent_branch_uuid": parent_branch_uuid,
                "head_operation_id": head_operation_id,
                "created_by_user_id": created_by_user_id,
                "description": description,
            },
        ).get("branch_uuid")

    def fork_branch(
        self,
        source_branch_uuid: str,
        name: str,
        branch_uuid: str | None = None,
        head_operation_id: str | None = None,
        created_by_user_id: str | None = None,
        description: str | None = None,
    ) -> str:
        """Create a parallel branch from an existing branch."""
        return self._call(
            "mfdb.v1.branches.fork",
            {
                "source_branch_uuid": source_branch_uuid,
                "name": name,
                "branch_uuid": branch_uuid,
                "head_operation_id": head_operation_id,
                "created_by_user_id": created_by_user_id,
                "description": description,
            },
        ).get("branch_uuid")

    def get_branch(self, branch_uuid_or_name: str) -> dict[str, Any]:
        return self._call("mfdb.v1.branches.get", {"branch_uuid_or_name": branch_uuid_or_name}).get("branch")

    def list_branches(self) -> list[dict[str, Any]]:
        return self._call("mfdb.v1.branches.list").get("branches", [])

    def update_branch_head(self, branch_uuid: str, head_operation_id: str | None) -> None:
        self._call("mfdb.v1.branches.update_head", {"branch_uuid": branch_uuid, "head_operation_id": head_operation_id})

    def delete_branch(self, branch_uuid: str) -> None:
        self._call("mfdb.v1.branches.delete", {"branch_uuid": branch_uuid})

    def set_user_active_branch(self, user_id: str, branch_uuid: str) -> None:
        self._call("mfdb.v1.users.set_active_branch", {"user_id": user_id, "branch_uuid": branch_uuid})

    def get_user_active_branch(self, user_id: str) -> dict[str, Any]:
        return self._call("mfdb.v1.users.get_active_branch", {"user_id": user_id}).get("branch")

    def jump_user_to_operation(
        self,
        user_id: str,
        operation_id: str,
        branch_name: str | None = None,
        branch_uuid: str | None = None,
        parent_branch_uuid: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create and activate a user branch at a historical operation."""
        return self._call(
            "mfdb.v1.users.jump_to_operation",
            {
                "user_id": user_id,
                "operation_id": operation_id,
                "branch_name": branch_name,
                "branch_uuid": branch_uuid,
                "parent_branch_uuid": parent_branch_uuid,
                "description": description,
            },
        ).get("branch")


    # ---- Auth methods ----

    @property
    def token(self) -> str | None:
        return self._token

    @token.setter
    def token(self, value: str | None) -> None:
        self._token = value

    def login(self, user_id: str, password: str = "", client_metadata: dict | None = None) -> dict[str, Any]:
        """Login and store the session token."""
        result = self._call_raw("mfdb.auth.login", {
            "user_id": user_id,
            "password": password,
            "client_metadata": client_metadata,
        })
        if result.get("ok"):
            self._token = result.get("token")
        return result

    def logout(self) -> dict[str, Any]:
        """Logout and clear the session token."""
        result = self._call_raw("mfdb.auth.logout")
        self._token = None
        return result

    def me(self) -> dict[str, Any]:
        """Return current authenticated user info."""
        return self._call("mfdb.auth.me")

    def sessions_list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List active sessions."""
        return self._call("mfdb.auth.sessions.list", {"user_id": user_id}).get("sessions", [])

    def sessions_revoke(self, session_id: str) -> dict[str, Any]:
        """Revoke a session by ID."""
        return self._call("mfdb.auth.sessions.revoke", {"session_id": session_id})

    # ---- Group methods ----

    def groups_list(self) -> list[dict[str, Any]]:
        return self._call("mfdb.groups.list").get("groups", [])

    def groups_get(self, group_id: str) -> dict[str, Any]:
        return self._call("mfdb.groups.get", {"group_id": group_id}).get("group")

    def groups_create(self, group: dict[str, Any]) -> dict[str, Any]:
        return self._call("mfdb.groups.create", {"group": group})

    def groups_update(self, group: dict[str, Any]) -> dict[str, Any]:
        return self._call("mfdb.groups.update", {"group": group})

    def groups_delete(self, group_id: str) -> dict[str, Any]:
        return self._call("mfdb.groups.delete", {"group_id": group_id})

    def members_list(self, group_id: str) -> list[dict[str, Any]]:
        return self._call("mfdb.groups.members.list", {"group_id": group_id}).get("members", [])

    def members_add(self, group_id: str, user_id: str, role: str = "member") -> dict[str, Any]:
        return self._call("mfdb.groups.members.add", {"group_id": group_id, "user_id": user_id, "role": role})

    def members_remove(self, group_id: str, user_id: str) -> dict[str, Any]:
        return self._call("mfdb.groups.members.remove", {"group_id": group_id, "user_id": user_id})

    # ---- Permission methods ----

    def permissions_get(self, object_type: str, object_id: str) -> dict[str, Any]:
        return self._call("mfdb.permissions.get", {"object_type": object_type, "object_id": object_id}).get("acl")

    def permissions_chmod(self, object_type: str, object_id: str, mode: int) -> dict[str, Any]:
        return self._call("mfdb.permissions.chmod", {"object_type": object_type, "object_id": object_id, "mode": mode})

    def permissions_chown(self, object_type: str, object_id: str, owner_user_id: str) -> dict[str, Any]:
        return self._call("mfdb.permissions.chown", {"object_type": object_type, "object_id": object_id, "owner_user_id": owner_user_id})

    def permissions_chgrp(self, object_type: str, object_id: str, owner_group_id: str) -> dict[str, Any]:
        return self._call("mfdb.permissions.chgrp", {"object_type": object_type, "object_id": object_id, "owner_group_id": owner_group_id})

    def permissions_grant(
        self,
        object_type: str,
        object_id: str,
        subject_type: str,
        subject_id: str,
        permissions: int,
        effect: str = "allow",
    ) -> dict[str, Any]:
        return self._call("mfdb.permissions.grant", {
            "object_type": object_type,
            "object_id": object_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "permissions": permissions,
            "effect": effect,
        })

    def permissions_revoke(self, entry_id: int) -> dict[str, Any]:
        return self._call("mfdb.permissions.revoke", {"entry_id": entry_id})

    # ---- Object Store ----

    def put_object(
        self,
        path: str | None = None,
        data: str | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a file or base64-encoded bytes in the object store.

        Parameters
        ----------
        path : str, optional
            Server-side file path to store.
        data : str, optional
            Base64-encoded binary data to store.
        filename : str, optional
            Original filename to record.
        mime_type : str, optional
            MIME type of the content.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        dict
            Object reference with uuid, md5, size, deduplicated flag.
        """
        params: dict[str, Any] = {}
        if path is not None:
            params["path"] = path
        if data is not None:
            params["data"] = data
        if filename is not None:
            params["filename"] = filename
        if mime_type is not None:
            params["mime_type"] = mime_type
        if metadata is not None:
            params["metadata"] = metadata
        return self._call("mfdb.objects.put", params)

    def put_object_bytes(
        self,
        data: str,
        filename: str,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store base64-encoded bytes in the object store.

        Parameters
        ----------
        data : str
            Base64-encoded binary data.
        filename : str
            Original filename to record.
        mime_type : str, optional
            MIME type of the content.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        dict
            Object reference with uuid, md5, size, deduplicated flag.
        """
        params: dict[str, Any] = {"data": data, "filename": filename}
        if mime_type is not None:
            params["mime_type"] = mime_type
        if metadata is not None:
            params["metadata"] = metadata
        return self._call("mfdb.objects.put_bytes", params)

    def get_object(self, object_uuid: str) -> dict[str, Any]:
        """Retrieve blob content by object UUID.

        Returns base64-encoded data.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        dict
            Result with base64-encoded ``data`` field.
        """
        return self._call("mfdb.objects.get", {"object_uuid": object_uuid})

    def get_object_info(self, object_uuid: str) -> dict[str, Any]:
        """Retrieve object metadata by UUID.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        dict
            Object metadata record.
        """
        return self._call("mfdb.objects.get_info", {"object_uuid": object_uuid})

    def delete_object(self, object_uuid: str) -> dict[str, Any]:
        """Delete an object or decrement its refcount.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        dict
            Result with ``deleted`` (bool) and ``refcount`` (int).
        """
        return self._call("mfdb.objects.delete", {"object_uuid": object_uuid})

    def list_objects(
        self,
        filename: str | None = None,
        user_uuid: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List objects with optional filtering.

        Parameters
        ----------
        filename : str, optional
            Filter by original filename (substring match).
        user_uuid : str, optional
            Filter by creator user UUID.
        limit : int
            Maximum number of results.
        offset : int
            Offset for pagination.

        Returns
        -------
        dict
            Result with ``objects`` list.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if filename is not None:
            params["filename"] = filename
        if user_uuid is not None:
            params["user_uuid"] = user_uuid
        return self._call("mfdb.objects.list", params)

    # ---- Internal ----

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Public RPC entry point (auth-injecting, envelope-unwrapping).

        External consumers (the dataset browser widget, plugin GUIs) call
        ``client.call(...)``; keep this in sync with ``_call``.
        """
        return self._call(method, params)

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        if self._token and "auth" not in params:
            params["auth"] = {"token": self._token}
        return self._call_raw(method, params)

    def _call_raw(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self._client.call(method, params or {})
        if not result.get("ok", True):
            error = result.get("error", method)
            logging.error("MFDB RPC failed: %s: %s", method, error)
            raise RuntimeError(error)
        return result.get("result", result)
