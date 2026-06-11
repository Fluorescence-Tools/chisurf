"""Local client wrapper for sample database services."""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient

from ..backend.services import register_services


class SampleDatabaseClient:
    """Client for sample database RPC handlers."""

    def __init__(self, client: Any | None = None):
        self._client = client or self._make_local_client()

    def _make_local_client(self) -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        dispatcher = ServiceDispatcher(SessionState())
        register_services(dispatcher)
        return InProcessClient(dispatcher)

    def status(self) -> dict[str, Any]:
        return self._call("sample_database.status")

    def list_samples(self) -> list[dict[str, Any]]:
        return self._call("sample_database.samples.list").get("samples", [])

    def get_sample(self, sample_id: str) -> dict[str, Any]:
        return self._call("sample_database.samples.get", {"sample_id": sample_id}).get("sample")

    def save_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._call("sample_database.samples.save", {"sample": sample}).get("sample")

    def delete_sample(self, sample_id: str) -> dict[str, Any]:
        return self._call("sample_database.samples.delete", {"sample_id": sample_id})

    def save_sample_key_values(
        self,
        sample_id: str,
        key_values: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Save sample-level key/value metadata."""
        return self._call(
            "sample_database.samples.key_values.save",
            {"sample_id": sample_id, "key_values": key_values},
        ).get("sample")

    def list_users(self) -> list[dict[str, Any]]:
        """Return known sample users/operators."""
        return self._call("sample_database.users.list").get("users", [])

    def save_user(self, user: dict[str, Any]) -> list[dict[str, Any]]:
        """Save a sample user/operator."""
        return self._call("sample_database.users.save", {"user": user}).get("users", [])

    def delete_user(self, user_id: str) -> list[dict[str, Any]]:
        """Delete a sample user/operator."""
        return self._call("sample_database.users.delete", {"user_id": user_id}).get("users", [])

    def list_devices(self) -> list[dict[str, Any]]:
        """Return known measurement devices."""
        return self._call("sample_database.devices.list").get("devices", [])

    def save_device(self, device: dict[str, Any]) -> list[dict[str, Any]]:
        """Save a measurement device."""
        return self._call("sample_database.devices.save", {"device": device}).get("devices", [])

    def delete_device(self, device_id: str) -> list[dict[str, Any]]:
        """Delete a sample user/operator."""
        return self._call("sample_database.devices.delete", {"device_id": device_id}).get(
            "devices", []
        )

    def list_experiment_types(self) -> list[dict[str, Any]]:
        """Return known experiment types."""
        return self._call("sample_database.experiment_types.list").get("experiment_types", [])

    def save_experiment_type(self, experiment_type: dict[str, Any]) -> list[dict[str, Any]]:
        """Save an experiment type."""
        return self._call(
            "sample_database.experiment_types.save",
            {"experiment_type": experiment_type},
        ).get("experiment_types", [])

    def delete_experiment_type(self, type_id: int) -> list[dict[str, Any]]:
        """Delete an experiment type."""
        return self._call(
            "sample_database.experiment_types.delete",
            {"type_id": type_id},
        ).get("experiment_types", [])

    def list_experiments(
        self,
        sample_id: str | None = None,
        project_id: str | None = None,
        type_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return experiments, optionally filtered by sample/project/type."""
        return self._call(
            "sample_database.experiments.list",
            {
                "sample_id": sample_id,
                "project_id": project_id,
                "type_id": type_id,
            },
        ).get("experiments", [])

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return a full experiment record."""
        return self._call(
            "sample_database.experiments.get",
            {"experiment_id": experiment_id},
        ).get("experiment")

    def save_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        """Save an experiment."""
        return self._call(
            "sample_database.experiments.save",
            {"experiment": experiment},
        ).get("experiment")

    def delete_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Delete an experiment."""
        return self._call(
            "sample_database.experiments.delete",
            {"experiment_id": experiment_id},
        )

    def save_experiment_key_values(
        self,
        experiment_id: str,
        key_values: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Save experiment-level key/value metadata."""
        return self._call(
            "sample_database.experiments.key_values.save",
            {"experiment_id": experiment_id, "key_values": key_values},
        ).get("experiment")

    def save_experiment_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Save experiment data link or embedded payload."""
        return self._call(
            "sample_database.experiments.data.save",
            {"data": data},
        ).get("experiment")

    def delete_experiment_data(self, data_id: int) -> dict[str, Any]:
        """Delete experiment data."""
        return self._call(
            "sample_database.experiments.data.delete",
            {"data_id": data_id},
        )

    def import_file(self, path: str) -> dict[str, Any]:
        return self._call("sample_database.import_file", {"path": path}).get("summary", {})

    def export_sample(
        self,
        sample_id: str,
        output_path: str | None = None,
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        """Export a sample as FLR CIF text or to a file."""
        return self._call(
            "sample_database.export_sample",
            {
                "sample_id": sample_id,
                "output_path": output_path,
                "analysis_id": analysis_id,
            },
        )

    def export_table(self, output_path: str, sample_id: str | None = None) -> dict[str, Any]:
        """Export sample table rows as CSV or Excel."""
        return self._call(
            "sample_database.export_table",
            {"output_path": output_path, "sample_id": sample_id},
        )

    def backup(self) -> str:
        return self._call("sample_database.backup").get("backup_path", "")

    def reset_from_source(self) -> dict[str, Any]:
        return self._call("sample_database.reset_from_source")

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self._client.call(method, params or {})
        if not result.get("ok", True):
            raise RuntimeError(result.get("error", method))
        return result.get("result", result)
