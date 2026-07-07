"""Operation-parameter access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``add_parameter`` / ``get_parameters`` / ``delete_parameter`` surface
as a mixin. Extracted verbatim from the former repository god-class; behaviour is
unchanged.
"""

from __future__ import annotations

from typing import Any

from mfdb.schema._sqlutil import _utc_now


class ParameterMixin:
    """Scalar operation-parameter reads and writes."""

    def add_parameter(self, param_id, name, value, param_type="string", unit=None, description=None, operation_id=None, artifact_id=None, details=None):
        if not param_id:
            raise ValueError("param_id is required")
        metadata = details or {}
        if isinstance(metadata, dict):
            metadata = metadata.copy()
            if unit is not None:
                metadata["unit"] = unit
            if description is not None:
                metadata["description"] = description
        return self.record_parameter(
            parameter_uuid=param_id,
            operation_id=operation_id,
            name=name,
            value=value,
            units=unit,
            parameter_type=param_type,
            metadata=metadata,
        )

    def get_parameters(self, operation_id=None, artifact_id=None):
        if artifact_id is not None:
            # raw: DISTINCT fan-out over the operation-artifact junction + IN(...)
            # over the resulting id set — not expressible via the equality DAO.
            operation_ids = [
                row["operation_id"]
                for row in self.conn.execute(
                    "SELECT DISTINCT operation_id FROM mfdb_operation_artifact WHERE artifact_id = ? AND deleted_at IS NULL",
                    (artifact_id,),
                ).fetchall()
            ]
            if not operation_ids:
                return []
            placeholders = ",".join("?" for _ in operation_ids)
            params: list[Any] = operation_ids
            query = f"SELECT * FROM mfdb_parameter WHERE operation_id IN ({placeholders}) AND deleted_at IS NULL ORDER BY parameter_id"
            return [dict(row) for row in self.conn.execute(query, params).fetchall()]
        filters = {"operation_id": operation_id} if operation_id is not None else None
        return self.dao.list("mfdb_parameter", filters=filters, order_by="parameter_id")

    def delete_parameter(self, param_id):
        # PRD-26 Task 2: schema-driven soft-delete (was two hand UPDATEs). param_id may
        # be either key, so soft-delete by each; the explicit _utc_now() keeps the
        # stored deleted_at marker format identical. The only delta is the now-idempotent
        # `AND deleted_at IS NULL` guard (callers ignore the rowcount).
        with self._transaction():
            now = _utc_now()
            self.dao.soft_delete("mfdb_parameter", param_id, pk_column="parameter_uuid", deleted_at=now)
            self.dao.soft_delete("mfdb_parameter", param_id, pk_column="parameter_id", deleted_at=now)
