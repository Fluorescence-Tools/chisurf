"""ServiceDispatcher-compatible handlers for the mfdb-admin."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from chisurf.core.fio.mmcif.db import (
    FluorescenceDatabase,
    import_structure_file,
)
from chisurf.core.mfdb.database_resolver import (
    backup_database,
    resolve_database_path,
    source_database_path,
    user_database_path,
)
from chisurf.plugins.core.mfdb_admin.backend.auth_services import (
    register_services as register_auth_services,
)
from chisurf.core.mfdb.auth import (
    AuthError,
    PERM_READ,
    AnonymousPrincipal,
    can_access,
    create_default_acl_for_object,
    filter_readable,
    principal_from_rpc_auth,
    require_access,
    require_authenticated,
)
from chisurf.plugins.core.mfdb_admin.backend.measurement_services import (
    register_measurement_services,
)
from chisurf.plugins.core.mfdb_admin.backend.ndxplorer_services import (
    register_ndxplorer_services,
)
from chisurf.plugins.core.mfdb_admin.backend.password_services import (
    evaluate_password,
    hash_password,
)
from chisurf.plugins.core.mfdb_admin.backend.setup_services import (
    register_setup_services,
)

VERSIONED_MFDB_METHODS = {
    "samples.register": "register_sample",
    "samples.get": "get_sample",
    "samples.list": "list_samples",
    "experiments.register": "register_experiment",
    "experiments.get": "get_experiment",
    "experiments.list": "list_experiments",
    "artifacts.register": "register_artifact",
    "artifacts.get": "get_artifact",
    "artifacts.list": "list_artifacts",
    "operations.record": "record_operation",
    "operations.record_with_artifacts": "record_operation_with_artifacts",
    "operations.get": "get_operation",
    "operations.list": "list_operations",
    "operations.link_artifact": "record_operation_link",
    "operations.link": "record_operation_link",
    "operations.transition_status": "transition_operation_status",
    "graph.upstream": "graph_upstream",
    "graph.downstream": "graph_downstream",
    "graph.export": "export_graph",
    "graph.traverse": "traverse_canonical_graph",
    "graph.traverse_legacy": "traverse_legacy_graph",
    "parameters.record": "record_parameter",
    "parameters.get": "get_parameter",
    "parameters.list": "list_parameters",
    "chinet.sessions.save": "save_chinet_session",
    "chinet.sessions.get": "get_chinet_session",
    "chinet.sessions.list": "list_chinet_sessions",
    "chinet.sessions.restore": "restore_chinet_session",
    "setups.save": "save_setup",
    "setups.get": "get_setup",
    "setups.list": "list_setups",
    "audit.list": "list_audit_logs",
    "branches.create": "create_branch",
    "branches.fork": "fork_branch",
    "branches.get": "get_branch",
    "branches.list": "list_branches",
    "branches.update_head": "update_branch_head",
    "branches.delete": "delete_branch",
    "users.set_active_branch": "set_user_active_branch",
    "users.get_active_branch": "get_user_active_branch",
    "users.jump_to_operation": "jump_user_to_operation",
}


def register_services(dispatcher_or_context: Any) -> None:
    """Register mfdb RPC handlers."""
    dispatcher = getattr(dispatcher_or_context, "dispatcher", dispatcher_or_context)
    register_auth_services(dispatcher)
    register_measurement_services(dispatcher)
    register_ndxplorer_services(dispatcher)
    register_setup_services(dispatcher)

    # Versioned mfdb.v1.* services with auth enforcement
    import chisurf.core.mfdb.api as mfdb_api
    for name, handler_name in VERSIONED_MFDB_METHODS.items():
        handler = getattr(mfdb_api, handler_name)

        def _make_v1_handler(h):
            def _v1_handler(params):
                auth = params.pop("auth", None) if isinstance(params, dict) else None
                with FluorescenceDatabase(resolve_database_path()) as db:
                    _require_auth(auth, db.conn)
                return h(**params, auth=auth)
            return _v1_handler

        dispatcher.register(
            f"mfdb.v1.{name}",
            _make_v1_handler(handler),
        )

    # Legacy sample_database.* aliases for backward compat
    for name in (
        "status", "samples.list", "samples.get", "samples.save", "samples.delete",
        "samples.search", "samples.key_values.save",
        "users.list", "users.save", "users.delete",
        "devices.list", "devices.save", "devices.delete",
        "experiment_types.list", "experiment_types.save", "experiment_types.delete",
        "experiments.list", "experiments.get", "experiments.save", "experiments.delete",
        "experiments.key_values.save", "experiments.data.save", "experiments.data.delete",
        "import_file", "export_sample", "export_table", "backup", "reset_from_source",
    ):
        dispatcher.register(
            f"sample_database.{name}",
            lambda params, _n=name: _delegate_mfdb(params, _n),
        )

    # Main mfdb.* services
    for name, handler in {
        "status": status_handler,
        "samples.list": list_samples_handler,
        "samples.get": get_sample_handler,
        "samples.save": save_sample_handler,
        "samples.delete": delete_sample_handler,
        "samples.search": search_samples_handler,
        "samples.key_values.save": save_sample_key_values_handler,
        "sample_conditions.get": get_sample_condition_handler,
        "sample_conditions.save": save_sample_condition_handler,
        "probes.list": list_probes_handler,
        "probes.get": get_probe_handler,
        "probes.optical_properties.get": get_probe_optical_properties_handler,
        "users.list": list_users_handler,
        "users.save": save_user_handler,
        "users.delete": delete_user_handler,
        "devices.list": list_devices_handler,
        "devices.save": save_device_handler,
        "devices.delete": delete_device_handler,
        "experiment_types.list": list_experiment_types_handler,
        "experiment_types.save": save_experiment_type_handler,
        "experiment_types.delete": delete_experiment_type_handler,
        "experiments.list": list_experiments_handler,
        "experiments.get": get_experiment_handler,
        "experiments.save": save_experiment_handler,
        "experiments.delete": delete_experiment_handler,
        "experiments.key_values.save": save_experiment_key_values_handler,
        "experiments.data.save": save_experiment_data_handler,
        "experiments.data.delete": delete_experiment_data_handler,
        "import_file": import_file_handler,
        "export_sample": export_sample_handler,
        "export_table": export_table_handler,
        "backup": backup_handler,
        "reset_from_source": reset_from_source_handler,
        "setups.list": list_setups_handler,
        "setups.get": get_setup_handler,
        "setups.save": save_setup_handler,
        "setups.delete": delete_setup_handler,
        "setups.validate": validate_setup_handler,
    }.items():
        dispatcher.register(f"mfdb.{name}", lambda params, _handler=handler: _handler(**params))


def _delegate_mfdb(params: dict[str, Any], name: str) -> dict[str, Any]:
    """Route sample_database.* calls to the corresponding mfdb.* handler."""
    handler_name = name.replace(".", "_") + "_handler"
    handler = globals().get(handler_name)
    if handler is None:
        raise ValueError(f"No handler for mfdb.{name}")
    return handler(**params)


def _validate_mfdb_methods_in_manifest(manifest_path: str | Path | None = None) -> list[str]:
    """Return versioned MFDB RPC methods missing from the plugin manifest.

    Parameters
    ----------
    manifest_path : str or pathlib.Path, optional
        Manifest path to validate. When omitted, the mfdb-admin manifest next
        to this package is used.

    Returns
    -------
    list of str
        Fully qualified ``mfdb.v1.*`` method names absent from the manifest.

    Examples
    --------
    >>> _validate_mfdb_methods_in_manifest()
    []
    """
    if manifest_path is None:
        path = Path(__file__).parents[1] / "manifest.json"
    else:
        path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared = {
        item.get("name")
        for item in manifest.get("rpc_methods", [])
        if isinstance(item, dict)
    }
    expected = {f"mfdb.v1.{name}" for name in VERSIONED_MFDB_METHODS}
    return sorted(expected - declared)


def _validate_fdb_methods_in_manifest(manifest_path: str | Path | None = None) -> list[str]:
    """Backward-compatible alias for manifest validation."""
    return _validate_mfdb_methods_in_manifest(manifest_path)


def status_handler() -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        return {
            "source_database": str(source_database_path()),
            "user_database": str(user_database_path()),
            "schema_version": db._get_schema_version(),
            "sample_count": len(db.list_samples()),
        }


def list_samples_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        rows = db.list_samples()
        filtered = _require_or_acl_filter(auth, db.conn, "sample", rows, id_key="sample_id")
        return {"samples": [_json_row(row) for row in filtered]}


def search_samples_handler(query: str | None = None, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        all_samples = db.list_samples()
        filtered = _require_or_acl_filter(auth, db.conn, "sample", all_samples, id_key="sample_id")
        if not query:
            return {"samples": [_json_row(row) for row in filtered]}
        q = query.lower().strip()
        matched = []
        for row in filtered:
            row_dict = {key: row[key] for key in row.keys()}
            if (q in str(row_dict.get("sample_id", "")).lower() or
                q in str(row_dict.get("sample_uuid", "")).lower() or
                q in str(row_dict.get("description", "")).lower() or
                q in str(row_dict.get("project_id", "")).lower()):
                matched.append(row_dict)
        return {"samples": matched}


def get_sample_handler(sample_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_or_acl_access(auth, db.conn, "sample", sample_id)
        sample = db.get_sample_full(sample_id)
        return {"sample": sample}


def get_sample_condition_handler(condition_id: str) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        return {"condition": _get_sample_condition_row(db, condition_id)}


def save_sample_condition_handler(condition: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    condition_id = str(condition.get("condition_id") or "").strip()
    if not condition_id:
        raise ValueError("condition_id is required")
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = _save_sample_condition_row(db, condition)
    return {"condition": _json_row(row)}


def get_probe_handler(probe_id: int) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        row = db.get_probe(probe_id)
        return {"probe": _json_row(row) if row else {}}


def get_probe_optical_properties_handler(probe_id: int) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        return {"optical_properties": [_json_row(row) for row in db.get_optical_properties(probe_id)]}


def list_probes_handler() -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        probes = []
        for row in db.get_probes():
            item = _json_row(row)
            item["optical_properties"] = [_json_row(prop) for prop in db.get_optical_properties(item["probe_id"])]
            probes.append(item)
        return {"probes": probes}


def save_sample_key_values_handler(
    sample_id: str, key_values: list[dict[str, Any]], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.clear_sample_key_values(sample_id)
        for item in key_values:
            key = str(item.get("key") or "").strip()
            if key:
                db.set_sample_key_value(
                    sample_id,
                    key,
                    item.get("value", ""),
                    item.get("details"),
                )
    return get_sample_handler(sample_id, auth=auth)

def _list_users_internal(db: Any = None) -> dict[str, Any]:
    """Internal list users — no auth check. Used by save/delete handlers as response."""
    close_db = False
    if db is None:
        db = FluorescenceDatabase(resolve_database_path())
        close_db = True
    try:
        users = []
        for row in db.get_users():
            d = _json_row(row)
            d["has_password"] = d.get("password_hash") is not None and d.get("password_hash") != ""
            d.pop("password_hash", None)
            users.append(d)
        return {"users": users}
    finally:
        if close_db:
            db.close()


def list_users_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    return _list_users_internal(db=None)


def save_user_handler(user: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    user_id = str(user.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("user_id is required")

    password = user.get("password")

    with FluorescenceDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        requester_is_admin = requester is not None and requester.is_admin
        requester_id = requester.user_id if requester else None

        # Check if there are any admins in the DB
        has_admin_res = db.conn.execute("SELECT 1 FROM flr_sample_users WHERE is_admin = 1 LIMIT 1").fetchone()
        has_admins = (has_admin_res is not None)

        # Load existing user properties to preserve if not updating
        row = db.conn.execute("SELECT * FROM flr_sample_users WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            existing = dict(row)
        else:
            existing = {}

        # Check permissions if admins exist
        if has_admins:
            if requester is None:
                raise ValueError("Unauthorized: authentication required")
            if not requester_is_admin:
                if requester_id != user_id:
                    raise ValueError("Unauthorized: Non-admin users can only edit their own profile")
                if "is_admin" in user and int(user["is_admin"]) == 1 and existing.get("is_admin", 0) != 1:
                    raise ValueError("Unauthorized: Non-admin users cannot grant admin privileges")

        def get_merged(key, default=None):
            if key in user:
                val = user[key]
                if val == "":
                    return default
                return val
            return existing.get(key, default)

        display_name = str(get_merged("display_name", user_id))
        email = get_merged("email")
        if email:
            email = str(email).strip()
            if not email:
                email = None
            else:
                if "@" not in email or "." not in email.split("@")[-1]:
                    raise ValueError(f"Invalid email format: '{email}'")

        affiliation = get_merged("affiliation")
        department = get_merged("department")
        role = get_merged("role")
        address = get_merged("address")
        website = get_merged("website")
        phone = get_merged("phone")
        details = get_merged("details")
        user_uuid = get_merged("user_uuid")
        allow_passwordless_login = get_merged("allow_passwordless_login")

        # Determine final is_admin and password_hash
        existing_is_admin = existing.get("is_admin", 0)
        existing_password_hash = existing.get("password_hash")

        final_is_admin = existing_is_admin
        if "is_admin" in user:
            is_admin_val = int(user["is_admin"])
            if has_admins:
                if requester_is_admin:
                    final_is_admin = is_admin_val
            else:
                final_is_admin = is_admin_val

        if password is not None:
            if password == "":
                if final_is_admin == 1:
                    raise ValueError("Admin password cannot be empty")
                final_password_hash = None
            else:
                if final_is_admin == 1:
                    strength = evaluate_password(password)
                    if strength["score"] < 4:
                        raise ValueError(f"Admin password is too weak. Requirements: {', '.join(strength['feedback'])}")
                final_password_hash = hash_password(password)
        else:
            final_password_hash = existing_password_hash

        db.add_user(
            user_id=user_id,
            display_name=display_name,
            email=email,
            affiliation=affiliation,
            department=department,
            role=role,
            address=address,
            website=website,
            phone=phone,
            details=details,
            user_uuid=user_uuid,
            is_admin=final_is_admin,
            password_hash=final_password_hash,
            allow_passwordless_login=allow_passwordless_login,
        )
        result = _list_users_internal(db)
    return result


def delete_user_handler(user_id: str, force: bool = False, requester_id: str = None, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    if user_id in ("user_default", "guest"):
        raise ValueError(f"The built-in user ('{user_id}') cannot be deleted.")

    import sqlite3
    with FluorescenceDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        requester_is_admin = requester is not None and requester.is_admin
        requester_id = requester.user_id if requester else None

        if force:
            if not requester_is_admin:
                raise ValueError("Only administrators can force-delete users with committed measurements.")
        else:
            # Check mfdb_operation
            res = db.conn.execute("SELECT 1 FROM mfdb_operation WHERE operator_user_id = ? LIMIT 1", (user_id,)).fetchone()
            if res:
                raise ValueError(f"User '{user_id}' has committed data and cannot be deleted.")
            # Check flr_experiment
            res = db.conn.execute("SELECT 1 FROM flr_experiment WHERE measured_by_user_id = ? LIMIT 1", (user_id,)).fetchone()
            if res:
                raise ValueError(f"User '{user_id}' has committed data and cannot be deleted.")
            # Check flr_sample
            res = db.conn.execute("SELECT 1 FROM flr_sample WHERE measured_by_user_id = ? LIMIT 1", (user_id,)).fetchone()
            if res:
                raise ValueError(f"User '{user_id}' has committed data and cannot be deleted.")
            # Check legacy/audit tables
            for table, col in [
                ("fdb_processing_run", "operator_user_id"),
                ("fdb_audit_log", "operator_user_id"),
                ("fdb_operation", "operator_user_id"),
                ("mfdb_audit_log", "operator_user_id")
            ]:
                try:
                    res = db.conn.execute(f"SELECT 1 FROM {table} WHERE {col} = ? LIMIT 1", (user_id,)).fetchone()
                    if res:
                        raise ValueError(f"User '{user_id}' has committed data and cannot be deleted.")
                except sqlite3.OperationalError:
                    pass
                except Exception:
                    pass

        db.delete_user(user_id)
        result = _list_users_internal(db)
    return result


def list_devices_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"devices": [_json_row(row) for row in db.get_devices()]}


def save_device_handler(device: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    device_id = str(device.get("device_id") or "").strip()
    if not device_id:
        raise ValueError("device_id is required")
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.add_device(
            device_id,
            str(device.get("name") or device_id),
            device.get("device_type") or None,
            device.get("model") or None,
            device.get("serial_number") or None,
            device.get("location") or None,
            device.get("owner") or None,
            device.get("details") or None,
        )
    return list_devices_handler(auth=auth)


def delete_device_handler(device_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_device(device_id)
    return list_devices_handler(auth=auth)


def list_experiment_types_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"experiment_types": [_json_row(row) for row in db.get_experiment_types()]}


def save_experiment_type_handler(experiment_type: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    name = str(experiment_type.get("name") or "").strip()
    if not name:
        raise ValueError("experiment type name is required")
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.add_experiment_type(
            name,
            category=experiment_type.get("category"),
            description=experiment_type.get("description"),
            details=experiment_type.get("details"),
        )
    return list_experiment_types_handler(auth=auth)


def delete_experiment_type_handler(type_id: int, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_experiment_type(int(type_id))
    return list_experiment_types_handler(auth=auth)


def list_experiments_handler(
    sample_id: str | None = None,
    project_id: str | None = None,
    type_id: int | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.get_experiments(sample_id=sample_id, project_id=project_id, type_id=type_id)
        return {"experiments": [_experiment_row_dict(row) for row in rows]}


def get_experiment_handler(experiment_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.get_experiment(experiment_id)
        if row is None:
            return {"experiment": None}
        exp = _experiment_row_dict(row)
        exp["data"] = _experiment_data_rows(db, experiment_id)
        exp["key_values"] = _experiment_key_values(db, experiment_id)
        return {"experiment": exp}


def save_experiment_handler(experiment: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    experiment_id = str(experiment.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("experiment_id is required")
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.add_experiment(
            experiment_id,
            type_id=_int_or_none(experiment.get("type_id")),
            sample_id=experiment.get("sample_id") or None,
            project_id=experiment.get("project_id") or None,
            measured_by_user_id=experiment.get("measured_by_user_id") or None,
            measured_by_device_id=experiment.get("measured_by_device_id") or None,
            started_at=experiment.get("started_at") or None,
            ended_at=experiment.get("ended_at") or None,
            status=experiment.get("status") or None,
            details=experiment.get("details") or None,
        )
        db.clear_experiment_key_values(experiment_id)
        for item in experiment.get("key_values", []):
            key = str(item.get("key") or "").strip()
            if key:
                db.set_experiment_key_value(
                    experiment_id,
                    key,
                    item.get("value", ""),
                    item.get("details"),
                )
    return get_experiment_handler(experiment_id, auth=auth)


def delete_experiment_handler(experiment_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            db.conn.execute("DELETE FROM flr_experiment WHERE experiment_id = ?", (experiment_id,))
    return {"ok": True, "experiment_id": experiment_id}


def save_experiment_key_values_handler(
    experiment_id: str, key_values: list[dict[str, Any]], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.clear_experiment_key_values(experiment_id)
        for item in key_values:
            key = str(item.get("key") or "").strip()
            if key:
                db.set_experiment_key_value(
                    experiment_id,
                    key,
                    item.get("value", ""),
                    item.get("details"),
                )
    return get_experiment_handler(experiment_id, auth=auth)


def save_experiment_data_handler(data: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    experiment_id = str(data.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("experiment_id is required")
    data_type = str(data.get("data_type") or "").strip()
    if not data_type:
        raise ValueError("data_type is required")
    storage_mode = str(data.get("storage_mode") or "link").strip()
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if data.get("data_id"):
            db.update_experiment_data(
                int(data["data_id"]),
                experiment_id,
                data_type,
                storage_mode,
                file_path=data.get("file_path") or None,
                url=data.get("url") or None,
                folder_path=data.get("folder_path") or None,
                mime_type=data.get("mime_type") or None,
                size_bytes=_int_or_none(data.get("size_bytes")),
                checksum=data.get("checksum") or None,
                data_json=data.get("data_json") or None,
                data_blob=_bytes_or_none(data.get("data_blob")),
                reading_options_json=data.get("reading_options_json") or None,
                details=data.get("details") or None,
            )
            int(data["data_id"])
        else:
            db.add_experiment_data(
                experiment_id,
                data_type,
                storage_mode,
                file_path=data.get("file_path") or None,
                url=data.get("url") or None,
                folder_path=data.get("folder_path") or None,
                mime_type=data.get("mime_type") or None,
                size_bytes=_int_or_none(data.get("size_bytes")),
                checksum=data.get("checksum") or None,
                data_json=data.get("data_json") or None,
                data_blob=_bytes_or_none(data.get("data_blob")),
                reading_options_json=data.get("reading_options_json") or None,
                details=data.get("details") or None,
            )
    return get_experiment_handler(experiment_id, auth=auth)


def delete_experiment_data_handler(data_id: int, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            "SELECT experiment_id FROM flr_experiment_data WHERE data_id = ?", (int(data_id),)
        ).fetchone()
        experiment_id = row["experiment_id"] if row else None
        db.delete_experiment_data(int(data_id))
    return {"ok": True, "data_id": int(data_id), "experiment_id": experiment_id}


def save_sample_handler(sample: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    with FluorescenceDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        owner_user_id = requester.user_id if requester else "user_default"
        with db.conn:
            condition = sample.get("condition") or {}
            condition_id = condition.get("condition_id") or sample.get("sample_condition_id")
            if condition_id:
                _save_sample_condition_row(db, condition)
            assembly_id = sample.get("entity_assembly_id")
            if assembly_id:
                assembly = sample.get("entity_assembly") or {}
                db.add_entity_assembly(
                    str(assembly_id),
                    assembly.get("description") or "",
                    assembly.get("details") or "",
                )
            for entity in sample.get("entities", []):
                entity_id = str(entity.get("entity_id") or "").strip()
                if not entity_id:
                    continue
                db.add_entity(
                    entity_id,
                    type=entity.get("type") or "polymer",
                    description=entity.get("description"),
                    common_name=entity.get("common_name"),
                    formula_weight=_float_or_none(entity.get("formula_weight")),
                )
                sequence = entity.get("sequence")
                if sequence:
                    db.set_sequence(entity_id, [str(item) for item in sequence])
            db.add_sample(
                sample_id,
                uuid=sample.get("sample_uuid"),
                description=sample.get("description") or "",
                details=sample.get("details") or "",
                num_of_probes=_int_or_none(sample.get("num_of_probes")),
                solvent_phase=sample.get("solvent_phase"),
                sample_condition_id=str(condition_id) if condition_id else None,
                entity_assembly_id=str(assembly_id) if assembly_id else None,
                project_id=sample.get("project_id") or None,
                measured_by_user_id=sample.get("measured_by_user_id") or None,
                measured_by_device_id=sample.get("measured_by_device_id") or None,
                measured_at=sample.get("measured_at") or None,
            )
            db.clear_sample_key_values(sample_id)
            for item in sample.get("key_values", []):
                key = str(item.get("key") or "").strip()
                if key:
                    db.set_sample_key_value(
                        sample_id,
                        key,
                        item.get("value", ""),
                        item.get("details"),
                    )
            db.clear_sample_probes(sample_id)
            for mapping in sample.get("sample_probes", []):
                db.add_sample_probe(
                    sample_id,
                    int(mapping["probe_id"]),
                    _int_or_none(mapping.get("poly_probe_position_id")),
                    mapping.get("fluorophore_type") or "unspecified",
                    mapping.get("description") or "",
                    _int_or_none(mapping.get("sample_probe_id")),
                )
            create_default_acl_for_object(
                db.conn, "sample", sample_id, owner_user_id=owner_user_id,
            )
    return get_sample_handler(sample_id, auth=auth)


def delete_sample_handler(sample_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_sample(sample_id)
    return {"ok": True, "sample_id": sample_id}


def import_file_handler(path: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        summary = import_structure_file(db, path)
    return {"summary": summary}


def export_sample_handler(
    sample_id: str,
    output_path: str | None = None,
    analysis_id: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if analysis_id is None:
            row = db.conn.execute(
                "SELECT analysis_id FROM flr_fret_analysis "
                "WHERE sample_id = ? ORDER BY analysis_id LIMIT 1",
                (sample_id,),
            ).fetchone()
            analysis_id = row["analysis_id"] if row else None
        if output_path:
            path = db.export_flr_cif(Path(output_path), analysis_id=analysis_id)
            return {"output_path": str(path)}
        return {"text": db.export_flr_cif_to_text(analysis_id=analysis_id)}


def export_table_handler(output_path: str, sample_id: str | None = None) -> dict[str, Any]:
    path = Path(output_path)
    with FluorescenceDatabase(resolve_database_path()) as db:
        rows = _sample_table_rows(db, sample_id)
    _write_table(path, rows)
    return {"output_path": str(path)}


def backup_handler() -> dict[str, Any]:
    path = backup_database(resolve_database_path())
    return {"backup_path": str(path)}


def reset_from_source_handler() -> dict[str, Any]:
    user_path = user_database_path()
    source_path = source_database_path()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    backup_path = backup_database(user_path) if user_path.exists() else None
    tmp_path = user_path.with_suffix(user_path.suffix + ".tmp")
    try:
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(user_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {"ok": True, "backup_path": str(backup_path) if backup_path else None}


def list_setups_handler() -> dict[str, Any]:
    from chisurf.core.mfdb.api import list_setups as _list_setups
    return _list_setups()


def get_setup_handler(setup_id: str) -> dict[str, Any]:
    from chisurf.core.mfdb.api import get_setup as _get_setup
    return _get_setup(setup_id)


def save_setup_handler(setup: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    from chisurf.core.mfdb.api import save_setup as _save_setup
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
    return _save_setup(setup)


def delete_setup_handler(setup_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    from chisurf.core.mfdb.api import delete_setup as _delete_setup
    with FluorescenceDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
    return _delete_setup(setup_id)


def validate_setup_handler(setup_id: str) -> dict[str, Any]:
    from chisurf.core.mfdb.api import validate_setup as _validate_setup
    return _validate_setup(setup_id)


def _sample_table_rows(
    db: FluorescenceDatabase, sample_id: str | None = None
) -> list[dict[str, Any]]:
    params: tuple[Any, ...] = ()
    where = ""
    if sample_id:
        where = "WHERE s.sample_id = ?"
        params = (sample_id,)
    rows = db.conn.execute(
        "SELECT s.*, u.display_name AS measured_by_user, d.name AS measured_by_device "
        "FROM flr_sample AS s "
        "LEFT JOIN flr_sample_users AS u ON u.user_id = s.measured_by_user_id "
        "LEFT JOIN flr_sample_devices AS d ON d.device_id = s.measured_by_device_id "
        f"{where} ORDER BY s.sample_id",
        params,
    ).fetchall()
    result = []
    for row in rows:
        item = {key: row[key] for key in row.keys()}
        key_values = {
            kv["key"]: kv["value"] for kv in db.get_sample_key_values(str(row["sample_id"]))
        }
        item["key_values"] = json.dumps(key_values, ensure_ascii=False)
        result.append(item)
    return result


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        _write_excel(path, rows)
        return
    if suffix not in {".csv", ".tsv", ".txt"}:
        suffix = ".csv"
        path = path.with_suffix(suffix)
    delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=_table_fieldnames(rows), delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def _write_excel(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for Excel export") from exc
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "samples"
    fieldnames = _table_fieldnames(rows)
    sheet.append(fieldnames)
    for row in rows:
        sheet.append([row.get(key, "") for key in fieldnames])
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = min(
            max(len(str(cell.value or "")) for cell in column) + 2, 60
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _table_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names or ["sample_id"]


def _json_row(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _get_sample_condition_row(db: FluorescenceDatabase, condition_id: str) -> dict[str, Any]:
    row = db.conn.execute(
        "SELECT * FROM flr_sample_condition WHERE condition_id = ?",
        (condition_id,),
    ).fetchone()
    return _json_row(row) if row else {}


def _save_sample_condition_row(
    db: FluorescenceDatabase,
    condition: dict[str, Any],
) -> Any:
    condition_id = str(condition.get("condition_id") or "").strip()
    if not condition_id:
        raise ValueError("condition_id is required")
    db.conn.execute(
        """
        INSERT OR REPLACE INTO flr_sample_condition (
            condition_id,
            ph,
            temperature,
            ionic_strength,
            buffer_composition,
            details
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            condition_id,
            _float_or_none(condition.get("ph")),
            _float_or_none(condition.get("temperature")),
            _float_or_none(condition.get("ionic_strength")),
            condition.get("buffer_composition") or None,
            condition.get("details") or None,
        ),
    )
    db.conn.commit()
    return db.conn.execute(
        "SELECT * FROM flr_sample_condition WHERE condition_id = ?",
        (condition_id,),
    ).fetchone()


def _experiment_row_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _experiment_data_rows(db: FluorescenceDatabase, experiment_id: str) -> list[dict[str, Any]]:
    return [
        {key: r[key] for key in r.keys()}
        for r in db.get_experiment_data(experiment_id)
    ]


def _experiment_key_values(db: FluorescenceDatabase, experiment_id: str) -> list[dict[str, Any]]:
    try:
        rows = db.get_experiment_key_values(experiment_id)
        return [{key: r[key] for key in r.keys()} for r in rows]
    except Exception:
        return []


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _require_auth(auth: dict[str, Any] | None, conn: Any) -> Any:
    """Require authenticated principal from *auth* dict. Returns the principal.

    If no admin users exist (bootstrap), auth is skipped to allow the first
    admin account creation. Once at least one admin exists, all writes require
    an authenticated session.
    """
    row = conn.execute("SELECT 1 FROM flr_sample_users WHERE is_admin = 1 LIMIT 1").fetchone()
    if not row:
        return None  # Bootstrap: no admin exists yet, allow seed writes
    principal = principal_from_rpc_auth(conn, auth)
    require_authenticated(principal)
    return principal


def _require_or_acl_access(
    auth: dict[str, Any] | None, conn: Any, object_type: str, object_id: str
) -> Any:
    """Check ACL if available; fall back to requiring admin.

    Returns the principal (may be anonymous if ACL allows public read).
    """
    principal = principal_from_rpc_auth(conn, auth)
    row = conn.execute(
        "SELECT 1 FROM mfdb_object_acl WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL",
        (object_type, object_id),
    ).fetchone()
    if row:
        require_access(conn, principal, object_type, object_id, PERM_READ)
    elif not principal.is_admin:
        raise AuthError("Authentication required")
    return principal


def _require_or_acl_filter(
    auth: dict[str, Any] | None, conn: Any, object_type: str, rows: list, id_key: str = "id"
) -> list:
    """Filter rows by ACL if any have ACLs; otherwise require admin.

    Returns the filtered/checked rows.
    """
    principal = principal_from_rpc_auth(conn, auth)
    if not rows:
        return rows
    has_acls = False
    for row in rows:
        obj_id = row[id_key] if isinstance(row, dict) else row[id_key]
        r = conn.execute(
            "SELECT 1 FROM mfdb_object_acl WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL",
            (object_type, obj_id),
        ).fetchone()
        if r:
            has_acls = True
            break
    if has_acls:
        return filter_readable(conn, principal, object_type, rows, id_key=id_key)
    if not principal.is_admin:
        raise AuthError("Authentication required")
    return rows


def _bytes_or_none(value: Any) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)
