"""ServiceDispatcher-compatible handlers for the mfdb-admin."""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging
from mfdb.store.database_resolver import (
    backup_database,
    resolve_database_path,
    source_database_path,
    user_database_path,
)
from mfdb.samples.importer import import_structure_file
from mfdb.models import (
    EntityDefinition,
    FretPairDefinition,
    ProbeDefinition,
    SampleDefinition,
)
from mfdb.samples import reagents
from mfdb.schema.pdbx_metadata import MmcifDictionary
from mfdb.repository import MFDatabase
from mfdb.samples.sample_manager import (
    create_sample,
    get_sample_full_description,
    suggest_pdbx_keys,
    validate_sample_for_export,
)
from mfdb.admin.backend.auth_services import (
    register_services as register_auth_services,
)
from mfdb.security.auth import (
    AuthError,
    PERM_READ,
    AnonymousPrincipal,
    create_default_acl_for_object,
    filter_readable,
    principal_from_rpc_auth,
    require_access,
    require_authenticated,
)
from mfdb.admin.backend.measurement_services import (
    register_measurement_services,
)
from mfdb.admin.backend.ndxplorer_services import (
    register_ndxplorer_services,
)
from mfdb.admin.backend.password_services import (
    evaluate_password,
    hash_password,
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
    # Fluorophore curation (fluorophores.*), migrated from the fluorophore_db plugin.
    from mfdb.admin.backend.fluorophore_services import (
        register_services as register_fluorophore_services,
    )
    register_fluorophore_services(dispatcher)

    # Versioned mfdb.v1.* services with auth enforcement
    import mfdb.api as mfdb_api
    for name, handler_name in VERSIONED_MFDB_METHODS.items():
        handler = getattr(mfdb_api, handler_name)

        def _make_v1_handler(h):
            def _v1_handler(params):
                auth = params.pop("auth", None) if isinstance(params, dict) else None
                with MFDatabase(resolve_database_path()) as db:
                    _require_auth(auth, db.conn)
                return h(**params, auth=auth)
            return _v1_handler

        dispatcher.register(
            f"mfdb.v1.{name}",
            _make_v1_handler(handler),
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
        "samples.full_description": get_sample_full_description_handler,
        "samples.validate_export": validate_sample_export_handler,
        "samples.create_structured": create_structured_sample_handler,
        "sample_conditions.list": list_sample_conditions_handler,
        "sample_conditions.get": get_sample_condition_handler,
        "sample_conditions.save": save_sample_condition_handler,
        "sample_conditions.delete": delete_sample_condition_handler,
        "lifecycle.state": lifecycle_state_handler,
        "lifecycle.history": lifecycle_history_handler,
        "lifecycle.transition": lifecycle_transition_handler,
        "lifecycle.definitions": lifecycle_definitions_handler,
        "protocols.list": list_protocols_handler,
        "protocols.get": get_protocol_handler,
        "protocols.versions": list_protocol_versions_handler,
        "protocols.create": create_protocol_handler,
        "protocols.for_operation": protocol_for_operation_handler,
        "studies.list": list_studies_handler,
        "studies.get": get_study_handler,
        "studies.create": create_study_handler,
        "studies.members.add": add_study_member_handler,
        "studies.fields.set": set_study_field_handler,
        "reagents.list": list_reagent_lots_handler,
        "reagents.create": create_reagent_lot_handler,
        "reagents.expired": expired_reagent_lots_handler,
        "reagents.usage.list": list_reagent_usage_handler,
        "reagents.usage.add": add_reagent_usage_handler,
        "calibrations.list": list_calibrations_handler,
        "calibrations.stale": stale_calibrations_handler,
        "calibrations.create": create_calibration_handler,
        "pipelines.list": list_pipelines_handler,
        "pipelines.get": get_pipeline_handler,
        "pipelines.runs": list_pipeline_runs_handler,
        "entities.list": list_entities_handler,
        "entities.get": get_entity_handler,
        "entities.save": save_entity_handler,
        "entities.delete": delete_entity_handler,
        "probes.list": list_probes_handler,
        "probes.get": get_probe_handler,
        "probes.save": save_probe_handler,
        "probes.delete": delete_probe_handler,
        "probes.optical_properties.get": get_probe_optical_properties_handler,
        "probes.optical_properties.save": save_probe_optical_properties_handler,
        "probes.positions.list": list_probe_positions_handler,
        "probes.positions.get": get_probe_position_handler,
        "probes.positions.save": save_probe_position_handler,
        "probes.positions.delete": delete_probe_position_handler,
        "fret_pairs.list": list_fret_pairs_handler,
        "fret_pairs.get": get_fret_pair_handler,
        "fret_pairs.save": save_fret_pair_handler,
        "fret_pairs.delete": delete_fret_pair_handler,
        "pdbx.suggest_keys": suggest_pdbx_keys_handler,
        "pdbx.validate_value": validate_pdbx_value_handler,
        "mock_data.populate": populate_mock_data_handler,
        "users.list": list_users_handler,
        "users.save": save_user_handler,
        "users.delete": delete_user_handler,
        "devices.list": list_devices_handler,
        "devices.get": get_device_handler,
        "devices.save": save_device_handler,
        "devices.delete": delete_device_handler,
        "experiment_types.list": list_experiment_types_handler,
        "experiment_types.get": get_experiment_type_handler,
        "experiment_types.save": save_experiment_type_handler,
        "experiment_types.delete": delete_experiment_type_handler,
        "experiments.list": list_experiments_handler,
        "experiments.get": get_experiment_handler,
        "experiments.save": save_experiment_handler,
        "experiments.delete": delete_experiment_handler,
        "experiments.key_values.save": save_experiment_key_values_handler,
        "experiments.data.save": save_experiment_data_handler,
        "experiments.data.delete": delete_experiment_data_handler,
        "raw_data.list": list_raw_data_handler,
        "raw_data.get": get_raw_data_handler,
        "artifacts.delete": delete_artifact_handler,
        "artifacts.validation.set": set_artifact_validation_handler,
        "processing.list": list_processing_handler,
        "processing.get": get_processing_handler,
        "processed_data.list": list_processed_data_handler,
        "processed_data.get": get_processed_data_handler,
        "analysis.list": list_analysis_handler,
        "analysis.get": get_analysis_handler,
        "analysis.full": get_analysis_full_handler,
        "projects.list": list_projects_handler,
        "projects.get": get_project_handler,
        "branches.list": list_branches_handler,
        "branches.get": get_branch_handler,
        "branches.save": save_branch_handler,
        "branches.delete": delete_branch_handler,
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
        "setups.detector_channels.list": list_detector_channels_handler,
        "setups.pie_windows.list": list_pie_windows_handler,
        "setups.fcs_pairs.list": list_fcs_pairs_handler,
        "objects.put": put_object_handler,
        "objects.put_bytes": put_object_bytes_handler,
        "objects.get": get_object_handler,
        "objects.get_info": get_object_info_handler,
        "objects.delete": delete_object_handler,
        "objects.list": list_objects_handler,
        "datasets.browse": datasets_browse_handler,
        "datasets.open": datasets_open_handler,
    }.items():
        dispatcher.register(f"mfdb.{name}", lambda params, _handler=handler: _handler(**params))


def _default_user_id() -> str | None:
    """Configured local default user (the one registration stamps ownership with).

    Delegates to the PRD-17 canonical resolver so the anonymous in-process client
    scopes reads to the very identity registration stamped writes with.
    """
    from mfdb.security.session import configured_default_user_id

    return configured_default_user_id()


def _resolve_owner_id(db: "MFDatabase", auth: dict[str, Any] | None) -> str | None:
    """Resolve the acting user for MFDB dataset access (PRD-17 canonical resolver).

    Uses the authenticated principal when present; otherwise (e.g. the in-process
    desktop client, which has no session) falls back to the configured
    ``mfdb.default_user_id`` — the same identity registration stamps — so reads and
    writes agree.
    """
    # The canonical resolver already returns the authenticated principal when
    # present and the configured default otherwise — no separate anonymous-fallback
    # branch needed (PRD-17).
    from mfdb.security.session import resolve_active_user_id

    return resolve_active_user_id(auth, conn=db.conn)


def datasets_browse_handler(
    scope: str = "all",
    query: str | None = None,
    kinds: list[str] | None = None,
    formats: list[str] | None = None,
    sample_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Browse datasets with scope-based access control and pagination.

    Parameters
    ----------
    scope : str, default='all'
        ``'own'``, ``'public'``, or ``'all'``.
    query : str, optional
        Free-text search.
    kinds : list of str, optional
        Artifact kind filter.
    formats : list of str, optional
        Data format filter.
    sample_id : str, optional
        Filter by linked sample.
    limit : int, default=50
        Max results per page.
    offset : int, default=0
        Pagination offset.
    auth : dict, optional
        Session auth dict.

    Returns
    -------
    dict
        ``datasets``, ``total``, ``sample_counts``.
    """
    with MFDatabase(resolve_database_path()) as db:
        owner_id = _resolve_owner_id(db, auth)
        return db.browse_datasets(
            scope=scope,
            query=query,
            kinds=kinds,
            formats=formats,
            sample_id=sample_id,
            owner_id=owner_id,
            limit=limit,
            offset=offset,
        )


def datasets_open_handler(
    artifact_id: str,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a dataset and return a local readable path.

    Parameters
    ----------
    artifact_id : str
        Artifact identifier.
    auth : dict, optional
        Session auth dict.

    Returns
    -------
    dict
        ``local_path`` (str) key.
    """
    with MFDatabase(resolve_database_path()) as db:
        # Allow the in-process/local client (anonymous) when a default user is
        # configured, consistent with datasets.browse; otherwise require auth.
        owner_id = _resolve_owner_id(db, auth)
        if not owner_id:
            require_authenticated(principal_from_rpc_auth(db.conn, auth))
        local_path = db.open_dataset(artifact_id)
        return {"local_path": local_path}


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
    >>> _validate_mfdb_methods_in_manifest(manifest_path)  # doctest: +SKIP
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


def status_handler(auth: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        res = {
            "source_database": str(source_database_path()),
            "user_database": str(user_database_path()),
            "schema_version": db._get_schema_version(),
            "sample_count": 0,
            "experiment_count": 0,
            "raw_data_count": 0,
            "processed_run_count": 0,
            "user_count": 0,
            "device_count": 0,
            "provenance_edge_count": 0,
        }
        def get_count(table_name: str) -> int:
            try:
                row = db.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                return row[0] if row else 0
            except Exception:
                return 0

        res["sample_count"] = get_count("flr_sample")
        res["experiment_count"] = get_count("flr_experiment")
        res["raw_data_count"] = get_count("fdb_raw_data")
        res["processed_run_count"] = get_count("fdb_processing_run")
        res["user_count"] = get_count("flr_sample_users")
        res["device_count"] = get_count("flr_sample_devices")
        res["provenance_edge_count"] = get_count("fdb_provenance_edge")
        return res


def list_samples_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        rows = db.list_samples()
        filtered = _require_or_acl_filter(auth, db.conn, "sample", rows, id_key="sample_id")
        return {"samples": [_json_row(row) for row in filtered]}


def search_samples_handler(query: str | None = None, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
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
    with MFDatabase(resolve_database_path()) as db:
        _require_or_acl_access(auth, db.conn, "sample", sample_id)
        sample = db.get_sample_full(sample_id)
        return {"sample": sample}


def list_sample_conditions_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List sample conditions for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.conn.execute(
            "SELECT * FROM flr_sample_condition WHERE deleted_at IS NULL ORDER BY condition_id"
        ).fetchall()
    return {"sample_conditions": [_json_row(row) for row in rows]}


def get_sample_condition_handler(
    condition_id: str,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        return {"condition": _get_sample_condition_row(db, condition_id)}


def delete_sample_condition_handler(
    condition_id: str,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Soft-delete a sample condition."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            db.dao.soft_delete("flr_sample_condition", condition_id, deleted_at=_utc_now())
    return {"ok": True, "condition_id": condition_id}


# -- protocols (PRD-14 Increment 3) -----------------------------------------

def _schema_to_jsonable(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a ``{name: OperationParameterDef}`` schema to a JSON-safe list."""
    out: list[dict[str, Any]] = []
    for name, d in schema.items():
        out.append({
            "name": name,
            "value_type": getattr(d, "value_type", ""),
            "units": getattr(d, "units", None),
            "default_value": getattr(d, "default_value", None),
            "required": bool(getattr(d, "required", False)),
            "repeatable": bool(getattr(d, "repeatable", False)),
            "description": getattr(d, "description", None),
        })
    return out


def list_protocols_handler(
    scope: str = "all", auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List the latest version of each protocol, scoped own/public/all."""
    with MFDatabase(resolve_database_path()) as db:
        return {"protocols": db.list_protocols(scope=scope)}


def get_protocol_handler(
    name: str, version: Any = "latest", auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a protocol (default latest) with its declared parameter schema."""
    with MFDatabase(resolve_database_path()) as db:
        protocol = db.get_protocol(name, version=version)
        schema = db.get_protocol_parameter_schema(protocol) if protocol else {}
        return {"protocol": protocol, "parameter_schema": _schema_to_jsonable(schema)}


def list_protocol_versions_handler(
    name: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return all versions of a protocol name (oldest first)."""
    with MFDatabase(resolve_database_path()) as db:
        return {"versions": db.list_protocol_versions(name)}


def create_protocol_handler(
    name: str,
    category: str,
    operation_type: str | None = None,
    setup_id: str | None = None,
    description: str = "",
    is_public: bool = False,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a protocol (or a new version); an invalid category returns an error."""
    with MFDatabase(resolve_database_path()) as db:
        try:
            protocol_id, version = db.create_protocol(
                name, category, operation_type=operation_type, setup_id=setup_id,
                description=description, is_public=is_public,
            )
            return {"protocol_id": protocol_id, "version": version}
        except ValueError as exc:
            return {"error": str(exc)}


def protocol_for_operation_handler(
    operation_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the protocol/version recorded on an operation (run provenance)."""
    with MFDatabase(resolve_database_path()) as db:
        row = db.conn.execute(
            "SELECT protocol_id, protocol_version FROM mfdb_operation "
            "WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if not row or not row[0]:
            return {"protocol": None}
        return {
            "protocol": db.get_protocol_by_id(row[0]),
            "protocol_version": row[1],
        }


# -- studies / projects (PRD-13 Increment 3) --------------------------------

def list_studies_handler(
    scope: str = "all", auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List studies scoped mine/public/all."""
    with MFDatabase(resolve_database_path()) as db:
        return {"studies": db.list_studies(scope=scope)}


def get_study_handler(
    study_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a study with its members and configurable fields."""
    with MFDatabase(resolve_database_path()) as db:
        return {
            "study": db.get_study(study_id),
            "members": db.list_study_members(study_id),
            "fields": db.get_study_fields(study_id),
        }


def create_study_handler(
    name: str,
    description: str = "",
    is_public: bool = False,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a study; a missing name returns an error."""
    with MFDatabase(resolve_database_path()) as db:
        try:
            return {"study_id": db.create_study(name, description, is_public=is_public)}
        except ValueError as exc:
            return {"error": str(exc)}


def add_study_member_handler(
    study_id: str,
    member_type: str,
    member_id: str,
    role: str = "member",
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a sample/artifact to a study; a bad member_type returns an error."""
    with MFDatabase(resolve_database_path()) as db:
        try:
            db.add_study_member(study_id, member_type, member_id, role=role)
            return {"members": db.list_study_members(study_id)}
        except ValueError as exc:
            return {"error": str(exc)}


def set_study_field_handler(
    study_id: str, key: str, value: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Set a configurable per-study field."""
    with MFDatabase(resolve_database_path()) as db:
        db.set_study_field(study_id, key, value)
        return {"fields": db.get_study_fields(study_id)}


# -- reagents / consumables (PRD-15 LIMS P4) --------------------------------

def list_reagent_lots_handler(
    kind: str | None = None,
    include_expired: bool = False,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List reagent lots, optionally filtered by kind and excluding expired ones."""
    with MFDatabase(resolve_database_path()) as db:
        return {"lots": reagents.list_lots(db, kind, include_expired=include_expired)}


def create_reagent_lot_handler(
    kind: str,
    name: str,
    lot_number: str = "",
    vendor: str = "",
    expiry: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a reagent lot; a bad kind / missing name returns an error."""
    with MFDatabase(resolve_database_path()) as db:
        try:
            lot_id = reagents.add_reagent_lot(
                db, kind=kind, name=name, lot_number=lot_number, vendor=vendor,
                expiry=expiry or None,
            )
            return {"lot_id": lot_id}
        except ValueError as exc:
            return {"error": str(exc)}


def expired_reagent_lots_handler(
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List lots whose expiry is in the past (QC)."""
    with MFDatabase(resolve_database_path()) as db:
        return {"lots": reagents.expired_lots(db)}


def list_reagent_usage_handler(
    target_type: str, target_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List the reagent lots used by an operation/setup/sample."""
    with MFDatabase(resolve_database_path()) as db:
        return {"lots": reagents.list_reagents_for(db, target_type, target_id)}


def add_reagent_usage_handler(
    lot_id: str,
    target_type: str,
    target_id: str,
    role: str = "used",
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Link a lot to an operation/setup/sample; a bad target_type returns an error."""
    with MFDatabase(resolve_database_path()) as db:
        try:
            reagents.link_reagent(db, lot_id, target_type, target_id, role=role)
            return {"lots": reagents.list_reagents_for(db, target_type, target_id)}
        except ValueError as exc:
            return {"error": str(exc)}


# -- calibration provenance (PRD-05) ----------------------------------------

def list_calibrations_handler(
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List calibration records (type/method/value/notes), newest first."""
    from mfdb.lifecycle.staleness import list_calibrations

    with MFDatabase(resolve_database_path()) as db:
        return {"calibrations": list_calibrations(db)}


def stale_calibrations_handler(
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List uses whose calibration is superseded by a newer one of the same type."""
    from mfdb.lifecycle.staleness import find_stale_calibration_uses

    with MFDatabase(resolve_database_path()) as db:
        stale = find_stale_calibration_uses(db)
        return {"stale": [dataclasses.asdict(s) for s in stale]}


def create_calibration_handler(
    calibration_type: str,
    value: float,
    method: str = "user_provided",
    notes: str = "",
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a calibration value (defaults to a user-provided / literature value)."""
    from mfdb.provenance.result_registry import register_calibration

    if not calibration_type:
        return {"error": "calibration_type is required"}
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return {"error": f"value {value!r} is not a number"}
    with MFDatabase(resolve_database_path()) as db:
        artifact_id = register_calibration(
            data={calibration_type: numeric},
            calibration_type=calibration_type,
            parameters={calibration_type: numeric},
            method=method,
            notes=notes,
            db=db,
        )
    return {"artifact_id": artifact_id}


# -- pipelines / workflows (PRD-22) -----------------------------------------

def list_pipelines_handler(
    scope: str = "all", auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List stored pipeline definitions scoped mine/public/all."""
    from chisurf.core.pipeline import list_pipelines

    with MFDatabase(resolve_database_path()) as db:
        return {"pipelines": list_pipelines(db, scope=scope)}


def get_pipeline_handler(
    pipeline_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a pipeline's structure (nodes + typed edges), or an error."""
    from chisurf.core.pipeline import get_pipeline

    with MFDatabase(resolve_database_path()) as db:
        pipeline = get_pipeline(db, pipeline_id)
    if pipeline is None:
        return {"error": f"unknown pipeline_id {pipeline_id!r}"}
    return {
        "name": pipeline.name,
        "version": pipeline.version,
        "nodes": [
            {"name": n.name, "operation_type": n.operation_type, "parameters": n.parameters}
            for n in pipeline.nodes
        ],
        "edges": [
            {"source": e.source, "source_port": e.source_port,
             "target": e.target, "target_port": e.target_port}
            for e in pipeline.edges
        ],
    }


def list_pipeline_runs_handler(
    pipeline_id: str | None = None, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List runs (newest first) with operation counts, optionally for one pipeline."""
    from chisurf.core.pipeline import list_pipeline_runs

    with MFDatabase(resolve_database_path()) as db:
        return {"runs": list_pipeline_runs(db, pipeline_id)}


# -- lifecycle state machine (PRD-12 Increment 4) ---------------------------

def lifecycle_state_handler(
    entity_type: str, entity_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return an entity's current lifecycle state (or ``None``)."""
    with MFDatabase(resolve_database_path()) as db:
        return {"state": db.get_state(entity_type, entity_id)}


def lifecycle_history_handler(
    entity_type: str, entity_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return an entity's ordered transition history (already JSON-safe dicts)."""
    with MFDatabase(resolve_database_path()) as db:
        return {"history": db.get_state_history(entity_type, entity_id)}


def lifecycle_transition_handler(
    entity_type: str,
    entity_id: str,
    to_state: str,
    reason: str = "",
    operator_user_id: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transition an entity; an illegal jump is returned as ``error`` (not raised)."""
    from mfdb.lifecycle.lifecycle import StateTransitionError

    with MFDatabase(resolve_database_path()) as db:
        try:
            changed = db.transition_state(
                entity_type, entity_id, to_state,
                reason=reason, operator_user_id=operator_user_id,
            )
            return {"changed": changed, "state": db.get_state(entity_type, entity_id)}
        except StateTransitionError as exc:
            return {
                "changed": False,
                "error": str(exc),
                "state": db.get_state(entity_type, entity_id),
            }


def lifecycle_definitions_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the authored lifecycle definitions (states + transitions per entity)."""
    from mfdb.lifecycle.lifecycle import load_lifecycle_defs

    defs = load_lifecycle_defs()
    return {
        "definitions": {
            et: {
                "states": list(ld.states),
                "transitions": [[f, t] for f, t in ld.transitions],
            }
            for et, ld in defs.items()
        }
    }


def save_sample_condition_handler(condition: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    condition_id = str(condition.get("condition_id") or "").strip()
    if not condition_id:
        raise ValueError("condition_id is required")
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = _save_sample_condition_row(db, condition)
    return {"condition": _json_row(row)}


def get_probe_handler(probe_id: int) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        row = db.get_probe(probe_id)
        return {"probe": _json_row(row) if row else {}}


def get_probe_optical_properties_handler(probe_id: int) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        return {"optical_properties": [_json_row(row) for row in db.get_optical_properties(probe_id)]}


def list_probes_handler() -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        probes = []
        for row in db.get_probes():
            item = _json_row(row)
            item["optical_properties"] = [_json_row(prop) for prop in db.get_optical_properties(item["probe_id"])]
            probes.append(item)
        return {"probes": probes}


def save_sample_key_values_handler(
    sample_id: str, key_values: list[dict[str, Any]], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
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
        db = MFDatabase(resolve_database_path())
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
    user_uuid_input = str(user.get("user_uuid") or "").strip()
    old_user_id_input = str(user.get("old_user_id") or "").strip()

    password = user.get("password")

    with MFDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        requester_is_admin = requester is not None and requester.is_admin
        requester_id = requester.user_id if requester else None

        # Check if there are any admins in the DB
        has_admin_res = db.conn.execute("SELECT 1 FROM flr_sample_users WHERE is_admin = 1 LIMIT 1").fetchone()
        has_admins = (has_admin_res is not None)

        # User UUID is the stable backend identity. user_id is the mutable
        # human-readable username/login name.
        if user_uuid_input:
            row = db.conn.execute("SELECT * FROM flr_sample_users WHERE user_uuid = ?", (user_uuid_input,)).fetchone()
        elif old_user_id_input:
            row = db.conn.execute("SELECT * FROM flr_sample_users WHERE user_id = ?", (old_user_id_input,)).fetchone()
        else:
            row = db.conn.execute("SELECT * FROM flr_sample_users WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            existing = dict(row)
        else:
            existing = {}
        old_user_id = existing.get("user_id") or old_user_id_input or user_id
        is_rename = bool(existing) and old_user_id != user_id

        if is_rename:
            if old_user_id in ("user_default", "guest"):
                raise ValueError(f"Built-in user '{old_user_id}' cannot be renamed")
            if db.conn.execute("SELECT 1 FROM flr_sample_users WHERE user_id = ?", (user_id,)).fetchone():
                raise ValueError(f"User ID '{user_id}' already exists")

        # Check permissions if admins exist
        if has_admins:
            if requester is None:
                raise ValueError("Unauthorized: authentication required")
            if not requester_is_admin:
                if requester_id != old_user_id:
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

        # Admin accounts can never use passwordless login (defense-in-depth: the
        # login path also refuses empty-password admin logins).
        if final_is_admin == 1:
            allow_passwordless_login = 0

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

        if is_rename:
            _rename_user_id_references(db.conn, old_user_id, user_id)

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
        result["user"] = next(
            (item for item in result["users"] if item.get("user_id") == user_id),
            {},
        )
    return result


def _rename_user_id_references(conn: Any, old_user_id: str, new_user_id: str) -> None:
    """Rename a user ID and all username-based MFDB references.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    old_user_id : str
        Existing username.
    new_user_id : str
        New username.
    """
    logging.info("MFDB users: renaming user_id '%s' to '%s'", old_user_id, new_user_id)
    table_columns = (
        ("flr_sample", "measured_by_user_id"),
        ("flr_experiment", "measured_by_user_id"),
        ("fdb_processing_run", "operator_user_id"),
        ("fdb_audit_log", "operator_user_id"),
        ("fdb_operation", "operator_user_id"),
        ("mfdb_operation", "operator_user_id"),
        ("mfdb_audit_log", "operator_user_id"),
        ("mfdb_branch", "created_by_user_id"),
        ("mfdb_group", "created_by_user_id"),
        ("mfdb_group_member", "user_id"),
        ("mfdb_group_member", "created_by_user_id"),
        ("mfdb_object_acl", "owner_user_id"),
        ("mfdb_acl_entry", "created_by_user_id"),
        ("mfdb_session", "user_id"),
        ("mfdb_auth_attempt", "user_id"),
    )
    foreign_keys_enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if foreign_keys_enabled:
        conn.execute("PRAGMA foreign_keys = OFF")
    try:
        with conn:
            conn.execute("UPDATE flr_sample_users SET user_id = ? WHERE user_id = ?", (new_user_id, old_user_id))
            for table, column in table_columns:
                if not _table_has_column(conn, table, column):
                    continue
                conn.execute(f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (new_user_id, old_user_id))
            if _table_has_column(conn, "mfdb_acl_entry", "subject_id"):
                conn.execute(
                    "UPDATE mfdb_acl_entry SET subject_id = ? WHERE subject_type = 'user' AND subject_id = ?",
                    (new_user_id, old_user_id),
                )
    finally:
        if foreign_keys_enabled:
            conn.execute("PRAGMA foreign_keys = ON")
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ValueError(f"User rename left foreign-key violations: {violations}")


def _table_has_column(conn: Any, table: str, column: str) -> bool:
    """Return whether *table* exists and contains *column*.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    table : str
        Table name.
    column : str
        Column name.

    Returns
    -------
    bool
        ``True`` when the table and column exist.
    """
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
        return False
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def delete_user_handler(user_id: str, force: bool = False, requester_id: str = None, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    if user_id in ("user_default", "guest"):
        raise ValueError(f"The built-in user ('{user_id}') cannot be deleted.")

    import sqlite3
    with MFDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        requester_is_admin = requester is not None and requester.is_admin

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
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"devices": [_json_row(row) for row in db.get_devices()]}


def get_device_handler(device_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one instrument/device row."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            "SELECT * FROM flr_sample_devices WHERE device_id = ? AND deleted_at IS NULL",
            (device_id,),
        ).fetchone()
    return {"device": _json_row(row) if row else {}}


def save_device_handler(device: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    device_id = str(device.get("device_id") or "").strip()
    if not device_id:
        raise ValueError("device_id is required")
    with MFDatabase(resolve_database_path()) as db:
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
    return get_device_handler(device_id, auth=auth)


def delete_device_handler(device_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_device(device_id)
    return list_devices_handler(auth=auth)


def list_experiment_types_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"experiment_types": [_json_row(row) for row in db.get_experiment_types()]}


def get_experiment_type_handler(type_id: int, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one experiment type."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            "SELECT * FROM flr_experiment_type WHERE type_id = ? AND deleted_at IS NULL",
            (int(type_id),),
        ).fetchone()
    return {"experiment_type": _json_row(row) if row else {}}


def save_experiment_type_handler(experiment_type: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    name = str(experiment_type.get("name") or "").strip()
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if not name:
            existing_names = {row["name"] for row in db.get_experiment_types()}
            index = 1
            while f"untitled_experiment_type_{index}" in existing_names:
                index += 1
            name = f"untitled_experiment_type_{index}"
        type_id = db.add_experiment_type(
            name,
            category=experiment_type.get("category"),
            description=experiment_type.get("description"),
            details=experiment_type.get("details"),
        )
    return get_experiment_type_handler(type_id, auth=auth)


def delete_experiment_type_handler(type_id: int, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_experiment_type(int(type_id))
    return list_experiment_types_handler(auth=auth)


def list_experiments_handler(
    sample_id: str | None = None,
    project_id: str | None = None,
    type_id: int | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.get_experiments(sample_id=sample_id, project_id=project_id, type_id=type_id)
        return {"experiments": [_experiment_row_dict(row) for row in rows]}


def get_experiment_handler(experiment_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
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
    with MFDatabase(resolve_database_path()) as db:
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
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            # raw: admin hard delete (dao.soft_delete would only set deleted_at).
            db.conn.execute("DELETE FROM flr_experiment WHERE experiment_id = ?", (experiment_id,))
    return {"ok": True, "experiment_id": experiment_id}


def save_experiment_key_values_handler(
    experiment_id: str, key_values: list[dict[str, Any]], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
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
    with MFDatabase(resolve_database_path()) as db:
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
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            "SELECT experiment_id FROM flr_experiment_data WHERE data_id = ?", (int(data_id),)
        ).fetchone()
        experiment_id = row["experiment_id"] if row else None
        db.delete_experiment_data(int(data_id))
    return {"ok": True, "data_id": int(data_id), "experiment_id": experiment_id}


def _artifact_location(row: dict[str, Any]) -> str:
    """Return the best display location for an artifact row."""
    return (
        row.get("file_path")
        or row.get("url")
        or row.get("folder_path")
        or row.get("storage_path")
        or ""
    )


def _raw_data_row_for_gui(row: dict[str, Any] | Any) -> dict[str, Any]:
    """Return a raw-data artifact row flattened to the generic GUI schema."""
    data = dict(row)
    meta = _json_loads_safe(data.get("metadata_json"))
    return {
        "raw_data_id": data.get("artifact_id", ""),
        "experiment_id": data.get("experiment_id", ""),
        "data_type": meta.get("data_type") or data.get("data_format") or data.get("artifact_kind", ""),
        "storage_mode": data.get("storage_mode", ""),
        "location": _artifact_location(data),
        "validation_status": data.get("validation_status", ""),
        "checksum": data.get("checksum", ""),
        "details": data.get("metadata_json") or data.get("data_json") or "",
    }


def _processed_data_row_for_gui(row: dict[str, Any] | Any) -> dict[str, Any]:
    """Return a processed-data artifact row flattened to the generic GUI schema."""
    data = dict(row)
    meta = _json_loads_safe(data.get("metadata_json"))
    return {
        "product_id": data.get("artifact_id", ""),
        "processed_data_id": data.get("artifact_id", ""),
        "processing_id": meta.get("processing_id") or data.get("processing_id") or "",
        "product_type": meta.get("product_type") or data.get("artifact_kind", ""),
        "storage_mode": data.get("storage_mode", ""),
        "location": _artifact_location(data),
        "validation_status": data.get("validation_status", ""),
        "checksum": data.get("checksum", ""),
        "experiment_id": data.get("experiment_id", ""),
    }


def _operation_row_for_gui(row: dict[str, Any] | Any, *, id_key: str, type_key: str) -> dict[str, Any]:
    """Return an operation row flattened to processing/analysis GUI schemas."""
    data = dict(row)
    metadata = _json_loads_safe(data.get("metadata_json"))
    settings = data.get("settings_json") or ""
    operation_id = data.get("operation_id") or data.get("analysis_id") or ""
    operation_type = data.get("operation_type") or data.get("analysis_type") or ""
    return {
        id_key: operation_id,
        "experiment_id": data.get("experiment_id", ""),
        type_key: operation_type,
        "type": operation_type,
        "status": data.get("status") or data.get("convergence_status", ""),
        "model_name": metadata.get("model_name", ""),
        "settings": settings,
    }


def list_raw_data_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List raw-data artifacts for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.get_raw_data_references()
    return {"raw_data": [_raw_data_row_for_gui(row) for row in rows]}


def get_raw_data_handler(raw_data_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one raw-data artifact row."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.get_raw_data(raw_data_id)
    return {"raw_data": _raw_data_row_for_gui(row) if row else {}}


def delete_artifact_handler(artifact_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Soft-delete an artifact and its direct provenance links."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return db.delete_artifact(artifact_id)


def set_artifact_validation_handler(
    artifact_id: str,
    validation_status: str,
    validation_message: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Set the validation status for a raw or processed artifact."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        artifact = db.set_artifact_validation(
            artifact_id,
            validation_status,
            validation_message=validation_message,
        )
    return {"artifact": artifact}


def list_processing_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List processing operations for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.conn.execute(
            """SELECT * FROM mfdb_operation
               WHERE operation_type NOT IN ('project', 'local_fit', 'global_fit', 'analysis', 'fitting', 'decay_fit')
                 AND deleted_at IS NULL
               ORDER BY created_at DESC"""
        ).fetchall()
    return {"processing": [_operation_row_for_gui(row, id_key="processing_id", type_key="type") for row in rows]}


def get_processing_handler(processing_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one processing operation."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.get_operation(processing_id)
    return {
        "processing": _operation_row_for_gui(row, id_key="processing_id", type_key="type")
        if row else {}
    }


def list_processed_data_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List processed-data artifacts for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.list_artifacts(artifact_kind="processed_data")
    return {"processed_data": [_processed_data_row_for_gui(row) for row in rows]}


def get_processed_data_handler(product_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one processed-data artifact."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.get_processed_data(product_id)
    return {"processed_data": _processed_data_row_for_gui(row) if row else {}}


def list_analysis_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List analysis operations for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.list_analysis_runs()
    return {"analysis": [_operation_row_for_gui(row, id_key="analysis_id", type_key="type") for row in rows]}


def get_analysis_handler(analysis_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one analysis operation."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.get_analysis_run(analysis_id)
    return {
        "analysis": _operation_row_for_gui(row, id_key="analysis_id", type_key="type")
        if row else {}
    }


def get_analysis_full_handler(analysis_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one analysis operation with parameters and linked products."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.get_analysis_run_full(analysis_id)
    return {"analysis": row or {}}


def _project_row_dict(row: Any) -> dict[str, Any]:
    """Return a GUI-friendly project record from an MFDB project operation row."""
    d = _json_row(row)
    meta = _json_loads_safe(d.get("metadata_json"))
    project_id = str(meta.get("project_id") or d.get("operation_id") or "")
    project_name = meta.get("project_name") or meta.get("model_name") or project_id
    return {
        "project_id": project_id,
        "name": project_name,
        "project_name": project_name,
        "version_id": d.get("operation_id"),
        "version_number": meta.get("version_number", 1),
        "branch_uuid": meta.get("branch_uuid"),
        "parent_version_id": meta.get("parent_version_id"),
        "owner_user_id": d.get("operator_user_id"),
        "status": d.get("status"),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        "description": meta.get("description") or meta.get("notes") or "",
    }


def list_projects_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List MFDB project records for the admin Project EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.conn.execute(
            """SELECT operation_id, operator_user_id, status, metadata_json,
                      created_at, updated_at
               FROM mfdb_operation
               WHERE operation_type = 'project' AND deleted_at IS NULL
               ORDER BY created_at DESC"""
        ).fetchall()
        projects = [_project_row_dict(row) for row in rows]
    return {"projects": projects}


def get_project_handler(project_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one MFDB project record by project ID or version operation ID."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        rows = db.conn.execute(
            """SELECT operation_id, operator_user_id, status, metadata_json,
                      created_at, updated_at
               FROM mfdb_operation
               WHERE operation_type = 'project' AND deleted_at IS NULL
               ORDER BY created_at DESC"""
        ).fetchall()
        for row in rows:
            project = _project_row_dict(row)
            if project.get("project_id") == project_id or project.get("version_id") == project_id:
                return {"project": project}
    return {"project": None}


def list_branches_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List MFDB branches for the admin Branch EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"branches": db.list_branches()}


def get_branch_handler(branch_uuid: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one branch by UUID or name."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"branch": db.get_branch(branch_uuid)}


def save_branch_handler(branch: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create or update a branch record."""
    branch_uuid = str(branch.get("branch_uuid") or "").strip()
    name = str(branch.get("name") or "").strip()
    with MFDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        existing = db.get_branch(branch_uuid) if branch_uuid else db.get_branch(name)
        if existing:
            with db.conn:
                db.dao.update(
                    "mfdb_branch",
                    existing["branch_uuid"],
                    {
                        "name": name,
                        "description": branch.get("description") or None,
                        "parent_branch_uuid": branch.get("parent_branch_uuid") or None,
                        "head_operation_id": branch.get("head_operation_id") or None,
                    },
                )
            return {"branch": db.get_branch(existing["branch_uuid"])}
        if not name:
            existing_names = {row["name"] for row in db.list_branches()}
            index = 1
            while f"untitled_branch_{index}" in existing_names:
                index += 1
            name = f"untitled_branch_{index}"
        created_by_user_id = requester.user_id if requester else branch.get("created_by_user_id")
        created_uuid = db.create_branch(
            branch_uuid=branch_uuid or None,
            name=name,
            parent_branch_uuid=branch.get("parent_branch_uuid") or None,
            head_operation_id=branch.get("head_operation_id") or None,
            created_by_user_id=created_by_user_id,
            description=branch.get("description") or None,
        )
        return {"branch": db.get_branch(created_uuid)}


def delete_branch_handler(branch_uuid: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Soft-delete a branch."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_branch(branch_uuid)
    return {"ok": True, "branch_uuid": branch_uuid}


def save_sample_handler(sample: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    if _is_structured_sample_payload(sample):
        structured = dict(sample)
        structured["name"] = sample_id
        create_structured_sample_handler(structured, auth=auth)
        return get_sample_handler(sample_id, auth=auth)
    with MFDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        owner_user_id = requester.user_id if requester else "user_default"
        with db.conn:
            condition = sample.get("condition") or {}
            condition_id = condition.get("condition_id") or sample.get("sample_condition_id")
            if condition_id and condition:
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
                sequence = entity.get("sequence")
                db.add_entity(
                    entity_id,
                    name=entity.get("common_name") or entity.get("name") or entity_id,
                    sequence=[str(item) for item in sequence] if sequence else None,
                    entity_type=entity.get("type") or entity.get("entity_type") or "polymer",
                    details=entity.get("description") or entity.get("details"),
                )
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
                    mapping.get("fluorophore_type") or "unspecified",
                    mapping.get("description") or "",
                    _int_or_none(mapping.get("poly_probe_position_id")),
                )
            create_default_acl_for_object(
                db.conn, "sample", sample_id, owner_user_id=owner_user_id,
            )
    return get_sample_handler(sample_id, auth=auth)


def delete_sample_handler(sample_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_sample(sample_id)
    return {"ok": True, "sample_id": sample_id}


def get_sample_full_description_handler(
    sample_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the PRD-02 nested public sample description."""
    with MFDatabase(resolve_database_path()) as db:
        _require_or_acl_access(auth, db.conn, "sample", sample_id)
        description = get_sample_full_description(db, sample_id)
    return {"description": description}


def validate_sample_export_handler(
    sample_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return export validation warnings for a sample."""
    with MFDatabase(resolve_database_path()) as db:
        _require_or_acl_access(auth, db.conn, "sample", sample_id)
        warnings = validate_sample_for_export(db, sample_id)
    return {"warnings": warnings, "valid": not warnings}


def create_structured_sample_handler(
    sample_data: dict[str, Any], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a structured PRD-02 sample through sample_manager."""
    definition = _sample_definition_from_dict(sample_data)
    with MFDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        owner_user_id = requester.user_id if requester else "user_default"
        sample_id = create_sample(db, definition)
        create_default_acl_for_object(
            db.conn, "sample", sample_id, owner_user_id=owner_user_id,
        )
        description = get_sample_full_description(db, sample_id)
    return {"sample_id": sample_id, "description": description}


def list_entities_handler(
    sample_id: str | None = None, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List entities globally or for one sample."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if sample_id:
            rows = db.conn.execute(
                """
                SELECT DISTINCT e.*
                FROM entities AS e
                JOIN flr_poly_probe_position AS ppp ON ppp.entity_id = e.entity_id
                JOIN flr_sample_probe AS sp ON sp.poly_probe_position_id = ppp.id
                WHERE sp.sample_id = ?
                  AND e.deleted_at IS NULL
                  AND ppp.deleted_at IS NULL
                  AND sp.deleted_at IS NULL
                ORDER BY e.entity_id
                """,
                (sample_id,),
            ).fetchall()
        else:
            rows = db.get_entities()
        entities = [_entity_row_with_sequence(db, row) for row in rows]
    return {"entities": entities}


def get_entity_handler(
    entity_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return one entity row for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ? AND deleted_at IS NULL",
            (entity_id,),
        ).fetchone()
        entity = _entity_row_with_sequence(db, row) if row else {}
    return {"entity": entity}


def save_entity_handler(
    entity: dict[str, Any], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create or update an entity and its sequence."""
    entity_id = str(entity.get("entity_id") or entity.get("id") or "").strip()
    if not entity_id:
        raise ValueError("entity_id is required")
    sequence = entity.get("sequence")
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.add_entity(
            entity_id,
            name=entity.get("common_name") or entity.get("name") or entity_id,
            sequence=[str(item) for item in sequence] if sequence else None,
            entity_type=entity.get("type") or entity.get("entity_type") or "polymer",
            details=entity.get("description") or entity.get("details"),
        )
        row = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ? AND deleted_at IS NULL",
            (entity_id,),
        ).fetchone()
        saved = _entity_row_with_sequence(db, row) if row else {}
    return {"entity": saved}


def delete_entity_handler(
    entity_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Soft-delete an entity."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            db.dao.soft_delete("entities", entity_id, deleted_at=_utc_now())
    return {"ok": True, "entity_id": entity_id}


def _probe_row_for_payload(db: MFDatabase, probe_id: int) -> dict[str, Any]:
    """Return one probe row as a JSON payload."""
    return _json_row(db.get_probe(int(probe_id)))


def save_probe_handler(
    probe: dict[str, Any], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create or update a probe identity row."""
    name = str(probe.get("chromophore_name") or probe.get("name") or "").strip()
    if not name:
        raise ValueError("probe name is required")
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        probe_id = _int_or_none(probe.get("probe_id"))
        if probe_id is not None:
            # raw: resurrecting update (resets deleted_at = NULL) — dao.update
            # refuses to touch soft-deleted rows, so it cannot re-enable a probe.
            with db.conn:
                db.conn.execute(
                    """UPDATE probes
                       SET chromophore_name = ?,
                           category = ?,
                           description = ?,
                           reactive_probe_flag = ?,
                           reactive_probe_name = ?,
                           probe_origin = ?,
                           probe_link_type = ?,
                           chromophore_center_atom = ?,
                           updated_at = ?,
                           deleted_at = NULL
                       WHERE probe_id = ?""",
                    (
                        name,
                        probe.get("category") or "other",
                        probe.get("description"),
                        probe.get("reactive_probe_flag") or "no",
                        probe.get("reactive_probe_name") or None,
                        probe.get("probe_origin") or "extrinsic",
                        probe.get("probe_link_type") or "covalent",
                        probe.get("chromophore_center_atom") or None,
                        _utc_now(),
                        int(probe_id),
                    ),
                )
        else:
            probe_id = db.find_or_add_probe(
                name,
                category=probe.get("category") or "other",
                description=probe.get("description"),
                reactive_probe_flag=probe.get("reactive_probe_flag") or "no",
                reactive_probe_name=probe.get("reactive_probe_name") or None,
                probe_origin=probe.get("probe_origin") or "extrinsic",
                probe_link_type=probe.get("probe_link_type") or "covalent",
                chromophore_center_atom=probe.get("chromophore_center_atom") or None,
            )
        saved = _probe_row_for_payload(db, int(probe_id))
    return {"probe": saved}


def delete_probe_handler(
    probe_id: int, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Soft-delete a probe identity row."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_probe(int(probe_id))
    return {"ok": True, "probe_id": int(probe_id)}


def save_probe_optical_properties_handler(
    probe_id: int,
    properties: list[dict[str, Any]],
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace editable optical properties for a probe."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            db.dao.soft_delete(
                "optical_properties", int(probe_id), pk_column="probe_id", deleted_at=_utc_now()
            )
        for prop in properties:
            name = str(prop.get("property_name") or prop.get("property_type") or "").strip()
            if not name:
                continue
            db.add_optical_property(
                int(probe_id),
                name,
                prop.get("property_value", prop.get("value")),
                unit=prop.get("unit"),
                method=prop.get("method"),
                condition_json=prop.get("condition_json"),
                details=prop.get("details"),
            )
        rows = db.get_optical_properties(int(probe_id))
    return {"optical_properties": [_json_row(row) for row in rows]}


def list_probe_positions_handler(
    sample_id: str | None = None,
    probe_id: int | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List probe positions by sample or probe."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        where = ["ppp.deleted_at IS NULL"]
        params: list[Any] = []
        if sample_id:
            where.append("sp.sample_id = ?")
            params.append(sample_id)
        if probe_id is not None:
            where.append("ppp.probe_id = ?")
            params.append(int(probe_id))
        rows = db.conn.execute(
            f"""
            SELECT
                ppp.*,
                sp.sample_id,
                sp.sample_probe_id,
                sp.fluorophore_type,
                p.chromophore_name
            FROM flr_poly_probe_position AS ppp
            LEFT JOIN flr_sample_probe AS sp
              ON sp.poly_probe_position_id = ppp.id AND sp.deleted_at IS NULL
            LEFT JOIN probes AS p
              ON p.probe_id = ppp.probe_id
            WHERE {' AND '.join(where)}
            ORDER BY ppp.id
            """,
            params,
        ).fetchall()
    return {"positions": [_json_row(row) for row in rows]}


def get_probe_position_handler(
    id: int, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return one probe-position row for the generic EntityDock."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            """
            SELECT
                ppp.*,
                sp.sample_id,
                sp.sample_probe_id,
                sp.fluorophore_type,
                p.chromophore_name
            FROM flr_poly_probe_position AS ppp
            LEFT JOIN flr_sample_probe AS sp
              ON sp.poly_probe_position_id = ppp.id AND sp.deleted_at IS NULL
            LEFT JOIN probes AS p
              ON p.probe_id = ppp.probe_id
            WHERE ppp.id = ? AND ppp.deleted_at IS NULL
            """,
            (int(id),),
        ).fetchone()
    return {"position": _json_row(row) if row else {}}


def save_probe_position_handler(
    position: dict[str, Any], auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create or update a polymer probe-position row."""
    probe_id = _int_or_none(position.get("probe_id"))
    entity_id = str(position.get("entity_id") or "").strip()
    residue_number = _int_or_none(position.get("residue_number"))
    if probe_id is None or not entity_id or residue_number is None:
        raise ValueError("probe_id, entity_id, and residue_number are required")

    position_id = _int_or_none(position.get("id"))
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if position_id is None:
            position_id = db.add_poly_probe_position(
                probe_id,
                entity_id,
                residue_number,
                asym_id=position.get("asym_id") or "A",
                residue_name=position.get("residue_name") or None,
                description=position.get("description") or None,
                atom_id=position.get("atom_id") or None,
                mutation_flag=position.get("mutation_flag") or "no",
                modification_flag=position.get("modification_flag") or "no",
                auth_name=position.get("auth_name") or None,
            )
        else:
            # raw: resurrecting update (resets deleted_at = NULL) — dao.update
            # refuses to touch soft-deleted rows, so it cannot re-enable a position.
            with db.conn:
                db.conn.execute(
                    """UPDATE flr_poly_probe_position
                       SET probe_id = ?,
                           entity_id = ?,
                           residue_number = ?,
                           asym_id = ?,
                           residue_name = ?,
                           description = ?,
                           atom_id = ?,
                           mutation_flag = ?,
                           modification_flag = ?,
                           auth_name = ?,
                           updated_at = ?,
                           deleted_at = NULL
                       WHERE id = ?""",
                    (
                        int(probe_id),
                        entity_id,
                        int(residue_number),
                        position.get("asym_id") or "A",
                        position.get("residue_name") or None,
                        position.get("description") or None,
                        position.get("atom_id") or None,
                        position.get("mutation_flag") or "no",
                        position.get("modification_flag") or "no",
                        position.get("auth_name") or None,
                        _utc_now(),
                        int(position_id),
                    ),
                )
    return get_probe_position_handler(int(position_id), auth=auth)


def delete_probe_position_handler(
    id: int, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Soft-delete a polymer probe-position row."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            db.dao.soft_delete("flr_poly_probe_position", int(id), deleted_at=_utc_now())
    return {"ok": True, "id": int(id)}


def list_detector_channels_handler(
    setup_id: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List detector channel definitions."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if setup_id:
            rows = db.list_detector_channels(setup_id)
        else:
            rows = [
                dict(r) for r in db.conn.execute(
                    "SELECT * FROM mfdb_setup_detector_channel WHERE deleted_at IS NULL ORDER BY id"
                ).fetchall()
            ]
    return {"detector_channels": [_json_row(row) for row in rows]}


def list_pie_windows_handler(
    setup_id: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List PIE window definitions."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if setup_id:
            rows = db.list_pie_windows(setup_id)
        else:
            rows = [
                dict(r) for r in db.conn.execute(
                    "SELECT * FROM mfdb_setup_pie_window WHERE deleted_at IS NULL ORDER BY id"
                ).fetchall()
            ]
    return {"pie_windows": [_json_row(row) for row in rows]}


def list_fcs_pairs_handler(
    setup_id: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List FCS channel-pair definitions."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        if setup_id:
            rows = [
                dict(row) for row in db.conn.execute(
                    "SELECT * FROM mfdb_setup_fcs_pair "
                    "WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
                    (setup_id,),
                ).fetchall()
            ]
        else:
            rows = [
                dict(row) for row in db.conn.execute(
                    "SELECT * FROM mfdb_setup_fcs_pair WHERE deleted_at IS NULL ORDER BY id"
                ).fetchall()
            ]
    return {"fcs_pairs": [_json_row(row) for row in rows]}


def list_fret_pairs_handler(
    sample_id: str | None = None, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """List FRET pair/Forster radius records globally or for one sample."""
    with MFDatabase(resolve_database_path()) as db:
        if sample_id:
            _require_or_acl_access(auth, db.conn, "sample", sample_id)
        else:
            _require_auth(auth, db.conn)
        where = ["f.deleted_at IS NULL"]
        params: list[Any] = []
        if sample_id:
            where.append("f.sample_id = ?")
            params.append(sample_id)
        rows = db.conn.execute(
            f"""
            SELECT
                f.*,
                donor.chromophore_name AS donor_probe,
                acceptor.chromophore_name AS acceptor_probe
            FROM flr_fret_forster_radius AS f
            LEFT JOIN probes AS donor ON donor.probe_id = f.donor_probe_id
            LEFT JOIN probes AS acceptor ON acceptor.probe_id = f.acceptor_probe_id
            WHERE {' AND '.join(where)}
            ORDER BY f.forster_radius_id
            """,
            params,
        ).fetchall()
    return {"fret_pairs": [_json_row(row) for row in rows]}


def get_fret_pair_handler(
    forster_radius_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return one FRET pair/Forster radius row."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        row = db.conn.execute(
            """
            SELECT
                f.*,
                donor.chromophore_name AS donor_probe,
                acceptor.chromophore_name AS acceptor_probe
            FROM flr_fret_forster_radius AS f
            LEFT JOIN probes AS donor ON donor.probe_id = f.donor_probe_id
            LEFT JOIN probes AS acceptor ON acceptor.probe_id = f.acceptor_probe_id
            WHERE f.forster_radius_id = ? AND f.deleted_at IS NULL
            """,
            (forster_radius_id,),
        ).fetchone()
    return {"fret_pair": _json_row(row) if row else {}}


def save_fret_pair_handler(
    pair: dict[str, Any] | None = None,
    fret_pair: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a FRET pair/Forster radius record."""
    pair = pair or fret_pair or {}
    sample_id = str(pair.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    donor_probe_id = _int_or_none(pair.get("donor_probe_id") or pair.get("probe_id_1"))
    acceptor_probe_id = _int_or_none(pair.get("acceptor_probe_id") or pair.get("probe_id_2"))
    if donor_probe_id is None or acceptor_probe_id is None:
        raise ValueError("donor_probe_id and acceptor_probe_id are required")
    forster_radius_id = (
        str(pair.get("forster_radius_id") or "").strip()
        or f"{sample_id}_forster_{donor_probe_id}_{acceptor_probe_id}"
    )
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            # raw: hard delete-then-reinsert — a soft delete would leave the row
            # occupying the UNIQUE forster_radius_id and block re-add.
            db.conn.execute(
                "DELETE FROM flr_fret_forster_radius WHERE forster_radius_id = ?",
                (forster_radius_id,),
            )
        db.add_fret_forster_radius(
            forster_radius_id,
            sample_id,
            donor_probe_id,
            acceptor_probe_id,
            _float_or_none(pair.get("forster_radius") or pair.get("forster_radius_nm")) or 5.0,
            kappa_squared=_float_or_none(pair.get("kappa_squared")),
            refractive_index=_float_or_none(pair.get("index_of_refraction") or pair.get("refractive_index")),
            details=pair.get("details"),
        )
    pairs = list_fret_pairs_handler(sample_id, auth=auth)["fret_pairs"]
    saved = next((item for item in pairs if item.get("forster_radius_id") == forster_radius_id), {})
    return {"fret_pair": saved}


def delete_fret_pair_handler(
    forster_radius_id: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Delete a FRET pair/Forster radius record."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        with db.conn:
            # raw: admin hard delete by the UNIQUE forster_radius_id (not the PK).
            db.conn.execute(
                "DELETE FROM flr_fret_forster_radius WHERE forster_radius_id = ?",
                (forster_radius_id,),
            )
    return {"ok": True, "forster_radius_id": forster_radius_id}


def suggest_pdbx_keys_handler(
    prefix: str = "", auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return matching PDBx/flrCIF dictionary keys."""
    del auth
    keys = [
        {"key": key, "description": description}
        for key, description in suggest_pdbx_keys(prefix)
    ]
    return {"keys": keys}


def validate_pdbx_value_handler(
    key: str, value: str, auth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one PDBx/flrCIF key/value pair."""
    del auth
    full_name = key if key.startswith("_") else f"_{key}"
    message = MmcifDictionary.load_bundled().validate_value(full_name, str(value))
    return {"valid": message is None, "message": message or ""}


def populate_mock_data_handler(
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Populate the MFDB with bundled demo data from plugin test fixtures."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
    from mfdb.admin.seed_example import seed_example

    return {"summary": seed_example(resolve_database_path())}


def import_file_handler(path: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        summary = import_structure_file(db, path)
    return {"summary": summary}


def export_sample_handler(
    sample_id: str,
    output_path: str | None = None,
    analysis_id: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
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
    with MFDatabase(resolve_database_path()) as db:
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


def _setup_row_for_gui(row: dict[str, Any] | Any) -> dict[str, Any]:
    """Return a setup row flattened to the generic GUI schema."""
    data = dict(row)
    config = _json_loads_safe(data.get("configuration_json"))
    data["instrument_type"] = data.get("instrument_type") or config.get("setup_type", "")
    data["details"] = data.get("details") or config.get("details", "")
    data["laser_wavelengths"] = config.get("laser_wavelengths", "")
    data["detector_channels"] = data.get("detectors_json") or ""
    return data


def list_setups_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        return {"setups": [_setup_row_for_gui(row) for row in db.list_setups()]}


def get_setup_handler(setup_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        setup = db.get_setup(setup_id)
    return {"setup": _setup_row_for_gui(setup) if setup else {}}


def save_setup_handler(setup: dict[str, Any], auth: dict[str, Any] | None = None) -> dict[str, Any]:
    setup_id = str(setup.get("setup_id") or "").strip()
    name = str(setup.get("name") or setup_id).strip()
    if not setup_id:
        raise ValueError("setup_id is required")
    if not name:
        raise ValueError("setup name is required")
    config = _json_loads_safe(setup.get("configuration_json"))
    if setup.get("instrument_type"):
        config["setup_type"] = setup.get("instrument_type")
    if setup.get("details"):
        config["details"] = setup.get("details")
    laser_wavelengths = _json_loads_safe(setup.get("laser_wavelengths"))
    if laser_wavelengths:
        config["laser_wavelengths"] = laser_wavelengths
    detectors = _json_loads_safe(setup.get("detector_channels"))
    with MFDatabase(resolve_database_path()) as db:
        requester = _require_auth(auth, db.conn)
        db.save_setup(
            setup_id=setup_id,
            name=name,
            description=setup.get("description") or None,
            configuration=config,
            detectors=detectors or None,
            created_by_user_id=requester.user_id if requester else setup.get("created_by_user_id"),
            is_public=bool(setup.get("is_public")),
        )
    return get_setup_handler(setup_id, auth=auth)


def delete_setup_handler(setup_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        db.delete_setup(setup_id)
    return {"ok": True, "setup_id": setup_id}


def validate_setup_handler(setup_id: str) -> dict[str, Any]:
    from mfdb.api import validate_setup as _validate_setup
    return _validate_setup(setup_id)


def _sample_table_rows(
    db: MFDatabase, sample_id: str | None = None
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


def _json_loads_safe(value: Any) -> Any:
    """Return decoded JSON for strings, or a sensible empty value on failure."""
    if value in (None, ""):
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _utc_now() -> str:
    """Return an ISO UTC timestamp for service-layer soft deletes."""
    return datetime.now(timezone.utc).isoformat()


def _is_structured_sample_payload(sample: dict[str, Any]) -> bool:
    """Return whether a sample payload contains PRD-02 structured fields."""
    return any(
        isinstance(sample.get(key), list) and sample.get(key)
        for key in ("probes", "fret_pairs")
    ) or bool(sample.get("entities"))


def _sample_definition_from_dict(sample_data: dict[str, Any]) -> SampleDefinition:
    """Build a SampleDefinition from an RPC/GUI dictionary payload."""
    condition = sample_data.get("condition") or {}
    extra = dict(sample_data.get("extra") or sample_data.get("metadata") or {})
    for item in sample_data.get("key_values", []) or []:
        key = str(item.get("key") or "").strip()
        if key:
            extra[key] = item.get("value", "")

    probes = [
        ProbeDefinition(**_dataclass_kwargs(ProbeDefinition, _normalize_probe_payload(probe)))
        for probe in sample_data.get("probes", []) or []
    ]
    entities = [
        EntityDefinition(**_dataclass_kwargs(EntityDefinition, _normalize_entity_payload(entity)))
        for entity in sample_data.get("entities", []) or []
    ]
    fret_pairs = [
        FretPairDefinition(**_dataclass_kwargs(FretPairDefinition, pair))
        for pair in _normalize_fret_pair_payloads(sample_data.get("fret_pairs", []) or [], probes)
    ]

    return SampleDefinition(
        name=str(
            sample_data.get("name")
            or sample_data.get("display_name")
            or sample_data.get("sample_id")
            or ""
        ),
        description=sample_data.get("description") or sample_data.get("details") or "",
        entities=entities,
        probes=probes,
        fret_pairs=fret_pairs,
        buffer_description=(
            sample_data.get("buffer_description")
            or condition.get("buffer_composition")
            or ""
        ),
        ph=_float_or_none(sample_data.get("ph", condition.get("ph"))),
        temperature_k=_float_or_none(
            sample_data.get("temperature_k", condition.get("temperature"))
        ),
        salt_concentration_m=_float_or_none(
            sample_data.get("salt_concentration_m", condition.get("ionic_strength"))
        ),
        solvent_phase=sample_data.get("solvent_phase"),
        extra=extra,
        validate_vocabulary=bool(sample_data.get("validate_vocabulary", False)),
    )


def _dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    """Filter a dictionary to fields accepted by a dataclass."""
    names = {field.name for field in dataclasses.fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


def _normalize_entity_payload(entity: dict[str, Any]) -> dict[str, Any]:
    """Normalize GUI/API entity aliases to EntityDefinition fields."""
    return {
        **entity,
        "name": entity.get("name") or entity.get("common_name") or entity.get("entity_id") or "",
        "entity_type": entity.get("entity_type") or entity.get("type") or "",
        "details": entity.get("details") or entity.get("description") or "",
    }


def _normalize_probe_payload(probe: dict[str, Any]) -> dict[str, Any]:
    """Normalize GUI/API probe aliases to ProbeDefinition fields."""
    return {
        **probe,
        "name": probe.get("name") or probe.get("probe_name") or probe.get("chromophore_name") or "",
        "position": probe.get("position") or probe.get("residue_number"),
        "chain_id": probe.get("chain_id") or probe.get("asym_id") or "A",
        "residue_name": probe.get("residue_name") or probe.get("comp_id") or "",
        "seq_id": probe.get("seq_id") or probe.get("residue_number") or probe.get("position"),
        "comp_id": probe.get("comp_id") or probe.get("residue_name") or "",
        "asym_id": probe.get("asym_id") or probe.get("chain_id") or "A",
    }


def _normalize_fret_pair_payloads(
    pairs: list[dict[str, Any]], probes: list[ProbeDefinition]
) -> list[dict[str, Any]]:
    """Normalize GUI/API FRET pair aliases to FretPairDefinition fields."""
    probe_index = {probe.name: index for index, probe in enumerate(probes)}
    normalized = []
    for pair in pairs:
        item = dict(pair)
        if "probe_1_index" not in item and item.get("donor_probe") in probe_index:
            item["probe_1_index"] = probe_index[item["donor_probe"]]
        if "probe_2_index" not in item and item.get("acceptor_probe") in probe_index:
            item["probe_2_index"] = probe_index[item["acceptor_probe"]]
        if "forster_radius_nm" not in item and "forster_radius" in item:
            item["forster_radius_nm"] = item["forster_radius"]
        if "refractive_index" not in item and "index_of_refraction" in item:
            item["refractive_index"] = item["index_of_refraction"]
        normalized.append(item)
    return normalized


def _entity_row_with_sequence(db: MFDatabase, row: Any) -> dict[str, Any]:
    """Return an entity row dictionary with sequence included."""
    entity = _json_row(row)
    seq_rows = db.conn.execute(
        """
        SELECT mon_id
        FROM entity_poly_seq
        WHERE entity_id = ? AND deleted_at IS NULL
        ORDER BY num
        """,
        (entity["entity_id"],),
    ).fetchall()
    entity["sequence"] = [seq["mon_id"] for seq in seq_rows]
    return entity


def _get_sample_condition_row(db: MFDatabase, condition_id: str) -> dict[str, Any]:
    row = db.conn.execute(
        "SELECT * FROM flr_sample_condition WHERE condition_id = ?",
        (condition_id,),
    ).fetchone()
    return _json_row(row) if row else {}


def _save_sample_condition_row(
    db: MFDatabase,
    condition: dict[str, Any],
) -> Any:
    condition_id = str(condition.get("condition_id") or "").strip()
    if not condition_id:
        raise ValueError("condition_id is required")
    db.add_sample_condition(
        condition_id,
        ph=_float_or_none(condition.get("ph")),
        temperature=_float_or_none(condition.get("temperature")),
        ionic_strength=_float_or_none(condition.get("ionic_strength")),
        buffer_composition=condition.get("buffer_composition") or None,
        details=condition.get("details") or None,
    )
    return db.conn.execute(
        "SELECT * FROM flr_sample_condition WHERE condition_id = ?",
        (condition_id,),
    ).fetchone()


def _experiment_row_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _experiment_data_rows(db: MFDatabase, experiment_id: str) -> list[dict[str, Any]]:
    return [
        {key: r[key] for key in r.keys()}
        for r in db.get_experiment_data(experiment_id)
    ]


def _experiment_key_values(db: MFDatabase, experiment_id: str) -> list[dict[str, Any]]:
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
        logging.info("MFDB auth: allowing bootstrap write because no admin users exist")
        return None  # Bootstrap: no admin exists yet, allow seed writes
    principal = principal_from_rpc_auth(conn, auth)
    if not auth or not auth.get("token"):
        logging.warning("MFDB auth: rejecting write without session token")
    elif isinstance(principal, AnonymousPrincipal):
        logging.warning("MFDB auth: rejecting write with invalid, expired, or revoked session token")
    require_authenticated(principal)
    return principal


def _require_or_acl_access(
    auth: dict[str, Any] | None, conn: Any, object_type: str, object_id: str
) -> Any:
    """Check ACL if available; fall back to requiring admin.

    Returns the principal (may be anonymous if ACL allows public read).
    """
    if not conn.execute("SELECT 1 FROM flr_sample_users WHERE is_admin = 1 LIMIT 1").fetchone():
        return None
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
    if not conn.execute("SELECT 1 FROM flr_sample_users WHERE is_admin = 1 LIMIT 1").fetchone():
        return rows
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


def put_object_handler(
    path: str | None = None,
    data: str | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Store a file or bytes in the object store.

    Parameters
    ----------
    path : str, optional
        Path to the file to store (server reads from disk).
    data : str, optional
        Base64-encoded binary data to store.
    filename : str, optional
        Original filename to record.
    mime_type : str, optional
        MIME type of the content.
    metadata : dict, optional
        Additional metadata.
    auth : dict, optional
        Authentication token.

    Returns
    -------
    dict
        Object reference with uuid, md5, size, deduplicated flag.
    """
    import base64
    principal = None
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        principal = principal_from_rpc_auth(db.conn, auth)

    data_bytes = None
    if data is not None:
        data_bytes = base64.b64decode(data)

    with MFDatabase(resolve_database_path()) as db:
        result = db.put_object(
            path=path,
            data=data_bytes,
            filename=filename,
            mime_type=mime_type,
            metadata=metadata,
            created_by_user_uuid=getattr(principal, "user_id", None),
        )
    return {"ok": True, "object": result}


def put_object_bytes_handler(
    data: str,
    filename: str,
    mime_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Store base64-encoded bytes in the object store."""
    import base64
    principal = None
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        principal = principal_from_rpc_auth(db.conn, auth)

    data_bytes = base64.b64decode(data)
    with MFDatabase(resolve_database_path()) as db:
        result = db.put_object(
            data=data_bytes,
            filename=filename,
            mime_type=mime_type,
            metadata=metadata,
            created_by_user_uuid=getattr(principal, "user_id", None),
        )
    return {"ok": True, "object": result}


def get_object_handler(
    object_uuid: str,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Retrieve blob content by object UUID.

    Returns base64-encoded data.
    """
    import base64
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        data = db.get_object(object_uuid)
    return {"ok": True, "data": base64.b64encode(data).decode("ascii")}


def get_object_info_handler(
    object_uuid: str,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Retrieve object metadata by UUID."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        info = db.get_object_info(object_uuid)
    if info is None:
        return {"ok": False, "error": "Object not found"}
    return {"ok": True, "object": _json_row(info)}


def delete_object_handler(
    object_uuid: str,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Delete an object or decrement its refcount."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        result = db.delete_object(object_uuid)
    return {"ok": True, **result}


def list_objects_handler(
    filename: str | None = None,
    user_uuid: str | None = None,
    limit: int = 100,
    offset: int = 0,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """List objects with optional filtering."""
    with MFDatabase(resolve_database_path()) as db:
        _require_auth(auth, db.conn)
        objects = db.list_objects(
            filename=filename,
            user_uuid=user_uuid,
            limit=limit,
            offset=offset,
        )
    return {"ok": True, "objects": [_json_row(o) for o in objects]}
