from __future__ import annotations

from typing import Any

from mfdb.security.auth import (
    PERM_READ,
    AuthError,
    can_access,
    create_default_acl_for_object,
    principal_from_rpc_auth,
    require_authenticated,
    require_access,
)
from mfdb.store.database_resolver import resolve_database_path
from mfdb.repository import MFDatabase


def _check_acl_access(conn: Any, principal: Any, object_type: str, object_id: str) -> None:
    """Check ACL if available; fall back to requiring admin."""
    row = conn.execute(
        "SELECT 1 FROM mfdb_object_acl WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL",
        (object_type, object_id),
    ).fetchone()
    if row:
        require_access(conn, principal, object_type, object_id, PERM_READ)
    elif not principal.is_admin:
        raise AuthError("Authentication required")


def _check_acl_filter(conn: Any, principal: Any, object_type: str, rows: list, id_key: str) -> list:
    """Filter rows by ACL if any have ACLs; otherwise require admin."""
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
        from mfdb.security.auth import filter_readable
        return filter_readable(conn, principal, object_type, rows, id_key=id_key)
    if not principal.is_admin:
        raise AuthError("Authentication required")
    return rows


def _check_api_auth(auth: dict[str, Any] | None) -> None:
    """Authenticate an RPC-style *auth* dict for direct api.py calls."""
    with MFDatabase(resolve_database_path()) as db:
        principal = principal_from_rpc_auth(db.conn, auth)
        require_authenticated(principal)
        return principal.user_id, principal.is_admin


def register_artifact(
    artifact_id: str,
    artifact_type: str | None = None,
    storage_mode: str = "local_file",
    experiment_id: str | None = None,
    file_path: str | None = None,
    url: str | None = None,
    folder_path: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    checksum: str | None = None,
    checksum_algorithm: str = "sha256",
    row_count: int | None = None,
    validation_status: str = "unvalidated",
    validation_message: str | None = None,
    metadata: dict[str, Any] | None = None,
    data_json: str | None = None,
    data_blob: bytes | None = None,
    artifact_kind: str | None = None,
    data_format: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register or update an MFDB artifact in the database.

    Parameters
    ----------
    artifact_id : str
        Unique identifier for the artifact.
    artifact_type : str, optional
        Compatibility alias for ``artifact_kind``.
    artifact_kind : str, optional
        Canonical artifact kind.
    storage_mode : str
        Storage location type ('local', 'remote', etc.).
    experiment_id : str, optional
        Associated experiment ID.
    file_path : str, optional
        Local file path if applicable.
    url : str, optional
        Remote URL if applicable.
    folder_path : str, optional
        Local folder path.
    mime_type : str, optional
        Mimetype of the file/data.
    size_bytes : int, optional
        Size of the artifact in bytes.
    checksum : str, optional
        Data verification checksum.
    checksum_algorithm : str, default='sha256'
        Algorithm used for checksum verification.
    row_count : int, optional
        Number of rows/events if applicable.
    validation_status : str, default='unvalidated'
        Validation status of the artifact.
    validation_message : str, optional
        Detailed message explaining validation status.
    metadata : dict, optional
        Arbitrary user metadata.
    data_json : str, optional
        Inline JSON data.
    data_blob : bytes, optional
        Raw binary data.
    data_format : str, optional
        Data format vocabulary value.
    auth : dict, optional
        Session auth dict for authorization.

    Returns
    -------
    dict
        RPC result dictionary with keys 'ok' and 'artifact_id'.
    """
    with MFDatabase(resolve_database_path()) as db:
        principal = principal_from_rpc_auth(db.conn, auth)
        require_authenticated(principal)
        user_id = principal.user_id
        kind = artifact_kind or artifact_type
        with db.conn:
            art_id = db.register_artifact(
                artifact_id=artifact_id,
                artifact_kind=kind,
                storage_mode=storage_mode,
                experiment_id=experiment_id,
                file_path=file_path,
                url=url,
                folder_path=folder_path,
                mime_type=mime_type,
                size_bytes=size_bytes,
                checksum=checksum,
                checksum_algorithm=checksum_algorithm,
                row_count=row_count,
                validation_status=validation_status,
                validation_message=validation_message,
                metadata=metadata,
                data_json=data_json,
                data_blob=data_blob,
                data_format=data_format,
                created_by_user_id=user_id,
            )
            create_default_acl_for_object(db.conn, "artifact", str(art_id), owner_user_id=user_id)
        return {"ok": True, "artifact_id": art_id}


def get_artifact(artifact_id: str, auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrieve an artifact by its identifier.

    Parameters
    ----------
    artifact_id : str
        Unique artifact identifier.
    auth : dict, optional
        Session auth dict for authorization.

    Returns
    -------
    dict
        RPC result containing the artifact dictionary under key 'artifact'.
    """
    with MFDatabase(resolve_database_path()) as db:
        principal = principal_from_rpc_auth(db.conn, auth)
        _check_acl_access(db.conn, principal, "artifact", artifact_id)
        art = db.get_artifact(artifact_id)
        return {"artifact": art}


def list_artifacts(
    artifact_type: str | None = None,
    experiment_id: str | None = None,
    artifact_kind: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List artifacts with optional filtering.

    Parameters
    ----------
    artifact_type : str, optional
        Compatibility alias for ``artifact_kind``.
    artifact_kind : str, optional
        Canonical artifact kind filter.
    experiment_id : str, optional
        Filter results by experiment ID.

    Returns
    -------
    dict
        RPC result containing a list of artifacts under key 'artifacts'.
    """
    with MFDatabase(resolve_database_path()) as db:
        principal = principal_from_rpc_auth(db.conn, auth)
        kind = artifact_kind or artifact_type
        artifacts = db.list_artifacts(
            artifact_kind=kind,
            experiment_id=experiment_id,
        )
        filtered = _check_acl_filter(db.conn, principal, "artifact", artifacts, id_key="artifact_id")
        return {"artifacts": filtered}


def record_operation(
    operation_id: str,
    operation_type: str,
    experiment_id: str | None = None,
    setup_id: str | None = None,
    settings: dict[str, Any] | None = None,
    operator_user_id: str | None = None,
    software_package: str | None = "chisurf",
    software_module: str | None = None,
    software_version: str | None = None,
    runtime_environment: dict[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    status: str = "pending",
    error_message: str | None = None,
    traceback_summary: str | None = None,
    metadata: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record or update a processing or analysis operation.

    Parameters
    ----------
    operation_id : str
        Unique operation identifier.
    operation_type : str
        Type of operation performed.
    experiment_id : str, optional
        Associated experiment ID.
    setup_id : str, optional
        ID of instrument setup used.
    settings : dict, optional
        Operation configuration settings.
    operator_user_id : str, optional
        User ID of the operator.
    software_package : str, default='chisurf'
        Software package used.
    software_module : str, optional
        Software module name.
    software_version : str, optional
        Version of the software.
    runtime_environment : dict, optional
        Detailed environment description.
    started_at : str, optional
        ISO timestamp of execution start.
    ended_at : str, optional
        ISO timestamp of execution end.
    status : str, default='pending'
        Status of the operation.
    error_message : str, optional
        Error message if execution failed.
    traceback_summary : str, optional
        Traceback summary if execution failed.
    metadata : dict, optional
        User metadata dict.

    Returns
    -------
    dict
        RPC result dictionary with keys 'ok' and 'operation_id'.
    """
    with MFDatabase(resolve_database_path()) as db:
        principal = principal_from_rpc_auth(db.conn, auth)
        require_authenticated(principal)
        op_id = db.record_operation(
            operation_id=operation_id,
            operation_type=operation_type,
            experiment_id=experiment_id,
            setup_id=setup_id,
            settings=settings,
            operator_user_id=operator_user_id,
            software_package=software_package,
            software_module=software_module,
            software_version=software_version,
            runtime_environment=runtime_environment,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            error_message=error_message,
            traceback_summary=traceback_summary,
            metadata=metadata,
        )
        return {"ok": True, "operation_id": op_id}


def record_operation_with_artifacts(
    operation_id: str,
    operation_type: str,
    status: str = "pending",
    experiment_id: str | None = None,
    setup_id: str | None = None,
    settings: dict[str, Any] | None = None,
    operator_user_id: str | None = None,
    software_package: str | None = None,
    software_module: str | None = None,
    software_version: str | None = None,
    runtime_environment: dict[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    input_artifacts: list[dict[str, Any]] | None = None,
    output_artifacts: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    error_message: str | None = None,
    traceback_summary: str | None = None,
    metadata: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an operation and related artifacts atomically.

    Parameters
    ----------
    operation_id : str
        Unique operation identifier.
    operation_type : str
        Operation vocabulary value.
    status : str, default='pending'
        Initial lifecycle status.
    experiment_id : str, optional
        Associated experiment identifier.
    setup_id : str, optional
        Associated setup identifier.
    settings : dict, optional
        Operation settings.
    operator_user_id : str, optional
        Operator identifier.
    software_package : str, optional
        Software package name.
    software_module : str, optional
        Software module name.
    software_version : str, optional
        Software version.
    runtime_environment : dict, optional
        Runtime metadata.
    started_at : str, optional
        Start timestamp.
    ended_at : str, optional
        End timestamp.
    input_artifacts : list of dict, optional
        Input artifact payloads.
    output_artifacts : list of dict, optional
        Output artifact payloads.
    parameters : list of dict, optional
        Parameter payloads.
    error_message : str, optional
        Failure message.
    traceback_summary : str, optional
        Traceback summary.
    metadata : dict, optional
        Operation metadata.
    auth : dict, optional
        Session auth dict for authorization.

    Returns
    -------
    dict
        RPC result dictionary with keys 'ok' and counts.
    """
    with MFDatabase(resolve_database_path()) as db:
        principal = principal_from_rpc_auth(db.conn, auth)
        require_authenticated(principal)
        result = db.record_operation_with_artifacts(
            operation_id=operation_id,
            operation_type=operation_type,
            status=status,
            experiment_id=experiment_id,
            setup_id=setup_id,
            settings=settings,
            operator_user_id=operator_user_id,
            software_package=software_package,
            software_module=software_module,
            software_version=software_version,
            runtime_environment=runtime_environment,
            started_at=started_at,
            ended_at=ended_at,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            parameters=parameters,
            error_message=error_message,
            traceback_summary=traceback_summary,
            metadata=metadata,
        )
        return {"ok": True, **result}


def transition_operation_status(
    operation_id: str,
    status: str,
    error_message: str | None = None,
    traceback_summary: str | None = None,
    operator_user_id: str | None = None,
) -> dict[str, Any]:
    """Transition an operation status through the canonical API.

    Parameters
    ----------
    operation_id : str
        Existing operation identifier.
    status : str
        Target lifecycle status.
    error_message : str, optional
        Failure message.
    traceback_summary : str, optional
        Traceback summary.
    operator_user_id : str, optional
        User performing the transition.

    Returns
    -------
    dict
        RPC result dictionary with keys 'ok' and 'operation_id'.
    """
    with MFDatabase(resolve_database_path()) as db:
        op_id = db.transition_operation_status(
            operation_id=operation_id,
            status=status,
            error_message=error_message,
            traceback_summary=traceback_summary,
            operator_user_id=operator_user_id,
        )
        return {"ok": True, "operation_id": op_id}


def register_sample(
    sample_id: str,
    uuid: str | None = None,
    description: str = "",
    details: str = "",
    num_of_probes: int | None = None,
    solvent_phase: str | None = None,
    sample_condition_id: str | None = None,
    entity_assembly_id: str | None = None,
    project_id: str | None = None,
    measured_by_user_id: str | None = None,
    measured_by_device_id: str | None = None,
    measured_at: str | None = None,
) -> dict[str, Any]:
    """Register or update a sample in the FLR domain layer."""
    with MFDatabase(resolve_database_path()) as db:
        db.add_sample(
            sample_id=sample_id,
            uuid=uuid,
            description=description,
            details=details,
            num_of_probes=num_of_probes,
            solvent_phase=solvent_phase,
            sample_condition_id=sample_condition_id,
            entity_assembly_id=entity_assembly_id,
            project_id=project_id,
            measured_by_user_id=measured_by_user_id,
            measured_by_device_id=measured_by_device_id,
            measured_at=measured_at,
        )
        return {"ok": True, "sample_id": sample_id}


def get_sample(sample_id: str) -> dict[str, Any]:
    """Retrieve a sample by identifier."""
    with MFDatabase(resolve_database_path()) as db:
        return {"sample": db.get_sample(sample_id)}


def list_samples() -> dict[str, Any]:
    """List samples."""
    with MFDatabase(resolve_database_path()) as db:
        return {"samples": [dict(row) for row in db.list_samples()]}


def register_experiment(
    experiment_id: str,
    type_id: int | None = None,
    sample_id: str | None = None,
    project_id: str | None = None,
    measured_by_user_id: str | None = None,
    measured_by_device_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    status: str | None = None,
    details: str = "",
    setup_definition_id: str | None = None,
) -> dict[str, Any]:
    """Register or update an experiment in the FLR domain layer."""
    with MFDatabase(resolve_database_path()) as db:
        db.add_experiment(
            experiment_id=experiment_id,
            type_id=type_id,
            sample_id=sample_id,
            project_id=project_id,
            measured_by_user_id=measured_by_user_id,
            measured_by_device_id=measured_by_device_id,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            details=details,
            setup_definition_id=setup_definition_id,
        )
        return {"ok": True, "experiment_id": experiment_id}


def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Retrieve an experiment by identifier."""
    with MFDatabase(resolve_database_path()) as db:
        return {"experiment": db.get_experiment(experiment_id)}


def list_experiments(
    sample_id: str | None = None,
    project_id: str | None = None,
    type_id: int | None = None,
) -> dict[str, Any]:
    """List experiments with optional filters."""
    with MFDatabase(resolve_database_path()) as db:
        return {"experiments": [dict(row) for row in db.get_experiments(sample_id, project_id, type_id)]}


def get_operation(operation_id: str) -> dict[str, Any]:
    """Retrieve an operation by its identifier.

    Parameters
    ----------
    operation_id : str
        Unique operation identifier.

    Returns
    -------
    dict
        RPC result containing the operation dictionary under key 'operation'.
    """
    with MFDatabase(resolve_database_path()) as db:
        op = db.get_operation(operation_id)
        return {"operation": op}


def list_operations(
    operation_type: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """List operations with optional filtering.

    Parameters
    ----------
    operation_type : str, optional
        Filter results by operation type.
    experiment_id : str, optional
        Filter results by experiment ID.

    Returns
    -------
    dict
        RPC result containing list of operations under key 'operations'.
    """
    with MFDatabase(resolve_database_path()) as db:
        ops = db.list_operations(
            operation_type=operation_type,
            experiment_id=experiment_id,
        )
        return {"operations": ops}


def record_operation_link(
    operation_id: str,
    artifact_id: str,
    direction: str,
    role: str | None = None,
    ordinal: int = 0,
    checksum_snapshot: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Link an operation and an artifact.

    Parameters
    ----------
    operation_id : str
        Operation identifier.
    artifact_id : str
        Artifact identifier.
    direction : str
        Direction of data flow ('input' or 'output').
    role : str, optional
        Data role (e.g. 'raw_data', 'fret_fit').
    ordinal : int, default=0
        Input/output order/ordinal.
    checksum_snapshot : str, optional
        Snapshot of the checksum at the time of linking.
    metadata : dict, optional
        Link-specific metadata.

    Returns
    -------
    dict
        RPC result dictionary with key 'ok'.
    """
    with MFDatabase(resolve_database_path()) as db:
        db.record_operation_link(
            operation_id=operation_id,
            artifact_id=artifact_id,
            direction=direction,
            role=role,
            ordinal=ordinal,
            checksum_snapshot=checksum_snapshot,
            metadata=metadata,
        )
        return {"ok": True}


def graph_upstream(
    node_type: str,
    node_id: str,
    max_depth: int = 100,
) -> dict[str, Any]:
    """Traverse upstream canonical graph edges.

    Parameters
    ----------
    node_type : str
        Starting node type.
    node_id : str
        Starting node identifier.
    max_depth : int, default=100
        Maximum traversal depth.

    Returns
    -------
    dict
        RPC result containing a list of canonical edge dictionaries.
    """
    with MFDatabase(resolve_database_path()) as db:
        from mfdb.provenance.graph import traverse_canonical_graph as traverse
        edges = traverse(
            db.conn,
            node_type,
            node_id,
            direction="upstream",
            max_depth=max_depth,
            canonical=True,
        )
        return {"edges": edges}


def graph_downstream(
    node_type: str,
    node_id: str,
    max_depth: int = 100,
) -> dict[str, Any]:
    """Traverse downstream canonical graph edges.

    Parameters
    ----------
    node_type : str
        Starting node type.
    node_id : str
        Starting node identifier.
    max_depth : int, default=100
        Maximum traversal depth.

    Returns
    -------
    dict
        RPC result containing a list of canonical edge dictionaries.
    """
    with MFDatabase(resolve_database_path()) as db:
        from mfdb.provenance.graph import traverse_canonical_graph as traverse
        edges = traverse(
            db.conn,
            node_type,
            node_id,
            direction="downstream",
            max_depth=max_depth,
            canonical=True,
        )
        return {"edges": edges}


def export_graph(
    seed_node_type: str,
    seed_node_id: str,
) -> dict[str, Any]:
    """Export a provenance graph as JSON-serializable nodes and edges.

    Parameters
    ----------
    seed_node_type : str
        Seed node type.
    seed_node_id : str
        Seed node identifier.

    Returns
    -------
    dict
        RPC result containing ``nodes`` and ``edges`` dictionaries.
    """
    with MFDatabase(resolve_database_path()) as db:
        graph = db.export_provenance_graph(seed_node_type, seed_node_id)
        return {"graph": graph}


def traverse_canonical_graph(
    start_node_type: str,
    start_node_id: str,
    direction: str = "upstream",
    max_depth: int = 100,
) -> dict[str, Any]:
    """Traverse the canonical graph recursively, cycle-safe.

    Parameters
    ----------
    start_node_type : str
        Starting node type (e.g. 'artifact', 'operation').
    start_node_id : str
        Starting node identifier.
    direction : str, default='upstream'
        Direction of traversal ('upstream' or 'downstream').
    max_depth : int, default=100
        Maximum search depth.

    Returns
    -------
    dict
        RPC result containing a list of edges under key 'edges'.
    """
    with MFDatabase(resolve_database_path()) as db:
        from mfdb.provenance.graph import traverse_canonical_graph as traverse
        edges = traverse(
            db.conn,
            start_node_type,
            start_node_id,
            direction,
            max_depth,
        )
        return {"edges": edges}




def record_parameter(
    parameter_uuid: str,
    operation_id: str,
    name: str,
    value: float | None = None,
    standard_error: float | None = None,
    confidence_interval_low: float | None = None,
    confidence_interval_high: float | None = None,
    initial_value: float | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    bounds_on: bool = False,
    units: str | None = None,
    parameter_type: str = "free",
    expression: str | None = None,
    prior: dict[str, Any] | None = None,
    mapping: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record or update a semantic parameter.

    Parameters
    ----------
    parameter_uuid : str
        Unique UUID for the parameter.
    operation_id : str
        Operation identifier linking this parameter.
    name : str
        Parameter name.
    value : float, optional
        Current parameter value.
    standard_error : float, optional
        Standard error of the fit value.
    confidence_interval_low : float, optional
        Lower bound of confidence interval.
    confidence_interval_high : float, optional
        Upper bound of confidence interval.
    initial_value : float, optional
        Initial value before fitting.
    lower_bound : float, optional
        Lower physical constraint bound.
    upper_bound : float, optional
        Upper physical constraint bound.
    bounds_on : bool, default=False
        Whether constraints are active.
    units : str, optional
        Units of measurement.
    parameter_type : str, default='free'
        Parameter type (e.g. 'free', 'fixed', 'calibrated').
    expression : str, optional
        Algebraic relation expression.
    prior : dict, optional
        Prior distribution dictionary.
    mapping : dict, optional
        Parameter mapping details.
    metadata : dict, optional
        User metadata dict.

    Returns
    -------
    dict
        RPC result dictionary with keys 'ok' and 'parameter_uuid'.
    """
    with MFDatabase(resolve_database_path()) as db:
        p_uuid = db.record_parameter(
            parameter_uuid=parameter_uuid,
            operation_id=operation_id,
            name=name,
            value=value,
            standard_error=standard_error,
            confidence_interval_low=confidence_interval_low,
            confidence_interval_high=confidence_interval_high,
            initial_value=initial_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bounds_on=bounds_on,
            units=units,
            parameter_type=parameter_type,
            expression=expression,
            prior=prior,
            mapping=mapping,
            metadata=metadata,
        )
        return {"ok": True, "parameter_uuid": p_uuid}


def get_parameter(parameter_uuid: str) -> dict[str, Any]:
    """Retrieve a parameter by its UUID.

    Parameters
    ----------
    parameter_uuid : str
        Unique parameter UUID.

    Returns
    -------
    dict
        RPC result containing parameter dictionary under key 'parameter'.
    """
    with MFDatabase(resolve_database_path()) as db:
        param = db.get_parameter(parameter_uuid)
        return {"parameter": param}


def list_parameters(
    operation_id: str | None = None,
    parameter_type: str | None = None,
) -> dict[str, Any]:
    """List parameters with optional filters.

    Parameters
    ----------
    operation_id : str, optional
        Filter by operation identifier.
    parameter_type : str, optional
        Filter by parameter vocabulary value.

    Returns
    -------
    dict
        RPC result containing a list of parameters under key 'parameters'.
    """
    with MFDatabase(resolve_database_path()) as db:
        parameters = db.list_parameters(
            operation_id=operation_id,
            parameter_type=parameter_type,
        )
        return {"parameters": parameters}


def save_chinet_session(
    session_payload: dict[str, Any],
    operation_id: str,
    experiment_id: str | None = None,
    fit_refs: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    store_node_artifacts: bool = True,
) -> dict[str, Any]:
    """Save a canonical chinet session schema payload to MFDB.

    Parameters
    ----------
    session_payload : dict
        Canonical ``chinet.session.v1`` payload.
    operation_id : str
        Operation identifier.
    experiment_id : str or None, optional
        Experiment identifier.
    fit_refs : list of dict or None, optional
        Fit references.
    parameters : list of dict or None, optional
        Parameter payloads.
    store_node_artifacts : bool, default=True
        Whether to write node artifacts.

    Returns
    -------
    dict
        RPC result containing storage summary.
    """
    from mfdb.adapters.chinet import store_chinet_session
    from chinet.schema import session_from_schema

    with MFDatabase(resolve_database_path()) as db:
        session = session_from_schema(session_payload)
        result = store_chinet_session(
            db,
            session,
            operation_id=operation_id,
            experiment_id=experiment_id,
            fit_refs=fit_refs,
            parameters=parameters,
            store_node_artifacts=store_node_artifacts,
        )
        return {"ok": True, **result}


def get_chinet_session(artifact_id: str) -> dict[str, Any]:
    """Load a chinet session artifact from MFDB.

    Parameters
    ----------
    artifact_id : str
        Chinet session artifact identifier.

    Returns
    -------
    dict
        RPC result containing artifact and session schema.
    """
    from mfdb.adapters.chinet import load_chinet_session

    with MFDatabase(resolve_database_path()) as db:
        session = load_chinet_session(db, artifact_id)
        artifact = db.get_artifact(artifact_id)
        return {"ok": True, "artifact": artifact, "session": session.to_schema()}


def list_chinet_sessions(experiment_id: str | None = None) -> dict[str, Any]:
    """List chinet session artifacts in MFDB.

    Parameters
    ----------
    experiment_id : str or None, optional
        Experiment filter.

    Returns
    -------
    dict
        RPC result containing artifact rows.
    """
    with MFDatabase(resolve_database_path()) as db:
        artifacts = db.list_artifacts(artifact_type="chinet_session", experiment_id=experiment_id)
        return {"ok": True, "artifacts": artifacts}


def restore_chinet_session(artifact_id: str) -> dict[str, Any]:
    """Restore a chinet session artifact from MFDB.

    Parameters
    ----------
    artifact_id : str
        Chinet session artifact identifier.

    Returns
    -------
    dict
        RPC result containing session schema and summary.
    """
    return get_chinet_session(artifact_id)


def save_setup(
    setup_id: str,
    name: str,
    version: int = 1,
    instrument_id: str | None = None,
    description: str | None = None,
    configuration: dict[str, Any] | None = None,
    detectors: dict[str, Any] | None = None,
    timing_calibration: dict[str, Any] | None = None,
    irf_definition: dict[str, Any] | None = None,
    dark_count: dict[str, Any] | None = None,
    timing_resolution: dict[str, Any] | None = None,
    burst_defaults: dict[str, Any] | None = None,
    fcs_calibration: dict[str, Any] | None = None,
    created_by_user_id: str | None = None,
    is_public: bool | int | None = None,
) -> dict[str, Any]:
    """Save or update an instrument/calibration setup configuration.

    Parameters
    ----------
    setup_id : str
        Unique setup identifier.
    name : str
        Display name of setup.
    version : int, default=1
        Setup configuration version.
    instrument_id : str, optional
        Associated instrument identifier.
    description : str, optional
        Description of the setup configuration.
    configuration : dict, optional
        Configuration settings dict.
    detectors : dict, optional
        Detector settings.
    timing_calibration : dict, optional
        Timing calibration coefficients.
    irf_definition : dict, optional
        IRF parameters.
    dark_count : dict, optional
        Dark count rates.
    timing_resolution : dict, optional
        Timing resolutions.
    burst_defaults : dict, optional
        Default burst parameters.
    fcs_calibration : dict, optional
        FCS calibration configuration.
    created_by_user_id : str, optional
        User who created this setup. NULL for shared/builtin setups.
    is_public : bool or int, optional
        GUI visibility flag. 1 (default) = visible to all, 0 = owner-only.

    Returns
    -------
    dict
        RPC result dictionary with key 'ok'.
    """
    with MFDatabase(resolve_database_path()) as db:
        db.save_setup(
            setup_id=setup_id,
            name=name,
            version=version,
            instrument_id=instrument_id,
            description=description,
            configuration=configuration,
            detectors=detectors,
            timing_calibration=timing_calibration,
            irf_definition=irf_definition,
            dark_count=dark_count,
            timing_resolution=timing_resolution,
            burst_defaults=burst_defaults,
            fcs_calibration=fcs_calibration,
            created_by_user_id=created_by_user_id,
            is_public=is_public,
        )
        return {"ok": True}


def get_setup(setup_id: str) -> dict[str, Any]:
    """Retrieve a setup configuration by its identifier.

    Parameters
    ----------
    setup_id : str
        Unique setup identifier.

    Returns
    -------
    dict
        RPC result containing setup dictionary under key 'setup'.
    """
    with MFDatabase(resolve_database_path()) as db:
        setup = db.get_setup(setup_id)
        return {"setup": setup}


def list_setups() -> dict[str, Any]:
    """List setup snapshots.

    Returns
    -------
    dict
        RPC result containing a list of setups under key 'setups'.
    """
    with MFDatabase(resolve_database_path()) as db:
        setups = db.list_setups()
        return {"setups": setups}


def add_setup_calibration(
    setup_id: str,
    channel_name: str,
    g_factor: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    g_factor_channels: list[int] | None = None,
    g_factor_calibration_id: str | None = None,
    calibrated_at: str | None = None,
    method: str | None = "manual",
    created_by_user_id: str | None = None,
) -> dict[str, Any]:
    """Append a calibration snapshot for one detector channel.

    Parameters
    ----------
    setup_id : str
        Setup identifier.
    channel_name : str
        Detector channel name.
    g_factor : float or None, optional
        G-factor value.
    l1 : float or None, optional
        Leakage parameter l1.
    l2 : float or None, optional
        Leakage parameter l2.
    g_factor_channels : list of int or None, optional
        Channel indices used for G-factor calculation.
    g_factor_calibration_id : str or None, optional
        Reference to the MFDB calibration artifact.
    calibrated_at : str or None, optional
        ISO-8601 timestamp. Defaults to current UTC time.
    method : str or None, optional
        Calibration method (e.g. ``manual``, ``migrated``).
    created_by_user_id : str or None, optional
        User creating this snapshot.

    Returns
    -------
    dict
        RPC result with the inserted snapshot row under key 'snapshot'.
    """
    with MFDatabase(resolve_database_path()) as db:
        snapshot = db.add_setup_calibration(
            setup_id=setup_id,
            channel_name=channel_name,
            g_factor=g_factor,
            l1=l1,
            l2=l2,
            g_factor_channels=g_factor_channels,
            g_factor_calibration_id=g_factor_calibration_id,
            calibrated_at=calibrated_at,
            method=method,
            created_by_user_id=created_by_user_id,
        )
        return {"ok": True, "snapshot": snapshot}


def list_setup_calibration_dates(setup_id: str) -> dict[str, Any]:
    """Return distinct calibration timestamps for a setup, newest first.

    Parameters
    ----------
    setup_id : str
        Setup identifier.

    Returns
    -------
    dict
        RPC result with a list of timestamps under key 'dates'.
    """
    with MFDatabase(resolve_database_path()) as db:
        dates = db.list_setup_calibration_dates(setup_id)
        return {"dates": dates}


def get_setup_calibration(
    setup_id: str,
    calibrated_at: str | None = None,
) -> dict[str, Any]:
    """Return calibration snapshots for a setup.

    Parameters
    ----------
    setup_id : str
        Setup identifier.
    calibrated_at : str or None, optional
        ISO-8601 timestamp. If None, returns the latest snapshot per channel.

    Returns
    -------
    dict
        RPC result with list of calibration rows under key 'calibration'.
    """
    with MFDatabase(resolve_database_path()) as db:
        calibration = db.get_setup_calibration(setup_id, calibrated_at=calibrated_at)
        return {"calibration": calibration}


def list_audit_logs(
    action: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """List audit log records with optional filtering.

    Parameters
    ----------
    action : str, optional
        Filter by action type (e.g. 'create', 'update').
    target_type : str, optional
        Filter by target entity type ('artifact', 'operation').
    target_id : str, optional
        Filter by target entity ID.
    limit : int, default=100
        Maximum logs to return.

    Returns
    -------
    dict
        RPC result containing a list of log dicts under key 'logs'.
    """
    with MFDatabase(resolve_database_path()) as db:
        logs = db.get_audit_logs(
            action=action,
            target_type=target_type,
            target_id=target_id,
            limit=limit,
        )
        return {"logs": logs}


def create_branch(
    branch_uuid: str | None = None,
    name: str | None = None,
    parent_branch_uuid: str | None = None,
    head_operation_id: str | None = None,
    created_by_user_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a branch pointer in the MFDB provenance graph."""
    with MFDatabase(resolve_database_path()) as db:
        uuid_val = db.create_branch(
            branch_uuid=branch_uuid,
            name=name,
            parent_branch_uuid=parent_branch_uuid,
            head_operation_id=head_operation_id,
            created_by_user_id=created_by_user_id,
            description=description,
        )
        return {"ok": True, "branch_uuid": uuid_val}


def fork_branch(
    source_branch_uuid: str,
    name: str,
    branch_uuid: str | None = None,
    head_operation_id: str | None = None,
    created_by_user_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a parallel branch from an existing branch.

    Parameters
    ----------
    source_branch_uuid : str
        Existing branch used as the parent branch.
    name : str
        Name for the created branch.
    branch_uuid : str, optional
        Explicit branch UUID.
    head_operation_id : str, optional
        Operation used as the branch point. Defaults to the source branch head.
    created_by_user_id : str, optional
        User creating the branch.
    description : str, optional
        Branch description.

    Returns
    -------
    dict
        RPC result containing the new branch UUID.
    """
    with MFDatabase(resolve_database_path()) as db:
        uuid_val = db.fork_branch(
            source_branch_uuid=source_branch_uuid,
            name=name,
            branch_uuid=branch_uuid,
            head_operation_id=head_operation_id,
            created_by_user_id=created_by_user_id,
            description=description,
        )
        return {"ok": True, "branch_uuid": uuid_val}


def get_branch(branch_uuid_or_name: str) -> dict[str, Any]:
    """Return one branch by UUID or name."""
    with MFDatabase(resolve_database_path()) as db:
        branch = db.get_branch(branch_uuid_or_name)
        return {"branch": branch}


def list_branches() -> dict[str, Any]:
    """Return all non-deleted branches."""
    with MFDatabase(resolve_database_path()) as db:
        branches = db.list_branches()
        return {"branches": branches}


def update_branch_head(branch_uuid: str, head_operation_id: str | None) -> dict[str, Any]:
    """Move a branch head to an operation or clear it."""
    with MFDatabase(resolve_database_path()) as db:
        db.update_branch_head(branch_uuid, head_operation_id)
        return {"ok": True}


def jump_user_to_operation(
    user_id: str,
    operation_id: str,
    branch_name: str | None = None,
    branch_uuid: str | None = None,
    parent_branch_uuid: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create and activate a user branch at a historical operation.

    Parameters
    ----------
    user_id : str
        User whose active branch changes.
    operation_id : str
        Operation used as the new branch head.
    branch_name : str, optional
        Name for the new branch.
    branch_uuid : str, optional
        Explicit branch UUID.
    parent_branch_uuid : str, optional
        Parent branch for provenance.
    description : str, optional
        Branch description.

    Returns
    -------
    dict
        RPC result containing the created active branch.
    """
    with MFDatabase(resolve_database_path()) as db:
        branch = db.jump_user_to_operation(
            user_id=user_id,
            operation_id=operation_id,
            branch_name=branch_name,
            branch_uuid=branch_uuid,
            parent_branch_uuid=parent_branch_uuid,
            description=description,
        )
        return {"ok": True, "branch": branch}


def delete_branch(branch_uuid: str) -> dict[str, Any]:
    """Soft-delete a branch when it is not protected or active."""
    with MFDatabase(resolve_database_path()) as db:
        db.delete_branch(branch_uuid)
        return {"ok": True}


def set_user_active_branch(user_id: str, branch_uuid: str) -> dict[str, Any]:
    """Set the active branch for a user."""
    with MFDatabase(resolve_database_path()) as db:
        db.set_user_active_branch(user_id, branch_uuid)
        return {"ok": True}


def get_user_active_branch(user_id: str) -> dict[str, Any]:
    """Return the active branch for a user."""
    with MFDatabase(resolve_database_path()) as db:
        branch = db.get_user_active_branch(user_id)
        return {"branch": branch}
