"""Project archiver for MFDB — decomposes projects into artifacts, parameters, and edges.

This module provides :func:`archive_project_to_mfdb` which stores a ChiSurf
project with full provenance: source files in the object store, derived datasets
as artifacts, fit results with chinet sessions, parameters in the parameter table,
and dependency edges.  It follows the patterns established by
:mod:`chisurf.core.mfdb.chinet_adapter` and
:class:`chisurf.core.experiments.core.reader.ExperimentReader`.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

from chisurf.core.mfdb.models import (
    RELATIONSHIP_TYPES,
    validate_vocabulary,
)
from chisurf.core.mfdb.repository import MFDatabase, _json_dumps, _json_loads

# Re-use constants from chinet_adapter when available
try:
    from chisurf.core.mfdb.chinet_adapter import (
        FIT_STATE_SCHEMA,
    )
except ImportError:
    FIT_STATE_SCHEMA = "chisurf.fit_state.v1"


def _encode_curve_arrays(ds: dict[str, Any]) -> dict[str, Any]:
    """Encode x/y/ex/ey arrays from a dataset payload to canonical JSON form.

    Parameters
    ----------
    ds : dict
        Dataset payload with ``x``, ``y``, ``ex``, ``ey`` keys.  Values may
        be NumPy arrays, plain lists, or dicts produced by
        :func:`~chisurf.core.experiments.core.serialize.encode_array`.

    Returns
    -------
    dict
        Same keys but with canonical ``{dtype, shape, data}`` dicts.
    """
    import numpy as np

    from chisurf.core.experiments.core.serialize import encode_array

    out: dict[str, Any] = {}
    for key in ("x", "y", "ex", "ey"):
        arr = ds.get(key)
        if arr is None:
            continue
        if isinstance(arr, dict) and "dtype" in arr and "data" in arr:
            out[key] = arr  # already encoded
        else:
            out[key] = encode_array(np.asarray(arr))
    return out


def archive_project_to_mfdb(
    db: MFDatabase,
    project_payload: dict[str, Any],
    version_id: str,
    project_id: str,
    version_number: int,
    parent_version_id: str | None = None,
    branch_uuid: str | None = None,
    user_id: str = "user_default",
    notes: str = "",
    project_name: str = "",
) -> dict[str, Any]:
    """Archive a ChiSurf project to MFDB with full artifact decomposition.

    Within a single transaction this function:

    1. Creates a ``project`` operation record.
    2. For every dataset stores the source file (when available) in the
       content-addressed object store and registers a ``raw_measurement``
       artifact, then stores the derived data (arrays) as a
       ``processed_data`` artifact.
    3. Delegates each fit to
       :func:`~chisurf.core.mfdb.chinet_adapter.archive_fit_to_mfdb`
       which stores chinet sessions, node artifacts, fit-result artifacts,
       parameters, and dependency edges.
    4. Creates ``project_contains`` edges linking the project operation to
       every output artifact.
    5. Creates ``derived_from`` edges linking each dataset back to its
       source file.
    6. If *parent_version_id* is given, creates a ``supersedes`` edge
       forming a DAG across versions.
    7. Updates the branch head when *branch_uuid* is provided.

    Parameters
    ----------
    db : MFDatabase
        Active database connection.
    project_payload : dict
        Full project state (output of ``get_project_payload()``).
    version_id : str
        Unique identifier for this version (``ver_…``).
    project_id : str
        Stable project identifier (``proj_…``).
    version_number : int
        Version number within the branch scope.
    parent_version_id : str or None, optional
        Version ID this one supersedes.
    branch_uuid : str or None, optional
        Branch to record this version on.
    user_id : str, default='user_default'
        Owner user id.
    notes : str, default=''
        Free-text notes.

    Returns
    -------
    dict
        Summary with keys: ``operation_id``, ``dataset_artifacts``,
        ``fit_artifacts``, ``chinet_artifacts``, ``parameter_count``,
        ``edge_count``, ``object_count``.
    """
    validate_vocabulary("project", [
        "measurement_import", "validation", "burst_selection",
        "filtering", "fcs_correlation", "microtime_histogram",
        "tcspc_fitting", "model_fitting", "ndxplorer_selection",
        "ndxplorer_clustering", "project_snapshot",
        "project_restore", "archive_export",
        "tcspc_histogram_computation", "pda_histogram_computation",
        "pch_histogram_computation", "fcs_correlation_load",
        "tcspc_curve_load",
        "import", "burst_filtering", "gmm_fitting", "analysis",
        "fitting", "project_archive", "local_fit", "global_fit",
        "project", "analysis_run",
    ], "operation_type")

    meta = project_payload.get("meta") or {}
    project_name = meta.get("name", "") or project_payload.get("name", "") or project_name
    datasets = project_payload.get("datasets") or {}
    fits = project_payload.get("fits") or []
    project_format_version = project_payload.get("project_format_version", 4)
    chisurf_version = meta.get("chisurf_version", "")
    description = meta.get("description", "")
    created = meta.get("created", "")

    dataset_artifacts: list[str] = []
    fit_artifacts: list[str] = []
    chinet_artifacts: list[str] = []
    object_count = 0

    with db.transaction():
        # -- 1. Project operation -------------------------------------------
        db.record_operation(
            operation_id=version_id,
            operation_type="project",
            operator_user_id=user_id,
            status="succeeded",
            acl_owner_user_id=user_id,
            metadata={
                "project_id": project_id,
                "version_number": version_number,
                "branch_uuid": branch_uuid,
                "parent_version_id": parent_version_id,
                "project_name": project_name,
                "model_name": project_name,
                "notes": notes,
                "fit_count": len(fits),
                "dataset_count": len(datasets),
                "chisurf_version": chisurf_version,
                "project_format_version": project_format_version,
                "description": description,
                "created": created,
                "ui_state": project_payload.get("ui", {}),
                "experiments": project_payload.get("experiments", {}),
            },
        )

        # -- 2. Per-dataset: source file + derived data --------------------
        ds_id_map: dict[str, str] = {}  # ds_id -> artifact_id
        for ds_idx, (ds_id, ds_payload) in enumerate(datasets.items()):
            if not isinstance(ds_payload, dict):
                continue
            filename = ds_payload.get("filename", "")
            source_object_uuid: str | None = None
            source_artifact_id: str | None = None

            # 2a. Source file → object store
            if filename and os.path.isfile(filename):
                source_ref = db.put_object(
                    path=filename,
                    filename=str(filename),
                )
                source_object_uuid = source_ref["object_uuid"]
                source_artifact_id = f"src_{source_object_uuid[:12]}"
                db.register_artifact(
                    artifact_id=source_artifact_id,
                    artifact_kind="raw_measurement",
                    storage_mode="local_file",
                    file_path=str(filename),
                    object_uuid=source_object_uuid,
                    size_bytes=source_ref.get("size_bytes"),
                    validation_status="unvalidated",
                )
                db.record_operation_link(
                    operation_id=version_id,
                    artifact_id=source_artifact_id,
                    direction="input",
                    role="source_file",
                    ordinal=ds_idx,
                )
                object_count += 1

            # 2b. Derived data → object store
            encoded = _encode_curve_arrays(ds_payload)
            reader_info = ds_payload.get("data_reader") or {}
            derived_payload = {
                "schema_version": "1.0",
                "data_type": ds_payload.get("experiment_name", "processed_data"),
                "created_by": f"{reader_info.get('module', '')}.{reader_info.get('class', '')}",
                "source_object_uuids": [source_object_uuid] if source_object_uuid else [],
                "curves": [{
                    "name": ds_payload.get("name", ds_id),
                    **encoded,
                }],
                "reader_settings": reader_info.get("state", {}),
            }
            derived_bytes = _json_dumps(derived_payload).encode("utf-8")
            derived_ref = db.put_object(
                data=derived_bytes,
                filename=f"{ds_id}.json",
                mime_type="application/json",
            )
            derived_object_uuid = derived_ref["object_uuid"]
            dataset_artifact_id = f"dataset:{version_id}:{ds_id}"
            db.register_artifact(
                artifact_id=dataset_artifact_id,
                artifact_kind="processed_data",
                storage_mode="embedded_json",
                object_uuid=derived_object_uuid,
                data_format="json",
                size_bytes=len(derived_bytes),
                metadata={
                    "ds_id": ds_id,
                    "name": ds_payload.get("name", ""),
                    "filename": filename,
                    "experiment_name": ds_payload.get("experiment_name", ""),
                    "data_reader_module": reader_info.get("module", ""),
                    "data_reader_class": reader_info.get("class", ""),
                },
            )
            db.record_operation_link(
                operation_id=version_id,
                artifact_id=dataset_artifact_id,
                direction="output",
                role="dataset",
                ordinal=ds_idx,
                metadata={"ds_id": ds_id},
            )
            ds_id_map[ds_id] = dataset_artifact_id
            dataset_artifacts.append(dataset_artifact_id)
            object_count += 1

            # 2c. project_contains edge
            db.add_edge(
                source_node_type="operation",
                source_node_id=version_id,
                target_node_type="artifact",
                target_node_id=dataset_artifact_id,
                relationship_type="project_contains",
                operation_id=version_id,
                metadata={"ds_id": ds_id},
            )

            # 2d. derived_from edge
            if source_artifact_id:
                db.add_edge(
                    source_node_type="artifact",
                    source_node_id=dataset_artifact_id,
                    target_node_type="artifact",
                    target_node_id=source_artifact_id,
                    relationship_type="derived_from",
                    operation_id=version_id,
                    metadata={"source_object_uuid": source_object_uuid},
                )

        # -- 3. Per-fit: decompose into chinet session, nodes, parameters -
        for fit_record in fits:
            if not isinstance(fit_record, dict):
                continue
            fit_uid = fit_record.get("id", "")
            for lf_idx, local_fit in enumerate(fit_record.get("local_fits", [])):
                if not isinstance(local_fit, dict):
                    continue
                dataset_ref_id = local_fit.get("dataset_id", "")
                linked_dataset = ds_id_map.get(dataset_ref_id)

                lf_id = local_fit.get("id") or str(lf_idx)
                fit_op_id = f"fit_{version_id}:{fit_uid}:{lf_id}"
                # Ensure the fit operation record exists before any
                # archiving attempt — the chinet path and the fallback
                # both need it for FK constraints on operation links.
                db.record_operation(
                    operation_id=fit_op_id,
                    operation_type="local_fit",
                    operator_user_id=user_id,
                    status="succeeded",
                    metadata={
                        "fit_id": fit_uid,
                        "project_id": project_id,
                        "version_id": version_id,
                    },
                )
                fit_state_payload = local_fit.get("fit_state") or {}

                # -- 3a. Build chinet session from serialized payload --
                from chisurf.core.mfdb.chinet_adapter import (
                    CHINET_NODE_ARTIFACT,
                    CHINET_SESSION_ARTIFACT,
                    _artifact_id,
                    _session_from_fit_state_payload,
                    _store_fit_state_links,
                    _store_fit_state_parameters,
                    _validate_fit_state_payload,
                )
                from chisurf.core.mfdb.chinet_adapter import (
                    _json_dumps as chinet_json,
                )

                chinet_session = None
                try:
                    _validate_fit_state_payload(fit_state_payload, None)
                    chinet_session = _session_from_fit_state_payload(
                        fit_state_payload
                    )
                except Exception:
                    logger.warning(
                        "Failed to build chinet session for fit %s",
                        fit_uid,
                        exc_info=True,
                    )

                # -- 3b. Store chinet session artifact --
                session_artifact_id = None
                if chinet_session is not None:
                    try:
                        schema = chinet_session.to_schema()
                        session_artifact_id = _artifact_id(
                            CHINET_SESSION_ARTIFACT, chinet_session.oid
                        )
                        db.register_artifact(
                            artifact_id=session_artifact_id,
                            artifact_kind=CHINET_SESSION_ARTIFACT,
                            storage_mode="embedded_json",
                            data_format="json",
                            data_json=chinet_json(schema),
                            metadata={
                                "schema_name": schema["schema_name"],
                                "schema_version": schema["schema_version"],
                                "session_id": chinet_session.oid,
                                "source": "chinet",
                                "operation_id": fit_op_id,
                            },
                        )
                        db.record_operation_link(
                            operation_id=fit_op_id,
                            artifact_id=session_artifact_id,
                            direction="output",
                            role="chinet_session",
                        )
                        chinet_artifacts.append(session_artifact_id)

                        # Store node artifacts
                        for node_key, node in chinet_session.nodes.items():
                            node_doc = {
                                "node_id": node.oid,
                                "session_id": chinet_session.oid,
                                "name": str(node_key or node.name),
                                "callback": node.callback,
                                "callback_type": (
                                    node.callback_type_string or ""
                                ),
                                "valid": bool(node.node_valid_),
                                "ports": list(node.ports.keys()),
                            }
                            node_art_id = _artifact_id(
                                CHINET_NODE_ARTIFACT, node.oid
                            )
                            db.register_artifact(
                                artifact_id=node_art_id,
                                artifact_kind=CHINET_NODE_ARTIFACT,
                                storage_mode="embedded_json",
                                data_format="json",
                                data_json=chinet_json(node_doc),
                                metadata={
                                    "schema_name": "chinet.node.v1",
                                    "session_id": chinet_session.oid,
                                    "node_id": node.oid,
                                    "source": "chinet",
                                    "operation_id": fit_op_id,
                                },
                            )
                            db.record_operation_link(
                                operation_id=fit_op_id,
                                artifact_id=node_art_id,
                                direction="output",
                                role="chinet_node",
                            )
                            db.add_edge(
                                source_node_type="artifact",
                                source_node_id=session_artifact_id,
                                target_node_type="artifact",
                                target_node_id=node_art_id,
                                relationship_type="contains",
                                operation_id=fit_op_id,
                            )
                            chinet_artifacts.append(node_art_id)
                    except Exception:
                        logger.warning(
                            "Failed to store chinet artifacts for fit %s",
                            fit_uid,
                            exc_info=True,
                        )
                        session_artifact_id = None

                # -- 3c. Store fit result artifact --
                full_fit_data = {
                    "id": fit_record.get("id", ""),
                    "name": fit_record.get("name", ""),
                    "model_name": fit_record.get("model_name", ""),
                    "fit_range": fit_record.get("fit_range"),
                    "plot_state": fit_record.get("plot_state"),
                    "local_fit": local_fit,
                    "fit_state": fit_state_payload,
                }
                fit_artifact_id = f"fit_result:{version_id}:{fit_uid}:{lf_id}"
                db.register_artifact(
                    artifact_id=fit_artifact_id,
                    artifact_kind="fit_result",
                    storage_mode="embedded_json",
                    data_format="json",
                    data_json=chinet_json(full_fit_data),
                    metadata={
                        "schema_name": FIT_STATE_SCHEMA,
                        "fit_id": fit_uid,
                        "fit_name": fit_record.get("name", ""),
                        "model_name": fit_record.get("model_name", ""),
                        "fit_range": fit_record.get("fit_range"),
                        "plot_state": fit_record.get("plot_state"),
                        "local_fits_count": len(fit_record.get("local_fits", [])),
                        "model_module": fit_state_payload.get(
                            "model_module"
                        ),
                        "model_class": fit_state_payload.get(
                            "model_class"
                        ),
                        "source": "chisurf.core.project.fit_state",
                    },
                )
                db.record_operation_link(
                    operation_id=fit_op_id,
                    artifact_id=fit_artifact_id,
                    direction="output",
                    role="fit_state",
                )
                fit_artifacts.append(fit_artifact_id)

                # -- 3d. Link dataset as input --
                if linked_dataset:
                    db.record_operation_link(
                        operation_id=fit_op_id,
                        artifact_id=linked_dataset,
                        direction="input",
                        role="input_data",
                    )

                # -- 3e. Store parameters and link edges --
                try:
                    _store_fit_state_parameters(
                        db, fit_op_id, fit_state_payload, None
                    )
                    _store_fit_state_links(db, fit_op_id, fit_state_payload)
                except Exception:
                    logger.warning(
                        "Failed to store parameters/links for fit %s",
                        fit_uid,
                        exc_info=True,
                    )

                # -- 3f. project_contains edges --
                db.add_edge(
                    source_node_type="operation",
                    source_node_id=version_id,
                    target_node_type="artifact",
                    target_node_id=fit_artifact_id,
                    relationship_type="project_contains",
                    operation_id=version_id,
                )
                if session_artifact_id:
                    db.add_edge(
                        source_node_type="operation",
                        source_node_id=version_id,
                        target_node_type="artifact",
                        target_node_id=session_artifact_id,
                        relationship_type="project_contains",
                        operation_id=version_id,
                    )

        # -- 4. Version lineage edge ----------------------------------------
        if parent_version_id:
            validate_vocabulary("supersedes", RELATIONSHIP_TYPES, "relationship_type")
            db.add_edge(
                source_node_type="operation",
                source_node_id=version_id,
                target_node_type="operation",
                target_node_id=parent_version_id,
                relationship_type="supersedes",
                operation_id=version_id,
                metadata={
                    "branch_uuid": branch_uuid,
                    "version_number": version_number,
                },
            )

        # -- 5. Update branch head ------------------------------------------
        if branch_uuid:
            try:
                db.update_branch_head(branch_uuid, version_id)
            except Exception:
                logger.warning(
                    "Failed to update branch head for %s",
                    branch_uuid,
                    exc_info=True,
                )

    # Count actual parameters and edges created for this version
    escaped = version_id.replace("_", "\\_")
    param_count = db.conn.execute(
        "SELECT COUNT(*) FROM mfdb_parameter WHERE operation_id = ? OR operation_id LIKE ? ESCAPE '\\'",
        (version_id, f"fit\\_{escaped}:%"),
    ).fetchone()[0]
    edge_count = db.conn.execute(
        "SELECT COUNT(*) FROM mfdb_edge WHERE operation_id = ?",
        (version_id,),
    ).fetchone()[0]

    return {
        "operation_id": version_id,
        "dataset_artifacts": dataset_artifacts,
        "fit_artifacts": fit_artifacts,
        "chinet_artifacts": chinet_artifacts,
        "parameter_count": param_count,
        "edge_count": edge_count,
        "object_count": object_count,
    }


def _parse_ds_id(aid: str) -> str | None:
    if aid.startswith("dataset:"):
        parts = aid.split(":", 2)
        if len(parts) == 3 and parts[2]:
            return parts[2]
    return None


def restore_project_from_artifacts(
    db: MFDatabase,
    version_id: str,
) -> dict[str, Any] | None:
    """Restore a project from its MFDB artifacts.

    Queries all output artifacts for the given version and reconstructs
    the project payload.  Returns ``None`` if no artifacts are found
    (indicating a legacy archive that should be restored via the JSON
    blob instead).

    Parameters
    ----------
    db : MFDatabase
        Active database connection.
    version_id : str
        Version identifier (``ver_…``).

    Returns
    -------
    dict or None
        Reconstructed project payload with ``datasets``, ``fits``, and
        ``chinet_sessions`` keys, or ``None`` if no artifacts exist.
    """
    # Query for artifacts linked to the project operation
    artifacts = db.get_operation_artifacts(version_id, direction="output")
    if not artifacts:
        return None

    datasets: dict[str, Any] = {}
    fits: list[dict[str, Any]] = []
    chinet_sessions: list[dict[str, Any]] = []
    fit_operation_ids: set[str] = set()
    project_metadata: dict[str, Any] = {}

    # Query for artifacts linked to fit operations scoped to this version
    # Use LIKE to match: fit_{version_id}:* with parameterized query and ESCAPE
    escaped_vid = version_id.replace("_", "\\_")
    fit_artifacts = db.conn.execute(
        """SELECT mfdb_artifact.*, mfdb_operation_artifact.role,
                 mfdb_operation_artifact.direction, mfdb_operation.operation_id
            FROM mfdb_operation_artifact
            JOIN mfdb_artifact ON mfdb_artifact.artifact_id = mfdb_operation_artifact.artifact_id
            JOIN mfdb_operation ON mfdb_operation.operation_id = mfdb_operation_artifact.operation_id
            WHERE mfdb_operation.operation_type = 'local_fit'
              AND mfdb_operation.operation_id LIKE ? ESCAPE '\\'
              AND mfdb_operation_artifact.deleted_at IS NULL
              AND mfdb_artifact.deleted_at IS NULL
              AND mfdb_operation_artifact.direction = 'output'""",
        (f"fit\\_{escaped_vid}:%",),
    ).fetchall()
    artifacts.extend(fit_artifacts)

    for art in artifacts:
        art = dict(art) if not isinstance(art, dict) else art
        kind = art.get("artifact_kind", "")
        data_json = art.get("data_json")

        if data_json and isinstance(data_json, str):
            data = _json_loads(data_json)
        elif art.get("object_uuid"):
            try:
                blob = db.get_object(art["object_uuid"])
                data = _json_loads(blob.decode("utf-8"))
            except Exception:
                logger.warning(
                    "Failed to read object %s from store",
                    art.get("object_uuid"),
                    exc_info=True,
                )
                data = None
        else:
            data = None

        if data is None:
            continue

        role = art.get("role", "")

        if kind == "processed_data" and role == "dataset":
            meta = art.get("metadata_json") or {}
            if isinstance(meta, str):
                meta = _json_loads(meta) or {}
            ds_id = (
                meta.get("ds_id")
                or meta.get("dataset_uid")
                or _parse_ds_id(art.get("artifact_id", ""))
                or art.get("artifact_id")
                or str(uuid.uuid4())
            )
            datasets[ds_id] = data

        elif kind == "fit_result":
            fit_data_is_full_record = isinstance(data, dict) and "local_fit" in data
            if fit_data_is_full_record:
                fit_record = {
                    "id": data.get("id", ""),
                    "name": data.get("name", ""),
                    "model_name": data.get("model_name", ""),
                    "fit_range": data.get("fit_range"),
                    "plot_state": data.get("plot_state"),
                    "local_fits": [data.get("local_fit", {})],
                }
                fits.append(fit_record)
            else:
                meta = art.get("metadata_json") or {}
                if isinstance(meta, str):
                    meta = _json_loads(meta) or {}
                fits.append({
                    "id": meta.get("fit_id", str(uuid.uuid4())),
                    "name": meta.get("fit_name", "Restored Fit"),
                    "model_name": meta.get("model_name", ""),
                    "fit_range": meta.get("fit_range"),
                    "plot_state": meta.get("plot_state"),
                    "local_fits": [data] if data else [],
                })
            fit_op_id = art.get("operation_id", "")
            if fit_op_id:
                fit_operation_ids.add(fit_op_id)

        elif kind == "chinet_session":
            chinet_sessions.append(data)

    if not datasets and not fits:
        return None

    # Query the project operation metadata for ui_state and experiments
    project_op = db.conn.execute(
        "SELECT metadata_json FROM mfdb_operation WHERE operation_id = ?",
        (version_id,),
    ).fetchone()
    if project_op:
        project_op = dict(project_op)
        op_meta = _json_loads(project_op.get("metadata_json")) or {}
        project_metadata = op_meta

    # Query parameters for each fit operation
    all_parameters: dict[str, list[dict[str, Any]]] = {}
    for fit_op_id in fit_operation_ids:
        rows = db.conn.execute(
            "SELECT * FROM mfdb_parameter WHERE operation_id = ?",
            (fit_op_id,),
        ).fetchall()
        all_parameters[fit_op_id] = [dict(r) for r in rows]

    # Query dependency edges scoped to this version's operations
    dependency_edges: list[dict[str, Any]] = []
    if fit_operation_ids:
        placeholders = ",".join("?" for _ in fit_operation_ids)
        rows = db.conn.execute(
            f"SELECT * FROM mfdb_edge WHERE relationship_type = 'parameter_depends_on' "
            f"AND operation_id IN ({placeholders})",
            list(fit_operation_ids),
        ).fetchall()
        dependency_edges = [dict(r) for r in rows]

    return {
        "datasets": datasets,
        "fits": fits,
        "chinet_sessions": chinet_sessions,
        "parameters": all_parameters,
        "dependency_edges": dependency_edges,
        "ui_state": project_metadata.get("ui_state", {}),
        "experiments": project_metadata.get("experiments", {}),
    }
