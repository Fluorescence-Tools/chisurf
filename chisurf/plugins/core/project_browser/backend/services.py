from __future__ import annotations

import json
import uuid
import base64
from pathlib import Path
from typing import Any

from chisurf import logging
from chisurf.core.mfdb.database_resolver import resolve_database_path
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.auth import (
    PERM_READ,
    PERM_MANAGE,
    filter_readable,
    principal_from_rpc_auth,
    require_authenticated,
    require_access,
)
from chisurf.core.mfdb.repository import _json_loads, _utc_now
from chisurf.core.project.archive import ProjectArchive, PROJECT_JSON, DATA_DIR
from chisurf.server.services import INVALID_INPUT, NOT_FOUND, OPERATION_FAILED, service_error

MFDB_EXPORT_JSON = "mfdb_export.json"


def register_services(dispatcher: Any) -> None:
    for name, handler in {
        "project_browser.list": list_projects_handler,
        "project_browser.save": save_project_handler,
        "project_browser.restore": restore_project_handler,
        "project_browser.export_csp": export_csp_handler,
        "project_browser.import_preview": import_preview_handler,
        "project_browser.import_csp": import_csp_handler,
        "project_browser.delete_version": delete_version_handler,
        "project_browser.create_branch": create_branch_handler,
        "project_browser.list_branches": list_branches_handler,
        "project_browser.version_graph": get_version_graph_handler,
        "project_browser.artifacts": list_project_artifacts_handler,
        "project_browser.parameters": list_project_parameters_handler,
    }.items():
        dispatcher.register(name, lambda params, _handler=handler: _handler(**params))


def _get_db() -> MFDatabase:
    return MFDatabase(resolve_database_path())


def _get_conn(db: MFDatabase) -> Any:
    return db.conn


def _require_auth(auth: dict[str, Any] | None = None) -> tuple[Any, Any, MFDatabase]:
    db = _get_db()
    conn = _get_conn(db)
    principal = principal_from_rpc_auth(conn, auth)
    require_authenticated(principal)
    return principal, conn, db


def _get_project_id_from_meta(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    return meta.get("project_id")


def _get_version_number_from_meta(meta: dict[str, Any] | None) -> int | None:
    if not meta:
        return None
    return meta.get("version_number")


def _get_parent_version_id_from_meta(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    return meta.get("parent_version_id")


def _counts_from_payload(fit_structure: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(fit_structure, dict):
        return 0, 0
    datasets = fit_structure.get("datasets", {})
    fits = fit_structure.get("fits", [])
    dataset_count = len(datasets) if hasattr(datasets, "__len__") else 0
    fit_count = len(fits) if hasattr(fits, "__len__") else 0
    return fit_count, dataset_count


def _decode_project_operation(row: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        row = dict(row)
    meta = _json_loads(row.get("metadata_json")) or {}
    fit_count, dataset_count = _counts_from_payload(meta.get("fit_structure"))
    return {
        "version_id": row.get("operation_id") or row.get("analysis_id", ""),
        "project_name": meta.get("model_name", meta.get("project_name", "")),
        "project_id": meta.get("project_id", row.get("operation_id", "")),
        "version_number": meta.get("version_number", 1),
        "parent_version_id": meta.get("parent_version_id"),
        "owner_user_id": row.get("operator_user_id", ""),
        "status": row.get("status", ""),
        "notes": meta.get("notes", ""),
        "fit_count": meta.get("fit_count") or fit_count,
        "dataset_count": meta.get("dataset_count") or dataset_count,
        "created_at": row.get("created_at", ""),
        "updated_at": row.get("updated_at", ""),
        "metadata_json": row.get("metadata_json"),
    }


def _get_operation_mode(conn: Any, operation_id: str) -> int | None:
    row = conn.execute(
        "SELECT mode FROM mfdb_object_acl WHERE object_type = 'mfdb_operation' AND object_id = ? AND deleted_at IS NULL",
        (operation_id,),
    ).fetchone()
    if row:
        return row[0] if not isinstance(row, dict) else row["mode"]
    return None


def _visibility_from_mode(mode: int | None) -> str:
    if mode is None:
        return "private"
    other_bits = mode & 0o7
    if other_bits & PERM_READ:
        return "public"
    conn_check = None
    return "private"


def _get_project_visibility(conn: Any, operation_id: str) -> str:
    mode = _get_operation_mode(conn, operation_id)
    if mode is None:
        return "private"
    other_bits = mode & 0o7
    if other_bits & PERM_READ:
        return "public"
    entry_rows = conn.execute(
        "SELECT entry_id FROM mfdb_acl_entry WHERE object_type = 'mfdb_operation' AND object_id = ? AND deleted_at IS NULL LIMIT 1",
        (operation_id,),
    ).fetchall()
    if entry_rows:
        return "shared"
    return "private"


def list_projects_handler(
    auth: dict[str, Any] | None = None,
    show_public: bool = True,
    search: str | None = None,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        projects_raw: list[dict[str, Any]] = []
        rows = conn.execute(
            """SELECT operation_id, operation_type, experiment_id,
                      operator_user_id, software_package, software_module, software_version,
                      settings_json, status, metadata_json, created_at, updated_at
               FROM mfdb_operation
               WHERE operation_type = 'project' AND deleted_at IS NULL
               ORDER BY created_at DESC""",
        ).fetchall()
        for row in rows:
            d = dict(row)
            meta = _json_loads(d.get("metadata_json")) or {}
            d["project_id"] = meta.get("project_id", d["operation_id"])
            d["version_number"] = meta.get("version_number", 1)
            d["parent_version_id"] = meta.get("parent_version_id")
            d["model_name"] = meta.get("model_name", meta.get("project_name", ""))
            fit_count, dataset_count = _counts_from_payload(meta.get("fit_structure"))
            d["fit_count"] = meta.get("fit_count") or fit_count
            d["dataset_count"] = meta.get("dataset_count") or dataset_count
            d["notes"] = meta.get("notes", "")
            d["analysis_id"] = d["operation_id"]
            projects_raw.append(d)

        if not principal.is_admin:
            projects_raw = filter_readable(
                conn, principal, "mfdb_operation",
                projects_raw, id_key="operation_id",
            )
            if not show_public:
                filtered = []
                for p in projects_raw:
                    mode = _get_operation_mode(conn, p["operation_id"])
                    if mode is not None:
                        other_bits = mode & 0o7
                        if other_bits & PERM_READ:
                            continue
                    filtered.append(p)
                projects_raw = filtered

        if search:
            sl = search.lower()
            projects_raw = [
                p for p in projects_raw
                if sl in (p.get("model_name") or "").lower()
                or sl in (p.get("project_id") or "").lower()
                or sl in (p.get("notes") or "").lower()
                or sl in (p.get("operator_user_id") or "").lower()
            ]

        grouped: dict[str, dict[str, Any]] = {}
        for p in projects_raw:
            pid = p["project_id"]
            if pid not in grouped:
                grouped[pid] = {
                    "project_id": pid,
                    "project_name": p.get("model_name", ""),
                    "owner_user_id": p.get("operator_user_id", ""),
                    "versions": [],
                    "version_count": 0,
                    "latest_version_number": 0,
                    "latest_version_id": "",
                    "visibility": "private",
                    "created_at": p.get("created_at", ""),
                    "updated_at": p.get("created_at", ""),
                }
            g = grouped[pid]
            decoded = _decode_project_operation(p)
            g["versions"].append(decoded)
            g["version_count"] += 1
            vn = decoded.get("version_number", 1)
            if vn > g["latest_version_number"]:
                g["latest_version_number"] = vn
                g["latest_version_id"] = decoded["version_id"]
                g["updated_at"] = decoded.get("created_at", "")
                if not g.get("project_name"):
                    g["project_name"] = decoded.get("project_name", "")
            if decoded.get("created_at", "") > g.get("created_at", ""):
                g["created_at"] = decoded["created_at"]

        for g in grouped.values():
            g["versions"].sort(key=lambda v: v.get("version_number", 1), reverse=True)
            latest = g["versions"][0] if g["versions"] else {}
            g["visibility"] = _get_project_visibility(conn, latest.get("version_id", ""))
            g["owner_user_id"] = latest.get("owner_user_id", "")

        return {
            "ok": True,
            "projects": sorted(grouped.values(), key=lambda x: x.get("updated_at", ""), reverse=True),
        }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def save_project_handler(
    auth: dict[str, Any] | None = None,
    project_name: str | None = None,
    project_payload: dict[str, Any] | None = None,
    project_id: str | None = None,
    parent_version_id: str | None = None,
    notes: str | None = None,
    visibility: str = "private",
    fit_count: int = 0,
    dataset_count: int = 0,
    branch_uuid: str | None = None,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        user_id = principal.user_id or "user_default"

        import uuid as _uuid
        if not project_id:
            project_id = f"proj_{_uuid.uuid4().hex[:12]}"
        version_id = f"ver_{_uuid.uuid4().hex[:12]}"

        # Ensure a default branch exists for this project
        if not branch_uuid:
            branch_row = conn.execute(
                "SELECT branch_uuid FROM mfdb_branch WHERE name = ? AND deleted_at IS NULL",
                (f"project_{project_id}",),
            ).fetchone()
            if branch_row:
                branch_uuid = branch_row[0] if isinstance(branch_row, dict) else branch_row.get("branch_uuid")
            else:
                branch_uuid = f"br_{_uuid.uuid4().hex[:12]}"
                conn.execute(
                    "INSERT OR IGNORE INTO mfdb_branch (branch_uuid, name, description, created_by_user_id) VALUES (?, ?, ?, ?)",
                    (branch_uuid, f"project_{project_id}", f"Default branch for {project_id}", user_id),
                )
                conn.commit()

        # Compute branch-scoped version number
        version_number = 1
        if parent_version_id:
            parent_row = conn.execute(
                "SELECT metadata_json FROM mfdb_operation WHERE operation_id = ? AND deleted_at IS NULL",
                (parent_version_id,),
            ).fetchone()
            if parent_row:
                parent_meta = _json_loads(parent_row[0] if not isinstance(parent_row, dict) else parent_row.get("metadata_json")) or {}
                parent_vn = parent_meta.get("version_number", 0)
                version_number = (parent_vn or 0) + 1
        else:
            max_vn_row = conn.execute(
                """SELECT MAX(json_extract(metadata_json, '$.version_number')) AS max_vn
                   FROM mfdb_operation
                   WHERE operation_type = 'project' AND deleted_at IS NULL
                     AND json_extract(metadata_json, '$.project_id') = ?
                     AND json_extract(metadata_json, '$.branch_uuid') = ?""",
                (project_id, branch_uuid),
            ).fetchone()
            if max_vn_row:
                max_vn = max_vn_row[0] if not isinstance(max_vn_row, dict) else max_vn_row.get("max_vn")
                if max_vn:
                    version_number = int(max_vn) + 1

        # Use the new project_archiver for full artifact decomposition
        from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb

        result = archive_project_to_mfdb(
            db=db,
            project_payload=project_payload or {},
            version_id=version_id,
            project_id=project_id,
            version_number=version_number,
            parent_version_id=parent_version_id,
            branch_uuid=branch_uuid,
            user_id=user_id,
            notes=notes or "",
        )

        if visibility == "public":
            import chisurf.core.mfdb.auth as authmod
            authmod.chmod(conn, principal, "mfdb_operation", version_id, 0o704)

        db.add_audit_log(
            action="archive",
            target_type="project",
            target_id=version_id,
            details={
                "project_id": project_id,
                "project_name": project_name,
                "version_number": version_number,
                "branch_uuid": branch_uuid,
                "user_id": user_id,
                "artifact_count": len(result.get("dataset_artifacts", [])) + len(result.get("fit_artifacts", [])),
            },
        )

        return {
            "ok": True,
            "project_id": project_id,
            "version_id": version_id,
            "version_number": version_number,
            "branch_uuid": branch_uuid,
            "project_name": project_name or "",
            "visibility": visibility,
            "artifact_count": len(result.get("dataset_artifacts", [])) + len(result.get("fit_artifacts", [])),
            "parameter_count": result.get("parameter_count", 0),
            "edge_count": result.get("edge_count", 0),
        }
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def restore_project_handler(
    auth: dict[str, Any] | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        if version_id:
            require_access(conn, principal, "mfdb_operation", version_id, PERM_READ)
            run = db.get_analysis_run_full(version_id)
            if not run:
                return service_error(f"Project version not found: {version_id}", error_code=NOT_FOUND)
            meta = run.get("metadata") or _json_loads(run.get("metadata_json")) or {}

            # Try artifact-based restore first (new-style)
            from chisurf.core.mfdb.project_archiver import restore_project_from_artifacts
            artifact_payload = restore_project_from_artifacts(db, version_id)

            if artifact_payload:
                # Reconstruct from artifacts
                payload = {
                    "project_format_version": 5,
                    "meta": {
                        "name": meta.get("project_name", ""),
                        "description": meta.get("notes", ""),
                        "project_id": meta.get("project_id", version_id),
                        "branch_uuid": meta.get("branch_uuid"),
                    },
                    "datasets": artifact_payload.get("datasets", {}),
                    "fits": artifact_payload.get("fits", []),
                    "experiments": {},
                    "ui": {},
                    "extra": {},
                }
            else:
                # Fallback: legacy JSON blob, or empty project payload
                payload = meta.get("fit_structure") or {
                    "project_format_version": 5,
                    "meta": {
                        "name": meta.get("project_name", ""),
                        "description": meta.get("notes", ""),
                        "project_id": meta.get("project_id", version_id),
                        "branch_uuid": meta.get("branch_uuid"),
                    },
                    "datasets": {},
                    "fits": [],
                    "experiments": {},
                    "ui": {},
                    "extra": {},
                }

            db.add_audit_log(
                action="restore",
                target_type="project",
                target_id=version_id,
                details={"project_id": meta.get("project_id"), "project_name": meta.get("model_name")},
            )
            return {
                "ok": True,
                "version_id": version_id,
                "project_id": meta.get("project_id", version_id),
                "version_number": meta.get("version_number", 1),
                "branch_uuid": meta.get("branch_uuid"),
                "project_name": meta.get("model_name", meta.get("project_name", "")),
                "project_payload": payload,
                "visibility": _get_project_visibility(conn, version_id),
            }
        else:
            rows = conn.execute(
                """SELECT operation_id FROM mfdb_operation
                   WHERE operation_type = 'project' AND deleted_at IS NULL
                   ORDER BY created_at DESC LIMIT 1""",
            ).fetchall()
            if not rows:
                return service_error("No project found", error_code=NOT_FOUND)
            latest_id = rows[0][0] if not isinstance(rows[0], dict) else rows[0].get("operation_id")
            require_access(conn, principal, "mfdb_operation", latest_id, PERM_READ)
            run = db.get_analysis_run_full(latest_id)
            if not run:
                return service_error(f"Project not found: {latest_id}", error_code=NOT_FOUND)
            meta = run.get("metadata") or _json_loads(run.get("metadata_json")) or {}

            # Try artifact-based restore first
            from chisurf.core.mfdb.project_archiver import restore_project_from_artifacts
            artifact_payload = restore_project_from_artifacts(db, latest_id)

            if artifact_payload:
                payload = {
                    "project_format_version": 5,
                    "meta": {
                        "name": meta.get("project_name", ""),
                        "project_id": meta.get("project_id", latest_id),
                        "branch_uuid": meta.get("branch_uuid"),
                    },
                    "datasets": artifact_payload.get("datasets", {}),
                    "fits": artifact_payload.get("fits", []),
                    "experiments": {},
                    "ui": {},
                    "extra": {},
                }
            else:
                payload = meta.get("fit_structure") or {
                    "project_format_version": 5,
                    "meta": {
                        "name": meta.get("project_name", ""),
                        "project_id": meta.get("project_id", latest_id),
                        "branch_uuid": meta.get("branch_uuid"),
                    },
                    "datasets": {},
                    "fits": [],
                    "experiments": {},
                    "ui": {},
                    "extra": {},
                }

            return {
                "ok": True,
                "version_id": latest_id,
                "project_id": meta.get("project_id", latest_id),
                "version_number": meta.get("version_number", 1),
                "branch_uuid": meta.get("branch_uuid"),
                "project_name": meta.get("model_name", meta.get("project_name", "")),
                "project_payload": payload,
                "visibility": _get_project_visibility(conn, latest_id),
            }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def _gather_project_dependencies(conn: Any, db: MFDatabase, version_id: str) -> dict[str, Any]:
    run = db.get_analysis_run_full(version_id)
    if not run:
        return {"operations": [], "artifacts": [], "objects": [], "parameters": [], "provenance_edges": []}

    meta = run.get("metadata") or _json_loads(run.get("metadata_json")) or {}
    operations = [{
        "operation_id": run.get("analysis_id", version_id),
        "operation_type": run.get("analysis_type", "project"),
        "experiment_id": run.get("experiment_id"),
        "operator_user_id": run.get("operator_user_id"),
        "settings": _json_loads(run.get("settings_json")) if isinstance(run.get("settings_json"), str) else run.get("settings_json"),
        "metadata": meta,
        "status": run.get("convergence_status", run.get("status")),
        "created_at": run.get("created_at"),
    }]

    artifacts = []
    objects = []
    parameters = run.get("parameters") or []
    provenance_edges = run.get("provenance_edges") or []

    seen_object_uuids: set[str] = set()
    for art in run.get("input_processed_data") or []:
        artifacts.append(art)
        ou = art.get("object_uuid")
        if ou and ou not in seen_object_uuids:
            seen_object_uuids.add(ou)
            obj = db.get_object_info(ou) if hasattr(db, "get_object_info") else None
            if obj:
                try:
                    blob = db.get_object(ou)
                except (KeyError, Exception):
                    blob = None
                obj_entry = {
                    "object_uuid": ou,
                    "content_md5": obj.get("content_md5"),
                    "size_bytes": obj.get("size_bytes"),
                    "filename": obj.get("original_filename", obj.get("filename", "")),
                    "mime_type": obj.get("mime_type"),
                }
                if blob:
                    obj_entry["data_base64"] = base64.b64encode(blob).decode("ascii")
                objects.append(obj_entry)

    for art in run.get("processed_data") or []:
        artifacts.append(art)
        ou = art.get("object_uuid")
        if ou and ou not in seen_object_uuids:
            seen_object_uuids.add(ou)
            obj = db.get_object_info(ou) if hasattr(db, "get_object_info") else None
            if obj:
                try:
                    blob = db.get_object(ou)
                except (KeyError, Exception):
                    blob = None
                obj_entry = {
                    "object_uuid": ou,
                    "content_md5": obj.get("content_md5"),
                    "size_bytes": obj.get("size_bytes"),
                    "filename": obj.get("original_filename", obj.get("filename", "")),
                    "mime_type": obj.get("mime_type"),
                }
                if blob:
                    obj_entry["data_base64"] = base64.b64encode(blob).decode("ascii")
                objects.append(obj_entry)

    return {
        "operations": operations,
        "artifacts": artifacts,
        "objects": objects,
        "parameters": parameters,
        "provenance_edges": provenance_edges,
    }


def export_csp_handler(
    auth: dict[str, Any] | None = None,
    version_id: str | None = None,
    target_path: str | None = None,
    project_name: str | None = None,
    fit_count: int = 0,
    dataset_count: int = 0,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        require_access(conn, principal, "mfdb_operation", version_id, PERM_READ)
        run = db.get_analysis_run_full(version_id)
        if not run:
            return service_error(f"Project version not found: {version_id}", error_code=NOT_FOUND)

        meta = run.get("metadata") or _json_loads(run.get("metadata_json")) or {}
        payload = meta.get("fit_structure")
        if not payload:
            return service_error("No project payload found in this version", error_code=NOT_FOUND)

        deps = _gather_project_dependencies(conn, db, version_id)
        export_meta = {
            "format_version": 1,
            "exported_at": _utc_now(),
            "exported_by_user_id": principal.user_id,
            "origin": {
                "project_id": meta.get("project_id", version_id),
                "version_id": version_id,
                "version_number": meta.get("version_number", 1),
                "parent_version_id": meta.get("parent_version_id"),
                "owner_user_id": run.get("operator_user_id"),
            },
            "dependencies": deps,
            "id_remap": {},
        }

        archive = ProjectArchive()
        archive.write_text(PROJECT_JSON, json.dumps(payload, indent=2, sort_keys=True))
        archive.write_text(MFDB_EXPORT_JSON, json.dumps(export_meta, indent=2, sort_keys=True))

        file_refs = _collect_file_refs(payload)
        project_root = Path(meta.get("project_root", "."))
        for ref in file_refs:
            src = project_root / ref if not Path(ref).is_absolute() else Path(ref)
            if src.exists():
                archive.write_file(str(src), str(src))

        if target_path:
            archive.save(target_path)
            return {
                "ok": True,
                "target_path": target_path,
                "project_id": meta.get("project_id", version_id),
                "version_id": version_id,
            }
        return {
            "ok": True,
            "archive_bytes": base64.b64encode(archive.to_bytes()).decode("ascii"),
            "project_id": meta.get("project_id", version_id),
            "version_id": version_id,
        }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def _collect_file_refs(payload: dict[str, Any] | None) -> list[str]:
    refs: list[str] = []
    if not payload:
        return refs
    for ds in (payload.get("datasets") or {}).values():
        path = (ds.get("parameters") or {}).get("path", "")
        if path:
            refs.append(path)
    return refs


def _find_collisions(conn: Any, export_meta: dict[str, Any]) -> dict[str, list[str]]:
    collisions: dict[str, list[str]] = {
        "operations": [],
        "artifacts": [],
        "objects": [],
        "parameters": [],
    }
    deps = export_meta.get("dependencies", {})
    for op in deps.get("operations", []):
        oid = op.get("operation_id", "")
        if oid:
            row = conn.execute("SELECT 1 FROM mfdb_operation WHERE operation_id = ? AND deleted_at IS NULL", (oid,)).fetchone()
            if row:
                collisions["operations"].append(oid)
    for art in deps.get("artifacts", []):
        aid = art.get("artifact_id", art.get("processed_data_id", art.get("raw_data_id", "")))
        if aid:
            row = conn.execute("SELECT 1 FROM mfdb_artifact WHERE artifact_id = ? AND deleted_at IS NULL", (aid,)).fetchone()
            if row:
                collisions["artifacts"].append(aid)
    for obj in deps.get("objects", []):
        ou = obj.get("object_uuid", "")
        if ou:
            row = conn.execute("SELECT 1 FROM mfdb_object WHERE object_uuid = ? AND deleted_at IS NULL", (ou,)).fetchone()
            if row:
                collisions["objects"].append(ou)
    for param in deps.get("parameters", []):
        pu = param.get("parameter_uuid", param.get("parameter_id", ""))
        if pu:
            row = conn.execute("SELECT 1 FROM mfdb_parameter WHERE parameter_uuid = ? AND deleted_at IS NULL", (pu,)).fetchone()
            if row:
                collisions["parameters"].append(pu)
    return collisions


def _generate_id_remap(collisions: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    remap: dict[str, dict[str, str]] = {
        "operations": {},
        "artifacts": {},
        "objects": {},
        "parameters": {},
    }
    for category, ids in collisions.items():
        for old_id in ids:
            remap[category][old_id] = f"{old_id[:4]}_{uuid.uuid4().hex[:12]}"
    return remap


def _apply_remap_to_export(export_meta: dict[str, Any], remap: dict[str, dict[str, str]]) -> dict[str, Any]:
    import copy
    meta = copy.deepcopy(export_meta)
    deps = meta.setdefault("dependencies", {})

    ops_map = remap.get("operations", {})
    for op in deps.get("operations", []):
        oid = op.get("operation_id", "")
        if oid in ops_map:
            op["operation_id"] = ops_map[oid]
            op["_original_operation_id"] = oid

    arts_map = remap.get("artifacts", {})
    for art in deps.get("artifacts", []):
        for key in ("artifact_id", "processed_data_id", "raw_data_id"):
            aid = art.get(key, "")
            if aid in arts_map:
                art[key] = arts_map[aid]
                art["_original_id"] = aid
                break

    objs_map = remap.get("objects", {})
    for obj in deps.get("objects", []):
        ou = obj.get("object_uuid", "")
        if ou in objs_map:
            obj["object_uuid"] = objs_map[ou]
            obj["_original_object_uuid"] = ou

    params_map = remap.get("parameters", {})
    for param in deps.get("parameters", []):
        pu = param.get("parameter_uuid", param.get("parameter_id", ""))
        if pu in params_map:
            param["parameter_uuid"] = params_map[pu]
            param["_original_parameter_uuid"] = pu

    for edge in deps.get("provenance_edges", []):
        for key in ("source_node_id", "target_node_id", "processing_id"):
            val = edge.get(key, "")
            for category in ("operations", "artifacts", "parameters"):
                if val in remap.get(category, {}):
                    edge[key] = remap[category][val]
                    break

    return meta


def _populate_mfdb_from_export(
    conn: Any,
    db: MFDatabase,
    export_meta: dict[str, Any],
    importing_user_id: str,
) -> dict[str, Any]:
    deps = export_meta.get("dependencies", {})
    origin = export_meta.get("origin", {})

    prev_project_id: str | None = None
    prev_version_number: int | None = None
    project_id = origin.get("project_id", f"proj_{uuid.uuid4().hex[:12]}")
    version_number = origin.get("version_number", 1)

    parent_version_id = None
    with db.transaction():
        for op in deps.get("operations", []):
            oid = op.get("operation_id", "")
            op_type = op.get("operation_type", "project")
            meta = dict(op.get("metadata", {}))
            meta["project_id"] = project_id
            meta["_imported_from"] = origin.get("version_id", "")
            meta["_imported_original_project_id"] = origin.get("project_id", "")
            meta["_imported_at"] = _utc_now()
            meta["_imported_by_user_id"] = importing_user_id

            if op_type == "project":
                meta["version_number"] = version_number
                if parent_version_id:
                    meta["parent_version_id"] = parent_version_id
                meta["project_id"] = project_id
            else:
                meta["version_number"] = 1

            db.record_operation(
                operation_id=oid,
                operation_type=op_type,
                experiment_id=op.get("experiment_id"),
                settings=op.get("settings"),
                operator_user_id=importing_user_id,
                status=op.get("status", "succeeded"),
                acl_owner_user_id=importing_user_id,
                metadata=meta,
            )

            if op_type == "project":
                parent_version_id = oid

        for obj in deps.get("objects", []):
            ou = obj.get("object_uuid", "")
            data_b64 = obj.get("data_base64", "")
            if ou and data_b64:
                data = base64.b64decode(data_b64)
                obj_result = db.put_object(
                    object_uuid=ou,
                    data=data,
                    filename=obj.get("filename", ""),
                    mime_type=obj.get("mime_type", ""),
                    created_by_user_id=importing_user_id,
                )

        for art in deps.get("artifacts", []):
            aid = art.get("artifact_id") or art.get("processed_data_id") or art.get("raw_data_id") or ""
            if not aid:
                continue
            ou = art.get("object_uuid", "")
            metadata = art.get("metadata") or {}
            if art.get("product_summary"):
                metadata["product_summary"] = art["product_summary"]
            db.register_artifact(
                artifact_id=aid,
                artifact_kind=art.get("artifact_type", art.get("product_type", "derived_product")),
                storage_mode="object_store" if ou else art.get("storage_mode", "embedded_json"),
                mime_type=art.get("mime_type", ""),
                size_bytes=art.get("size_bytes"),
                object_uuid=ou if ou else None,
                metadata=metadata,
            )

        for param in deps.get("parameters", []):
            pu = param.get("parameter_uuid", param.get("parameter_id", ""))
            if not pu:
                continue
            try:
                db.add_analysis_parameter(
                    analysis_id=param.get("analysis_id", parent_version_id or ""),
                    name=param.get("name", ""),
                    value=param.get("value"),
                    standard_error=param.get("standard_error"),
                    parameter_uuid=pu,
                )
            except Exception:
                pass

        for edge in deps.get("provenance_edges", []):
            try:
                db.add_provenance_edge(
                    source_node_type=edge.get("source_node_type", ""),
                    source_node_id=edge.get("source_node_id", ""),
                    target_node_type=edge.get("target_node_type", ""),
                    target_node_id=edge.get("target_node_id", ""),
                    relationship_type=edge.get("relationship_type", ""),
                    processing_id=edge.get("processing_id", parent_version_id or ""),
                )
            except Exception:
                pass

        db.add_audit_log(
            action="import",
            target_type="project",
            target_id=parent_version_id or "",
            details={
                "project_id": project_id,
                "original_project_id": origin.get("project_id", ""),
                "original_version_id": origin.get("version_id", ""),
                "imported_by_user_id": importing_user_id,
            },
        )

    return {
        "project_id": project_id,
        "version_id": parent_version_id or "",
        "version_number": version_number,
    }


def import_preview_handler(
    auth: dict[str, Any] | None = None,
    archive_base64: str | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        if archive_base64:
            data = base64.b64decode(archive_base64)
        elif file_path:
            data = Path(file_path).read_bytes()
        else:
            return service_error("Either archive_base64 or file_path is required", error_code=INVALID_INPUT)

        archive = ProjectArchive.open_bytes(data)
        if not archive.has_entry(MFDB_EXPORT_JSON):
            return service_error("Not a valid MFDB export archive (missing mfdb_export.json)", error_code=INVALID_INPUT)

        export_text = archive.read_text(MFDB_EXPORT_JSON)
        export_meta = json.loads(export_text)

        export_meta["id_remap"] = {}
        collisions = _find_collisions(conn, export_meta)
        has_collisions = any(v for v in collisions.values())
        total_ops = len(export_meta.get("dependencies", {}).get("operations", []))
        total_arts = len(export_meta.get("dependencies", {}).get("artifacts", []))
        total_objs = len(export_meta.get("dependencies", {}).get("objects", []))

        origin = export_meta.get("origin", {})
        return {
            "ok": True,
            "has_collisions": has_collisions,
            "collisions": collisions,
            "origin": origin,
            "entity_counts": {
                "operations": total_ops,
                "artifacts": total_arts,
                "objects": total_objs,
            },
        }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def import_csp_handler(
    auth: dict[str, Any] | None = None,
    archive_base64: str | None = None,
    file_path: str | None = None,
    resolve_collisions: bool = False,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        user_id = principal.user_id or "user_default"

        if archive_base64:
            data = base64.b64decode(archive_base64)
        elif file_path:
            data = Path(file_path).read_bytes()
        else:
            return service_error("Either archive_base64 or file_path is required", error_code=INVALID_INPUT)

        archive = ProjectArchive.open_bytes(data)
        if not archive.has_entry(MFDB_EXPORT_JSON):
            return service_error("Not a valid MFDB export archive (missing mfdb_export.json)", error_code=INVALID_INPUT)

        export_text = archive.read_text(MFDB_EXPORT_JSON)
        export_meta = json.loads(export_text)

        collisions = _find_collisions(conn, export_meta)
        if any(v for v in collisions.values()):
            if not resolve_collisions:
                return service_error(
                    "Collisions detected. Set resolve_collisions=true to remap conflicting IDs.",
                    error_code=INVALID_INPUT,
                )
            remap = _generate_id_remap(collisions)
            export_meta = _apply_remap_to_export(export_meta, remap)
            export_meta["id_remap"] = {
                cat: mapping for cat, mapping in remap.items() if mapping
            }
        else:
            export_meta["id_remap"] = {}

        result = _populate_mfdb_from_export(conn, db, export_meta, user_id)

        return {
            "ok": True,
            **result,
            "collisions_resolved": collisions,
            "id_remap": export_meta.get("id_remap", {}),
        }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def delete_version_handler(
    auth: dict[str, Any] | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    try:
        principal, conn, db = _require_auth(auth)
        if not version_id:
            return service_error("version_id is required", error_code=INVALID_INPUT)
        require_access(conn, principal, "mfdb_operation", version_id, PERM_MANAGE)
        with db.transaction():
            conn.execute(
                "UPDATE mfdb_operation SET deleted_at = ? WHERE operation_id = ?",
                (_utc_now(), version_id),
            )
            conn.execute(
                "UPDATE mfdb_object_acl SET deleted_at = ? WHERE object_type = 'mfdb_operation' AND object_id = ?",
                (_utc_now(), version_id),
            )
            db.add_audit_log(
                action="delete",
                target_type="project_version",
                target_id=version_id,
                details={"user_id": principal.user_id},
            )
        return {"ok": True, "deleted_version_id": version_id}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


# ── Branch Management ──────────────────────────────────────────────────


def create_branch_handler(
    auth: dict[str, Any] | None = None,
    project_id: str | None = None,
    from_version_id: str | None = None,
    branch_name: str | None = None,
) -> dict[str, Any]:
    """Create a new branch from an existing version.

    Parameters
    ----------
    auth : dict
        Authentication context.
    project_id : str
        Project identifier.
    from_version_id : str
        Version ID to fork from (becomes the branch head).
    branch_name : str
        Human-readable branch name.

    Returns
    -------
    dict
        ``{ok, branch_uuid, branch_name, head_version_id}``.
    """
    try:
        principal, conn, db = _require_auth(auth)
        user_id = principal.user_id or "user_default"

        if not project_id or not from_version_id or not branch_name:
            return service_error("project_id, from_version_id, and branch_name are required", error_code=INVALID_INPUT)

        require_access(conn, principal, "mfdb_operation", from_version_id, PERM_READ)

        import uuid as _uuid
        branch_uuid = f"br_{_uuid.uuid4().hex[:12]}"

        with db.transaction():
            conn.execute(
                """INSERT INTO mfdb_branch (branch_uuid, name, description, head_operation_id, created_by_user_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (branch_uuid, branch_name, f"Forked from {from_version_id}", from_version_id, user_id),
            )
            db.add_audit_log(
                action="create_branch",
                target_type="branch",
                target_id=branch_uuid,
                details={
                    "project_id": project_id,
                    "from_version_id": from_version_id,
                    "branch_name": branch_name,
                    "user_id": user_id,
                },
            )

        return {
            "ok": True,
            "branch_uuid": branch_uuid,
            "branch_name": branch_name,
            "head_version_id": from_version_id,
        }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def list_branches_handler(
    auth: dict[str, Any] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """List all branches that contain versions for a project.

    Parameters
    ----------
    auth : dict
        Authentication context.
    project_id : str
        Project identifier.

    Returns
    -------
    dict
        ``{ok, branches: [{branch_uuid, name, head_version_id, version_count}]}``.
    """
    try:
        principal, conn, db = _require_auth(auth)

        if not project_id:
            return service_error("project_id is required", error_code=INVALID_INPUT)

        # Find all branches that have project operations
        rows = conn.execute(
            """SELECT DISTINCT
                   json_extract(m.metadata_json, '$.branch_uuid') AS branch_uuid,
                   json_extract(m.metadata_json, '$.project_name') AS project_name
               FROM mfdb_operation m
               WHERE m.operation_type = 'project'
                 AND m.deleted_at IS NULL
                 AND json_extract(m.metadata_json, '$.project_id') = ?""",
            (project_id,),
        ).fetchall()

        branches = []
        for row in rows:
            buuid = row["branch_uuid"] if isinstance(row, dict) else row[0]
            if not buuid:
                continue
            branch_row = conn.execute(
                "SELECT branch_uuid, name, head_operation_id FROM mfdb_branch WHERE branch_uuid = ?",
                (buuid,),
            ).fetchone()
            if not branch_row:
                continue

            # Count versions on this branch for this project
            vn_count = conn.execute(
                """SELECT COUNT(*) FROM mfdb_operation
                   WHERE operation_type = 'project' AND deleted_at IS NULL
                     AND json_extract(metadata_json, '$.project_id') = ?
                     AND json_extract(metadata_json, '$.branch_uuid') = ?""",
                (project_id, buuid),
            ).fetchone()
            count = vn_count[0] if vn_count else 0

            branches.append({
                "branch_uuid": branch_row["branch_uuid"] if isinstance(branch_row, dict) else branch_row[0],
                "name": branch_row["name"] if isinstance(branch_row, dict) else branch_row[1],
                "head_version_id": branch_row["head_operation_id"] if isinstance(branch_row, dict) else branch_row[2],
                "version_count": count,
            })

        return {"ok": True, "branches": branches}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def get_version_graph_handler(
    auth: dict[str, Any] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Get the full version DAG for a project.

    Parameters
    ----------
    auth : dict
        Authentication context.
    project_id : str
        Project identifier.

    Returns
    -------
    dict
        ``{ok, graph: {nodes: [...], edges: [...], roots: [...], leaves: [...]}}``.
    """
    try:
        principal, conn, db = _require_auth(auth)

        if not project_id:
            return service_error("project_id is required", error_code=INVALID_INPUT)

        # Get all versions for this project
        rows = conn.execute(
            """SELECT operation_id, metadata_json, created_at
               FROM mfdb_operation
               WHERE operation_type = 'project' AND deleted_at IS NULL
                 AND json_extract(metadata_json, '$.project_id') = ?
               ORDER BY created_at""",
            (project_id,),
        ).fetchall()

        nodes = []
        node_ids = set()
        for row in rows:
            op_id = row["operation_id"] if isinstance(row, dict) else row[0]
            meta_raw = row["metadata_json"] if isinstance(row, dict) else row[1]
            created = row["created_at"] if isinstance(row, dict) else row[2]
            meta = _json_loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})

            node_ids.add(op_id)
            nodes.append({
                "version_id": op_id,
                "version_number": meta.get("version_number", 0),
                "branch_uuid": meta.get("branch_uuid"),
                "project_name": meta.get("project_name", ""),
                "notes": meta.get("notes", ""),
                "fit_count": meta.get("fit_count", 0),
                "dataset_count": meta.get("dataset_count", 0),
                "created_at": created,
            })

        # Build edges from parent_version_id metadata
        edges = []
        for node in nodes:
            parent_id = None
            # Find parent from metadata
            for row in rows:
                op_id = row["operation_id"] if isinstance(row, dict) else row[0]
                if op_id == node["version_id"]:
                    meta_raw = row["metadata_json"] if isinstance(row, dict) else row[1]
                    meta = _json_loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                    parent_id = meta.get("parent_version_id")
                    break
            if parent_id and parent_id in node_ids:
                edges.append({
                    "source": node["version_id"],
                    "target": parent_id,
                    "relationship": "supersedes",
                })

        # Also query mfdb_edge for supersedes edges
        edge_rows = conn.execute(
            """SELECT source_node_id, target_node_id, metadata_json
               FROM mfdb_edge
               WHERE relationship_type = 'supersedes' AND deleted_at IS NULL
                 AND source_node_id IN ({})""".format(",".join("?" * len(node_ids))),
            list(node_ids),
        ).fetchall()
        existing_edge_keys = {(e["source"], e["target"]) for e in edges}
        for erow in edge_rows:
            src = erow["source_node_id"] if isinstance(erow, dict) else erow[0]
            tgt = erow["target_node_id"] if isinstance(erow, dict) else erow[1]
            if (src, tgt) not in existing_edge_keys:
                edges.append({"source": src, "target": tgt, "relationship": "supersedes"})
                existing_edge_keys.add((src, tgt))

        # Identify roots (no parent) and leaves (no children)
        child_ids = {e["target"] for e in edges}
        parent_ids = {e["source"] for e in edges}
        roots = [n for n in nodes if n["version_id"] not in child_ids]
        leaves = [n for n in nodes if n["version_id"] not in parent_ids]

        return {
            "ok": True,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "roots": [r["version_id"] for r in roots],
                "leaves": [l["version_id"] for l in leaves],
            },
        }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def list_project_artifacts_handler(
    auth: dict[str, Any] | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    """List all artifacts for a project version.

    Parameters
    ----------
    auth : dict
        Authentication context.
    version_id : str
        Version identifier.

    Returns
    -------
    dict
        ``{ok, artifacts: [...]}``.
    """
    try:
        principal, conn, db = _require_auth(auth)
        if not version_id:
            return service_error("version_id is required", error_code=INVALID_INPUT)
        require_access(conn, principal, "mfdb_operation", version_id, PERM_READ)

        artifacts = db.get_operation_artifacts(version_id)
        result = []
        for art in artifacts:
            result.append({
                "artifact_id": art.get("artifact_id"),
                "artifact_kind": art.get("artifact_kind"),
                "role": art.get("role"),
                "direction": art.get("direction"),
                "storage_mode": art.get("storage_mode"),
                "object_uuid": art.get("object_uuid"),
                "size_bytes": art.get("size_bytes"),
                "data_format": art.get("data_format"),
                "file_path": art.get("file_path"),
            })

        return {"ok": True, "artifacts": result}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def list_project_parameters_handler(
    auth: dict[str, Any] | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    """List all parameters for a project version (across all fits).

    Parameters
    ----------
    auth : dict
        Authentication context.
    version_id : str
        Version identifier.

    Returns
    -------
    dict
        ``{ok, parameters: [...]}``.
    """
    try:
        principal, conn, db = _require_auth(auth)
        if not version_id:
            return service_error("version_id is required", error_code=INVALID_INPUT)
        require_access(conn, principal, "mfdb_operation", version_id, PERM_READ)

        # Find all fit operations that belong to this version
        fit_ops = conn.execute(
            """SELECT operation_id FROM mfdb_operation
               WHERE operation_id LIKE ? AND deleted_at IS NULL""",
            (f"fit_{version_id}:%",),
        ).fetchall()

        parameters = []
        for frow in fit_ops:
            op_id = frow["operation_id"] if isinstance(frow, dict) else frow[0]
            params = conn.execute(
                """SELECT parameter_uuid, name, value, initial_value,
                          lower_bound, upper_bound, bounds_on, parameter_type, metadata_json
                   FROM mfdb_parameter
                   WHERE operation_id = ? AND deleted_at IS NULL""",
                (op_id,),
            ).fetchall()
            for prow in params:
                meta = _json_loads(prow["metadata_json"] if isinstance(prow, dict) else prow[8]) or {}
                parameters.append({
                    "parameter_uuid": prow["parameter_uuid"] if isinstance(prow, dict) else prow[0],
                    "operation_id": op_id,
                    "name": prow["name"] if isinstance(prow, dict) else prow[1],
                    "value": prow["value"] if isinstance(prow, dict) else prow[2],
                    "initial_value": prow["initial_value"] if isinstance(prow, dict) else prow[3],
                    "lower_bound": prow["lower_bound"] if isinstance(prow, dict) else prow[4],
                    "upper_bound": prow["upper_bound"] if isinstance(prow, dict) else prow[5],
                    "bounds_on": prow["bounds_on"] if isinstance(prow, dict) else prow[6],
                    "parameter_type": prow["parameter_type"] if isinstance(prow, dict) else prow[7],
                    "link_target": meta.get("link_target"),
                    "fit_parameter_uid": meta.get("fit_parameter_uid"),
                })

        return {"ok": True, "parameters": parameters}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)
