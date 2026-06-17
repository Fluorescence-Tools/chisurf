"""JSON-RPC handlers for fdb measurement provenance."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sqlite3
import traceback
from pathlib import Path
from typing import Any

from chisurf.core.mfdb.database_resolver import resolve_database_path
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.server.services import INVALID_INPUT, NOT_FOUND, OPERATION_FAILED, service_error


def register_measurement_services(dispatcher: Any) -> None:
    """Register fdb Phase 1 RPC handlers.

    Parameters
    ----------
    dispatcher : object
        Service dispatcher exposing a ``register`` method.

    """
    for name, handler in {
        "raw_data.register": register_raw_data_handler,
        "raw_data.list": list_raw_data_handler,
        "raw_data.get": get_raw_data_handler,
        "processing.burst_selection.record": record_burst_selection_handler,
        "processing.burst_selection.run": run_burst_selection_handler,
        "processing.burst_selection.get": get_processing_run_handler,
        "processing.burst_selection.list": list_processing_runs_handler,
        "processed_data.register": register_processed_data_handler,
        "processed_data.list": list_processed_data_handler,
        "processed_data.get": get_processed_data_handler,
        "provenance.edges.list": list_provenance_edges_handler,
        "provenance.trace_processed_data": trace_processed_data_handler,
        "archive.burst_processing_manifest.export": export_burst_manifest_handler,
        "provenance.dependencies.upstream": get_upstream_dependencies_handler,
        "provenance.dependencies.downstream": get_downstream_dependencies_handler,
        "processing.run.record": record_general_processing_run_handler,
        "analysis.run.record": record_analysis_run_handler,
        "analysis.run.get": get_analysis_run_handler,
        "analysis.run.list": list_analysis_runs_handler,
        "analysis.run.delete": delete_analysis_run_handler,
        "provenance.graph.export": export_provenance_graph_handler,
        "database.backup": database_backup_handler,
        "archive.zip.export": export_zip_archive_handler,
        "audit_log.list": list_audit_logs_handler,
    }.items():
        dispatcher.register(name, lambda params, _handler=handler: _handler(**params))


def register_raw_data_handler(
    raw_data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Register a raw TTTR/PTU/SPC/BH data reference.

    Parameters
    ----------
    raw_data : dict, optional
        Raw-data fields. Keyword arguments are merged on top.
    **kwargs : Any
        Additional raw-data fields.

    Returns
    -------
    dict
        JSON-RPC result containing the registered raw-data row.

    """
    payload = {**(raw_data or {}), **kwargs}
    try:
        payload = _fill_location_metadata(payload)
        with MFDatabase(resolve_database_path()) as db:
            raw_data_id = db.add_raw_data_reference(
                experiment_id=str(payload.get("experiment_id") or ""),
                data_type=str(payload.get("data_type") or payload.get("file_type") or "TTTR"),
                storage_mode=str(payload.get("storage_mode") or _storage_mode_for(payload)),
                raw_data_id=payload.get("raw_data_id") or None,
                file_path=payload.get("file_path") or None,
                url=payload.get("url") or None,
                folder_path=payload.get("folder_path") or None,
                mime_type=payload.get("mime_type") or payload.get("content_type") or None,
                size_bytes=_int_or_none(payload.get("size_bytes")),
                checksum=payload.get("checksum") or None,
                checksum_algorithm=payload.get("checksum_algorithm") or "sha256",
                header_metadata=payload.get("header_metadata") or None,
                detector_mapping=payload.get("detector_mapping") or None,
                acquisition_software=payload.get("acquisition_software") or None,
                acquisition_software_version=payload.get("acquisition_software_version") or None,
                acquired_at=payload.get("acquired_at") or None,
                validation_status=payload.get("validation_status") or "unvalidated",
                validation_message=payload.get("validation_message") or None,
            )
            row = db.get_raw_data(raw_data_id)
            return {"ok": True, "raw_data": db._decode_raw_data_row(row)}
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def list_raw_data_handler(
    experiment_id: str | None = None,
    data_type: str | None = None,
) -> dict[str, Any]:
    """List raw-data references.

    Parameters
    ----------
    experiment_id : str, optional
        Experiment filter.
    data_type : str, optional
        Raw-data type filter.

    Returns
    -------
    dict
        JSON-RPC result containing raw-data rows.

    """
    with MFDatabase(resolve_database_path()) as db:
        rows = db.get_raw_data_references(experiment_id=experiment_id, data_type=data_type)
        return {"ok": True, "raw_data": [db._decode_raw_data_row(row) for row in rows]}


def get_raw_data_handler(raw_data_id: str) -> dict[str, Any]:
    """Return one raw-data reference.

    Parameters
    ----------
    raw_data_id : str
        Raw-data identifier.

    Returns
    -------
    dict
        JSON-RPC result containing one raw-data row.

    """
    with MFDatabase(resolve_database_path()) as db:
        row = db.get_raw_data(raw_data_id)
        if row is None:
            return service_error(f"raw data not found: {raw_data_id}", error_code=NOT_FOUND)
        return {"ok": True, "raw_data": db._decode_raw_data_row(row)}


def record_burst_selection_handler(
    experiment_id: str,
    raw_data_ids: list[str] | None = None,
    settings: dict[str, Any] | None = None,
    selected_setup: str | None = None,
    detectors: dict[str, Any] | None = None,
    windows: dict[str, Any] | None = None,
    output_paths: dict[str, str] | None = None,
    products: list[dict[str, Any]] | None = None,
    result_metadata: dict[str, Any] | None = None,
    processing_id: str | None = None,
    operator_user_id: str | None = None,
    software_version: str | None = None,
    status: str = "succeeded",
    error_message: str | None = None,
    traceback_summary: str | None = None,
) -> dict[str, Any]:
    """Record a Burst Selection processing run and its products.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier.
    raw_data_ids : list of str, optional
        Input raw-data identifiers.
    settings : dict, optional
        Serialized Burst Selection settings.
    selected_setup : str, optional
        Selected detector/setup name.
    detectors : dict, optional
        Detector definitions.
    windows : dict, optional
        PIE or microtime window definitions.
    output_paths : dict, optional
        Output paths keyed by product type.
    products : list of dict, optional
        Explicit product records.
    result_metadata : dict, optional
        Aggregate Burst Selection result metadata.
    processing_id : str, optional
        Explicit processing-run identifier.
    operator_user_id : str, optional
        Operator/user identifier.
    software_version : str, optional
        Software version string.
    status : str
        Processing status.
    error_message : str, optional
        Failure message.
    traceback_summary : str, optional
        Compact failure traceback.

    Returns
    -------
    dict
        JSON-RPC result containing the expanded processing run.

    """
    try:
        product_specs = list(products or [])
        product_specs.extend(_products_from_output_paths(output_paths or {}, result_metadata or {}))
        counts = _counts_from_metadata(result_metadata or {}, raw_data_ids or [])
        with MFDatabase(resolve_database_path()) as db:
            with db.transaction():
                run_id = db.add_processing_run(
                    experiment_id=experiment_id,
                    processing_type="burst_selection",
                    processing_id=processing_id,
                    input_raw_data_ids=raw_data_ids or [],
                    settings=settings or {},
                    selected_setup_name=selected_setup,
                    detector_definitions=detectors or {},
                    pie_window_definitions=windows or {},
                    operator_user_id=operator_user_id,
                    software_module="chisurf.plugins.burst.burst_selection",
                    software_version=software_version,
                    started_at=(result_metadata or {}).get("started_at"),
                    ended_at=(result_metadata or {}).get("ended_at"),
                    status=status,
                    error_message=error_message,
                    traceback_summary=traceback_summary,
                    **counts,
                )
                for product in product_specs:
                    _register_product(db, run_id, product)
            return {"ok": True, "processing_run": db.get_processing_run_full(run_id)}
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def run_burst_selection_handler(
    files: list[str],
    experiment_id: str,
    filetype: str | None = None,
    windows: dict[str, list[int]] | None = None,
    detectors: dict[str, dict[str, Any]] | None = None,
    settings: dict[str, Any] | None = None,
    output_dir: str | None = None,
    legacy_output: bool = False,
    legacy_output_folder_name: str | None = None,
    selected_setup: str | None = None,
    legacy_parameters: dict[str, Any] | None = None,
    raw_data_ids: list[str] | None = None,
    operator_user_id: str | None = None,
    software_version: str | None = None,
) -> dict[str, Any]:
    """Run Burst Selection and persist provenance for the result.

    Parameters
    ----------
    files : list of str
        TTTR input paths.
    experiment_id : str
        Experiment identifier.
    filetype : str, optional
        Explicit TTTR file type.
    windows : dict, optional
        PIE or microtime window definitions.
    detectors : dict, optional
        Detector definitions.
    settings : dict, optional
        Serialized Burst Selection settings.
    output_dir : str, optional
        Output directory for generated products.
    legacy_output : bool
        Whether to write the legacy folder layout.
    legacy_output_folder_name : str, optional
        Legacy output folder name.
    selected_setup : str, optional
        Selected setup name.
    legacy_parameters : dict, optional
        Legacy Info metadata.
    raw_data_ids : list of str, optional
        Existing raw-data identifiers to reuse.
    operator_user_id : str, optional
        Operator/user identifier.
    software_version : str, optional
        Software version string.

    Returns
    -------
    dict
        JSON-RPC result containing Burst Selection output and database records.

    """
    from chisurf.plugins.burst.burst_selection.backend.services import analyze_files_handler

    raw_ids = raw_data_ids or []
    try:
        raw_ids = raw_ids or _register_input_files(files, experiment_id, filetype)
        result = analyze_files_handler(
            files=files,
            filetype=filetype,
            windows=windows,
            detectors=detectors,
            settings=settings,
            output_dir=output_dir,
            legacy_output=legacy_output,
            legacy_output_folder_name=legacy_output_folder_name,
            selected_setup=selected_setup,
            legacy_parameters=legacy_parameters,
        )
        if not result.get("ok", True):
            failure_record = record_burst_selection_handler(
                experiment_id=experiment_id,
                raw_data_ids=raw_ids,
                settings=settings or {},
                selected_setup=selected_setup,
                detectors=detectors or {},
                windows=windows or {},
                operator_user_id=operator_user_id,
                software_version=software_version,
                status="failed",
                error_message=result.get("error") or "Burst Selection failed",
            )
            if failure_record.get("ok"):
                result["processing_run"] = failure_record.get("processing_run")
            return result
        analysis_result = result.get("result", {})
        record = record_burst_selection_handler(
            experiment_id=experiment_id,
            raw_data_ids=raw_ids,
            settings=settings or {},
            selected_setup=selected_setup,
            detectors=detectors or {},
            windows=windows or {},
            output_paths=analysis_result.get("output_paths") or {},
            result_metadata=analysis_result.get("metadata") or {},
            operator_user_id=operator_user_id,
            software_version=software_version,
            status="succeeded",
        )
        record["burst_selection_result"] = analysis_result
        return record
    except Exception as exc:
        failure_record = record_burst_selection_handler(
            experiment_id=experiment_id,
            raw_data_ids=raw_ids,
            settings=settings or {},
            selected_setup=selected_setup,
            detectors=detectors or {},
            windows=windows or {},
            operator_user_id=operator_user_id,
            software_version=software_version,
            status="failed",
            error_message=str(exc),
            traceback_summary=traceback_summary_for_current_exception(),
        )
        error = service_error(
            str(exc),
            error_code=OPERATION_FAILED,
            exception=exc,
        )
        if failure_record.get("ok"):
            error["processing_run"] = failure_record.get("processing_run")
        return error


def get_processing_run_handler(processing_id: str) -> dict[str, Any]:
    """Return one Burst Selection processing run.

    Parameters
    ----------
    processing_id : str
        Processing-run identifier.

    Returns
    -------
    dict
        JSON-RPC result containing the expanded run.

    """
    with MFDatabase(resolve_database_path()) as db:
        run = db.get_processing_run_full(processing_id)
        if run is None:
            return service_error(f"processing run not found: {processing_id}", error_code=NOT_FOUND)
        return {"ok": True, "processing_run": run}


def list_processing_runs_handler(
    experiment_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """List Burst Selection processing runs.

    Parameters
    ----------
    experiment_id : str, optional
        Experiment filter.
    status : str, optional
        Status filter.

    Returns
    -------
    dict
        JSON-RPC result containing processing runs.

    """
    with MFDatabase(resolve_database_path()) as db:
        rows = db.get_processing_runs(
            experiment_id=experiment_id,
            processing_type="burst_selection",
            status=status,
        )
        return {
            "ok": True,
            "processing_runs": [
                db.get_processing_run_full(row["processing_id"]) for row in rows
            ],
        }


def register_processed_data_handler(
    processed_data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Register a processed data product.

    Parameters
    ----------
    processed_data : dict, optional
        Processed-data fields. Keyword arguments are merged on top.
    **kwargs : Any
        Additional processed-data fields.

    Returns
    -------
    dict
        JSON-RPC result containing the registered product.

    """
    payload = {**(processed_data or {}), **kwargs}
    try:
        with MFDatabase(resolve_database_path()) as db:
            product_id = _register_product(
                db,
                str(payload.get("processing_id") or ""),
                payload,
            )
            return {
                "ok": True,
                "processed_data": db._decode_processed_data_row(db.get_processed_data(product_id)),
            }
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def list_processed_data_handler(
    processing_id: str | None = None,
    product_type: str | None = None,
) -> dict[str, Any]:
    """List processed-data products.

    Parameters
    ----------
    processing_id : str, optional
        Processing-run filter.
    product_type : str, optional
        Product-type filter.

    Returns
    -------
    dict
        JSON-RPC result containing products.

    """
    with MFDatabase(resolve_database_path()) as db:
        rows = db.get_processed_data_products(
            processing_id=processing_id,
            product_type=product_type,
        )
        return {"ok": True, "processed_data": [db._decode_processed_data_row(row) for row in rows]}


def get_processed_data_handler(processed_data_id: str) -> dict[str, Any]:
    """Return one processed-data product.

    Parameters
    ----------
    processed_data_id : str
        Processed-data identifier.

    Returns
    -------
    dict
        JSON-RPC result containing one product.

    """
    with MFDatabase(resolve_database_path()) as db:
        row = db.get_processed_data(processed_data_id)
        if row is None:
            return service_error(
                f"processed data not found: {processed_data_id}",
                error_code=NOT_FOUND,
            )
        return {"ok": True, "processed_data": db._decode_processed_data_row(row)}


def list_provenance_edges_handler(**filters: Any) -> dict[str, Any]:
    """List provenance edges.

    Parameters
    ----------
    **filters : Any
        Optional provenance filters accepted by the repository.

    Returns
    -------
    dict
        JSON-RPC result containing provenance edges.

    """
    allowed = {
        "source_node_type",
        "source_node_id",
        "target_node_type",
        "target_node_id",
        "relationship_type",
        "processing_id",
    }
    with MFDatabase(resolve_database_path()) as db:
        rows = db.get_provenance_edges(
            **{key: value for key, value in filters.items() if key in allowed}
        )
        return {
            "ok": True,
            "provenance_edges": [db._decode_provenance_edge_row(row) for row in rows],
        }


def trace_processed_data_handler(processed_data_id: str) -> dict[str, Any]:
    """Trace a product back through raw inputs and settings.

    Parameters
    ----------
    processed_data_id : str
        Processed-data identifier.

    Returns
    -------
    dict
        JSON-RPC result containing the trace.

    """
    with MFDatabase(resolve_database_path()) as db:
        trace = db.trace_processed_data(processed_data_id)
        if trace is None:
            return service_error(
                f"processed data not found: {processed_data_id}",
                error_code=NOT_FOUND,
            )
        return {"ok": True, "trace": trace}


def export_burst_manifest_handler(
    processing_id: str,
    output_path: str | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """Export an archive manifest for one Burst Selection run.

    Parameters
    ----------
    processing_id : str
        Processing-run identifier.
    output_path : str, optional
        Optional JSON output path.
    register : bool
        If ``True``, register the manifest as a processed-data product.

    Returns
    -------
    dict
        JSON-RPC result containing the manifest and optional product id.

    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            manifest = db.export_burst_processing_manifest(processing_id)
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            product_id = (
                db.register_archive_manifest(processing_id, manifest, output_path=output_path)
                if register
                else None
            )
            return {
                "ok": True,
                "manifest": manifest,
                "processed_data_id": product_id,
                "output_path": output_path,
            }
    except KeyError as exc:
        return service_error(str(exc), error_code=NOT_FOUND, exception=exc)
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def _register_input_files(
    files: list[str],
    experiment_id: str,
    filetype: str | None,
) -> list[str]:
    """Register raw input files for a Burst Selection run.

    Parameters
    ----------
    files : list of str
        Raw TTTR input file paths.
    experiment_id : str
        Experiment identifier.
    filetype : str, optional
        Explicit file type.

    Returns
    -------
    list of str
        Raw-data identifiers.

    """
    raw_ids: list[str] = []
    with MFDatabase(resolve_database_path()) as db:
        for file_name in files:
            payload = _fill_location_metadata(
                {
                    "experiment_id": experiment_id,
                    "data_type": (filetype or Path(file_name).suffix.lstrip(".") or "TTTR").upper(),
                    "storage_mode": "local_file",
                    "file_path": file_name,
                    "validation_status": "valid" if Path(file_name).exists() else "missing",
                }
            )
            raw_ids.append(
                db.add_raw_data_reference(
                    experiment_id=experiment_id,
                    data_type=payload["data_type"],
                    storage_mode=payload["storage_mode"],
                    file_path=payload["file_path"],
                    mime_type=payload.get("mime_type"),
                    size_bytes=payload.get("size_bytes"),
                    checksum=payload.get("checksum"),
                    checksum_algorithm=payload.get("checksum_algorithm") or "sha256",
                    validation_status=payload["validation_status"],
                )
            )
    return raw_ids


def _register_product(
    db: MFDatabase,
    processing_id: str,
    product: dict[str, Any],
) -> str:
    """Register a product dictionary in the repository.

    Parameters
    ----------
    db : MFDatabase
        Open repository.
    processing_id : str
        Processing-run identifier.
    product : dict
        Product fields.

    Returns
    -------
    str
        Processed-data identifier.

    """
    payload = _fill_location_metadata(product)
    data_json = payload.get("data_json")
    if isinstance(payload.get("data"), (dict, list)):
        data_json = json.dumps(payload["data"], sort_keys=True)
    return db.add_processed_data_product(
        processing_id=processing_id,
        product_type=str(payload.get("product_type") or _infer_product_type(payload)),
        storage_mode=str(payload.get("storage_mode") or _storage_mode_for(payload)),
        processed_data_id=payload.get("processed_data_id") or None,
        file_path=payload.get("file_path") or None,
        url=payload.get("url") or None,
        folder_path=payload.get("folder_path") or None,
        mime_type=payload.get("mime_type") or payload.get("content_type") or None,
        size_bytes=_int_or_none(payload.get("size_bytes")),
        checksum=payload.get("checksum") or None,
        checksum_algorithm=payload.get("checksum_algorithm") or "sha256",
        row_count=_int_or_none(payload.get("row_count")),
        product_summary=payload.get("product_summary") or None,
        metadata=payload.get("metadata") or None,
        data_json=data_json,
        validation_status=payload.get("validation_status") or "unvalidated",
        validation_message=payload.get("validation_message") or None,
    )


def _fill_location_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill size, checksum, and MIME metadata from local paths.

    Parameters
    ----------
    payload : dict
        Raw or processed data payload.

    Returns
    -------
    dict
        Payload with inferred metadata.

    """
    payload = dict(payload)
    path_text = payload.get("file_path") or payload.get("folder_path")
    if path_text and not payload.get("mime_type"):
        guessed, _ = mimetypes.guess_type(str(path_text))
        if guessed:
            payload["mime_type"] = guessed
    if not path_text or payload.get("checksum"):
        return payload
    path = Path(path_text)
    if path.exists():
        payload.setdefault("size_bytes", _path_size(path))
        payload["checksum"] = _path_checksum(path)
        payload.setdefault("checksum_algorithm", "sha256")
    return payload


def _path_checksum(path: Path) -> str:
    """Compute a SHA-256 checksum for a file or directory.

    Parameters
    ----------
    path : Path
        File or directory path.

    Returns
    -------
    str
        SHA-256 digest.

    """
    digest = hashlib.sha256()
    if path.is_dir():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            _update_digest(digest, item)
    else:
        _update_digest(digest, path)
    return digest.hexdigest()


def _update_digest(digest: Any, path: Path) -> None:
    """Update a digest from a file path.

    Parameters
    ----------
    digest : hashlib object
        Digest object to update.
    path : Path
        File path.

    """
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _path_size(path: Path) -> int:
    """Return file or directory size in bytes.

    Parameters
    ----------
    path : Path
        File or directory path.

    Returns
    -------
    int
        Total byte size.

    """
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def _storage_mode_for(payload: dict[str, Any]) -> str:
    """Infer a storage mode from location fields.

    Parameters
    ----------
    payload : dict
        Data payload.

    Returns
    -------
    str
        Storage mode.

    """
    if payload.get("url"):
        return "url"
    if payload.get("folder_path"):
        return "folder"
    if payload.get("data") or payload.get("data_json"):
        return "embedded_json"
    return "local_file"


def _products_from_output_paths(
    output_paths: dict[str, str],
    result_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert Burst Selection output paths to product specs.

    Parameters
    ----------
    output_paths : dict
        Output paths keyed by Burst Selection result key.
    result_metadata : dict
        Aggregate result metadata.

    Returns
    -------
    list of dict
        Product specifications.

    """
    products: list[dict[str, Any]] = []
    for key, path_text in output_paths.items():
        if not path_text:
            continue
        path = Path(path_text)
        location_key = "folder_path" if path.exists() and path.is_dir() else "file_path"
        products.append(
            {
                "product_type": _product_type_from_key(key, path),
                location_key: path_text,
                "row_count": result_metadata.get("n_bursts") if key == "bur" else None,
                "product_summary": result_metadata,
                "validation_status": "valid" if path.exists() else "missing",
            }
        )
    if result_metadata.get("gmm_fit"):
        products.append(
            {
                "product_type": "gmm_summary",
                "storage_mode": "embedded_json",
                "data": result_metadata["gmm_fit"],
                "product_summary": result_metadata,
                "validation_status": "valid",
            }
        )
    return products


def _product_type_from_key(key: str, path: Path) -> str:
    """Infer a product type from a Burst Selection output key.

    Parameters
    ----------
    key : str
        Burst Selection output key.
    path : Path
        Output path.

    Returns
    -------
    str
        Product type.

    """
    normalized = key.lower()
    if normalized in {"bur", "hdf5", "zip"}:
        return normalized
    if normalized == "mti_dir":
        return "mti_summary"
    if normalized == "output_folder":
        if path.is_dir():
            return "legacy_info_folder"
        return _infer_product_type({"file_path": str(path)})
    return _infer_product_type({"file_path": str(path), "product_type": normalized})


def _infer_product_type(payload: dict[str, Any]) -> str:
    """Infer product type from payload fields.

    Parameters
    ----------
    payload : dict
        Product payload.

    Returns
    -------
    str
        Product type.

    """
    if payload.get("product_type"):
        return str(payload["product_type"])
    path_text = payload.get("file_path") or payload.get("folder_path") or ""
    suffix = Path(path_text).suffix.lower()
    return {
        ".bur": "bur",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".zip": "zip",
        ".mti": "mti_summary",
        ".json": "json_summary",
        ".fcs": "fcs_correlation",
        ".dec": "tcspc_decay",
        ".tcspc": "tcspc_decay",
        ".irf": "irf_curve",
        ".spc": "spectra",
        ".spectra": "spectra",
        ".aniso": "anisotropy_curve",
        ".pda": "pda_histogram",
        ".fit": "fit_results",
    }.get(suffix, "derived_product")


def _counts_from_metadata(
    metadata: dict[str, Any],
    raw_data_ids: list[str],
) -> dict[str, int | None]:
    """Extract processing aggregate counts.

    Parameters
    ----------
    metadata : dict
        Burst Selection metadata.
    raw_data_ids : list of str
        Input raw-data identifiers.

    Returns
    -------
    dict
        Count fields accepted by the repository.

    """
    return {
        "file_count": _int_or_none(metadata.get("n_files")) or len(raw_data_ids) or None,
        "photon_count": _int_or_none(metadata.get("n_photons")),
        "selected_photon_count": _int_or_none(metadata.get("n_selected")),
        "burst_count": _int_or_none(metadata.get("n_bursts")),
    }


def _int_or_none(value: Any) -> int | None:
    """Convert a value to ``int`` when possible.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    int or None
        Converted integer or ``None``.

    """
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def traceback_summary_for_current_exception() -> str:
    """Return a compact traceback summary for the active exception.

    Returns
    -------
    str
        Traceback text.

    """
    return "".join(traceback.format_exc(limit=8))


def record_general_processing_run_handler(
    experiment_id: str,
    processing_type: str,
    input_raw_data_ids: list[str] | None = None,
    input_processed_data_ids: list[str] | None = None,
    settings: dict[str, Any] | None = None,
    selected_setup: str | None = None,
    detectors: dict[str, Any] | None = None,
    windows: dict[str, Any] | None = None,
    products: list[dict[str, Any]] | None = None,
    result_metadata: dict[str, Any] | None = None,
    processing_id: str | None = None,
    operator_user_id: str | None = None,
    software_package: str | None = "chisurf",
    software_module: str | None = None,
    software_version: str | None = None,
    status: str = "succeeded",
    error_message: str | None = None,
    traceback_summary: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Record a general processing run and its inputs/products.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier.
    processing_type : str
        The type of processing run (e.g., 'fcs_correlation', 'tcspc_fitting').
    input_raw_data_ids : list of str, optional
        Identifiers of raw input files.
    input_processed_data_ids : list of str, optional
        Identifiers of processed data inputs.
    settings : dict, optional
        Parameters/settings used in the processing run.
    selected_setup : str, optional
        Instrument setup name.
    detectors : dict, optional
        Detector configuration.
    windows : dict, optional
        Window definitions.
    products : list of dict, optional
        Product specifications to be registered.
    result_metadata : dict, optional
        Additional metadata about the processing run.
    processing_id : str, optional
        Explicit identifier for the run.
    operator_user_id : str, optional
        Identifier of the operator.
    software_package : str, optional
        Software package name.
    software_module : str, optional
        Software module name.
    software_version : str, optional
        Software version.
    status : str
        Run status (e.g., 'succeeded', 'failed').
    error_message : str, optional
        Error message if failed.
    traceback_summary : str, optional
        Traceback summary if failed.
    **kwargs : Any
        Catch-all for extra parameters.

    Returns
    -------
    dict
        JSON-RPC result with the recorded processing run and its products.
    """
    try:
        raw_ids = input_raw_data_ids or []
        proc_ids = input_processed_data_ids or []
        product_specs = products or []
        settings_hash = None
        if settings:
            settings_hash = hashlib.sha256(
                json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()

        with MFDatabase(resolve_database_path()) as db:
            with db.transaction():
                missing_processed_inputs = [
                    input_id for input_id in proc_ids
                    if db.get_processed_data(input_id) is None
                ]
                if missing_processed_inputs:
                    raise sqlite3.IntegrityError(
                        "Missing input processed data artifact(s): " + ", ".join(missing_processed_inputs)
                    )
                run_id = db.add_processing_run(
                    experiment_id=experiment_id,
                    processing_type=processing_type,
                    processing_id=processing_id,
                    input_raw_data_ids=raw_ids,
                    settings=settings,
                    selected_setup_name=selected_setup,
                    detector_definitions=detectors,
                    pie_window_definitions=windows,
                    operator_user_id=operator_user_id,
                    software_package=software_package,
                    software_module=software_module,
                    software_version=software_version,
                    started_at=(result_metadata or {}).get("started_at"),
                    ended_at=(result_metadata or {}).get("ended_at"),
                    status=status,
                    error_message=error_message,
                    traceback_summary=traceback_summary,
                )

                # Record input processed data dependencies
                for input_id in proc_ids:
                    in_prod = db.get_processed_data(input_id)
                    db.add_provenance_edge(
                        source_node_type="processed_data",
                        source_node_id=input_id,
                        target_node_type="processing_run",
                        target_node_id=run_id,
                        relationship_type="input_to",
                        processing_id=run_id,
                        settings_hash=settings_hash,
                        software_version=software_version,
                        checksum_snapshot={
                            "input_processed_data": in_prod["checksum"] if in_prod else None,
                            "settings": settings_hash,
                        },
                    )

                # Record output products
                registered_products = []
                for product in product_specs:
                    prod_id = _register_product(db, run_id, product)
                    registered_products.append(
                        db._decode_processed_data_row(db.get_processed_data(prod_id))
                    )

                return {
                    "ok": True,
                    "processing_run": db.get_processing_run_full(run_id),
                    "products": registered_products,
                }
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def get_upstream_dependencies_handler(node_type: str, node_id: str) -> dict[str, Any]:
    """Retrieve upstream dependencies for a node.

    Parameters
    ----------
    node_type : str
        The node type.
    node_id : str
        The node identifier.

    Returns
    -------
    dict
        JSON-RPC result with the list of upstream provenance edges.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            rows = db.get_upstream_dependencies(node_type, node_id)
            edges = [db._decode_provenance_edge_row(row) for row in rows]
            return {"ok": True, "edges": edges}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def get_downstream_dependencies_handler(node_type: str, node_id: str) -> dict[str, Any]:
    """Retrieve downstream dependencies for a node.

    Parameters
    ----------
    node_type : str
        The node type.
    node_id : str
        The node identifier.

    Returns
    -------
    dict
        JSON-RPC result with the list of downstream provenance edges.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            rows = db.get_downstream_dependencies(node_type, node_id)
            edges = [db._decode_provenance_edge_row(row) for row in rows]
            return {"ok": True, "edges": edges}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def record_analysis_run_handler(
    analysis_type: str,
    experiment_id: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    model_version: str | None = None,
    fit_structure: list[dict[str, Any]] | None = None,
    parameter_links: list[tuple[Any, ...]] | None = None,
    software_package: str | None = "chisurf",
    software_module: str | None = None,
    software_version: str | None = None,
    optimizer_settings: dict[str, Any] | None = None,
    covariance_matrix: list[list[float]] | dict[str, Any] | None = None,
    convergence_status: str | None = None,
    goodness_of_fit: dict[str, Any] | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    analysis_id: str | None = None,
    input_processed_data_ids: list[str] | None = None,
    products: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    grouped_fit_uuids: list[str] | None = None,
    parameter_linkages: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Record an analysis run with its parameters, inputs, products, and linkages.

    Parameters
    ----------
    analysis_type : str
        Type of analysis run (e.g. 'local_fit', 'global_fit').
    experiment_id : str, optional
        Optional experiment ID.
    model_name : str, optional
        Model name.
    model_type : str, optional
        Model type.
    model_version : str, optional
        Model version.
    fit_structure : list of dict, optional
        Structured grouped fits and dataset maps.
    parameter_links : list of tuple, optional
        Formula parameter links.
    software_package : str, optional
        Software package name.
    software_module : str, optional
        Software module.
    software_version : str, optional
        Software version.
    optimizer_settings : dict, optional
        Optimizer settings.
    covariance_matrix : list of list of float or dict, optional
        Covariance matrix.
    convergence_status : str, optional
        Convergence status.
    goodness_of_fit : dict, optional
        Goodness-of-fit metrics.
    notes : str, optional
        Operator notes.
    metadata : dict, optional
        Fit parameters widget settings/properties.
    analysis_id : str, optional
        Stable identifier. Generates UUID if omitted.
    input_processed_data_ids : list of str, optional
        Processed data inputs consumed by this analysis.
    products : list of dict, optional
        Products produced by this analysis.
    parameters : list of dict, optional
        Model parameter dictionaries to register.
    grouped_fit_uuids : list of str, optional
        UUIDs of local fits to group into this run.
    parameter_linkages : list of tuple of str, optional
        Explicit parameter linkages `(source_uuid, target_uuid)`.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            with db.transaction():
                # Add analysis run
                run_id = db.add_analysis_run(
                    analysis_type=analysis_type,
                    experiment_id=experiment_id,
                    model_name=model_name,
                    model_type=model_type,
                    model_version=model_version,
                    fit_structure=fit_structure,
                    parameter_links=parameter_links,
                    software_package=software_package,
                    software_module=software_module,
                    software_version=software_version,
                    optimizer_settings=optimizer_settings,
                    covariance_matrix=covariance_matrix,
                    convergence_status=convergence_status,
                    goodness_of_fit=goodness_of_fit,
                    notes=notes,
                    metadata=metadata,
                    analysis_id=analysis_id,
                )

                # Record input dependencies
                for input_id in (input_processed_data_ids or []):
                    in_prod = db.get_processed_data(input_id)
                    db.add_provenance_edge(
                        source_node_type="processed_data",
                        source_node_id=input_id,
                        target_node_type="analysis_run",
                        target_node_id=run_id,
                        relationship_type="input_to",
                        processing_id=run_id,
                        checksum_snapshot={
                            "input_processed_data": in_prod["checksum"] if in_prod else None,
                        },
                    )

                # Record parameters
                registered_parameters = []
                for param in (parameters or []):
                    param_uuid = db.add_analysis_parameter(
                        analysis_id=run_id,
                        name=param["name"],
                        value=param.get("value"),
                        standard_error=param.get("standard_error"),
                        confidence_interval_low=param.get("confidence_interval_low"),
                        confidence_interval_high=param.get("confidence_interval_high"),
                        initial_value=param.get("initial_value"),
                        lower_bound=param.get("lower_bound"),
                        upper_bound=param.get("upper_bound"),
                        bounds_on=param.get("bounds_on", False),
                        units=param.get("units"),
                        parameter_type=param.get("parameter_type", "free"),
                        expression=param.get("expression"),
                        prior=param.get("prior"),
                        mapping=param.get("mapping"),
                        metadata=param.get("metadata"),
                        parameter_uuid=param.get("parameter_uuid"),
                    )
                    db.add_provenance_edge(
                        source_node_type="analysis_parameter",
                        source_node_id=param_uuid,
                        target_node_type="analysis_run",
                        target_node_id=run_id,
                        relationship_type="parameter_of",
                        processing_id=run_id,
                    )
                    registered_parameters.append(
                        db._decode_analysis_parameter_row(db.get_analysis_parameter(param_uuid))
                    )

                # Record products
                registered_products = []
                for prod in (products or []):
                    prod_payload = _fill_location_metadata(prod)
                    data_json = prod_payload.get("data_json")
                    if isinstance(prod_payload.get("data"), (dict, list)):
                        data_json = json.dumps(prod_payload["data"], sort_keys=True)
                    prod_id = db.add_analysis_product(
                        analysis_id=run_id,
                        product_type=str(prod_payload.get("product_type") or _infer_product_type(prod_payload)),
                        storage_mode=str(prod_payload.get("storage_mode") or _storage_mode_for(prod_payload)),
                        processed_data_id=prod_payload.get("processed_data_id"),
                        file_path=prod_payload.get("file_path"),
                        url=prod_payload.get("url"),
                        folder_path=prod_payload.get("folder_path"),
                        mime_type=prod_payload.get("mime_type"),
                        size_bytes=_int_or_none(prod_payload.get("size_bytes")),
                        checksum=prod_payload.get("checksum"),
                        checksum_algorithm=prod_payload.get("checksum_algorithm") or "sha256",
                        row_count=_int_or_none(prod_payload.get("row_count")),
                        product_summary=prod_payload.get("product_summary"),
                        metadata=prod_payload.get("metadata"),
                        data_json=data_json,
                        validation_status=prod_payload.get("validation_status") or "unvalidated",
                        validation_message=prod_payload.get("validation_message"),
                    )
                    registered_products.append(
                        db._decode_processed_data_row(db.get_processed_data(prod_id))
                    )

                # Link grouped sub-fits
                for sub_uuid in (grouped_fit_uuids or []):
                    db.link_grouped_fits(sub_uuid, run_id)

                # Link analysis parameters
                for src_param, tgt_param in (parameter_linkages or []):
                    db.link_analysis_parameters(src_param, tgt_param)

            return {
                "ok": True,
                "analysis_run": db.get_analysis_run_full(run_id),
                "products": registered_products,
                "parameters": registered_parameters,
            }
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def get_analysis_run_handler(analysis_id: str) -> dict[str, Any]:
    """Retrieve one analysis run with all details.

    Parameters
    ----------
    analysis_id : str
        Analysis run identifier.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            run = db.get_analysis_run_full(analysis_id)
            if run is None:
                return service_error(f"analysis run not found: {analysis_id}", error_code=NOT_FOUND)
            return {"ok": True, "analysis_run": run}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def list_analysis_runs_handler(
    experiment_id: str | None = None,
    analysis_type: str | None = None,
) -> dict[str, Any]:
    """List registered analysis runs.

    Parameters
    ----------
    experiment_id : str, optional
        Filter by experiment ID.
    analysis_type : str, optional
        Filter by analysis type.

    Returns
    -------
    dict
        JSON-RPC result list.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            rows = db.list_analysis_runs(experiment_id=experiment_id, analysis_type=analysis_type)
            runs = [db._decode_analysis_run_row(row) for row in rows]
            return {"ok": True, "analysis_runs": runs}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def delete_analysis_run_handler(analysis_id: str) -> dict[str, Any]:
    """Delete an analysis run.

    Parameters
    ----------
    analysis_id : str
        Analysis run identifier.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            db.delete_analysis_run(analysis_id)
            return {"ok": True, "deleted_analysis_id": analysis_id}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def archive_project_handler(
    project_id: str,
    project_name: str,
    project_payload: dict[str, Any],
    project_archive_data: str | None = None,
    project_archive_filename: str | None = None,
    experiment_id: str | None = None,
    input_processed_data_ids: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Archive a complete project state to the database."""
    try:
        with MFDatabase(resolve_database_path()) as db:
            with db.transaction():
                archive_object = None
                if project_archive_data:
                    import base64

                    archive_bytes = base64.b64decode(project_archive_data)
                    archive_object = db.put_object(
                        data=archive_bytes,
                        filename=project_archive_filename or f"{project_id}.csp",
                        mime_type="application/vnd.chisurf.project+zip",
                        metadata={
                            "project_id": project_id,
                            "project_name": project_name,
                            "project_format": "csp",
                        },
                    )
                    db.register_artifact(
                        artifact_id=f"{project_id}_archive",
                        artifact_kind="project_archive",
                        data_format="zip",
                        storage_mode="managed_archive",
                        mime_type="application/vnd.chisurf.project+zip",
                        size_bytes=int(archive_object.get("size_bytes") or len(archive_bytes)),
                        checksum=archive_object.get("content_md5"),
                        checksum_algorithm="md5",
                        object_uuid=archive_object.get("object_uuid"),
                        metadata={
                            "project_id": project_id,
                            "project_name": project_name,
                            "filename": archive_object.get("original_filename"),
                        },
                    )

                stored_payload = dict(project_payload or {})
                if archive_object:
                    stored_payload.setdefault("extra", {})
                    if isinstance(stored_payload["extra"], dict):
                        stored_payload["extra"]["mfdb_project_archive"] = {
                            "object_uuid": archive_object.get("object_uuid"),
                            "artifact_id": f"{project_id}_archive",
                            "filename": archive_object.get("original_filename"),
                            "size_bytes": archive_object.get("size_bytes"),
                            "content_md5": archive_object.get("content_md5"),
                        }
                run_id = db.add_analysis_run(
                    analysis_type="project",
                    experiment_id=experiment_id,
                    model_name=project_name,
                    model_type="project_archive",
                    fit_structure=stored_payload,
                    notes=notes,
                    analysis_id=project_id,
                )

                # Record input dependencies from processed datasets
                for input_id in (input_processed_data_ids or []):
                    in_prod = db.get_processed_data(input_id)
                    if in_prod is None:
                        raise ValueError(f"Processed data artifact not found: {input_id}")
                    db.add_provenance_edge(
                        source_node_type="processed_data",
                        source_node_id=input_id,
                        target_node_type="analysis_run",
                        target_node_id=run_id,
                        relationship_type="input_to",
                        processing_id=run_id,
                        checksum_snapshot={
                            "input_processed_data": in_prod["checksum"] if in_prod else None,
                        },
                    )

                db.add_audit_log(
                    action="archive",
                    target_type="project",
                    target_id=run_id,
                    details={
                        "project_name": project_name,
                        "experiment_id": experiment_id,
                        "archive_object_uuid": archive_object.get("object_uuid") if archive_object else None,
                    },
                )

            return {
                "ok": True,
                "project_id": run_id,
                "project_name": project_name,
                "archive_object": archive_object,
            }
    except Exception as exc:
        return service_error(str(exc), error_code=INVALID_INPUT, exception=exc)


def restore_project_handler(project_id: str) -> dict[str, Any]:
    """Retrieve an archived project state from the database."""
    try:
        with MFDatabase(resolve_database_path()) as db:
            run = db.get_analysis_run_full(project_id)
            if not run:
                return service_error(f"Project not found: {project_id}", error_code=NOT_FOUND)

            payload = run.get("fit_structure")
            archive_data = None
            archive_info = None
            if isinstance(payload, dict):
                extra = payload.get("extra")
                if isinstance(extra, dict):
                    archive_info = extra.get("mfdb_project_archive")
            if isinstance(archive_info, dict) and archive_info.get("object_uuid"):
                import base64

                archive_bytes = db.get_object(str(archive_info["object_uuid"]))
                archive_data = base64.b64encode(archive_bytes).decode("ascii")
            db.add_audit_log(
                action="restore",
                target_type="project",
                target_id=project_id,
                details={"project_name": run.get("model_name")},
            )
            return {
                "ok": True,
                "project_id": project_id,
                "project_name": run.get("model_name"),
                "project_payload": payload,
                "project_archive_data": archive_data,
                "project_archive": archive_info,
            }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def export_provenance_graph_handler(
    seed_node_type: str,
    seed_node_id: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Export the provenance subgraph as a JSON-serializable structure."""
    try:
        with MFDatabase(resolve_database_path()) as db:
            graph = db.export_provenance_graph(seed_node_type, seed_node_id)
            if output_path:
                import json
                with open(output_path, "w", encoding="utf-8") as f:
                    if output_path.lower().endswith(".jsonl"):
                        for node in graph["nodes"]:
                            f.write(json.dumps({"type": "node", "data": node}) + "\n")
                        for edge in graph["edges"]:
                            f.write(json.dumps({"type": "edge", "data": edge}) + "\n")
                    else:
                        json.dump(graph, f, indent=2)

            return {
                "ok": True,
                "graph": graph,
                "output_path": output_path,
            }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def database_backup_handler(target_path: str) -> dict[str, Any]:
    """Create a hot backup of the SQLite database to the specified target path."""
    try:
        with MFDatabase(resolve_database_path()) as db:
            db.backup_database(target_path)
            db.add_audit_log(
                action="backup",
                target_type="database",
                target_id=target_path,
            )
            return {"ok": True, "backup_path": target_path}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def export_zip_archive_handler(
    target_zip_path: str,
    seed_node_type: str,
    seed_node_id: str,
    include_external_data: bool = False,
    base_path_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Package a full ZIP archive containing DB snapshot, graph, manifest and optionally data."""
    import json
    import os
    import shutil
    import tempfile
    import zipfile
    from datetime import datetime

    try:
        with MFDatabase(resolve_database_path()) as db:
            graph = db.export_provenance_graph(seed_node_type, seed_node_id)

            with tempfile.TemporaryDirectory() as tmpdir:
                db_snapshot_path = os.path.join(tmpdir, "database_snapshot.db")
                db.backup_database(db_snapshot_path)

                graph_json_path = os.path.join(tmpdir, "provenance_graph.json")
                with open(graph_json_path, "w", encoding="utf-8") as f:
                    json.dump(graph, f, indent=2)

                manifest = {
                    "archive_format_version": 1,
                    "seed_node": {"type": seed_node_type, "id": seed_node_id},
                    "created_at": datetime.now().isoformat(),
                    "files": [],
                }

                def resolve_path(p: str) -> str:
                    if not p:
                        return p
                    if base_path_map:
                        for old_prefix, new_prefix in base_path_map.items():
                            if p.startswith(old_prefix):
                                return p.replace(old_prefix, new_prefix, 1)
                    return p

                manifest["files"].append({
                    "relative_path": "database_snapshot.db",
                    "type": "database_snapshot",
                })
                manifest["files"].append({
                    "relative_path": "provenance_graph.json",
                    "type": "provenance_graph",
                })

                if include_external_data:
                    os.makedirs(os.path.join(tmpdir, "external_data"), exist_ok=True)

                for node in graph["nodes"]:
                    node_type = node.get("node_type")
                    orig_path = node.get("file_path") or node.get("path")
                    if not orig_path:
                        continue

                    resolved_path = resolve_path(orig_path)
                    checksum = node.get("checksum")

                    file_info = {
                        "node_type": node_type,
                        "node_id": node.get("node_id") or node.get("artifact_id") or node.get("raw_data_id") or node.get("processed_data_id") or node.get("analysis_id"),
                        "original_path": orig_path,
                        "resolved_path": resolved_path,
                        "checksum": checksum,
                        "copied": False,
                    }

                    if include_external_data and resolved_path and os.path.isfile(resolved_path):
                        filename = os.path.basename(resolved_path)
                        dest_rel = f"external_data/{filename}"
                        dest_full = os.path.join(tmpdir, dest_rel)
                        shutil.copy2(resolved_path, dest_full)
                        file_info["relative_path"] = dest_rel
                        file_info["copied"] = True

                    manifest["files"].append(file_info)

                manifest_path = os.path.join(tmpdir, "manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                target_dir = os.path.dirname(os.path.abspath(target_zip_path))
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)

                with zipfile.ZipFile(target_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(manifest_path, "manifest.json")
                    zf.write(db_snapshot_path, "database_snapshot.db")
                    zf.write(graph_json_path, "provenance_graph.json")
                    if include_external_data:
                        for file_info in manifest["files"]:
                            if file_info.get("copied"):
                                zf.write(
                                    os.path.join(tmpdir, file_info["relative_path"]),
                                    file_info["relative_path"]
                                )

                db.add_audit_log(
                    action="archive",
                    target_type="zip_archive",
                    target_id=target_zip_path,
                    details={"seed_node_type": seed_node_type, "seed_node_id": seed_node_id, "include_external_data": include_external_data},
                )

                return {
                    "ok": True,
                    "zip_path": target_zip_path,
                    "manifest": manifest,
                }
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def list_audit_logs_handler(
    action: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Retrieve audit log records with optional filtering.

    Parameters
    ----------
    action : str, optional
        Filter by action type.
    target_type : str, optional
        Filter by target entity type.
    target_id : str, optional
        Filter by target entity ID.
    limit : int, default=100
        Maximum number of logs to return.

    Returns
    -------
    dict
        JSON-RPC result containing audit log rows.
    """
    try:
        with MFDatabase(resolve_database_path()) as db:
            logs = db.get_audit_logs(
                action=action,
                target_type=target_type,
                target_id=target_id,
                limit=limit,
            )
            return {"ok": True, "audit_logs": logs}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)

