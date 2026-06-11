"""JSON-RPC handlers for fdb4chembio measurement provenance."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import traceback
from pathlib import Path
from typing import Any

from chisurf.core.fio.mmcif.db import FluorophoreDatabase, resolve_database_path
from chisurf.server.services import INVALID_INPUT, NOT_FOUND, OPERATION_FAILED, service_error


def register_measurement_services(dispatcher: Any) -> None:
    """Register fdb4chembio Phase 1 RPC handlers.

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
        with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
        with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
        with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
        with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    db: FluorophoreDatabase,
    processing_id: str,
    product: dict[str, Any],
) -> str:
    """Register a product dictionary in the repository.

    Parameters
    ----------
    db : FluorophoreDatabase
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
