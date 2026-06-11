"""ServiceDispatcher-compatible handlers for the sample database plugin."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from chisurf.core.fio.mmcif.db import (
    FluorophoreDatabase,
    backup_database,
    import_structure_file,
    resolve_database_path,
    source_database_path,
    user_database_path,
)
from chisurf.plugins.sample_database.backend.measurement_services import (
    register_measurement_services,
)


def register_services(dispatcher: Any) -> None:
    """Register sample database RPC handlers."""
    register_measurement_services(dispatcher)
    dispatcher.register("sample_database.status", lambda params: status_handler(**params))
    dispatcher.register(
        "sample_database.samples.list", lambda params: list_samples_handler(**params)
    )
    dispatcher.register("sample_database.samples.get", lambda params: get_sample_handler(**params))
    dispatcher.register(
        "sample_database.samples.save", lambda params: save_sample_handler(**params)
    )
    dispatcher.register(
        "sample_database.samples.delete", lambda params: delete_sample_handler(**params)
    )
    dispatcher.register(
        "sample_database.samples.key_values.save",
        lambda params: save_sample_key_values_handler(**params),
    )
    dispatcher.register("sample_database.users.list", lambda params: list_users_handler(**params))
    dispatcher.register("sample_database.users.save", lambda params: save_user_handler(**params))
    dispatcher.register(
        "sample_database.users.delete", lambda params: delete_user_handler(**params)
    )
    dispatcher.register(
        "sample_database.devices.list", lambda params: list_devices_handler(**params)
    )
    dispatcher.register(
        "sample_database.devices.save", lambda params: save_device_handler(**params)
    )
    dispatcher.register(
        "sample_database.devices.delete", lambda params: delete_device_handler(**params)
    )
    dispatcher.register(
        "sample_database.experiment_types.list",
        lambda params: list_experiment_types_handler(**params),
    )
    dispatcher.register(
        "sample_database.experiment_types.save",
        lambda params: save_experiment_type_handler(**params),
    )
    dispatcher.register(
        "sample_database.experiment_types.delete",
        lambda params: delete_experiment_type_handler(**params),
    )
    dispatcher.register(
        "sample_database.experiments.list", lambda params: list_experiments_handler(**params)
    )
    dispatcher.register(
        "sample_database.experiments.get", lambda params: get_experiment_handler(**params)
    )
    dispatcher.register(
        "sample_database.experiments.save", lambda params: save_experiment_handler(**params)
    )
    dispatcher.register(
        "sample_database.experiments.delete", lambda params: delete_experiment_handler(**params)
    )
    dispatcher.register(
        "sample_database.experiments.key_values.save",
        lambda params: save_experiment_key_values_handler(**params),
    )
    dispatcher.register(
        "sample_database.experiments.data.save",
        lambda params: save_experiment_data_handler(**params),
    )
    dispatcher.register(
        "sample_database.experiments.data.delete",
        lambda params: delete_experiment_data_handler(**params),
    )
    dispatcher.register("sample_database.import_file", lambda params: import_file_handler(**params))
    dispatcher.register(
        "sample_database.export_sample", lambda params: export_sample_handler(**params)
    )
    dispatcher.register(
        "sample_database.export_table", lambda params: export_table_handler(**params)
    )
    dispatcher.register("sample_database.backup", lambda params: backup_handler(**params))
    dispatcher.register(
        "sample_database.reset_from_source", lambda params: reset_from_source_handler(**params)
    )


def status_handler() -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        return {
            "source_database": str(source_database_path()),
            "user_database": str(user_database_path()),
            "schema_version": db._get_schema_version(),
            "sample_count": len(db.list_samples()),
        }


def list_samples_handler() -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        return {"samples": [_json_row(row) for row in db.list_samples()]}


def get_sample_handler(sample_id: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        sample = db.get_sample_full(sample_id)
        return {"sample": sample}


def save_sample_key_values_handler(
    sample_id: str, key_values: list[dict[str, Any]]
) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    return get_sample_handler(sample_id)


def list_users_handler() -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        return {"users": [_json_row(row) for row in db.get_users()]}


def save_user_handler(user: dict[str, Any]) -> dict[str, Any]:
    user_id = str(user.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("user_id is required")
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.add_user(
            user_id,
            str(user.get("display_name") or user_id),
            user.get("email") or None,
            user.get("affiliation") or None,
            user.get("details") or None,
        )
    return list_users_handler()


def delete_user_handler(user_id: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.delete_user(user_id)
    return list_users_handler()


def list_devices_handler() -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        return {"devices": [_json_row(row) for row in db.get_devices()]}


def save_device_handler(device: dict[str, Any]) -> dict[str, Any]:
    device_id = str(device.get("device_id") or "").strip()
    if not device_id:
        raise ValueError("device_id is required")
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    return list_devices_handler()


def delete_device_handler(device_id: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.delete_device(device_id)
    return list_devices_handler()


def list_experiment_types_handler() -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        return {"experiment_types": [_json_row(row) for row in db.get_experiment_types()]}


def save_experiment_type_handler(experiment_type: dict[str, Any]) -> dict[str, Any]:
    name = str(experiment_type.get("name") or "").strip()
    if not name:
        raise ValueError("experiment type name is required")
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.add_experiment_type(
            name,
            category=experiment_type.get("category"),
            description=experiment_type.get("description"),
            details=experiment_type.get("details"),
        )
    return list_experiment_types_handler()


def delete_experiment_type_handler(type_id: int) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.delete_experiment_type(int(type_id))
    return list_experiment_types_handler()


def list_experiments_handler(
    sample_id: str | None = None,
    project_id: str | None = None,
    type_id: int | None = None,
) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        return {
            "experiments": [
                _json_row(row)
                for row in db.get_experiments(
                    sample_id=sample_id, project_id=project_id, type_id=type_id
                )
            ]
        }


def get_experiment_handler(experiment_id: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        experiment = db.get_experiment_full(experiment_id)
    return {"experiment": experiment}


def save_experiment_handler(experiment: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(experiment.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("experiment_id is required")
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    return get_experiment_handler(experiment_id)


def delete_experiment_handler(experiment_id: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.delete_experiment(experiment_id)
    return {"ok": True, "experiment_id": experiment_id}


def save_experiment_key_values_handler(
    experiment_id: str, key_values: list[dict[str, Any]]
) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    return get_experiment_handler(experiment_id)


def save_experiment_data_handler(data: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(data.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("experiment_id is required")
    data_type = str(data.get("data_type") or "").strip()
    if not data_type:
        raise ValueError("data_type is required")
    storage_mode = str(data.get("storage_mode") or "link").strip()
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    return get_experiment_handler(experiment_id)


def delete_experiment_data_handler(data_id: int) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        row = db.conn.execute(
            "SELECT experiment_id FROM flr_experiment_data WHERE data_id = ?", (int(data_id),)
        ).fetchone()
        experiment_id = row["experiment_id"] if row else None
        db.delete_experiment_data(int(data_id))
    return {"ok": True, "data_id": int(data_id), "experiment_id": experiment_id}


def save_sample_handler(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    with FluorophoreDatabase(resolve_database_path()) as db:
        condition = sample.get("condition") or {}
        condition_id = condition.get("condition_id") or sample.get("sample_condition_id")
        if condition_id:
            db.add_sample_condition(
                str(condition_id),
                _float_or_none(condition.get("ph")),
                _float_or_none(condition.get("temperature")),
                _float_or_none(condition.get("ionic_strength")),
                condition.get("buffer_composition"),
                condition.get("details"),
            )
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
    return get_sample_handler(sample_id)


def delete_sample_handler(sample_id: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.delete_sample(sample_id)
    return {"ok": True, "sample_id": sample_id}


def import_file_handler(path: str) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
        summary = import_structure_file(db, path)
    return {"summary": summary}


def export_sample_handler(
    sample_id: str,
    output_path: str | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    with FluorophoreDatabase(resolve_database_path()) as db:
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
    with FluorophoreDatabase(resolve_database_path()) as db:
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


def _sample_table_rows(
    db: FluorophoreDatabase, sample_id: str | None = None
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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _bytes_or_none(value: Any) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)
