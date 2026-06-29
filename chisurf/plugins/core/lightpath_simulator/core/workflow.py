"""Shared light-path simulator workflows used by GUI, CLI, and RPC."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chisurf.core.mfdb.database_resolver import resolve_database_path
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.plugins.core.lightpath_simulator.backend.simulator import (
    OpticalPathSimulator,
)




class MFDatabaseAdapter:
    """Adapt MFDB records to the simulator's small spectra/probe interface."""

    def __init__(self, db: MFDatabase) -> None:
        """Wrap an open MFDB connection."""
        self.db = db

    def __enter__(self) -> MFDatabaseAdapter:
        """Return the adapter without closing the outer MFDB connection."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Leave transaction ownership to the outer MFDB context."""
        return None

    def get_probe_spectrum(
        self,
        probe_id: int,
        spectrum_type: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return one MFDB probe spectrum as ``(wavelengths, values)`` arrays."""
        rec = self.db.get_spectrum_record(probe_id, spectrum_type)
        if rec is None:
            return None
        return (rec["wavelengths"], rec["intensity_values"])

    def get_standardized_optical_properties(self, probe_id: int) -> dict[str, Any]:
        """Return simulator-relevant optical properties under canonical keys."""
        rows = self.db.get_optical_properties(probe_id)
        props = {}
        for raw_row in rows:
            row = dict(raw_row)
            name = row.get("property_name")
            val = row.get("property_value")
            if name in ("ext_coeff", "extinction_coefficient"):
                props["ext_coeff"] = val
            elif name in ("qy", "quantum_yield"):
                props["qy"] = val
        return props

    def get_probe_by_id(self, probe_id: int) -> dict[str, Any] | None:
        """Return one probe row as a dictionary."""
        row = self.db.get_probe(probe_id)
        if row is None:
            return None
        return dict(row)


def serialize_numpy(obj: Any) -> Any:
    """Recursively serialize NumPy values into JSON-compatible objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_numpy(v) for v in obj]
    return obj


def json_dumps(data: Any) -> str:
    """Serialize lightpath payloads consistently for embedded MFDB JSON."""
    return json.dumps(serialize_numpy(data), sort_keys=True, separators=(",", ":"))


def resolve_db_path(db_path: str | None = None) -> str:
    """Resolve the MFDB database used by light-path spectra workflows."""
    if db_path:
        return str(db_path)

    configured = _configured_spectra_db_path()
    if configured:
        return str(configured)

    return str(resolve_database_path())


def _configured_spectra_db_path() -> Path | None:
    """Return an optional LPS spectra database path from settings."""
    try:
        from chisurf.core.settings import cs_settings
    except Exception:
        return None

    for section_name in ("lightpath_simulator", "mfdb"):
        section = cs_settings.get(section_name) or {}
        configured = section.get("spectra_db_path") or section.get("spectra_database")
        if configured:
            path = Path(str(configured)).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            return path
    return None





def _utc_stamp() -> str:
    """Return a compact UTC timestamp for operation IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _simulate_with_db(graph: dict[str, Any], db: MFDatabase) -> dict[str, Any]:
    """Run the simulator against an open MFDB connection."""
    adapter = MFDatabaseAdapter(db)
    sim = OpticalPathSimulator(adapter)
    sim.load_from_dict(graph)
    states = sim.propagate()
    setting = sim.to_instrument_setting()

    serialized_states = {}
    for node_id, ns in states.items():
        serialized_states[node_id] = {
            "id": ns.id,
            "node_type": ns.node_type,
            "config": serialize_numpy(ns.config),
            "input_spectra": serialize_numpy(ns.input_spectra),
            "output_spectra": serialize_numpy(ns.output_spectra),
            "node_char": serialize_numpy(ns.node_char),
        }
    return {
        "states": serialized_states,
        "detector_signals": serialize_numpy(sim.get_detector_signals()),
        "crosstalk_matrices": serialize_numpy(sim.get_crosstalk_matrices()),
        "instrument_setting": setting.to_dict() if setting else None,
    }


def simulate_lightpath(
    graph: dict[str, Any],
    db_path: str | None = None,
) -> dict[str, Any]:
    """Run a light-path simulation for a JSON graph."""
    with MFDatabase(resolve_db_path(db_path)) as db:
        return _simulate_with_db(graph, db)


def save_lightpath(
    graph: dict[str, Any],
    name: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Persist a graph and its simulated outputs as MFDB artifacts."""
    with MFDatabase(resolve_db_path(db_path)) as db:
        result = _simulate_with_db(graph, db)
        operation_id = f"lightpath_{_utc_stamp()}_{uuid.uuid4().hex[:8]}"
        base_metadata = {
            "analysis_type": "lightpath_simulation",
            "name": name or "Light path simulation",
            "source": "chisurf.plugins.core.lightpath_simulator",
        }
        graph_artifact_id = f"{operation_id}_graph"
        setting_artifact_id = f"{operation_id}_instrument_setting"
        signals_artifact_id = f"{operation_id}_detector_signals"
        matrices_artifact_id = f"{operation_id}_crosstalk_matrices"
        output_artifacts = [
            {
                "artifact_id": graph_artifact_id,
                "artifact_kind": "project_snapshot",
                "storage_mode": "embedded_json",
                "data_format": "json",
                "data_json": json_dumps(graph),
                "metadata": {**base_metadata, "payload": "graph"},
                "role": "graph",
                "ordinal": 0,
            },
            {
                "artifact_id": setting_artifact_id,
                "artifact_kind": "analysis_result",
                "storage_mode": "embedded_json",
                "data_format": "json",
                "data_json": json_dumps(result["instrument_setting"]),
                "metadata": {**base_metadata, "payload": "instrument_setting"},
                "role": "instrument_setting",
                "ordinal": 1,
            },
            {
                "artifact_id": signals_artifact_id,
                "artifact_kind": "analysis_result",
                "storage_mode": "embedded_json",
                "data_format": "json",
                "data_json": json_dumps(result["detector_signals"]),
                "metadata": {**base_metadata, "payload": "detector_signals"},
                "role": "detector_signals",
                "ordinal": 2,
            },
            {
                "artifact_id": matrices_artifact_id,
                "artifact_kind": "analysis_result",
                "storage_mode": "embedded_json",
                "data_format": "json",
                "data_json": json_dumps(result["crosstalk_matrices"]),
                "metadata": {**base_metadata, "payload": "crosstalk_matrices"},
                "role": "crosstalk_matrices",
                "ordinal": 3,
            },
        ]
        record = db.record_operation_with_artifacts(
            operation_id=operation_id,
            operation_type="analysis",
            status="succeeded",
            software_package="chisurf",
            software_module="lightpath_simulator",
            settings={"graph_artifact_id": graph_artifact_id},
            output_artifacts=output_artifacts,
            metadata={
                **base_metadata,
                "graph_artifact_id": graph_artifact_id,
                "instrument_setting_artifact_id": setting_artifact_id,
                "detector_signals_artifact_id": signals_artifact_id,
                "crosstalk_matrices_artifact_id": matrices_artifact_id,
            },
        )
        return {
            "operation_id": operation_id,
            "artifacts": {
                "graph": graph_artifact_id,
                "instrument_setting": setting_artifact_id,
                "detector_signals": signals_artifact_id,
                "crosstalk_matrices": matrices_artifact_id,
            },
            "record": record,
            **result,
        }


def list_lightpaths(db_path: str | None = None) -> dict[str, Any]:
    """List saved light-path simulations from MFDB."""
    with MFDatabase(resolve_db_path(db_path)) as db:
        rows = db.get_operations(operation_type="analysis", status="succeeded")
        simulations = []
        for row in rows:
            item = dict(row)
            metadata = json.loads(item.get("metadata_json") or "{}")
            if metadata.get("analysis_type") != "lightpath_simulation":
                continue
            simulations.append(
                {
                    "operation_id": item["operation_id"],
                    "name": metadata.get("name") or item["operation_id"],
                    "created_at": item.get("created_at"),
                    "graph_artifact_id": metadata.get("graph_artifact_id"),
                }
            )
        return {"simulations": simulations}


def get_lightpath(operation_id: str, db_path: str | None = None) -> dict[str, Any]:
    """Return the graph and saved outputs for one light-path simulation."""
    with MFDatabase(resolve_db_path(db_path)) as db:
        operation = db.get_operation(operation_id)
        if operation is None:
            raise KeyError(f"Operation not found: {operation_id}")
        metadata = json.loads(operation.get("metadata_json") or "{}")
        if metadata.get("analysis_type") != "lightpath_simulation":
            raise ValueError(f"Not a lightpath simulation: {operation_id}")
        artifacts = {}
        for role, metadata_key in (
            ("graph", "graph_artifact_id"),
            ("instrument_setting", "instrument_setting_artifact_id"),
            ("detector_signals", "detector_signals_artifact_id"),
            ("crosstalk_matrices", "crosstalk_matrices_artifact_id"),
        ):
            artifact_id = metadata.get(metadata_key)
            if not artifact_id:
                artifacts[role] = None
                continue
            artifact = db.get_artifact(artifact_id)
            artifacts[role] = json.loads((artifact or {}).get("data_json") or "null")
        return {
            "operation_id": operation_id,
            "name": metadata.get("name"),
            "graph": artifacts["graph"],
            "instrument_setting": artifacts["instrument_setting"],
            "detector_signals": artifacts["detector_signals"],
            "crosstalk_matrices": artifacts["crosstalk_matrices"],
            "artifacts": artifacts,
        }


def _spectra_types_by_probe_id(db: MFDatabase) -> dict[int, set[str]]:
    """Return available spectrum types keyed by MFDB probe id.

    MFDB databases in the wild use ``id`` as the spectra primary key, while
    older repository helpers still assume ``spectrum_id``.  The lightpath
    palette only needs probe ownership and spectrum type, so query those stable
    fields directly.
    """
    columns = {
        row["name"]
        for row in db.conn.execute("PRAGMA table_info(spectra)").fetchall()
    }
    if "probe_id" in columns:
        probe_column = "probe_id"
    elif "item_id" in columns:
        probe_column = "item_id"
    else:
        return {}

    where = [f"{probe_column} IS NOT NULL"]
    if "deleted_at" in columns:
        where.append("deleted_at IS NULL")
    rows = db.conn.execute(
        f"""SELECT {probe_column} AS probe_id, spectrum_type
            FROM spectra
            WHERE {' AND '.join(where)}"""
    ).fetchall()

    by_probe: dict[int, set[str]] = {}
    for row in rows:
        spectrum_type = row["spectrum_type"]
        if spectrum_type:
            by_probe.setdefault(int(row["probe_id"]), set()).add(str(spectrum_type))
    return by_probe


def get_probes_info(db_path: str | None = None) -> dict[str, Any]:
    """Return probe names, ids, spectra availability, QY, and extinction."""
    with MFDatabase(resolve_db_path(db_path)) as db:
        probes = db.get_probes()
        spectra_by_probe = _spectra_types_by_probe_id(db)
        res = []
        for p in probes:
            probe = dict(p)
            probe_id = probe["probe_id"]
            types = spectra_by_probe.get(int(probe_id), set())

            qy = 1.0
            ec = 1.0
            for raw_row in db.get_optical_properties(probe_id):
                row = dict(raw_row)
                pname = row.get("property_name")
                val = row.get("property_value")
                if pname in ("qy", "quantum_yield") and val is not None:
                    try:
                        qy = float(val)
                    except ValueError:
                        pass
                elif pname in ("ext_coeff", "extinction_coefficient") and val is not None:
                    try:
                        ec = float(val)
                    except ValueError:
                        pass

            res.append(
                {
                    "probe_id": probe_id,
                    "name": probe.get("chromophore_name") or probe.get("name") or f"Probe {probe_id}",
                    "category": probe.get("category", "other"),
                    "has_abs": "absorption" in types,
                    "has_em": "emission" in types,
                    "has_trans": "transmission" in types,
                    "has_qe": "quantum_efficiency" in types,
                    "qy": qy,
                    "ec": ec,
                }
            )
        return {"probes": res}
