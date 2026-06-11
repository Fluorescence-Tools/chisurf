"""FLR database service — wraps FluorophoreDatabase behind JSON-RPC.

All methods are exposed via a single generic ``flr.call`` entry point
for flexibility, plus dedicated methods for the most common operations
used by FitInfo.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from chisurf.server.services import ServiceResult, service_error, NOT_FOUND, OPERATION_FAILED
from chisurf.server.session import SessionState

# ── helpers ──────────────────────────────────────────────────────────


def _db(state: SessionState):
    db = state.flr_database
    if db is None:
        raise RuntimeError("No FLR database configured on this server")
    return db


def _ensure_analysis(state: SessionState, analysis_id: str):
    db = _db(state)
    db.update_analysis_record(analysis_id)
    return db


# ── dedicated RPC methods ────────────────────────────────────────────


def flr_metadata_get(state: SessionState, analysis_id: str = "analysis_1", **kwargs) -> ServiceResult:
    try:
        db = _db(state)
        meta = db.get_analysis_metadata(analysis_id)
        return {"ok": True, "metadata": meta}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_metadata_set(state: SessionState, analysis_id: str = "analysis_1", metadata: Optional[Dict[str, str]] = None, **kwargs) -> ServiceResult:
    try:
        db = _ensure_analysis(state, analysis_id)
        db.set_analysis_metadata(analysis_id, metadata or {})
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_metadata_add(state: SessionState, analysis_id: str = "analysis_1", key: str = "", value: str = "", **kwargs) -> ServiceResult:
    try:
        db = _ensure_analysis(state, analysis_id)
        db.add_analysis_metadata(analysis_id, key, value)
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_metadata_delete(state: SessionState, analysis_id: str = "analysis_1", key: str = "", **kwargs) -> ServiceResult:
    try:
        db = _ensure_analysis(state, analysis_id)
        db.delete_analysis_metadata(analysis_id, key)
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_analysis_update(state: SessionState, analysis_id: str = "analysis_1", **kwargs) -> ServiceResult:
    try:
        db = _ensure_analysis(state, analysis_id)
        update_kwargs = {k: v for k, v in kwargs.items() if k != "analysis_id"}
        db.update_analysis_record(analysis_id, **update_kwargs)
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_photon_stream_add(state: SessionState, analysis_id: str = "analysis_1", file_path: str = "", detector_id: str = "", stream_id: str = "", file_format: str = "", **kwargs) -> ServiceResult:
    try:
        db = _ensure_analysis(state, analysis_id)
        from pathlib import Path
        db.add_photon_stream(analysis_id, Path(file_path), detector_id=detector_id, stream_id=stream_id or None, file_format=file_format or None)
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_photon_stream_list(state: SessionState, analysis_id: str = "analysis_1", **kwargs) -> ServiceResult:
    try:
        db = _db(state)
        streams = db.get_photon_streams(analysis_id)
        rows = [dict(r) for r in streams]
        return {"ok": True, "streams": rows}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_export(state: SessionState, analysis_id: str = "analysis_1", file_path: str = "", **kwargs) -> ServiceResult:
    try:
        db = _db(state)
        from pathlib import Path
        out = db.export_flr_cif(Path(file_path), analysis_id=analysis_id)
        return {"ok": True, "path": str(out)}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_probe_types(state: SessionState, **kwargs) -> ServiceResult:
    try:
        db = _db(state)
        types = [dict(r) for r in db.get_probe_types()]
        return {"ok": True, "probe_types": types}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)


def flr_probes(state: SessionState, type_id: Optional[int] = None, **kwargs) -> ServiceResult:
    try:
        db = _db(state)
        probes = [dict(r) for r in db.get_probes(type_id=type_id)]
        return {"ok": True, "probes": probes}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED)
