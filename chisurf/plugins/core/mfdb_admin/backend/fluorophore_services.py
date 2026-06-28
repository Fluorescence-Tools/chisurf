"""ZMQ RPC backend services for the Fluorophore DB plugin.

Registered via the manifest's ``services`` entrypoint. Every handler
receives ``auth`` as the last keyword argument (injected by the
dispatcher).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.database_resolver import resolve_database_path


@contextlib.contextmanager
def _db():
    """Open the configured MFDB for one handler call, closing it afterwards.

    Matches the per-call ``with MFDatabase(resolve_database_path()) as db:``
    lifecycle used by the other mfdb-admin services. The old non-context form
    leaked a SQLite connection per call, which stalled the GUI's blocking RPC
    against the embedded server.
    """
    with MFDatabase(resolve_database_path()) as db:
        yield db


def handle_list_probes(
    verification_status: str | None = None,
    category: str | None = None,
    source: str | None = None,
    search: str | None = None,
    limit: int = 200,
    offset: int = 0,
    auth: dict | None = None,
) -> dict[str, Any]:
    """List fluorophore probes with optional filtering."""
    clauses = ["p.deleted_at IS NULL"]
    params: list[Any] = []
    if verification_status:
        clauses.append("p.verification_status = ?")
        params.append(verification_status)
    if category:
        clauses.append("p.category = ?")
        params.append(category)
    if source:
        clauses.append("p.source = ?")
        params.append(source)
    if search:
        clauses.append("p.chromophore_name LIKE ?")
        params.append(f"%{search}%")
    where = " AND ".join(clauses)
    # Surface the common optical properties (stored in optical_properties, not on
    # the probe row) so the list table can show Abs max / Em max / QY.
    op = (
        "(SELECT property_value FROM optical_properties o "
        "WHERE o.probe_id = p.probe_id AND o.property_name = ? AND o.deleted_at IS NULL LIMIT 1)"
    )
    with _db() as db:
        total = db.conn.execute(
            f"SELECT COUNT(*) FROM probes p WHERE {where}", params
        ).fetchone()[0]
        rows = db.conn.execute(
            f"SELECT p.*, {op} AS abs_max, {op} AS em_max, {op} AS qy "
            f"FROM probes p WHERE {where} ORDER BY p.chromophore_name LIMIT ? OFFSET ?",
            ["abs_max", "em_max", "qy"] + params + [limit, offset],
        ).fetchall()
    return {
        "probes": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def handle_get_probe(
    probe_id: int,
    auth: dict | None = None,
) -> dict[str, Any]:
    """Get a single probe with its optical properties and spectra."""
    with _db() as db:
        probe = db.conn.execute(
            "SELECT * FROM probes WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        ).fetchone()
        if not probe:
            raise ValueError(f"Probe {probe_id} not found")
        props = db.conn.execute(
            "SELECT * FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        ).fetchall()
        spectra = db.conn.execute(
            "SELECT * FROM spectra WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        ).fetchall()
    return {
        "probe": dict(probe),
        "optical_properties": [dict(p) for p in props],
        "spectra": [
            {
                "spectrum_type": s["spectrum_type"],
                "wavelengths": list(np.frombuffer(s["wavelengths"], dtype=np.float64)),
                "intensity": list(np.frombuffer(s["intensity_values"], dtype=np.float64)),
                "wavelength_unit": s["wavelength_unit"] or "nm",
                "intensity_unit": s["intensity_unit"] or "normalized",
            }
            for s in spectra
        ],
    }


def handle_approve_probe(
    probe_id: int,
    verified_by: str = "admin",
    auth: dict | None = None,
) -> dict[str, Any]:
    """Approve a fluorophore probe."""
    with _db() as db:
        db.approve_probe(probe_id, verified_by=verified_by)
    return {"ok": True, "probe_id": probe_id, "verification_status": "approved"}


def handle_reject_probe(
    probe_id: int,
    verified_by: str = "admin",
    auth: dict | None = None,
) -> dict[str, Any]:
    """Reject a fluorophore probe."""
    with _db() as db:
        db.reject_probe(probe_id, verified_by=verified_by)
    return {"ok": True, "probe_id": probe_id, "verification_status": "rejected"}


def handle_set_probe_quality(
    probe_id: int,
    quality: str,
    auth: dict | None = None,
) -> dict[str, Any]:
    """Set the quality level of a probe."""
    with _db() as db:
        db.set_probe_quality(probe_id, quality)
    return {"ok": True, "probe_id": probe_id, "quality": quality}


def handle_import_reference_set(
    mark_verified: bool = False,
    auth: dict | None = None,
) -> dict[str, Any]:
    """Import fluorophore reference data from the bundled spectra.db."""
    with _db() as db:
        counts = db.import_reference_set(mark_verified=mark_verified)
    return {"ok": True, **counts}


def handle_lookup_forster_radius(
    donor_name: str,
    acceptor_name: str,
    auth: dict | None = None,
) -> dict[str, Any]:
    """Look up the Förster radius R0 for a donor-acceptor pair."""
    with _db() as db:
        r0 = db.lookup_forster_radius(donor_name, acceptor_name)
    return {"forster_radius": r0}


def handle_run_ai_triage(
    probe_id: int,
    auth: dict | None = None,
) -> dict[str, Any]:
    """Run AI-assisted triage on a single probe."""
    from chisurf.core.fluorescence.curation.ai_triage import run_deterministic_checks

    with _db() as db:
        probe = db.conn.execute(
            "SELECT * FROM probes WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        ).fetchone()
        if not probe:
            raise ValueError(f"Probe {probe_id} not found")

        props = db.conn.execute(
            "SELECT property_name, property_value FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        ).fetchall()
        prop_dict = {r["property_name"]: r["property_value"] for r in props}

        has_abs = db.conn.execute(
            "SELECT 1 FROM spectra WHERE probe_id = ? AND spectrum_type = 'absorption' AND deleted_at IS NULL",
            (probe_id,),
        ).fetchone() is not None
        has_em = db.conn.execute(
            "SELECT 1 FROM spectra WHERE probe_id = ? AND spectrum_type = 'emission' AND deleted_at IS NULL",
            (probe_id,),
        ).fetchone() is not None

        probe_data = {
            "name": probe["chromophore_name"],
            "abs_max": prop_dict.get("abs_max", ""),
            "em_max": prop_dict.get("em_max", ""),
            "qy": prop_dict.get("qy", ""),
            "ext_coeff": prop_dict.get("ext_coeff", ""),
            "has_abs": has_abs,
            "has_em": has_em,
        }

        result = run_deterministic_checks(probe_data)

        # AI proposes, human disposes (PRD-06 Task 9): triage never auto-approves.
        # Record the proposed quality and queue the probe for human review;
        # approval stays an explicit human action via ``fluorophores.approve``.
        db.set_probe_quality(probe_id, result["proposed_quality"])
        db.conn.execute(
            "UPDATE probes SET verification_status = 'needs_review' "
            "WHERE probe_id = ? AND verification_status != 'approved'",
            (probe_id,),
        )
        db.conn.commit()

    return {
        "probe_id": probe_id,
        "issues": result["issues"],
        "proposed_quality": result["proposed_quality"],
        "applied": False,
    }


def handle_list_probe_types(
    auth: dict | None = None,
) -> dict[str, Any]:
    """List all available probe types."""
    with _db() as db:
        types = db.get_probe_types()
    return {"probe_types": [dict(r) for r in types]}


def register_services(dispatcher: Any) -> None:
    """Register all fluorophore RPC handlers with the dispatcher.

    The mfdb-admin ``ServiceDispatcher`` invokes a handler with a single
    positional ``params`` dict, so each keyword-style handler is wrapped to
    expand ``**params`` (matching the convention used by the other mfdb-admin
    service registrations).
    """
    def _kw(handler):
        return lambda params, _h=handler: _h(**(params or {}))

    dispatcher.register("fluorophores.list", _kw(handle_list_probes))
    dispatcher.register("fluorophores.get", _kw(handle_get_probe))
    dispatcher.register("fluorophores.approve", _kw(handle_approve_probe))
    dispatcher.register("fluorophores.reject", _kw(handle_reject_probe))
    dispatcher.register("fluorophores.set_quality", _kw(handle_set_probe_quality))
    dispatcher.register("fluorophores.import_reference_set", _kw(handle_import_reference_set))
    dispatcher.register("fluorophores.forster_radius.lookup", _kw(handle_lookup_forster_radius))
    dispatcher.register("fluorophores.ai_triage", _kw(handle_run_ai_triage))
    dispatcher.register("fluorophores.probe_types.list", _kw(handle_list_probe_types))
