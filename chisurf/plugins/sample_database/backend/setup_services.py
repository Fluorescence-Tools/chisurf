"""JSON-RPC handlers for fdb setup definitions."""

from __future__ import annotations

import uuid
from typing import Any

from chisurf.core.fio.mmcif.db import FluorophoreDatabase, resolve_database_path
from chisurf.server.services import NOT_FOUND, OPERATION_FAILED, service_error


def register_setup_services(dispatcher: Any) -> None:
    """Register fdb Phase 3 RPC handlers for setup definitions.

    Parameters
    ----------
    dispatcher : object
        Service dispatcher exposing a ``register`` method.
    """
    for name, handler in {
        "sample_database.setups.save": save_setup_handler,
        "sample_database.setups.get": get_setup_handler,
        "sample_database.setups.list": list_setups_handler,
        "sample_database.setups.delete": delete_setup_handler,
        "sample_database.setups.validate": validate_setup_handler,
    }.items():
        dispatcher.register(name, lambda params, _handler=handler: _handler(**params))


def validate_setup_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate setup configuration fields for TTTR/burst-wise workflows.

    Parameters
    ----------
    config : dict
        The configuration dictionary to validate.

    Returns
    -------
    dict
        A validation summary containing "valid" (bool) and "errors" (list of str).
    """
    errors = []

    # 1. Laser wavelengths validation
    lasers = config.get("laser_wavelengths")
    if lasers is not None:
        if not isinstance(lasers, list):
            errors.append("laser_wavelengths must be a list of numbers")
        elif len(lasers) == 0:
            errors.append("laser_wavelengths list cannot be empty")
        else:
            for i, wl in enumerate(lasers):
                if not isinstance(wl, (int, float)) or wl <= 0:
                    errors.append(f"laser_wavelengths[{i}] must be a positive number: {wl}")

    # 2. Detector definitions validation
    detectors = config.get("detector_channels")
    if detectors is not None:
        if not isinstance(detectors, (dict, list)):
            errors.append("detector_channels must be a dictionary or list representing detector mappings")

    # 3. PIE validation
    pie_enabled = config.get("pie_enabled")
    if pie_enabled:
        pie_window = config.get("pie_window")
        if pie_window is None:
            errors.append("pie_window must be defined when pie_enabled is True")
        elif not isinstance(pie_window, (dict, list)):
            errors.append("pie_window must be a dictionary or list of window bounds")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def save_setup_handler(setup: dict[str, Any]) -> dict[str, Any]:
    """Save or update a setup definition.

    Parameters
    ----------
    setup : dict
        Setup definition fields.

    Returns
    -------
    dict
        JSON-RPC result with the saved setup.
    """
    try:
        setup_id = str(setup.get("setup_id") or "").strip()
        setup_id = setup_id or f"setup_{uuid.uuid4()}"
        name = str(setup.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")

        config = setup.get("configuration") or {}
        validation = validate_setup_config(config)

        with FluorophoreDatabase(resolve_database_path()) as db:
            db.add_setup_definition(
                setup_id=setup_id,
                name=name,
                version=int(setup.get("version") or 1),
                instrument_id=setup.get("instrument_id") or None,
                description=setup.get("description") or None,
                configuration=config,
                detectors=setup.get("detectors") or None,
                timing_calibration=setup.get("timing_calibration") or None,
                irf_definition=setup.get("irf_definition") or None,
                burst_defaults=setup.get("burst_defaults") or None,
                fcs_calibration=setup.get("fcs_calibration") or None,
            )
            row = db.get_setup_definition(setup_id)
            decoded = db._decode_setup_definition_row(row)
            decoded["validation"] = validation
            return {"ok": True, "setup": decoded}
    except Exception as exc:
        return service_error(str(exc), error_code=OPERATION_FAILED, exception=exc)


def get_setup_handler(setup_id: str) -> dict[str, Any]:
    """Retrieve one setup definition.

    Parameters
    ----------
    setup_id : str
        Setup identifier.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    with FluorophoreDatabase(resolve_database_path()) as db:
        row = db.get_setup_definition(setup_id)
        if row is None:
            return service_error(f"setup not found: {setup_id}", error_code=NOT_FOUND)
        decoded = db._decode_setup_definition_row(row)
        decoded["validation"] = validate_setup_config(decoded.get("configuration") or {})
        return {"ok": True, "setup": decoded}


def list_setups_handler() -> dict[str, Any]:
    """List all setup definitions.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    with FluorophoreDatabase(resolve_database_path()) as db:
        rows = db.list_setup_definitions()
        setups = []
        for row in rows:
            decoded = db._decode_setup_definition_row(row)
            decoded["validation"] = validate_setup_config(decoded.get("configuration") or {})
            setups.append(decoded)
        return {"ok": True, "setups": setups}


def delete_setup_handler(setup_id: str) -> dict[str, Any]:
    """Delete a setup definition.

    Parameters
    ----------
    setup_id : str
        Setup identifier.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    with FluorophoreDatabase(resolve_database_path()) as db:
        db.delete_setup_definition(setup_id)
    return {"ok": True, "setup_id": setup_id}


def validate_setup_handler(configuration: dict[str, Any]) -> dict[str, Any]:
    """Validate a setup configuration.

    Parameters
    ----------
    configuration : dict
        Configuration to validate.

    Returns
    -------
    dict
        JSON-RPC result.
    """
    return validate_setup_config(configuration)
