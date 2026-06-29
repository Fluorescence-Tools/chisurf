"""Central RPC service for detector / channel setup definitions.

This is the transport-agnostic, Qt-free access layer over the canonical
detector-setup store (``detector_setups.json`` / MFDB, managed by
``chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups``).

It serves two purposes:

* **Persistent named setups** — ``list_setups`` / ``get_setup`` / ``save_setup``
  expose the on-disk / MFDB store so any plugin (or the planned
  ``SetupSelectorWidget``) can read and write named detector setups over RPC
  instead of embedding the full :class:`DetectorWizardPage`.
* **Session-active definition** — ``get_current`` / ``set_current`` hold the
  *live* (possibly unsaved) detector definition in
  ``state.plugins["detector_setups"]["active"]`` so coordinating tools (e.g. the
  Imaging Tools window) can broadcast the current setup to their sub-tools
  without forcing the user to save a named setup first.

The underlying ``tttr_detector_setups`` module is import-time Qt-free; only its
missing-file warning dialog imports Qt lazily, so importing this service in the
headless server is safe.
"""

from __future__ import annotations

from typing import Any

from chisurf.server.services import (
    INVALID_INPUT,
    OPERATION_FAILED,
    ServiceResult,
    service_error,
)
from chisurf.server.session import SessionState

_PLUGIN_KEY = "detector_setups"


def _store(state: SessionState) -> dict[str, Any]:
    """Return (creating if needed) this service's slot in session state."""
    slot = state.plugins.get(_PLUGIN_KEY)
    if not isinstance(slot, dict):
        slot = {}
        state.plugins[_PLUGIN_KEY] = slot
    return slot


def _load_all(file_path: str | None = None) -> dict[str, Any]:
    """Load ``{"setups": {...}, "last_used": str}`` from the canonical store."""
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
        load_detector_setups,
    )

    data = load_detector_setups(file_path=file_path) or {}
    setups = data.get("setups") or {}
    if not isinstance(setups, dict):
        setups = {}
    return {"setups": setups, "last_used": data.get("last_used", "") or ""}


def list_setups(state: SessionState, file_path: str | None = None) -> ServiceResult:
    """List the names of all saved detector setups.

    Returns
    -------
    dict
        ``{"ok": True, "result": {"setups": [names], "last_used": name}}``.
    """
    try:
        data = _load_all(file_path)
    except Exception as exc:  # pragma: no cover - depends on MFDB/disk
        return service_error(
            f"Failed to load detector setups: {exc}",
            error_code=OPERATION_FAILED,
            exception=exc,
        )
    names: list[str] = sorted(data["setups"].keys())
    return {"ok": True, "result": {"setups": names, "last_used": data["last_used"]}}


def get_setup(
    state: SessionState,
    name: str | None = None,
    file_path: str | None = None,
) -> ServiceResult:
    """Return the full settings dict for one saved setup.

    When ``name`` is ``None`` the session-active definition is returned (falling
    back to the persisted ``last_used`` setup).
    """
    if name is None:
        active = _store(state).get("active")
        if isinstance(active, dict) and active:
            return {"ok": True, "result": dict(active)}
    try:
        data = _load_all(file_path)
    except Exception as exc:  # pragma: no cover - depends on MFDB/disk
        return service_error(
            f"Failed to load detector setups: {exc}",
            error_code=OPERATION_FAILED,
            exception=exc,
        )
    resolved = name if name is not None else data["last_used"]
    if not resolved:
        return {"ok": True, "result": {}}
    setup = data["setups"].get(resolved)
    if setup is None:
        return service_error(
            f"Detector setup not found: {resolved!r}",
            error_code=INVALID_INPUT,
        )
    return {"ok": True, "result": dict(setup)}


def save_setup(
    state: SessionState,
    name: str,
    settings: dict[str, Any],
    is_public: bool | None = None,
    file_path: str | None = None,
) -> ServiceResult:
    """Persist ``settings`` under ``name`` in the canonical setup store."""
    if not name:
        return service_error("A setup name is required.", error_code=INVALID_INPUT)
    if not isinstance(settings, dict):
        return service_error("settings must be an object.", error_code=INVALID_INPUT)
    try:
        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
            save_detector_setups,
        )

        payload = {"setups": {name: dict(settings)}, "last_used": name}
        save_detector_setups(payload, file_path=file_path, is_public=is_public)
    except Exception as exc:  # pragma: no cover - depends on MFDB/disk
        return service_error(
            f"Failed to save detector setup {name!r}: {exc}",
            error_code=OPERATION_FAILED,
            exception=exc,
        )
    # Mirror the saved definition into the session-active slot.
    _store(state)["active"] = dict(settings)
    return {"ok": True, "result": {"name": name}}


def get_current(state: SessionState, file_path: str | None = None) -> ServiceResult:
    """Return the live session-active detector definition.

    Falls back to the persisted ``last_used`` setup when no live definition has
    been published this session.
    """
    active = _store(state).get("active")
    if isinstance(active, dict) and active:
        return {"ok": True, "result": dict(active)}
    return get_setup(state, name=None, file_path=file_path)


def set_current(
    state: SessionState,
    settings: dict[str, Any] | None = None,
    name: str | None = None,
    file_path: str | None = None,
) -> ServiceResult:
    """Publish the live session-active detector definition.

    Pass ``settings`` to store an explicit (possibly unsaved) definition, or
    ``name`` to activate a saved setup by name.  This is the live broadcast
    channel used by coordinating tools to push the current setup to sub-tools.
    """
    if settings is not None:
        if not isinstance(settings, dict):
            return service_error("settings must be an object.", error_code=INVALID_INPUT)
        _store(state)["active"] = dict(settings)
        return {"ok": True, "result": {"active": True}}
    if name is not None:
        resolved = get_setup(state, name=name, file_path=file_path)
        if not resolved.get("ok"):
            return resolved
        _store(state)["active"] = dict(resolved.get("result") or {})
        return {"ok": True, "result": {"active": True, "name": name}}
    return service_error(
        "Provide either 'settings' or 'name'.", error_code=INVALID_INPUT
    )
