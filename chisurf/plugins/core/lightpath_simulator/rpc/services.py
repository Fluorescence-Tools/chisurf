"""ServiceDispatcher RPC handlers for the light-path simulator plugin."""

from __future__ import annotations

from typing import Any

from ..api.contract import (
    METHOD_DESCRIBE_CONTRACT,
    METHOD_GET,
    METHOD_GET_PROBES_INFO,
    METHOD_LIST,
    METHOD_SAVE,
    METHOD_SIMULATE,
    contract_descriptor,
    service_error,
    service_success,
)
from ..core.session import LightPathSessionState
from ..core.workflow import (
    get_lightpath,
    get_probes_info,
    list_lightpaths,
    save_lightpath,
    simulate_lightpath,
)


def register_services(dispatcher: Any) -> None:
    """Register light-path simulator RPC handlers."""
    session = _session_state(dispatcher)
    dispatcher.register(
        METHOD_SIMULATE,
        lambda params: simulate_handler(session=session, **(params or {})),
    )
    dispatcher.register(METHOD_SAVE, lambda params: save_handler(session=session, **(params or {})))
    dispatcher.register(METHOD_LIST, lambda params: list_handler(**(params or {})))
    dispatcher.register(METHOD_GET, lambda params: get_handler(**(params or {})))
    dispatcher.register(
        METHOD_GET_PROBES_INFO,
        lambda params: get_probes_info_handler(session=session, **(params or {})),
    )
    dispatcher.register(
        METHOD_DESCRIBE_CONTRACT,
        lambda params: contract_handler(**(params or {})),
    )


def _session_state(dispatcher: Any) -> LightPathSessionState | None:
    """Return the lightpath plugin namespace from the server session."""
    state = getattr(dispatcher, "_state", None)
    if state is None:
        return None

    plugins = getattr(state, "plugins", None)
    if plugins is None:
        plugins = {}
        setattr(state, "plugins", plugins)

    plugin_state = plugins.get("lightpath_simulator")
    if not isinstance(plugin_state, LightPathSessionState):
        plugin_state = LightPathSessionState()
        plugins["lightpath_simulator"] = plugin_state
    return plugin_state


def list_methods() -> dict[str, str]:
    """Return the light-path simulator RPC method catalogue."""
    return {
        METHOD_SIMULATE: "Run a light-path simulation for a JSON graph.",
        METHOD_SAVE: "Persist a light-path graph and simulated outputs in MFDB.",
        METHOD_LIST: "List saved light-path simulations from MFDB.",
        METHOD_GET: "Load one saved light-path simulation from MFDB.",
        METHOD_GET_PROBES_INFO: "Return probe metadata for the light-path simulator.",
        METHOD_DESCRIBE_CONTRACT: "Return the light-path simulator workflow contract.",
    }


def simulate_handler(
    graph: dict[str, Any],
    db_path: str | None = None,
    auth: dict[str, Any] | None = None,
    session: LightPathSessionState | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Run a light-path simulation."""
    try:
        result = simulate_lightpath(graph, db_path=db_path)
        if session is not None:
            session.record_simulation(graph, result)
        return service_success(result)
    except Exception as exc:
        return service_error(str(exc))


def save_handler(
    graph: dict[str, Any],
    name: str | None = None,
    db_path: str | None = None,
    auth: dict[str, Any] | None = None,
    session: LightPathSessionState | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Persist a graph and simulation outputs in MFDB."""
    try:
        result = save_lightpath(graph, name=name, db_path=db_path)
        if session is not None:
            session.record_simulation(graph, result)
            session.record_save(result)
        return service_success(result)
    except Exception as exc:
        return service_error(str(exc))


def list_handler(
    db_path: str | None = None,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """List saved light-path simulations."""
    try:
        return service_success(list_lightpaths(db_path=db_path))
    except Exception as exc:
        return service_error(str(exc))


def get_handler(
    operation_id: str,
    db_path: str | None = None,
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Load one saved light-path simulation."""
    try:
        return service_success(get_lightpath(operation_id, db_path=db_path))
    except Exception as exc:
        return service_error(str(exc))


def get_probes_info_handler(
    db_path: str | None = None,
    auth: dict[str, Any] | None = None,
    session: LightPathSessionState | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return probe metadata for light-path GUI palettes."""
    try:
        if session is not None:
            return service_success(session.get_probe_catalogue(db_path=db_path))
        return service_success(get_probes_info(db_path=db_path))
    except Exception as exc:
        return service_error(str(exc))


def contract_handler(
    auth: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return the plugin workflow contract descriptor."""
    return service_success(contract_descriptor())
