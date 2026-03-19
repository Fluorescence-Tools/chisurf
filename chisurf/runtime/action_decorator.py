from __future__ import annotations
import functools
import inspect
import threading
from chisurf import typing

_threading_local = threading.local()


def is_dispatching() -> bool:
    """Check if an action is currently being dispatched on the current thread."""
    return getattr(_threading_local, "is_dispatching", False)


def action(
    name: str,
    schema: typing.Optional[typing.Dict[str, typing.Any]] = None,
    replayable: bool = True,
    debounce_ms: int = 0,
    side_effect_class: str = "state",
):
    """
    Decorator to register an action.

    Parameters
    ----------
    name : str
        Unique dot-separated action name, e.g. ``"parameter.value"``.
    schema : dict, optional
        Map of required payload keys to their expected types (or ``None`` for any).
    replayable : bool
        Whether the action can be replayed from history.
    debounce_ms : int
        Coalesce repeated identical calls within this window (milliseconds).
        ``0`` disables debouncing — every call is recorded independently.
    side_effect_class : str
        Broad category of side-effect: ``"state"``, ``"execution"``, ``"diagnostic"``.
    """
    def decorator(func: typing.Callable[..., typing.Any]):
        from chisurf.runtime.actions import ActionSpec
        import chisurf

        effective_schema = schema
        if effective_schema is None:
            effective_schema = {}
            sig = inspect.signature(func)
            for pname, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    ann = param.annotation
                    if ann is not inspect.Parameter.empty:
                        effective_schema[pname] = ann

        spec = ActionSpec(
            name=name,
            schema=effective_schema,
            replayable=replayable,
            debounce_ms=debounce_ms,
            side_effect_class=side_effect_class,
            handler=func,
        )

        # Register immediately if the registry is already available
        try:
            registry = getattr(chisurf, "action_registry", None)
            if registry is not None:
                registry.register(spec)
        except (AttributeError, ImportError):
            pass

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If already inside dispatcher.execute, run the function directly
            # to avoid re-entering dispatch.
            if getattr(wrapper, "_executing", False):
                return func(*args, **kwargs)

            # Build payload from call signature and route through dispatch
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            payload = dict(bound.arguments)

            wrapper._executing = True
            try:
                return dispatch(name, payload)
            finally:
                wrapper._executing = False

        wrapper._action_spec = spec
        wrapper._executing = False
        return wrapper

    return decorator


def dispatch(name: str, payload: typing.Optional[typing.Dict[str, typing.Any]] = None):
    """Dispatch an action by name with a payload dict."""
    import chisurf
    dispatcher = getattr(chisurf, "action_dispatcher", None)
    if dispatcher:
        was = getattr(_threading_local, "is_dispatching", False)
        _threading_local.is_dispatching = True
        try:
            return dispatcher.execute(name, payload)
        finally:
            _threading_local.is_dispatching = was
    raise RuntimeError("Action dispatcher not initialized")
