from __future__ import annotations

from chisurf.history.core import OperationHistory
from chisurf.history.projection import DomainState, build_target_state
from chisurf.history.replay import (
    capture_domain_snapshot,
    reconstruct_fit_range_state,
    reconstruct_model_state,
    reconstruct_navigation_state,
    reconstruct_parameter_state,
    reconstruct_setup_state,
    snapshot_to_replay_state,
    sync_domain_entities,
    touched_parameter_keys,
)

__all__ = [
    "OperationHistory",
    "reconstruct_navigation_state",
    "reconstruct_parameter_state",
    "reconstruct_fit_range_state",
    "reconstruct_setup_state",
    "reconstruct_model_state",
    "capture_domain_snapshot",
    "snapshot_to_replay_state",
    "sync_domain_entities",
    "touched_parameter_keys",
    "DomainState",
    "build_target_state",
    "get_history",
]


# ---------------------------------------------------------------------------
# Package / singleton name-collision shim.
#
# ``chisurf.history`` is BOTH this subpackage and the name under which
# ``chisurf/__init__.py`` exposes the process-wide ``OperationHistory`` singleton
# (``cs.history``).  Once any ``from chisurf.history import ...`` runs, Python
# binds the ``history`` attribute on the ``chisurf`` package to *this module*,
# permanently shadowing the lazy singleton attribute — so ``cs.history`` becomes
# the module and ``cs.history.record(...)`` / ``.subscribe(...)`` silently fail.
#
# To make that collision benign, the singleton lives HERE and the module
# delegates any OperationHistory attribute access (record/list_events/subscribe/
# set_checkpoint_capture/...) to it.  ``chisurf/__init__.py`` returns this same
# instance, so there is exactly one history object whether ``cs.history`` resolves
# to the module or to the instance.
# ---------------------------------------------------------------------------

_history_singleton: OperationHistory | None = None


def get_history() -> OperationHistory:
    """Return the process-wide :class:`OperationHistory` singleton."""
    global _history_singleton
    if _history_singleton is None:
        _history_singleton = OperationHistory()
    return _history_singleton


def __getattr__(name: str):
    # Only fires for attributes this module does not already define. Delegate
    # OperationHistory instance members so ``cs.history.<method>`` works even when
    # ``cs.history`` resolved to this module rather than the singleton instance.
    obj = get_history()
    if hasattr(obj, name):
        return getattr(obj, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
