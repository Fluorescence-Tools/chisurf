"""Pure, Qt-free replay projection.

Reconstructs the target domain state for a given history cursor position by
seeding from a checkpoint snapshot (when available) and merging the deltas
reconstructed from the events that follow it.

This module is deliberately free of any GUI dependency: it consumes history
events and a checkpoint snapshot and returns a :class:`DomainState` of plain
dicts.  Applying that state to live widgets is the job of the GUI adapter
(``chisurf/gui/main_helper.py``).  Keeping the merge here makes the whole
replay path testable headless and reusable for server-side reconstruction.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from chisurf.history import replay as _hr


@dataclass
class DomainState:
    """Reconstructed scientific state at a history cursor position.

    Each field is a plain JSON-serializable dict produced by the
    ``reconstruct_*_state`` functions in :mod:`chisurf.history.replay`.
    """

    navigation: dict[str, typing.Any] = field(default_factory=dict)
    parameters: dict[str, typing.Any] = field(default_factory=dict)
    fit_ranges: dict[str, typing.Any] = field(default_factory=dict)
    setup: dict[str, typing.Any] = field(default_factory=dict)
    models: dict[str, typing.Any] = field(default_factory=dict)


def build_target_state(
    checkpoint_snapshot: dict[str, typing.Any] | None,
    events_to_replay: list[dict[str, typing.Any]],
    all_events: list[dict[str, typing.Any]] | None = None,
) -> DomainState:
    """Reconstruct the :class:`DomainState` for a history cursor position.

    Parameters
    ----------
    checkpoint_snapshot:
        A checkpoint snapshot dict (as returned by
        :meth:`OperationHistory.get_events_from_checkpoint`) to seed the state,
        or ``None`` to reconstruct purely from ``events_to_replay``.
    events_to_replay:
        Events to replay on top of the checkpoint.  When ``checkpoint_snapshot``
        is ``None`` this is the full list of events up to the cursor.
    all_events:
        Reserved for future use (e.g. folding ``sync_domain_entities`` into the
        pure layer).  Currently unused by the merge.

    Returns
    -------
    DomainState
        The merged navigation / parameter / fit-range / setup / model state.
    """
    del all_events  # reserved; the merge does not need it

    if checkpoint_snapshot is not None:
        replay_state = _hr.snapshot_to_replay_state(checkpoint_snapshot)
        nav_state = replay_state.get("navigation", {})
        parameter_state = replay_state.get("parameters", {})
        fit_range_state = replay_state.get("fit_ranges", {})
        setup_state = replay_state.get("setup", {})
        model_state = replay_state.get("models", {})
        nav_delta = _hr.reconstruct_navigation_state(events_to_replay)
        param_delta = _hr.reconstruct_parameter_state(events_to_replay)
        range_delta = _hr.reconstruct_fit_range_state(events_to_replay)
        setup_delta = _hr.reconstruct_setup_state(events_to_replay)
        model_delta = _hr.reconstruct_model_state(events_to_replay)
        for key in ["datasets", "dataset_uids", "fits", "fit_uids"]:
            if key in nav_delta:
                nav_state[key] = nav_delta[key]
        if nav_delta.get("selected_dataset"):
            nav_state["selected_dataset"] = nav_delta["selected_dataset"]
        if nav_delta.get("selected_dataset_uid"):
            nav_state["selected_dataset_uid"] = nav_delta["selected_dataset_uid"]
        if nav_delta.get("selected_fit"):
            nav_state["selected_fit"] = nav_delta["selected_fit"]
        if nav_delta.get("selected_fit_uid"):
            nav_state["selected_fit_uid"] = nav_delta["selected_fit_uid"]
        parameter_state.update(param_delta)
        fit_range_state.update(range_delta)
        setup_state.update(setup_delta)
        for fg_uid, fg_data in model_delta.items():
            if fg_uid not in model_state:
                model_state[fg_uid] = fg_data
            else:
                for local_uid, local_data in fg_data.get("local_fits", {}).items():
                    if local_uid not in model_state[fg_uid].get("local_fits", {}):
                        if "local_fits" not in model_state[fg_uid]:
                            model_state[fg_uid]["local_fits"] = {}
                        model_state[fg_uid]["local_fits"][local_uid] = local_data
                    else:
                        existing = model_state[fg_uid]["local_fits"][local_uid]
                        if "components" in local_data:
                            if "components" not in existing:
                                existing["components"] = []
                            for comp in local_data["components"]:
                                comp_name = comp.get("name", "")
                                found = False
                                for i, existing_comp in enumerate(existing["components"]):
                                    if existing_comp.get("name") == comp_name:
                                        existing["components"][i] = comp
                                        found = True
                                        break
                                if not found:
                                    existing["components"].append(comp)
                        if "config" in local_data:
                            if "config" not in existing:
                                existing["config"] = {}
                            existing["config"].update(local_data["config"])
    else:
        nav_state = _hr.reconstruct_navigation_state(events_to_replay)
        parameter_state = _hr.reconstruct_parameter_state(events_to_replay)
        fit_range_state = _hr.reconstruct_fit_range_state(events_to_replay)
        setup_state = _hr.reconstruct_setup_state(events_to_replay)
        model_state = _hr.reconstruct_model_state(events_to_replay)

    return DomainState(
        navigation=nav_state,
        parameters=parameter_state,
        fit_ranges=fit_range_state,
        setup=setup_state,
        models=model_state,
    )
