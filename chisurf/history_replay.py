from __future__ import annotations

from chisurf import typing


def reconstruct_navigation_state(events: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
    datasets: typing.List[str] = []
    dataset_uids: typing.List[str] = []
    fits: typing.List[str] = []
    fit_uids: typing.List[str] = []
    selected_dataset: typing.Optional[str] = None
    selected_dataset_uid: typing.Optional[str] = None
    selected_fit: typing.Optional[str] = None
    selected_fit_uid: typing.Optional[str] = None

    def append_unique(lst: typing.List[str], value: str) -> None:
        v = str(value)
        if v and v not in lst:
            lst.append(v)

    def remove_values(lst: typing.List[str], values: typing.Set[str]) -> typing.List[str]:
        return [v for v in lst if v not in values]

    for event in events:
        action_type = str(event.get("action_type", ""))
        payload = event.get("payload", {}) or {}

        if action_type == "dataset_add":
            loaded = payload.get("loaded_names", [])
            loaded_uids = payload.get("loaded_uids", [])
            if isinstance(loaded, list):
                for name in loaded:
                    append_unique(datasets, str(name))
                if loaded:
                    selected_dataset = str(loaded[-1])
            if isinstance(loaded_uids, list):
                for uid in loaded_uids:
                    append_unique(dataset_uids, str(uid))
                if loaded_uids:
                    selected_dataset_uid = str(loaded_uids[-1])

        elif action_type == "dataset_group":
            name = str(payload.get("group_name", ""))
            uid = str(payload.get("group_uid", ""))
            if name:
                append_unique(datasets, name)
                selected_dataset = name
            if uid:
                append_unique(dataset_uids, uid)
                selected_dataset_uid = uid

        elif action_type == "dataset_remove":
            removed = payload.get("removed_names", [])
            removed_uids = payload.get("removed_uids", [])
            if isinstance(removed, list):
                removed_set = {str(n) for n in removed}
                datasets = remove_values(datasets, removed_set)
                if selected_dataset in removed_set:
                    selected_dataset = datasets[-1] if datasets else None
            if isinstance(removed_uids, list):
                removed_uid_set = {str(n) for n in removed_uids}
                dataset_uids = remove_values(dataset_uids, removed_uid_set)
                if selected_dataset_uid in removed_uid_set:
                    selected_dataset_uid = dataset_uids[-1] if dataset_uids else None

        elif action_type == "dataset_ungroup":
            group_names = payload.get("group_names", [])
            group_uids = payload.get("group_uids", [])
            expanded_names = payload.get("expanded_names", [])
            expanded_uids = payload.get("expanded_uids", [])
            if isinstance(group_names, list):
                remove_set = {str(n) for n in group_names}
                datasets = remove_values(datasets, remove_set)
                if selected_dataset in remove_set:
                    selected_dataset = None
            if isinstance(group_uids, list):
                remove_uid_set = {str(n) for n in group_uids}
                dataset_uids = remove_values(dataset_uids, remove_uid_set)
                if selected_dataset_uid in remove_uid_set:
                    selected_dataset_uid = None
            if isinstance(expanded_names, list):
                for name in expanded_names:
                    append_unique(datasets, str(name))
                if expanded_names:
                    selected_dataset = str(expanded_names[-1])
            if isinstance(expanded_uids, list):
                for uid in expanded_uids:
                    append_unique(dataset_uids, str(uid))
                if expanded_uids:
                    selected_dataset_uid = str(expanded_uids[-1])

        elif action_type == "fit_add":
            fit_name = str(payload.get("fit_group_name") or payload.get("fit_name") or "")
            fit_uid = str(event.get("source_uid") or payload.get("fit_uid") or "")
            if fit_name:
                append_unique(fits, fit_name)
                selected_fit = fit_name
            if fit_uid:
                append_unique(fit_uids, fit_uid)
                selected_fit_uid = fit_uid

        elif action_type == "fit_close":
            fit_name = str(payload.get("fit_name") or "")
            fit_uid = str(event.get("source_uid") or payload.get("fit_uid") or "")
            if fit_name:
                fits = [f for f in fits if f != fit_name]
                if selected_fit == fit_name:
                    selected_fit = fits[-1] if fits else None
            if fit_uid:
                fit_uids = [u for u in fit_uids if u != fit_uid]
                if selected_fit_uid == fit_uid:
                    selected_fit_uid = fit_uids[-1] if fit_uids else None

        elif action_type in {"fit_run_start", "fit_run_finish", "fit_run_abort"}:
            fit_name = str(payload.get("fit_name") or "")
            fit_uid = str(event.get("source_uid") or payload.get("fit_uid") or "")
            if fit_name:
                selected_fit = fit_name
            if fit_uid:
                selected_fit_uid = fit_uid

        elif action_type in {
            "parameter_value",
            "parameter_fixed",
            "parameter_bounds_set",
            "parameter_bounds_on",
            "parameter_link",
            "parameter_unlink",
            "fit_range_set",
            "fit_mask_set",
        }:
            fit_name = str(payload.get("fit_group") or payload.get("source_fit_group") or "")
            if fit_name:
                selected_fit = fit_name

    return {
        "datasets": datasets,
        "dataset_uids": dataset_uids,
        "fits": fits,
        "fit_uids": fit_uids,
        "selected_dataset": selected_dataset,
        "selected_dataset_uid": selected_dataset_uid,
        "selected_fit": selected_fit,
        "selected_fit_uid": selected_fit_uid,
    }


def reconstruct_parameter_state(
        events: typing.List[typing.Dict[str, typing.Any]],
) -> typing.Dict[typing.Tuple[str, str, str], typing.Dict[str, typing.Any]]:
    """Build parameter state map up to cursor from history events.

    Key is (fit_group_name, local_fit_name, parameter_name).
    """

    state: typing.Dict[typing.Tuple[str, str, str], typing.Dict[str, typing.Any]] = {}

    def get_key(payload: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Tuple[str, str, str]]:
        fit_group = str(payload.get("fit_group") or payload.get("source_fit_group") or "")
        local_fit = str(payload.get("local_fit") or payload.get("source_local_fit") or "")
        param_name = str(payload.get("parameter_name") or payload.get("source_parameter") or "")
        if not fit_group or not local_fit or not param_name:
            return None
        return fit_group, local_fit, param_name

    def apply_snapshot_rows(rows: typing.Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            fit_group = str(row.get("fit_group") or "")
            local_fit = str(row.get("local_fit") or "")
            param_name = str(row.get("parameter_name") or "")
            if not fit_group or not local_fit or not param_name:
                continue
            key = (fit_group, local_fit, param_name)
            entry = state.setdefault(key, {})
            if "value" in row:
                entry["value"] = row.get("value")
            if "fixed" in row:
                entry["fixed"] = bool(row.get("fixed"))
            if "bounds_on" in row:
                entry["bounds_on"] = bool(row.get("bounds_on"))
            lower = row.get("lower")
            upper = row.get("upper")
            if lower is not None and upper is not None:
                try:
                    entry["bounds"] = (float(lower), float(upper))
                except Exception:
                    pass

    for event in events:
        action_type = str(event.get("action_type", ""))
        payload = event.get("payload", {}) or {}

        if action_type == "fit_run_start":
            apply_snapshot_rows(payload.get("parameter_snapshot_before"))
            continue
        if action_type in {"fit_run_finish", "fit_run_abort"}:
            apply_snapshot_rows(payload.get("parameter_snapshot_after"))
            continue

        key = get_key(payload)
        if key is None:
            continue
        entry = state.setdefault(key, {})

        src_fit_uid = str(payload.get("fit_uid") or payload.get("source_fit_uid") or "")
        src_local_uid = str(payload.get("local_fit_uid") or payload.get("source_local_fit_uid") or "")
        src_param_uid = str(payload.get("parameter_uid") or payload.get("source_parameter_uid") or "")
        if src_fit_uid:
            entry["source_fit_uid"] = src_fit_uid
        if src_local_uid:
            entry["source_local_fit_uid"] = src_local_uid
        if src_param_uid:
            entry["source_parameter_uid"] = src_param_uid

        if action_type == "parameter_value":
            if "new_value" in payload:
                entry["value"] = payload.get("new_value")

        elif action_type == "parameter_fixed":
            if "fixed" in payload:
                entry["fixed"] = bool(payload.get("fixed"))

        elif action_type == "parameter_bounds_on":
            if "bounds_on" in payload:
                entry["bounds_on"] = bool(payload.get("bounds_on"))

        elif action_type == "parameter_bounds_set":
            lower = payload.get("lower")
            upper = payload.get("upper")
            if lower is not None and upper is not None:
                try:
                    entry["bounds"] = (float(lower), float(upper))
                except Exception:
                    pass

        elif action_type == "parameter_link":
            target_fit_group = str(payload.get("target_fit_group") or "")
            target_local_fit = str(payload.get("target_local_fit") or "")
            target_parameter = str(payload.get("target_parameter") or "")
            if target_fit_group and target_local_fit and target_parameter:
                entry["link"] = (target_fit_group, target_local_fit, target_parameter)
            target_fit_uid = str(payload.get("target_fit_uid") or "")
            target_local_uid = str(payload.get("target_local_fit_uid") or "")
            target_param_uid = str(payload.get("target_parameter_uid") or "")
            if target_fit_uid or target_local_uid or target_param_uid:
                entry["link_uid"] = (target_fit_uid, target_local_uid, target_param_uid)

        elif action_type == "parameter_unlink":
            entry["link"] = None

    return state


def reconstruct_fit_range_state(
        events: typing.List[typing.Dict[str, typing.Any]],
) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    """Build fit-range state per fit-group name up to cursor."""

    state: typing.Dict[str, typing.Dict[str, typing.Any]] = {}

    def apply_range_rows(rows: typing.Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            fit_group = str(row.get("fit_group") or "")
            if not fit_group:
                continue
            xmin = row.get("xmin")
            xmax = row.get("xmax")
            if xmin is None or xmax is None:
                continue
            try:
                state[fit_group] = {
                    "xmin": int(xmin),
                    "xmax": int(xmax),
                }
            except Exception:
                pass

    for event in events:
        action_type = str(event.get("action_type", ""))
        payload = event.get("payload", {}) or {}

        if action_type == "fit_run_start":
            apply_range_rows(payload.get("fit_range_snapshot_before"))
            continue
        if action_type in {"fit_run_finish", "fit_run_abort"}:
            apply_range_rows(payload.get("fit_range_snapshot_after"))
            continue

        if action_type == "fit_range_set":
            fit_group = str(payload.get("fit_group") or "")
            xmin = payload.get("xmin")
            xmax = payload.get("xmax")
            if fit_group and xmin is not None and xmax is not None:
                try:
                    state[fit_group] = {
                        "xmin": int(xmin),
                        "xmax": int(xmax),
                    }
                except Exception:
                    pass

    return state


def touched_parameter_keys(
        events: typing.List[typing.Dict[str, typing.Any]],
        include_actions: typing.Optional[typing.Set[str]] = None,
) -> typing.Set[typing.Tuple[str, str, str]]:
    keys: typing.Set[typing.Tuple[str, str, str]] = set()
    default_actions = {
        "parameter_value",
        "parameter_fixed",
        "parameter_bounds_set",
        "parameter_bounds_on",
        "parameter_link",
        "parameter_unlink",
    }
    actions = include_actions or default_actions

    for event in events:
        action_type = str(event.get("action_type", ""))
        if action_type not in actions:
            continue
        payload = event.get("payload", {}) or {}
        fit_group = str(payload.get("fit_group") or payload.get("source_fit_group") or "")
        local_fit = str(payload.get("local_fit") or payload.get("source_local_fit") or "")
        param_name = str(payload.get("parameter_name") or payload.get("source_parameter") or "")
        if fit_group and local_fit and param_name:
            keys.add((fit_group, local_fit, param_name))
    return keys


def reconstruct_setup_state(
        events: typing.List[typing.Dict[str, typing.Any]],
) -> typing.Dict[str, typing.Any]:
    experiment_name: typing.Optional[str] = None
    setup_name: typing.Optional[str] = None
    params: typing.Dict[str, typing.Any] = {}

    for event in events:
        action_type = str(event.get("action_type", ""))
        payload = event.get("payload", {}) or {}

        if action_type == "experiment_set":
            name = str(payload.get("name") or "")
            if name:
                experiment_name = name
            continue

        if action_type == "setup_select":
            name = str(payload.get("name") or "")
            if name:
                setup_name = name
            continue

        if action_type == "setup_params_set":
            values = payload.get("params") or {}
            if isinstance(values, dict):
                for key, value in values.items():
                    k = str(key)
                    if k:
                        params[k] = value

    return {
        "experiment": experiment_name,
        "setup": setup_name,
        "params": params,
    }


def capture_domain_snapshot() -> typing.Dict[str, typing.Any]:
    """Capture current domain state for checkpoint storage.

    Returns a JSON-serializable dict containing:
    - navigation: datasets, fits, selections
    - parameters: all parameter values, bounds, fixed, links
    - fit_ranges: fit range state per fit group
    - setup: experiment and setup state
    """
    import chisurf

    snapshot: typing.Dict[str, typing.Any] = {
        "navigation": {},
        "parameters": {},
        "fit_ranges": {},
        "setup": {},
    }

    try:
        datasets = list(getattr(chisurf, "imported_datasets", []))
        dataset_names = []
        dataset_uids = []
        for ds in datasets:
            name = str(getattr(ds, "name", ""))
            uid = str(getattr(ds, "unique_identifier", ""))
            if name:
                dataset_names.append(name)
            if uid:
                dataset_uids.append(uid)
        snapshot["navigation"]["datasets"] = dataset_names
        snapshot["navigation"]["dataset_uids"] = dataset_uids
    except Exception:
        pass

    try:
        current_ds = getattr(chisurf, "current_data", None)
        if current_ds is not None:
            snapshot["navigation"]["selected_dataset"] = str(getattr(current_ds, "name", ""))
            snapshot["navigation"]["selected_dataset_uid"] = str(getattr(current_ds, "unique_identifier", ""))
    except Exception:
        pass

    try:
        fits = list(getattr(chisurf, "fits", []))
        fit_names = []
        fit_uids = []
        for fg in fits:
            name = str(getattr(fg, "name", ""))
            uid = str(getattr(fg, "unique_identifier", ""))
            if name:
                fit_names.append(name)
            if uid:
                fit_uids.append(uid)
        snapshot["navigation"]["fits"] = fit_names
        snapshot["navigation"]["fit_uids"] = fit_uids
    except Exception:
        pass

    try:
        current_fit = getattr(chisurf, "current_fit", None)
        if current_fit is not None:
            snapshot["navigation"]["selected_fit"] = str(getattr(current_fit, "name", ""))
            snapshot["navigation"]["selected_fit_uid"] = str(getattr(current_fit, "unique_identifier", ""))
    except Exception:
        pass

    try:
        fits = list(getattr(chisurf, "fits", []))
        param_state: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        for fg in fits:
            fg_name = str(getattr(fg, "name", ""))
            fg_uid = str(getattr(fg, "unique_identifier", ""))
            local_fits = list(getattr(fg, "local_fits", []))
            for local in local_fits:
                local_name = str(getattr(local, "name", ""))
                local_uid = str(getattr(local, "unique_identifier", ""))
                params = getattr(local, "parameters_all_dict", {})
                if not params:
                    continue
                for p_name, param in params.items():
                    if param is None:
                        continue
                    key = f"{fg_name}/{local_name}/{p_name}"
                    entry: typing.Dict[str, typing.Any] = {
                        "fit_group": fg_name,
                        "fit_group_uid": fg_uid,
                        "local_fit": local_name,
                        "local_fit_uid": local_uid,
                        "parameter_name": str(p_name),
                        "parameter_uid": str(getattr(param, "unique_identifier", "")),
                    }
                    try:
                        entry["value"] = float(getattr(param, "value", 0.0))
                    except Exception:
                        pass
                    try:
                        entry["fixed"] = bool(getattr(param, "fixed", False))
                    except Exception:
                        pass
                    try:
                        entry["bounds_on"] = bool(getattr(param, "bounds_on", False))
                    except Exception:
                        pass
                    try:
                        lb = getattr(param, "lb", None)
                        ub = getattr(param, "ub", None)
                        if lb is not None and ub is not None:
                            entry["bounds"] = [float(lb), float(ub)]
                    except Exception:
                        pass
                    try:
                        link = getattr(param, "link", None)
                        if link is not None:
                            link_param = link
                            link_fg = ""
                            link_local = ""
                            link_pname = ""
                            try:
                                parent = getattr(link_param, "parent", None)
                                if parent is not None:
                                    link_local = str(getattr(parent, "name", ""))
                                    grandparent = getattr(parent, "parent", None)
                                    if grandparent is not None:
                                        link_fg = str(getattr(grandparent, "name", ""))
                            except Exception:
                                pass
                            try:
                                link_pname = str(getattr(link_param, "name", ""))
                            except Exception:
                                pass
                            if link_fg and link_local and link_pname:
                                entry["link"] = {
                                    "fit_group": link_fg,
                                    "local_fit": link_local,
                                    "parameter_name": link_pname,
                                }
                    except Exception:
                        pass
                    param_state[key] = entry
        snapshot["parameters"] = param_state
    except Exception:
        pass

    try:
        fits = list(getattr(chisurf, "fits", []))
        fit_range_state: typing.Dict[str, typing.Dict[str, int]] = {}
        for fg in fits:
            fg_name = str(getattr(fg, "name", ""))
            xmin = getattr(fg, "xmin", None)
            xmax = getattr(fg, "xmax", None)
            if xmin is not None and xmax is not None:
                try:
                    fit_range_state[fg_name] = {
                        "xmin": int(xmin),
                        "xmax": int(xmax),
                    }
                except Exception:
                    pass
        snapshot["fit_ranges"] = fit_range_state
    except Exception:
        pass

    try:
        experiment = getattr(chisurf, "experiment", None)
        if experiment is not None:
            snapshot["setup"]["experiment"] = str(getattr(experiment, "name", ""))
        setup = getattr(chisurf, "setup", None)
        if setup is not None:
            snapshot["setup"]["setup"] = str(getattr(setup, "name", ""))
    except Exception:
        pass

    return snapshot


def snapshot_to_replay_state(
        snapshot: typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    """Convert a domain snapshot to the replay state format.

    Returns a dict with:
    - navigation: same as reconstruct_navigation_state output
    - parameters: same as reconstruct_parameter_state output
    - fit_ranges: same as reconstruct_fit_range_state output
    - setup: same as reconstruct_setup_state output
    """
    result: typing.Dict[str, typing.Any] = {
        "navigation": {},
        "parameters": {},
        "fit_ranges": {},
        "setup": {},
    }

    nav = snapshot.get("navigation", {})
    result["navigation"] = {
        "datasets": nav.get("datasets", []),
        "dataset_uids": nav.get("dataset_uids", []),
        "fits": nav.get("fits", []),
        "fit_uids": nav.get("fit_uids", []),
        "selected_dataset": nav.get("selected_dataset"),
        "selected_dataset_uid": nav.get("selected_dataset_uid"),
        "selected_fit": nav.get("selected_fit"),
        "selected_fit_uid": nav.get("selected_fit_uid"),
    }

    params = snapshot.get("parameters", {})
    param_state: typing.Dict[typing.Tuple[str, str, str], typing.Dict[str, typing.Any]] = {}
    for key, entry in params.items():
        if not isinstance(entry, dict):
            continue
        fg = str(entry.get("fit_group", ""))
        local = str(entry.get("local_fit", ""))
        pname = str(entry.get("parameter_name", ""))
        if not fg or not local or not pname:
            continue
        param_key = (fg, local, pname)
        state_entry: typing.Dict[str, typing.Any] = {}
        if "value" in entry:
            state_entry["value"] = entry["value"]
        if "fixed" in entry:
            state_entry["fixed"] = entry["fixed"]
        if "bounds_on" in entry:
            state_entry["bounds_on"] = entry["bounds_on"]
        if "bounds" in entry:
            bounds = entry["bounds"]
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                state_entry["bounds"] = (float(bounds[0]), float(bounds[1]))
        if "link" in entry:
            link = entry["link"]
            if isinstance(link, dict):
                link_fg = str(link.get("fit_group", ""))
                link_local = str(link.get("local_fit", ""))
                link_pname = str(link.get("parameter_name", ""))
                if link_fg and link_local and link_pname:
                    state_entry["link"] = (link_fg, link_local, link_pname)
        if entry.get("fit_group_uid"):
            state_entry["source_fit_uid"] = str(entry.get("fit_group_uid"))
        if entry.get("local_fit_uid"):
            state_entry["source_local_fit_uid"] = str(entry.get("local_fit_uid"))
        if entry.get("parameter_uid"):
            state_entry["source_parameter_uid"] = str(entry.get("parameter_uid"))
        param_state[param_key] = state_entry
    result["parameters"] = param_state

    result["fit_ranges"] = snapshot.get("fit_ranges", {})

    result["setup"] = snapshot.get("setup", {})

    return result


def sync_domain_entities(
        target_nav_state: typing.Dict[str, typing.Any],
        all_events: typing.List[typing.Dict[str, typing.Any]],
) -> None:
    """Synchronize live domain entities (datasets, fits) with the target state.

    This identifies missing or extra entities by UID and uses action services
    to reconcile them, with history recording suppressed.
    """
    import chisurf
    from chisurf.controllers.services import dataset_service, fit_service

    target_ds_uids = set(target_nav_state.get("dataset_uids", []))
    target_fit_uids = set(target_nav_state.get("fit_uids", []))

    current_ds_uids = {
        str(getattr(ds, "unique_identifier", ""))
        for ds in getattr(chisurf, "imported_datasets", [])
    }
    current_fit_uids = {
        str(getattr(f, "unique_identifier", ""))
        for f in getattr(chisurf, "fits", [])
    }

    # Identify missing UIDs
    missing_ds = target_ds_uids - current_ds_uids
    missing_fits = target_fit_uids - current_fit_uids

    # Identify extra UIDs
    extra_ds_indices = [
        i for i, ds in enumerate(getattr(chisurf, "imported_datasets", []))
        if str(getattr(ds, "unique_identifier", "")) not in target_ds_uids
        and str(getattr(ds, "name", "")) != "Global Dataset"
    ]
    extra_fit_indices = [
        i for i, f in enumerate(getattr(chisurf, "fits", []))
        if str(getattr(f, "unique_identifier", "")) not in target_fit_uids
    ]

    history = getattr(chisurf, "history", None)
    if history is None:
        return

    with history.suppress_recording():
        # 1. Remove extra entities (reverse order to keep indices valid)
        if extra_fit_indices:
            for idx in sorted(extra_fit_indices, reverse=True):
                fit_service.close_fit(idx=idx)

        if extra_ds_indices:
            # dataset_service.remove_datasets takes a list
            dataset_service.remove_datasets(dataset_indices=extra_ds_indices)

            # 2a. Build UID -> Current Index map for resolving dependencies
            uid_to_idx = {
                str(getattr(ds, "unique_identifier", "")): i
                for i, ds in enumerate(getattr(chisurf, "imported_datasets", []))
            }

            # Map UID -> Event for creation actions
            creation_map: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
            uid_to_event_uids: typing.Dict[str, typing.List[str]] = {} # Map UID to all UIDs created in same event

            for event in all_events:
                atype = str(event.get("action_type", ""))
                payload = event.get("payload", {}) or {}
                if atype == "dataset_add":
                    uids = [str(u) for u in payload.get("loaded_uids", [])]
                    for uid in uids:
                        creation_map[uid] = event
                        uid_to_event_uids[uid] = uids
                elif atype == "fit_add":
                    uid = str(event.get("target_uid") or payload.get("fit_uid") or "")
                    if uid:
                        creation_map[uid] = event
                elif atype == "dataset_group":
                    uid = str(payload.get("group_uid", ""))
                    if uid:
                        creation_map[uid] = event

            def resolve_indices(old_indices: list, creator_event: dict) -> list:
                # This is tricky: old events have indices valid at that time.
                # We need to find the UIDs of the datasets at those indices in the past.
                # But history events don't usually store the FULL state of indices.
                # However, many events store 'loaded_uids'.
                # For now, we'll try to use the most recent uid_to_idx.
                # If the payload has 'member_uids', we use those.
                payload = creator_event.get("payload", {})
                member_uids = payload.get("member_uids", [])
                if member_uids:
                    return [uid_to_idx[str(u)] for u in member_uids if str(u) in uid_to_idx]
                return [int(i) for i in old_indices]

            # Replay missing datasets
            processed_events: typing.Set[str] = set()
            for uid in sorted(missing_ds): # Deterministic order
                event = creation_map.get(uid)
                if event and event["event_id"] not in processed_events:
                    atype = str(event.get("action_type", ""))
                    if atype == "dataset_add":
                        dataset_service.add_dataset(payload=event.get("payload", {}))
                    elif atype == "dataset_group":
                        payload = event.get("payload", {})
                        new_indices = resolve_indices(payload.get("dataset_indices", []), event)
                        dataset_service.group_datasets(dataset_indices=new_indices)
                    processed_events.add(event["event_id"])
                    
                    # Update uid_to_idx after adding
                    uid_to_idx = {
                        str(getattr(ds, "unique_identifier", "")): i
                        for i, ds in enumerate(getattr(chisurf, "imported_datasets", []))
                    }

            # Replay missing fits
            for uid in sorted(missing_fits):
                event = creation_map.get(uid)
                if event and event["event_id"] not in processed_events:
                    payload = event.get("payload", {})
                    new_indices = resolve_indices(payload.get("dataset_indices", []), event)
                    fit_service.add_fit(
                        dataset_indices=new_indices,
                        model_name=payload.get("model_name"),
                        model_kw=payload.get("model_kw"),
                    )
                    processed_events.add(event["event_id"])
