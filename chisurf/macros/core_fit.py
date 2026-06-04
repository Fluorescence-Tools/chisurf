from __future__ import annotations

import os
import gc
import shutil
import json
import pathlib
import importlib
import numpy as np

import chisurf
import chisurf.base
import chisurf.data
import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets
from chisurf.runtime.actions import record_action, get_action_catalog

from chisurf import typing
from chisurf import logging
from chisurf.project import Project as CSProject, save_project as project_save_json, load_project as project_load_json
from chisurf.project import fit_state as project_fit_state
from chisurf.experiments.core.reader import ExperimentReader


def _iter_group_members(group):
    if isinstance(group, (list, tuple)):
        yield from group
    else:
        yield group


def _coerce_finite_float(value: typing.Any) -> typing.Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _resolve_dataset_anisotropy_calibration(data_group):
    calibration: typing.Dict[str, typing.Optional[float]] = {
        'g_factor': None,
        'l1': None,
        'l2': None,
    }

    reader = getattr(data_group, 'data_reader', None)
    if reader is None:
        for member in _iter_group_members(data_group):
            reader = getattr(member, 'data_reader', None)
            if reader is not None:
                break
    if reader is not None:
        for key in ('g_factor', 'l1', 'l2'):
            value = getattr(reader, key, None)
            if value is None:
                continue
            v = _coerce_finite_float(value)
            if v is not None:
                calibration[key] = v

    metas = []
    meta_data = getattr(data_group, 'meta_data', None)
    if isinstance(meta_data, dict):
        metas.append(meta_data)
    for member in _iter_group_members(data_group):
        meta = getattr(member, 'meta_data', None)
        if isinstance(meta, dict):
            metas.append(meta)
    for meta in metas:
        for key in ('g_factor', 'l1', 'l2'):
            if calibration.get(key) is not None:
                continue
            if key not in meta:
                continue
            v = _coerce_finite_float(meta[key])
            if v is not None:
                calibration[key] = v
    return calibration


def _resolve_dataset_g_factor(data_group):
    calibration = _resolve_dataset_anisotropy_calibration(data_group)
    return calibration.get('g_factor')


def _apply_g_factor_to_fit(fit_group, g_factor: float) -> None:
    _apply_anisotropy_calibration_to_fit(fit_group, {'g_factor': g_factor})


def _apply_anisotropy_calibration_to_fit(fit_group, calibration: typing.Dict[str, typing.Any]) -> None:
    if not isinstance(calibration, dict):
        return

    resolved = {}
    for source_key in ('g_factor', 'l1', 'l2'):
        value = _coerce_finite_float(calibration.get(source_key))
        if value is None:
            continue
        resolved[source_key] = value
    if not resolved:
        return

    mapping = {
        'g_factor': ('_g', 'g'),
        'l1': ('_l1', 'l1'),
        'l2': ('_l2', 'l2'),
    }

    def _set_anisotropy(anisotropy):
        if anisotropy is None:
            return
        params = getattr(anisotropy, 'parameters_all_dict', None)
        for source_key, value in resolved.items():
            private_name, public_name = mapping[source_key]
            private_param = getattr(anisotropy, private_name, None)
            applied = False
            if private_param is not None and hasattr(private_param, 'value'):
                private_param.value = value
                applied = True
            elif isinstance(private_param, (int, float, np.floating)):
                setattr(anisotropy, private_name, value)
                applied = True
            if (not applied) and isinstance(params, dict):
                param_entry = params.get(public_name)
                if param_entry is not None and hasattr(param_entry, 'value'):
                    param_entry.value = value
                    applied = True
            if not applied:
                try:
                    setattr(anisotropy, public_name, value)
                except Exception:
                    pass

    _set_anisotropy(getattr(getattr(fit_group, 'model', None), 'anisotropy', None))
    for member in getattr(fit_group, 'grouped_fits', []):
        _set_anisotropy(getattr(member.model, 'anisotropy', None))


def _collect_group_nuisance_parameter_names(model: typing.Any) -> typing.Set[str]:
    names: typing.Set[str] = set()
    if model is None:
        return names
    for attr_name, attr_value in getattr(model, "__dict__", {}).items():
        lname = str(attr_name).lower()
        is_nuisance_attr = (
            lname in {"nuisance", "nusiance", "generic", "corrections", "convolve"}
            or "nuisance" in lname
            or "nusiance" in lname
        )
        if not is_nuisance_attr:
            continue
        try:
            params = getattr(attr_value, "parameters_all", None)
            if isinstance(params, (list, tuple)):
                for p in params:
                    pname = str(getattr(p, "name", ""))
                    if pname:
                        names.add(pname)
                continue
            params_dict = getattr(attr_value, "parameters_all_dict", None)
            if isinstance(params_dict, dict):
                for pname in params_dict.keys():
                    if pname:
                        names.add(str(pname))
        except Exception:
            continue
    return names


def _auto_link_non_nuisance_group_parameters(fit_group) -> typing.Tuple[int, int]:
    grouped_fits = list(getattr(fit_group, "grouped_fits", []) or [])
    if len(grouped_fits) <= 1:
        return 0, 0

    master_fit = grouped_fits[0]
    master_model = getattr(master_fit, "model", None)
    master_params = getattr(master_model, "parameters_all_dict", None)
    if not isinstance(master_params, dict) or not master_params:
        return 0, 0

    nuisance_names = _collect_group_nuisance_parameter_names(master_model)
    linked_master_parameters = 0
    linked_followers = 0

    for parameter_name, master_parameter in master_params.items():
        if parameter_name in nuisance_names:
            continue
        if bool(getattr(master_parameter, "is_output", False)):
            continue
        if not hasattr(master_parameter, "link"):
            continue

        try:
            master_parameter.is_link_master = True
        except Exception:
            pass

        linked_this_parameter = False
        for local_fit in grouped_fits[1:]:
            try:
                local_params = getattr(getattr(local_fit, "model", None), "parameters_all_dict", None)
                if not isinstance(local_params, dict):
                    continue
                follower_parameter = local_params.get(parameter_name)
                if follower_parameter is None:
                    continue
                try:
                    follower_parameter.is_link_master = False
                except Exception:
                    pass
                follower_parameter.link = master_parameter
                linked_followers += 1
                linked_this_parameter = True
            except Exception:
                continue

        if linked_this_parameter:
            linked_master_parameters += 1

    return linked_master_parameters, linked_followers

HISTORY_FILENAME = "history.jsonl"


def _save_history_snapshot(project_dir: typing.Union[str, pathlib.Path]) -> typing.Optional[pathlib.Path]:
    try:
        history_obj = getattr(chisurf, "history", None)
        if history_obj is None or not hasattr(history_obj, "save_jsonl"):
            return None
        history_path = pathlib.Path(project_dir).resolve() / HISTORY_FILENAME
        return history_obj.save_jsonl(history_path)
    except Exception:
        return None


def _load_history_snapshot(
        project_dir: typing.Union[str, pathlib.Path],
        replace: bool = True,
) -> bool:
    try:
        history_obj = getattr(chisurf, "history", None)
        if history_obj is None or not hasattr(history_obj, "load_jsonl"):
            return False
        history_path = pathlib.Path(project_dir).resolve() / HISTORY_FILENAME
        if not history_path.exists():
            return False
        history_obj.load_jsonl(history_path, replace=replace)
        return True
    except Exception:
        return False


def _refresh_history_browser() -> None:
    try:
        cs = getattr(chisurf, "cs", None)
        browser = getattr(cs, "historyBrowser", None)
        if browser is not None and hasattr(browser, "reload"):
            browser.reload()
    except Exception:
        pass


def _record_history(
        action_type: str,
        summary: str,
        payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
        source_uid: str = "",
        target_uid: str = "",
) -> None:
    try:
        record_action(
            action_type=action_type,
            summary=summary,
            payload=payload,
            source_uid=source_uid or "",
            target_uid=target_uid or "",
        )
        return
    except Exception:
        pass


def _history_event_count() -> int:
    try:
        history_obj = getattr(chisurf, "history", None)
        if history_obj is not None and hasattr(history_obj, "list_events"):
            return int(len(history_obj.list_events()))
    except Exception:
        pass
    return 0


def export_action_catalog(
        target_path: str = "",
        file_type: str = "yaml",
) -> pathlib.Path:
    catalog = get_action_catalog()

    def _normalize_format(path_obj: pathlib.Path, raw: str) -> str:
        value = str(raw or "").strip().lower()
        if value in {"yml", "yaml"}:
            return "yaml"
        if value == "json":
            return "json"
        suffix = path_obj.suffix.lower().lstrip(".")
        if suffix in {"yml", "yaml"}:
            return "yaml"
        if suffix == "json":
            return "json"
        return "yaml"

    out_path: pathlib.Path
    if target_path:
        out_path = pathlib.Path(str(target_path))
    else:
        working = getattr(chisurf, "working_path", None)
        base_dir = pathlib.Path(working) if working else pathlib.Path.cwd()
        out_path = base_dir / "action_catalog"

    fmt = _normalize_format(out_path, file_type)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".yaml" if fmt == "yaml" else ".json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    else:
        import yaml

        out_path.write_text(
            yaml.safe_dump(catalog, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    _record_history(
        action_type="action_catalog_export",
        summary=f"export action catalog to '{out_path.as_posix()}'",
        payload={
            "target_path": out_path.as_posix(),
            "format": fmt,
            "action_count": int(len(catalog)),
        },
    )

    return out_path


def _serialize_reader(reader: typing.Any) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Serialize an ExperimentReader into a JSON-friendly dict.

    We keep this minimal to avoid recursion loops (e.g. experiment._readers
    holding the same reader). Only elementary attributes are retained; Qt
    widgets, experiment references, and controllers are skipped.
    """
    if not isinstance(reader, ExperimentReader):
        return None

    def _is_basic(v: typing.Any) -> bool:
        return isinstance(v, (str, int, float, bool, type(None)))

    def _to_basic(v: typing.Any):
        if _is_basic(v):
            return v
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            out = []
            for item in v:
                if _is_basic(item) or isinstance(item, (np.integer, np.floating)):
                    out.append(_to_basic(item))
                else:
                    # skip non-basic entries in sequences
                    continue
            return out
        return None

    state: typing.Dict[str, typing.Any] = {}
    banned_keys = {"experiment", "_experiment", "controller", "_readers", "setup"}
    for k, v in getattr(reader, "__dict__", {}).items():
        if k in banned_keys or (k.startswith("_") and k not in {"_irf"}):
            continue
        basic = _to_basic(v)
        if basic is not None:
            state[k] = basic

    rec: typing.Dict[str, typing.Any] = {
        "module": type(reader).__module__,
        "class": type(reader).__name__,
        "state": state,
    }
    return rec


def _deserialize_reader(reader_info: typing.Dict[str, typing.Any]) -> typing.Optional[ExperimentReader]:
    """Reconstruct an ExperimentReader from serialized info."""
    if not isinstance(reader_info, dict):
        return None
    mod_name = reader_info.get("module")
    cls_name = reader_info.get("class")
    state = reader_info.get("state") or {}
    if not mod_name or not cls_name:
        return None
    try:
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    except Exception:
        return None
    try:
        reader = cls.__new__(cls)
    except Exception:
        return None
    try:
        if isinstance(state, dict):
            reader.__dict__.update(state)
    except Exception:
        pass
    # Attach current experiment if available so autofitrange works
    try:
        exp_obj = getattr(chisurf.cs, "current_experiment", None)
        if exp_obj is not None:
            reader.experiment = exp_obj
    except Exception:
        pass
    return reader


def add_fit(
        dataset_indices: typing.List[int] = None,
        model_name: str = None,
        model_kw: typing.Dict = None,
        _defer_cs_update: bool = False,
        _ui_updates_frozen: bool = False,
        _force_local: bool = False,
):
    # Phase 8: in server mode, route through the API so the server
    # creates the fit object.  The proxy list will pick it up on the
    # next refresh.
    import chisurf as _cs_guard
    _api_guard = getattr(_cs_guard, "api", None)
    if (
            not _force_local and
            _api_guard is not None and
            getattr(_api_guard, "mode", None) == "server"
    ):
        return _api_guard.add_fit(
            dataset_indices=list(dataset_indices or [0]),
            model_name=model_name,
            model_kw=model_kw,
        )
    def _resolve_model_name_from_cs(main_window) -> str:
        try:
            v = str(getattr(main_window, "current_model_name", "") or "").strip()
            if v:
                return v
        except Exception:
            pass

        try:
            mc = getattr(main_window, "current_model_class", None)
            if mc is not None:
                v = str(getattr(mc, "name", "") or "").strip()
                if v:
                    return v
        except Exception:
            pass

        try:
            exp = getattr(main_window, "current_experiment", None)
            models = list(getattr(exp, "models", []) or [])
            if models:
                v = str(getattr(models[0], "name", "") or "").strip()
                if v:
                    return v
        except Exception:
            pass

        return ""

    cs = getattr(chisurf, "cs", None)
    # Process inputs of macro and replace None
    # with more sensible values that are read
    # from the GUI or fallback defaults
    if dataset_indices is None:
        if cs is not None:
            dataset_indices = [cs.dataset_selector.selected_curve_index]
        else:
            dataset_indices = [0] if chisurf.imported_datasets else []
    if model_name is None:
        if cs is not None:
            model_name = _resolve_model_name_from_cs(cs)
        else:
            model_name = ""

    # Do nothing of no dataset is selected
    if len(dataset_indices) == 0:
        chisurf.logging.warning("add_fit: no dataset index selected; aborting")
        return {"ok": False, "error": "no dataset index selected"}

    # If multiple datasets were requested, build each fit independently
    # using the already-stable single-dataset code path.
    if len(dataset_indices) > 1:
        batched_frozen = bool(_ui_updates_frozen)
        mdl_parent = None
        plo_parent = None
        
        if cs is not None:
            mdl_parent = getattr(cs.modelLayout, 'parentWidget', lambda: None)()
            plo_parent = getattr(cs.plotOptionsLayout, 'parentWidget', lambda: None)()
            try:
                if not batched_frozen:
                    if mdl_parent:
                        mdl_parent.setUpdatesEnabled(False)
                    if plo_parent:
                        plo_parent.setUpdatesEnabled(False)
                    cs.mdiarea.setUpdatesEnabled(False)
            except Exception:
                pass

        try:
            for idx in dataset_indices:
                try:
                    add_fit(
                        dataset_indices=[idx],
                        model_name=model_name,
                        model_kw=model_kw,
                        _defer_cs_update=True,
                        _ui_updates_frozen=True,
                    )
                except Exception as e:
                    chisurf.logging.warning(f"add_fit: failed for dataset index {idx}: {e}")
        finally:
            if cs is not None and not batched_frozen:
                try:
                    cs.mdiarea.setUpdatesEnabled(True)
                    if mdl_parent:
                        mdl_parent.setUpdatesEnabled(True)
                    if plo_parent:
                        plo_parent.setUpdatesEnabled(True)
                except Exception:
                    pass
        if cs is not None and not _defer_cs_update:
            try:
                cs.update()
            except Exception:
                pass
        return

    # create a list of data sets to which a fit with
    # a particular model is added. Avoid GUI access if missing.
    try:
        data_sets = [chisurf.imported_datasets[i] for i in dataset_indices]
    except IndexError:
        chisurf.logging.error("add_fit: dataset indices out of bounds of chisurf.imported_datasets")
        return {"ok": False, "error": "dataset indices out of bounds"}

    # Prefer the experiment attached to the dataset; fall back to the
    # globally selected experiment if necessary (e.g. after project load).
    exp = getattr(data_sets[0], "experiment", None)
    if exp is None and cs is not None:
        exp = getattr(cs, "current_experiment", None)
    if exp is None:
        # Headless fallback: trying to use first registered experiment
        try:
            exp = list(chisurf.experiments.types.values())[0] if chisurf.experiments.types else None
        except Exception:
            exp = None
    if exp is None:
        chisurf.logging.warning("add_fit: no experiment available on dataset or cs.current_experiment; aborting")
        return {"ok": False, "error": "no experiment available"}

    model_names = exp.model_names
    model_class = None

    # Try to find the model by name in the experiment type
    for model_idx, mn in enumerate(model_names):
        if mn == model_name:
            model_class = exp.model_classes[model_idx]
            break

    # If not found and we have a specific name, search globally in all Model subclasses.
    # This ensures headless project loading works for any registered model class in the environment.
    if model_class is None and model_name != "None":
        from chisurf.models.model import Model
        def get_all_subclasses(cls):
            all_subclasses = []
            for subclass in cls.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_all_subclasses(subclass))
            return all_subclasses

        for cls in get_all_subclasses(Model):
            if getattr(cls, 'name', None) == model_name:
                model_class = cls
                break

    # Fallback to experiment's default model if still not found and no specific name requested
    if model_class is None and exp.model_classes:
        model_class = exp.model_classes[0]

    if model_class is None:
        chisurf.logging.warning(f"add_fit: could not resolve model '{model_name}'; aborting")
        return {"ok": False, "error": f"could not resolve model '{model_name}'"}

    base_model_kw = dict(model_kw or {})

    for data_set in data_sets:
        if data_set.experiment is data_sets[0].experiment:
            # Make sure the data set is a DataGroup
            if not isinstance(data_set, chisurf.data.DataGroup):
                data_group = chisurf.data.ExperimentDataCurveGroup([data_set])
            else:
                data_group = data_set

            # Propagate data_reader/experiment to the group for restored projects
            try:
                if getattr(data_group, "data_reader", None) is None:
                    data_group.data_reader = getattr(data_set, "data_reader", None)
            except Exception:
                pass
            try:
                if getattr(data_group, "experiment", None) is None:
                    data_group.experiment = getattr(data_set, "experiment", None)
            except Exception:
                pass
            try:
                # Ensure contained curves have the reader attached (legacy projects)
                reader_obj = getattr(data_set, "data_reader", None)
                if reader_obj is not None:
                    for dc in data_group:
                        if getattr(dc, "data_reader", None) is None:
                            dc.data_reader = reader_obj
            except Exception:
                pass

            _record_history(
                action_type="fit_add_start",
                summary=(
                    f"start add fit for dataset '{getattr(data_set, 'name', '')}' "
                    f"with model '{model_name}'"
                ),
                payload={
                    "dataset_name": str(getattr(data_set, "name", "")),
                    "model_name": str(model_name),
                    "dataset_indices": [int(i) for i in dataset_indices],
                },
            )

            dataset_model_kw = dict(base_model_kw)
            dataset_calibration = _resolve_dataset_anisotropy_calibration(data_group)
            for key in ('g_factor', 'l1', 'l2'):
                value = _coerce_finite_float(dataset_calibration.get(key))
                if value is not None and key not in dataset_model_kw:
                    dataset_model_kw[key] = value

            # Create the fit
            fit_group = chisurf.fitting.fit.FitGroup(
                data=data_group,
                model_class=model_class,
                model_kw=dataset_model_kw
            )
            _apply_anisotropy_calibration_to_fit(fit_group, dataset_calibration)
            linked_masters, linked_followers = _auto_link_non_nuisance_group_parameters(fit_group)
            chisurf.fits.append(fit_group)
            _record_history(
                action_type="fit_add",
                summary=(
                    f"add fit group '{getattr(fit_group, 'name', '')}' for dataset "
                    f"'{getattr(data_set, 'name', '')}' with model '{model_name}'"
                ),
                payload={
                    "fit_group_name": str(getattr(fit_group, "name", "")),
                    "dataset_name": str(getattr(data_set, "name", "")),
                    "model_name": str(model_name),
                    "dataset_indices": [int(i) for i in dataset_indices],
                },
                source_uid=str(getattr(fit_group, "unique_identifier", "")) or None,
            )
            if linked_followers > 0:
                _record_history(
                    action_type="fit_group_auto_link",
                    summary=(
                        f"auto-link non-nuisance parameters for fit group '{getattr(fit_group, 'name', '')}' "
                        f"({linked_masters} master parameter(s), {linked_followers} follower link(s))"
                    ),
                    payload={
                        "fit_group_name": str(getattr(fit_group, "name", "")),
                        "linked_master_parameters": int(linked_masters),
                        "linked_followers": int(linked_followers),
                        "policy": "non_nuisance_default_link",
                    },
                    source_uid=str(getattr(fit_group, "unique_identifier", "")) or "",
                )

            # Batch UI updates to avoid repeated repaints while constructing widgets
            if cs is not None:
                mdl_parent = getattr(cs.modelLayout, 'parentWidget', lambda: None)()
                plo_parent = getattr(cs.plotOptionsLayout, 'parentWidget', lambda: None)()
                try:
                    if not _ui_updates_frozen:
                        if mdl_parent:
                            mdl_parent.setUpdatesEnabled(False)
                        if plo_parent:
                            plo_parent.setUpdatesEnabled(False)
                        cs.mdiarea.setUpdatesEnabled(False)

                    fit_control_widget = chisurf.gui.widgets.fitting.FittingControllerWidget(
                        fit=fit_group
                    )
                    header_layout = getattr(cs, "analysisHeaderLayout", None)
                    if header_layout is not None:
                        header_layout.addWidget(fit_control_widget)
                    else:
                        cs.modelLayout.addWidget(fit_control_widget)
                    for fit in fit_group:
                        cs.modelLayout.addWidget(fit.model)

                    fit_window = chisurf.gui.widgets.fitting.FitSubWindow(
                        fit=fit_group,
                        control_layout=cs.plotOptionsLayout,
                        fit_widget=fit_control_widget
                    )

                    fit_window.setWindowTitle(fit.name)
                    fit_window = cs.mdiarea.addSubWindow(fit_window)
                    chisurf.gui.fit_windows.append(fit_window)
                    cs.current_fit = fit_group
                    # Run auto-fit range synchronously so that each fit completes
                    # its range setup and model/plot updates before the next fit
                    # is created. This mirrors the stable sequential behaviour.
                    try:
                        fit_control_widget.onAutoFitRange()
                    except Exception:
                        pass
                finally:
                    # Re-enable updates and show
                    if not _ui_updates_frozen:
                        cs.mdiarea.setUpdatesEnabled(True)
                        if mdl_parent:
                            mdl_parent.setUpdatesEnabled(True)
                        if plo_parent:
                            plo_parent.setUpdatesEnabled(True)
                    try:
                        fit_window.show()
                    except Exception:
                        pass

    if cs is not None and not _defer_cs_update:
        try:
            cs.update()
        except Exception:
            pass


def save_fit(
        target_path: str = None,
        use_complex_name: bool = False,
        fit_window=None):
    log = chisurf.logging
    log.debug("save_fit: start (target_path=%r, use_complex_name=%r)",
              target_path, use_complex_name)

    cs = getattr(chisurf, "cs", None)
    if cs is None:
        log.error("save_fit: no active main window (chisurf.cs is missing)")
        return
    if fit_window is None:
        log.debug("No fit_window passed—taking current MDI subwindow")
        fit_window = cs.mdiarea.currentSubWindow()

    fit       = fit_window.fit
    widget    = fit_window.fit_widget
    fit_group = widget.fit

    # decide on save directory & base name
    if target_path is None:
        target_path = chisurf.working_path
        log.debug("No target_path passed—using working_path=%r", target_path)

    if use_complex_name:
        save_name = chisurf.base.clean_string(fit.name)
        log.debug("Using complex fit.name → %r", save_name)
    else:
        save_name = os.path.basename(fit.data.name)
        log.debug("Using simple data name → %r", save_name)

    # Keep output filenames readable by dropping any trailing source extension
    # from dataset-based names (e.g. "*.dat VV" -> "* VV").
    save_stem = os.path.splitext(save_name)[0]
    basename = os.path.join(target_path, save_stem)
    log.info("Will write files with base %r", basename)

    # 1) dump numeric data
    log.debug("Saving fit CSV and curves to %r.csv", basename)
    fit.save(basename, 'csv', save_curves=True)
    # Also persist a fit.json (single-fit project-style state without global links)
    try:
        fg_key, fit_payload = _build_fitgroup_payload_from_window(
            fit_window,
            register_datacurve=lambda dc: "",
            log=log,
            group_index=0,
            include_global_links=False,
        )
        # embed dataset in-place to avoid cross-fit collisions; reuse fit.data arrays
        datasets: typing.Dict[str, typing.Dict] = {}
        try:
            data_obj = getattr(fit, "data", None)
            if isinstance(data_obj, chisurf.data.DataCurve):
                try:
                    x = np.asarray(getattr(data_obj, "x", []), dtype=float)
                    y = np.asarray(getattr(data_obj, "y", []), dtype=float)
                    ex = np.asarray(getattr(data_obj, "ex", np.zeros_like(x)), dtype=float)
                    ey = np.asarray(getattr(data_obj, "ey", np.ones_like(y)), dtype=float)
                except Exception:
                    x = np.asarray([], dtype=float)
                    y = np.asarray([], dtype=float)
                    ex = np.asarray([], dtype=float)
                    ey = np.asarray([], dtype=float)
                datasets["ds000"] = {
                    "name": getattr(data_obj, "name", ""),
                    "filename": getattr(data_obj, "filename", ""),
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "ex": ex.tolist(),
                    "ey": ey.tolist(),
                }
                # attach dataset_id directly to the fit payload
                if fit_payload and fit_payload.get("local_fits"):
                    try:
                        fit_payload["local_fits"][0]["dataset_id"] = "ds000"
                    except Exception:
                        pass
        except Exception as exc:
            log.warning(f"save_fit: could not build dataset payload for fit.json: {exc}")

        if fit_payload:
            fit_ui_state = {"current_fit_index": getattr(cs, "fit_idx", 0)}
            try:
                history_browser = getattr(cs, "historyBrowser", None)
                get_hist_state = getattr(history_browser, "get_ui_state", None)
                if callable(get_hist_state):
                    fit_ui_state["history_browser"] = get_hist_state()
            except Exception:
                pass

            proj = CSProject(
                name=fit.name or save_stem,
                description=f"ChiSurf fit '{fit.name or save_stem}'",
                chisurf_version=getattr(chisurf.info, "__version__", None),
                datasets=datasets,
                experiments={},
                fits={fg_key: fit_payload},
                ui_state=fit_ui_state,
            )
            # Write fit.json using the base name without the original data extension
            base_no_ext = os.path.join(target_path, save_stem)
            fit_json_path = base_no_ext + ".fit.json"
            try:
                with open(fit_json_path, "w", encoding="utf-8") as f:
                    json.dump(proj.to_dict(), f, indent=2, sort_keys=True)
                log.info(f"Saved fit state to {fit_json_path}")
            except Exception as exc:
                log.warning(f"save_fit: could not write fit.json: {exc}")
    except Exception as exc:
        log.warning(f"save_fit: failed to build fit.json payload: {exc}")
    #log.debug("Saving fit data object to %r_data.pkl", basename)
    #fit.data.save(basename + "_data", 'pkl')

    # 2) build the Word report
    log.debug("Building Word document")
    import docx
    from docx.shared import Inches
    document = docx.Document()
    document.add_heading(cs.current_fit.name, 0)

    if not os.path.isdir(target_path):
        log.warning("Target folder %r does not exist, aborting report", target_path)
        return

    document.add_heading('Fit‑Results', level=1)
    overlay_prev = bool(getattr(chisurf, "_suspend_plot_metrics_overlay", False))
    setattr(chisurf, "_suspend_plot_metrics_overlay", True)
    try:
        for i, f in enumerate(fit):
            widget.selected_fit = i
            log.debug("Adding screenshots for fit #%d", i+1)
            document.add_paragraph(f"Fit #{i+1}", style='ListNumber')

            for suffix, source in (
                ("_screenshot_fit.png",   fit_window),
                ("_screenshot_model.png", f.model),
            ):
                png_path = basename + suffix
                log.debug(" Grabbing %r → %r", source, png_path)

                pix = source.grab()
                pix.save(png_path)
                del pix
                log.debug("  Saved and deleted QPixmap")

                document.add_picture(png_path, width=Inches(2.0))
                log.debug("  Embedded picture %r", png_path)
    finally:
        setattr(chisurf, "_suspend_plot_metrics_overlay", overlay_prev)

    # 3) summary table
    log.debug("Adding summary table for %d grouped fits", len(fit_group.grouped_fits))
    document.add_heading('Summary', level=1)
    p = document.add_paragraph("Parameters which are fitted are given in ")
    p.add_run('bold').bold = True
    p.add_run(', linked parameters in ')
    p.add_run('italic.').italic = True
    p.add_run(' Fixed parameters are plain name.')

    n = len(fit_group.grouped_fits)
    table = document.add_table(rows=1, cols=n+1)
    hdr = table.rows[0].cells
    hdr[0].text = "Param"
    for col in range(n):
        hdr[col+1].text = str(col+1)

    parameters = sorted(fit.model.parameters_all_dict.keys())
    for k in parameters:
        row = table.add_row().cells
        row[0].text = k
        for col, f in enumerate(fit_group):
            val = f.model.parameters_all_dict[k]
            run = row[col+1].paragraphs[0].add_run(f"{val.value:.3f}")
            if val.fixed:
                style = "fixed"
            elif val.link is not None:
                run.italic = True
                style = "linked"
            else:
                run.bold = True
                style = "fitted"
            log.debug(" Table cell [%r, fit #%d] = %.3f (%s)", k, col+1, val.value, style)

    # chi² row
    chi_row = table.add_row().cells
    chi_row[0].text = "Chi2r"
    for col, f in enumerate(fit_group):
        val = f.chi2r
        chi_row[col+1].paragraphs[0].add_run(f"{val:.4f}")
        log.debug(" Table cell [Chi2r, fit #%d] = %.4f", col+1, val)

    # finally save the document
    docx_path = basename + '.docx'
    log.info("Saving report document to %r", docx_path)
    document.save(docx_path)

    # ——— force Qt cleanup —————
    log.debug("Processing pending Qt events and cleaning up")
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()

    # drop Qt references and run GC
    fit_window = widget = fit_group = document = None
    gc.collect()
    log.debug("save_fit: done")




def load_fit_result(
        fit_index: int,
        filename: str
) -> bool:
    if os.path.isfile(filename):
        chisurf.fits[fit_index].model.load(filename)
        chisurf.fits[fit_index].update()
        return True
    else:
        return False


def _merge_docx(docx_files, out_path: str) -> bool:
    """Merge multiple DOCX files into a single document by stacking their bodies.

    Returns True on success, False otherwise.
    Note: This simple merge appends XML bodies and may not keep images/styles perfectly,
    but is sufficient to "simply stack" documents.
    """
    try:
        from docx import Document
    except Exception as e:
        chisurf.logging.info(f"python-docx not available, skipping combined DOCX creation: {e}")
        return False

    from copy import deepcopy
    try:
        master = Document()
        first = True
        # Helper to get body element compatibly
        def _body(doc):
            try:
                return doc.element.body
            except Exception:
                return doc._element.body
        for fp in docx_files:
            if not fp or not os.path.exists(fp):
                continue
            sub = Document(fp)
            if not first:
                master.add_page_break()
            first = False
            src_body = _body(sub)
            dst_body = _body(master)
            # Append deep copies of all children (paragraphs, tables, images)
            for child in list(src_body):
                dst_body.append(deepcopy(child))
        master.save(out_path)
        return True
    except Exception as e:
        chisurf.logging.warning(f"Failed to merge DOCX files: {e}")
        return False


def save_fits(target_path: str, use_complex_name: bool = False):
    if os.path.isdir(target_path):
        created_docx = []
        for fit_window in chisurf.gui.fit_windows:
            fit = fit_window.fit

            # Skip global fits
            setup = getattr(fit.data, 'setup', None)
            if isinstance(setup, chisurf.experiments.globalfit.GlobalFitSetup):
                continue

            if use_complex_name:
                save_name = chisurf.base.clean_string(fit.name)
            else:
                save_name = os.path.basename(fit.data.name)
            save_stem = os.path.splitext(save_name)[0]

            fit_name = fit.name
            p2 = os.path.join(target_path, save_stem)

            # Ensure per-fit directory handling with overwrite/skip/cancel dialog
            if os.path.exists(p2):
                try:
                    from qtpy import QtWidgets
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Question)
                    msg.setWindowTitle("Folder exists")
                    msg.setText(f"The folder '{p2}' already exists.")
                    msg.setInformativeText("Do you want to overwrite it?")
                    overwrite_btn = msg.addButton("Overwrite", QtWidgets.QMessageBox.AcceptRole)
                    skip_btn = msg.addButton("Skip", QtWidgets.QMessageBox.RejectRole)
                    cancel_btn = msg.addButton("Cancel", QtWidgets.QMessageBox.DestructiveRole)
                    msg.setDefaultButton(skip_btn)
                    msg.exec_()
                    clicked = msg.clickedButton()
                    if clicked is overwrite_btn:
                        chisurf.logging.info(f"Overwriting existing folder: {p2}")
                        try:
                            if os.path.isdir(p2):
                                shutil.rmtree(p2)
                            else:
                                os.remove(p2)
                        except Exception as e:
                            chisurf.logging.warning(f"Failed to remove existing path {p2}: {e}")
                        os.makedirs(p2, exist_ok=True)
                    elif clicked is skip_btn:
                        chisurf.logging.info(f"Skipping existing folder: {p2}")
                        continue
                    else:
                        chisurf.logging.info("Save all fits cancelled by user")
                        return
                except Exception as e:
                    # Headless or dialog failed: default to skipping
                    chisurf.logging.warning(f"Could not show overwrite dialog or handle existing folder ({e}). Skipping fit.")
                    continue
            else:
                os.makedirs(p2, exist_ok=True)

            save_fit(target_path=p2, fit_window=fit_window)

            # Track created per-fit DOCX path for merging
            per_fit_docx = os.path.join(p2, f"{save_stem}.docx")
            if os.path.exists(per_fit_docx):
                created_docx.append(per_fit_docx)

        # After saving all, create combined DOCX by stacking
        if created_docx:
            combined_path = os.path.join(target_path, "all_fits.docx")
            ok = _merge_docx(created_docx, combined_path)
            if ok:
                chisurf.logging.info(f"Combined DOCX created: {combined_path}")
            else:
                chisurf.logging.info("Combined DOCX could not be created.")


def close_fit(idx: int = None):
    cs = getattr(chisurf, "cs", None)
    if cs is None:
        chisurf.logging.error("close_fit: no active main window (chisurf.cs is missing)")
        return

    # Resolve index from current subwindow if not explicitly provided.
    if idx is None:
        sub_window = None
        try:
            mdi = getattr(cs, "mdiarea", None)
            if mdi is not None:
                sub_window = mdi.currentSubWindow()
        except Exception:
            sub_window = None

        if sub_window is not None:
            for i, w in enumerate(chisurf.gui.fit_windows):
                if w is sub_window:
                    idx = i
                    break

        # Fallback: try to resolve via cs.current_fit if available.
        if idx is None:
            current_fit = getattr(cs, "current_fit", None)
            if current_fit is not None:
                try:
                    idx = chisurf.fits.index(current_fit)
                except ValueError:
                    idx = None

    # If we still do not have a valid index, log and bail out gracefully.
    try:
        idx_int = int(idx) if idx is not None else None
    except Exception:
        idx_int = None

    if idx_int is None:
        chisurf.logging.warning("close_fit: no active fit to close (idx is None); ignoring request")
        return

    if idx_int < 0 or idx_int >= len(chisurf.fits) or idx_int >= len(chisurf.gui.fit_windows):
        chisurf.logging.warning(f"close_fit: index {idx_int} out of range; ignoring request")
        return

    fit_name = ""
    fit_uid = None
    try:
        fit_obj = chisurf.fits[idx_int]
        fit_name = str(getattr(fit_obj, "name", ""))
        uid = str(getattr(fit_obj, "unique_identifier", ""))
        fit_uid = uid or None
    except Exception:
        pass

    _record_history(
        action_type="fit_close",
        summary=f"close fit '{fit_name}'",
        payload={
            "fit_index": int(idx_int),
            "fit_name": str(fit_name),
        },
        source_uid=fit_uid or "",
    )

    # Remove the fit object and its corresponding window.
    try:
        chisurf.fits.pop(idx_int)
    except Exception as e:
        chisurf.logging.warning(f"close_fit: failed to remove fit at index {idx_int}: {e}")

    try:
        sub_window = chisurf.gui.fit_windows.pop(idx_int)
    except Exception as e:
        chisurf.logging.warning(f"close_fit: failed to pop fit window at index {idx_int}: {e}")
        sub_window = None

    if sub_window is not None:
        try:
            sub_window.close()
        except Exception:
            pass

    try:
        cs.update()
    except Exception:
        pass


def close_all_fits():
    """Close all currently active fits."""
    import chisurf
    cs = getattr(chisurf, "cs", None)
    if cs is None:
        chisurf.logging.error("close_all_fits: no active main window (chisurf.cs is missing)")
        return
    for fit_window in list(chisurf.gui.fit_windows):
        try:
            fit_window.close_confirm = False
            fit_window.close()
        except Exception:
            pass
    chisurf.fits.clear()
    chisurf.gui.fit_windows.clear()
    cs.update()


def link_fit_group(
        fitting_parameter_name: str,
        csi: int = 0
) -> None:
    """
    This macro links the parameters with a name
    specified by fitting_parameter_name within
    a FitGroup

    :param fitting_parameter_name:
    :param csi:
    :return:
    """
    cs = getattr(chisurf, "cs", None)
    if cs is None:
        chisurf.logging.error("add_fit: no active main window (chisurf.cs is missing)")
        return
    linked_count = 0
    unlinked_count = 0
    master_uid = None
    if csi == 2:
        # Establish a fit-group link with a stable master: always use the
        # first local fit in the group as the master parameter source.
        current_fit = cs.current_fit
        grouped_fits = list(getattr(current_fit, "grouped_fits", []))
        if grouped_fits:
            master_fit = grouped_fits[0]
        else:
            master_fit = current_fit
        try:
            parameter = master_fit.model.parameters_all_dict[fitting_parameter_name]
        except Exception:
            chisurf.logging.warning(
                f"link_fit_group: first fit has no parameter '{fitting_parameter_name}', cannot link group"
            )
            return
        try:
            uid = str(getattr(parameter, "unique_identifier", ""))
            master_uid = uid or None
        except Exception:
            master_uid = None

        # Reset all master flags first to avoid stale GUI state.
        for f in current_fit:
            try:
                p_reset = f.model.parameters_all_dict[fitting_parameter_name]
                p_reset.is_link_master = False
            except Exception:
                continue

        # Mark master for GUI purposes only; numerical behaviour is governed
        # by the underlying parameter links.
        try:
            parameter.is_link_master = True
        except Exception:
            pass

        for f in current_fit:
            try:
                p = f.model.parameters_all_dict[fitting_parameter_name]
            except KeyError:
                chisurf.logging.warning(f"The fit {f.name} has no parameter {fitting_parameter_name}")
                continue
            if p is parameter:
                # Master remains unlinked but flagged as such for the GUI.
                continue
            try:
                p.is_link_master = False
            except Exception:
                pass
            p.link = parameter
            linked_count += 1

    if csi == 0:
        # Unlink the entire fit group for this parameter name and clear any
        # master flags so the GUI shows the unchecked state everywhere.
        for f in cs.current_fit:
            try:
                p = f.model.parameters_all_dict[fitting_parameter_name]
            except KeyError:
                continue
            try:
                p.is_link_master = False
            except Exception:
                pass
            p.link = None
            unlinked_count += 1

    if csi == 2:
        _record_history(
            action_type="fit_group_link",
            summary=(
                f"link fit group parameter '{fitting_parameter_name}' across "
                f"{linked_count} follower(s)"
            ),
            payload={
                "parameter_name": str(fitting_parameter_name),
                "linked_followers": int(linked_count),
                "mode": int(csi),
            },
            source_uid=master_uid,
        )
    elif csi == 0:
        _record_history(
            action_type="fit_group_unlink",
            summary=(
                f"unlink fit group parameter '{fitting_parameter_name}' across "
                f"{unlinked_count} local fit(s)"
            ),
            payload={
                "parameter_name": str(fitting_parameter_name),
                "unlinked": int(unlinked_count),
                "mode": int(csi),
            },
        )


def change_selected_fit_of_group(
    selected_fit: int
) -> None:
    """
    Changes the currently selected fit

    :param selected_fit:
    :return:
    """
    cs = getattr(chisurf, "cs", None)
    if cs is None:
        chisurf.logging.error("change_selected_fit_of_group: no active main window (chisurf.cs is missing)")
        return

    # Switching local fits changes the underlying model/parameter objects.
    # Ensure any derived output parameters are recomputed and the parameter
    # widgets refresh accordingly.
    try:
        cs.current_fit.model.hide()
    except Exception:
        pass

    cs.current_fit.selected_fit = selected_fit
    cs.current_fit.update()
    try:
        cs.current_fit.model.finalize()
    except Exception:
        pass

    # Refresh parameter controllers for the associated (selected) fit/model only
    # (includes output/result parameters).
    try:
        model = getattr(cs.current_fit, "model", None)
        params = getattr(model, "parameters_all", None)
        if isinstance(params, (list, tuple)):
            for p in params:
                try:
                    ctrl = getattr(p, "controller", None)
                    if ctrl is not None and hasattr(ctrl, "finalize"):
                        ctrl.finalize()
                except (AttributeError, RuntimeError, TypeError):
                    continue
    except Exception:
        pass

    try:
        cs.current_fit.model.show()
    except Exception:
        pass


def save_project(target_path: str, project_name: str = "chisurf_project"):
    """Save the current state of the application as a project.

    This implementation writes a JSON ``project.json`` file using
    :class:`chisurf.project.Project` together with a snapshot of all datasets,
    fits and UI state. It no longer creates per-fit folders, screenshots or
    DOCX reports; everything is embedded in ``project.json``.

    Works in headless mode (without GUI / ``chisurf.cs``). UI state is
    only captured when a main window is available.

    Parameters
    ----------
    target_path : str
        Directory in which the project folder will be created.
    project_name : str
        Name of the project subfolder (default: ``"chisurf_project"``).
    """

    log = chisurf.logging
    cs = getattr(chisurf, "cs", None)

    base_dir = os.path.abspath(str(target_path))
    project_dir = os.path.join(base_dir, project_name)

    # Clean the target directory to avoid stale files from earlier saves
    try:
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir)
    except Exception as exc:
        log.error(f"save_project: could not clean existing project directory {project_dir}: {exc}")
        return

    try:
        os.makedirs(project_dir, exist_ok=True)
    except Exception as exc:
        log.error(f"save_project: could not create project directory {project_dir}: {exc}")
        return

    # --- Collect datasets as plain arrays ---------------------------------
    datasets: typing.Dict[str, typing.Dict] = {}
    dataset_id_by_obj: typing.Dict[int, str] = {}
    ds_counter = 0

    def register_datacurve(dc: chisurf.data.DataCurve) -> str:
        nonlocal ds_counter
        key = id(dc)
        if key in dataset_id_by_obj:
            return dataset_id_by_obj[key]
        ds_id = f"ds{ds_counter:03d}"
        ds_counter += 1
        try:
            x = np.asarray(getattr(dc, "x", []), dtype=float)
            y = np.asarray(getattr(dc, "y", []), dtype=float)
            ex = np.asarray(getattr(dc, "ex", np.zeros_like(x)), dtype=float)
            ey = np.asarray(getattr(dc, "ey", np.ones_like(y)), dtype=float)
        except Exception:
            x = np.asarray([], dtype=float)
            y = np.asarray([], dtype=float)
            ex = np.asarray([], dtype=float)
            ey = np.asarray([], dtype=float)
        # Track a human-readable dataset name and, if available, the
        # original filename/path the data was loaded from. The filename is
        # stored as-is (typically an absolute path for file-based imports)
        # but is *not* re-read on project load; it is only restored to the
        # DataCurve.filename attribute for user reference.
        filename = getattr(dc, "filename", "")
        datasets[ds_id] = {
            "name": getattr(dc, "name", ""),
            "filename": filename,
            "x": x.tolist(),
            "y": y.tolist(),
            "ex": ex.tolist(),
            "ey": ey.tolist(),
            "data_reader": _serialize_reader(getattr(dc, "data_reader", None)),
            "experiment_name": getattr(getattr(dc, "experiment", None), "name", None),
        }
        dataset_id_by_obj[key] = ds_id
        return ds_id

    # Register all DataCurve objects reachable from imported_datasets
    for item in chisurf.imported_datasets:
        if isinstance(item, chisurf.data.DataCurve):
            register_datacurve(item)
        elif isinstance(item, chisurf.data.DataGroup):
            for dc in item:
                if isinstance(dc, chisurf.data.DataCurve):
                    register_datacurve(dc)

    dataset_layout: typing.List[typing.Dict[str, typing.Any]] = []
    for item in chisurf.imported_datasets:
        if isinstance(item, chisurf.data.DataCurve):
            ds_id = register_datacurve(item)
            dataset_layout.append({
                "kind": "dataset",
                "dataset_id": ds_id,
            })
            continue

        if isinstance(item, chisurf.data.DataGroup):
            member_ids: typing.List[str] = []
            for dc in item:
                if isinstance(dc, chisurf.data.DataCurve):
                    member_ids.append(register_datacurve(dc))
            if not member_ids:
                continue

            rec: typing.Dict[str, typing.Any] = {
                "kind": "group",
                "group_type": type(item).__name__,
                "dataset_ids": member_ids,
            }
            try:
                group_name = getattr(item, "name", "")
                if group_name:
                    rec["name"] = str(group_name)
            except Exception:
                pass
            try:
                current_dataset_idx = int(getattr(item, "_current_dataset", 0))
                if 0 <= current_dataset_idx < len(member_ids):
                    rec["current_dataset_index"] = current_dataset_idx
            except Exception:
                pass
            dataset_layout.append(rec)

    # --- Collect fits & global links --------------------------------------
    manifest_fits: typing.List[typing.Dict[str, typing.Any]] = []

    # Headless: iterate core model list (chisurf.fits) directly.
    # Falls back to chisurf.gui.fit_windows only when fits list is empty
    # but GUI windows exist (legacy compat).
    fit_sources = chisurf.fits
    for i, fit_group in enumerate(fit_sources):
        if fit_group is None:
            continue

        base_name = getattr(fit_group, "name", "") or f"fitgroup_{i:03d}"
        fg_id = chisurf.base.clean_string(str(base_name)) or f"fitgroup_{i:03d}"

        local_fits_state: typing.List[typing.Dict[str, typing.Any]] = []
        model_name = None

        grouped = getattr(fit_group, "grouped_fits", [])
        for local_fit in grouped:
            data_obj = getattr(local_fit, "data", None)
            ds_id = None
            if isinstance(data_obj, chisurf.data.DataCurve):
                ds_id = register_datacurve(data_obj)

            # Derive human-readable model label from experiment, if available
            if model_name is None and data_obj is not None:
                try:
                    exp = getattr(data_obj, "experiment", None)
                    mn = getattr(exp, "model_names", [])
                    mc = getattr(exp, "model_classes", [])
                    for name, cls in zip(mn, mc):
                        try:
                            if isinstance(local_fit.model, cls):
                                model_name = name
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            # Headless fallback: use the model class's own 'name' attribute if experiment mapping is missing
            if model_name is None and local_fit is not None:
                try:
                    model_name = getattr(local_fit.model, "name", None)
                except Exception:
                    pass

            # Capture per-fit model state and current x-range
            try:
                get_state = getattr(local_fit, "get_state", None)
                if callable(get_state):
                    fit_state = get_state()
                else:
                    fit_state = project_fit_state.fit_to_state(local_fit)
            except Exception as exc:
                log.warning(f"save_project: could not serialize fit group {fg_id}: {exc}")
                fit_state = {}

            try:
                fr = getattr(local_fit, "fit_range", None)
                if isinstance(fr, tuple) and len(fr) == 2:
                    fit_range = [int(fr[0]), int(fr[1])]
                else:
                    fit_range = None
            except Exception:
                fit_range = None

            rec = {
                "dataset_id": ds_id,
                "fit_state": fit_state,
            }
            if fit_range is not None:
                rec["fit_range"] = fit_range

            local_fits_state.append(rec)

        # Capture global links if a GlobalFitModel is present
        global_model = getattr(fit_group, "_model", None)
        global_links_state: typing.Dict[str, typing.Any] = {}
        if global_model is not None:
            try:
                global_links_state = project_fit_state.global_links_to_state(global_model)
            except Exception as exc:
                log.warning(f"save_project: could not serialize global links for {fg_id}: {exc}")

        manifest_fits.append({
            "id": fg_id,
            "name": getattr(fit_group, "name", fg_id),
            "model_name": model_name,
            "local_fits": local_fits_state,
            "global_links": global_links_state,
        })

    ui_state = {}
    if dataset_layout:
        ui_state["dataset_layout"] = dataset_layout

    # Capture UI state only when a main window is available (headless-safe)
    if cs is not None:
        ui_state["current_fit_index"] = getattr(cs, "fit_idx", 0)
        ui_state["current_experiment_idx"] = getattr(cs, "current_experiment_idx", 0)
        ui_state["current_setup_idx"] = getattr(cs, "current_setup_idx", 0)
        try:
            from chisurf.project.ui_state import get_ui_state
            gui_state = get_ui_state(cs)
            if gui_state:
                ui_state.update(gui_state)
        except Exception as exc:
            log.warning(f"save_project: could not capture UI state: {exc}")

    proj = CSProject(
        name=project_name,
        description=f"ChiSurf project '{project_name}'",
        chisurf_version=getattr(chisurf.info, "__version__", None),
        datasets=datasets,
        experiments={},  # reserved for future structured experiment state
        fits=manifest_fits,
        ui_state=ui_state,
    )

    try:
        proj.extra["history"] = {
            "filename": HISTORY_FILENAME,
            "event_count": _history_event_count(),
        }
    except Exception:
        pass

    try:
        proj.extra["action_catalog"] = {
            "entries": get_action_catalog(),
        }
    except Exception:
        pass

    project_save_json(proj, project_dir)
    hist_path = _save_history_snapshot(project_dir)
    _record_history(
        action_type="project_save",
        summary=f"save project '{project_name}' to '{project_dir}'",
        payload={
            "project_dir": project_dir,
            "project_name": project_name,
            "history_file": str(hist_path) if hist_path is not None else None,
        },
    )
    if hist_path is not None:
        _save_history_snapshot(project_dir)
    log.info(f"Project saved to {project_dir}")


def _write_fit_docx(
        fit_window,
        fit_group,
        local_fit,
        lf_dir: pathlib.Path,
        clean_name: str,
        fit_index: int,
) -> pathlib.Path | None:
    """Create a per-local-fit DOCX + screenshots inside the local-fit folder."""
    log = chisurf.logging
    try:
        import docx
        from docx.shared import Inches
    except Exception as exc:
        log.info(f"python-docx not available; skipping DOCX for {clean_name}: {exc}")
        return None

    widget = getattr(fit_window, "fit_widget", None)
    if widget is None or fit_group is None or local_fit is None:
        return None

    lf_dir.mkdir(parents=True, exist_ok=True)
    basename = lf_dir / clean_name

    document = docx.Document()
    document.add_heading(local_fit.name or clean_name, 0)

    # One fit => single section
    try:
        widget.selected_fit = fit_index
    except Exception:
        pass

    for suffix, source in (
        ("_screenshot_fit.png", fit_window),
        ("_screenshot_model.png", local_fit.model),
    ):
        png_path = basename.parent / f"{basename.name}{suffix}"
        try:
            pix = source.grab()
            pix.save(str(png_path))
            del pix
            document.add_picture(str(png_path), width=Inches(2.0))
        except Exception as exc:
            log.debug(f"Could not add screenshot {png_path}: {exc}")

    # Compact summary table for this fit only
    document.add_heading('Summary', level=1)
    p = document.add_paragraph("Parameters: fitted in ")
    p.add_run('bold').bold = True
    p.add_run(', linked in ')
    p.add_run('italic.').italic = True
    p.add_run(' Fixed are plain.')

    table = document.add_table(rows=1, cols=2)
    hdr = table.rows[0].cells
    hdr[0].text = "Param"
    hdr[1].text = "#1"

    parameters = sorted(local_fit.model.parameters_all_dict.keys())
    for k in parameters:
        row = table.add_row().cells
        row[0].text = k
        try:
            val = local_fit.model.parameters_all_dict[k]
            run = row[1].paragraphs[0].add_run(f"{val.value:.3f}")
            if val.fixed:
                pass
            elif val.link is not None:
                run.italic = True
            else:
                run.bold = True
        except Exception:
            continue

    chi_row = table.add_row().cells
    chi_row[0].text = "Chi2r"
    try:
        chi_row[1].paragraphs[0].add_run(f"{local_fit.chi2r:.4f}")
    except Exception:
        pass

    docx_path = basename.with_suffix(".docx")
    document.save(str(docx_path))
    return docx_path

def _build_fitgroup_payload(
        fit_group,
        register_datacurve: typing.Callable[[chisurf.data.DataCurve], str],
        log: typing.Any,
        group_index: int = 0,
        include_global_links: bool = True,
) -> typing.Tuple[str, typing.Dict]:
    """Helper to serialize a FitGroup into the Project payload.

    Works headlessly — no GUI window required.
    """
    if fit_group is None:
        return None, {}

    local_fits_state = []
    model_name = None

    grouped = getattr(fit_group, "grouped_fits", [])
    for local_fit in grouped:
        data_obj = getattr(local_fit, "data", None)
        ds_id = None
        if isinstance(data_obj, chisurf.data.DataCurve):
            ds_id = register_datacurve(data_obj)

        # Derive human-readable model label from experiment, if available
        if model_name is None and data_obj is not None:
            try:
                exp = getattr(data_obj, "experiment", None)
                mn = getattr(exp, "model_names", [])
                mc = getattr(exp, "model_classes", [])
                for name, cls in zip(mn, mc):
                    try:
                        if isinstance(local_fit.model, cls):
                            model_name = name
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        # Capture per-fit model state and current x-range so that a
        # round-trip restores both parameters and fit limits.
        try:
            get_state = getattr(local_fit, "get_state", None)
            if callable(get_state):
                fit_state = get_state()
            else:
                fit_state = project_fit_state.fit_to_state(local_fit)
        except Exception as exc:
            log.warning(f"save_fit: could not serialize fit group #{group_index}: {exc}")
            fit_state = {}

        try:
            fr = getattr(local_fit, "fit_range", None)
            if isinstance(fr, tuple) and len(fr) == 2:
                fit_range = [int(fr[0]), int(fr[1])]
            else:
                fit_range = None
        except Exception:
            fit_range = None

        rec = {
            "dataset_id": ds_id,
            "fit_state": fit_state,
        }
        if fit_range is not None:
            rec["fit_range"] = fit_range

        local_fits_state.append(rec)

    # Capture global links if a GlobalFitModel is present and requested
    global_links_state: typing.Dict[str, typing.Any] = {}
    if include_global_links:
        global_model = getattr(fit_group, "_model", None)
        if global_model is not None:
            try:
                global_links_state = project_fit_state.global_links_to_state(global_model)
            except Exception as exc:
                log.warning(f"save_fit: could not serialize global links for fit group #{group_index}: {exc}")

    fg_key = f"fitgroup_{group_index:03d}"
    return fg_key, {
        "type": "fit_group",
        "name": getattr(fit_group, "name", fg_key),
        "model_name": model_name,
        "local_fits": local_fits_state,
        "global_links": global_links_state,
    }


def _build_fitgroup_payload_from_window(
        fit_window,
        register_datacurve: typing.Callable[[chisurf.data.DataCurve], str],
        log: typing.Any,
        group_index: int = 0,
        include_global_links: bool = True,
) -> typing.Tuple[str, typing.Dict]:
    """Legacy wrapper: extract FitGroup from a GUI window, then delegate."""
    fit_group = getattr(fit_window, "fit", None)
    return _build_fitgroup_payload(fit_group, register_datacurve, log, group_index, include_global_links)


def save_fit_project(target_path: str, fit_window=None, fit_name: str = "chisurf_fit"):
    """Save a single fit (data + model state + window) in the project format.

    The output is a folder containing ``project.json`` so it can be reloaded
    with :func:`load_fit_project` similarly to full projects, but without
    touching other open fits.
    """
    log = chisurf.logging
    cs = getattr(chisurf, "cs", None)
    # Headless-safe: cs may be None
    if fit_window is None and cs is not None:
        fit_window = getattr(cs.mdiarea, "currentSubWindow", lambda: None)()
    if fit_window is None:
        # Headless: try to use the last fit group directly
        if chisurf.fits:
            fit_group = chisurf.fits[-1]
        else:
            log.error("save_fit_project: no fit window or fit group available")
            return
    else:
        fit_group = getattr(fit_window, "fit", None)
        if fit_group is None:
            log.error("save_fit_project: fit window has no fit group")
            return

    base_dir = os.path.abspath(str(target_path))
    project_dir = os.path.join(base_dir, fit_name)

    try:
        os.makedirs(project_dir, exist_ok=True)
    except Exception as exc:
        log.error(f"save_fit_project: could not create directory {project_dir}: {exc}")
        return

    datasets: typing.Dict[str, typing.Dict] = {}
    dataset_id_by_obj: typing.Dict[int, str] = {}
    ds_counter = 0

    def register_datacurve(dc: chisurf.data.DataCurve) -> str:
        nonlocal ds_counter
        key = id(dc)
        if key in dataset_id_by_obj:
            return dataset_id_by_obj[key]
        ds_id = f"ds{ds_counter:03d}"
        ds_counter += 1
        try:
            x = np.asarray(getattr(dc, "x", []), dtype=float)
            y = np.asarray(getattr(dc, "y", []), dtype=float)
            ex = np.asarray(getattr(dc, "ex", np.zeros_like(x)), dtype=float)
            ey = np.asarray(getattr(dc, "ey", np.ones_like(y)), dtype=float)
        except Exception:
            x = np.asarray([], dtype=float)
            y = np.asarray([], dtype=float)
            ex = np.asarray([], dtype=float)
            ey = np.asarray([], dtype=float)
        filename = getattr(dc, "filename", "")
        datasets[ds_id] = {
            "name": getattr(dc, "name", ""),
            "filename": filename,
            "x": x.tolist(),
            "y": y.tolist(),
            "ex": ex.tolist(),
            "ey": ey.tolist(),
        }
        dataset_id_by_obj[key] = ds_id
        return ds_id

    fg_key, fit_payload = _build_fitgroup_payload(
        fit_group, register_datacurve, log, group_index=0
    )
    if not fit_payload:
        log.error("save_fit_project: fit payload empty; aborting")
        return

    fit_ui_state = {}
    if cs is not None:
        fit_ui_state["current_fit_index"] = getattr(cs, "fit_idx", 0)
        try:
            history_browser = getattr(cs, "historyBrowser", None)
            get_hist_state = getattr(history_browser, "get_ui_state", None)
            if callable(get_hist_state):
                fit_ui_state["history_browser"] = get_hist_state()
        except Exception:
            pass

    proj = CSProject(
        name=fit_name,
        description=f"ChiSurf fit '{fit_name}'",
        chisurf_version=getattr(chisurf.info, "__version__", None),
        datasets=datasets,
        experiments={},
        fits={fg_key: fit_payload},
        ui_state=fit_ui_state,
    )

    try:
        proj.extra["history"] = {
            "filename": HISTORY_FILENAME,
            "event_count": _history_event_count(),
        }
    except Exception:
        pass

    try:
        proj.extra["action_catalog"] = {
            "entries": get_action_catalog(),
        }
    except Exception:
        pass

    # Write both project.json (compat) and fit.json (explicit single-fit entry point)
    project_save_json(proj, project_dir)
    hist_path = _save_history_snapshot(project_dir)
    _record_history(
        action_type="fit_save",
        summary=f"save fit '{fit_name}' to '{project_dir}'",
        payload={
            "project_dir": project_dir,
            "fit_name": fit_name,
            "history_file": str(hist_path) if hist_path is not None else None,
        },
    )
    if hist_path is not None:
        _save_history_snapshot(project_dir)
    try:
        fit_json_path = os.path.join(project_dir, "fit.json")
        with open(fit_json_path, "w", encoding="utf-8") as f:
            json.dump(proj.to_dict(), f, indent=2, sort_keys=True)
    except Exception as exc:
        log.warning(f"save_fit_project: could not write fit.json: {exc}")
    log.info(f"Fit saved to {project_dir}")


def load_fit_project(project_path: str):
    """Load a single-fit project and append its fit/data to the current session.

    Existing datasets and fits are left untouched. The stored dataset(s) are
    appended, then the fit group is rebuilt and its state restored.

    Works in headless mode (without GUI / ``chisurf.cs``).
    """
    log = chisurf.logging
    cs = getattr(chisurf, "cs", None)

    proj = None
    history_base_dir = None
    if project_path.endswith(".json") and os.path.isfile(project_path):
        # Accept direct fit.json/project.json paths
        try:
            with open(project_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            proj = CSProject.from_dict(data)
            history_base_dir = os.path.dirname(project_path)
        except Exception as exc:
            log.error(f"load_fit_project: failed to read {project_path}: {exc}")
            return
    else:
        base_dir = project_path
        if not os.path.isdir(base_dir):
            log.error(f"load_fit_project: path {project_path} does not exist")
            return
        try:
            proj = project_load_json(base_dir)
            history_base_dir = base_dir
        except Exception as exc:
            log.error(f"load_fit_project: failed to read project.json from {base_dir}: {exc}")
            return

    # Append project-local history to current session history (fit import is additive).
    history_loaded = False
    try:
        if history_base_dir:
            history_loaded = bool(_load_history_snapshot(history_base_dir, replace=False))
    except Exception:
        pass

    # --- Reconstruct datasets and append to existing imports ---------------
    dataset_objects: typing.Dict[str, chisurf.data.DataCurve] = {}
    dataset_indices: typing.Dict[str, int] = {}

    for ds_id, payload in (proj.datasets or {}).items():
        try:
            name = payload.get("name", ds_id)
            filename = payload.get("filename", "")
            x = np.asarray(payload.get("x", []), dtype=float)
            y = np.asarray(payload.get("y", []), dtype=float)
            ex = np.asarray(payload.get("ex", np.zeros_like(x)), dtype=float)
            ey = np.asarray(payload.get("ey", np.ones_like(y)), dtype=float)
            dc = chisurf.data.DataCurve(x=x, y=y, ex=ex, ey=ey, name=name)

            if filename:
                try:
                    dc.filename = filename
                except Exception:
                    pass

            try:
                exp_obj = getattr(cs, "current_experiment", None) if cs is not None else None
            except Exception:
                exp_obj = None
            if exp_obj is not None:
                try:
                    dc.experiment = exp_obj
                except Exception as e_exp:
                    log.warning(f"load_fit_project: could not attach experiment to dataset {ds_id}: {e_exp}")

            dataset_objects[ds_id] = dc
            chisurf.imported_datasets.append(dc)
            dataset_indices[ds_id] = len(chisurf.imported_datasets) - 1
        except Exception as exc:
            log.warning(f"load_fit_project: could not reconstruct dataset {ds_id}: {exc}")
            continue

    if cs is not None:
        try:
            cs.dataset_selector.update()
        except Exception:
            pass

    # --- Rebuild fit groups and restore their state -----------------------
    fits_map = proj.fits or {}
    for key, rec in fits_map.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("type") != "fit_group":
            continue

        local_fits = rec.get("local_fits") or []
        if not isinstance(local_fits, list) or not local_fits:
            log.warning(f"load_fit_project: fit record {key} has no local_fits; skipping")
            continue

        group_indices: typing.List[int] = []
        for lf in local_fits:
            if not isinstance(lf, dict):
                continue
            ds_id = lf.get("dataset_id")
            if not ds_id:
                continue
            idx = dataset_indices.get(ds_id)
            if idx is not None:
                group_indices.append(idx)

        if not group_indices:
            log.warning(f"load_fit_project: fit record {key} has no valid datasets; skipping")
            continue

        model_name = rec.get("model_name")
        try:
            add_fit(dataset_indices=group_indices, model_name=model_name)
        except Exception as exc:
            log.warning(f"load_fit_project: add_fit failed for record {key}: {exc}")
            continue

        # Newly created FitGroup is appended to chisurf.fits
        try:
            fit_group = chisurf.fits[-1]
        except Exception:
            continue

        grouped_new = getattr(fit_group, "grouped_fits", [])
        for lf_rec, new_fit in zip(local_fits, grouped_new):
            state = lf_rec.get("fit_state") or {}
            if isinstance(state, dict):
                try:
                    set_state = getattr(new_fit, "set_state", None)
                    if callable(set_state):
                        set_state(state)
                    else:
                        project_fit_state.apply_state_to_fit(new_fit, state)
                except Exception as exc:
                    log.warning(f"load_fit_project: could not restore state for local fit in {key}: {exc}")

            fr = lf_rec.get("fit_range")
            if isinstance(fr, (list, tuple)) and len(fr) == 2:
                try:
                    new_fit.fit_range = (int(fr[0]), int(fr[1]))
                except Exception as exc:
                    log.warning(f"load_fit_project: could not restore fit_range for local fit in {key}: {exc}")

        global_links_state = rec.get("global_links") or {}
        if isinstance(global_links_state, dict):
            global_model = getattr(fit_group, "_model", None)
            if global_model is not None:
                try:
                    project_fit_state.apply_global_links_state(global_model, global_links_state)
                except Exception as exc:
                    log.warning(f"load_fit_project: could not restore global links for {key}: {exc}")

    # Best-effort: bring the newest fit window to front (GUI only)
    if cs is not None:
        try:
            if chisurf.gui.fit_windows:
                win = chisurf.gui.fit_windows[-1]
                win.show()
                win.setFocus()
        except Exception:
            pass

    if cs is not None:
        try:
            fit_ui_state = proj.ui_state or {}
            history_browser_state = fit_ui_state.get("history_browser") or {}
            history_browser = getattr(cs, "historyBrowser", None)
            set_hist_state = getattr(history_browser, "set_ui_state", None)
            if callable(set_hist_state) and isinstance(history_browser_state, dict):
                set_hist_state(history_browser_state)
        except Exception:
            pass

    _refresh_history_browser()
    _record_history(
        action_type="fit_load",
        summary=f"load fit project from '{project_path}'",
        payload={
            "project_path": str(project_path),
            "history_loaded": bool(history_loaded),
        },
    )


def load_project(project_path: str):
    """Load a project from a JSON-based project folder.

    The folder must contain a ``project.json`` file created by
    :func:`save_project`. Datasets are reconstructed from the stored x/y/ex/ey
    arrays, :func:`add_fit` is used to rebuild each :class:`FitGroup`, and
    per-fit parameter state plus global links are restored via
    :mod:`chisurf.project.fit_state`.

    Parameters
    ----------
    project_path : str
        Path to the project folder containing ``project.json``.
    """

    log = chisurf.logging
    cs = getattr(chisurf, "cs", None)

    if not os.path.isdir(project_path):
        log.error(f"Project path {project_path} does not exist")
        return

    try:
        proj = project_load_json(project_path)
    except Exception as exc:
        log.error(f"load_project: failed to read project.json from {project_path}: {exc}")
        return

    # Full project load replaces current operation history when available.
    history_loaded = False
    try:
        history_loaded = bool(_load_history_snapshot(project_path, replace=True))
    except Exception:
        pass

    if cs is not None:
        try:
            reinit = getattr(cs, "reinitialize", None)
            if callable(reinit):
                reinit()
            else:
                try:
                    cs.onCloseAllFits()
                except Exception:
                    pass
        except Exception:
            pass

    # Unconditionally clear headless state
    try:
        chisurf.fits.clear()
    except Exception:
        pass
    try:
        chisurf.gui.fit_windows.clear()
    except Exception:
        pass
    # We rely on reinit() to have cleared headless state as needed.
    # chisurf.imported_datasets.clear() is NOT called here because reinit()
    # already clears it while preserving the Global Dataset instance if possible.
    # We will overwrite it later with [:] slice assignment.

    # --- Restore experiment/setup state early so we can attach it to datasets
    ui_state = proj.ui_state or {}

    if cs is not None:
        try:
            current_experiment_idx = ui_state.get("current_experiment_idx", 0)
            total_exp = cs.comboBox_experimentSelect.count()
            if 0 <= current_experiment_idx < total_exp:
                cs.set_current_experiment_idx(current_experiment_idx)
        except Exception:
            pass

        try:
            current_setup_idx = ui_state.get("current_setup_idx", 0)
            total_setup = cs.comboBox_setupSelect.count()
            if 0 <= current_setup_idx < total_setup:
                cs.set_current_setup_idx(current_setup_idx)
        except Exception:
            pass

    # --- Reconstruct datasets ---------------------------------------------
    dataset_objects: typing.Dict[str, chisurf.data.DataCurve] = {}
    dataset_indices: typing.Dict[str, int] = {}

    for ds_id, payload in (proj.datasets or {}).items():
        try:
            name = payload.get("name", ds_id)
            filename = payload.get("filename", "")
            x = np.asarray(payload.get("x", []), dtype=float)
            y = np.asarray(payload.get("y", []), dtype=float)
            ex = np.asarray(payload.get("ex", np.zeros_like(x)), dtype=float)
            ey = np.asarray(payload.get("ey", np.ones_like(y)), dtype=float)
            dc = chisurf.data.DataCurve(x=x, y=y, ex=ex, ey=ey, name=name)

            # Preserve the original filename for user reference without
            # triggering a reload from disk (DataCurve.__init__ only loads
            # when given a filename argument).
            if filename:
                try:
                    dc.filename = filename
                except Exception:
                    pass

            # Rehydrate and attach the experiment reader if stored
            reader_info = payload.get("data_reader")
            reader_obj = _deserialize_reader(reader_info)
            if reader_obj is not None:
                try:
                    dc.data_reader = reader_obj
                except Exception:
                    pass

            # Re-associate with the correct experiment object if possible.
            exp_obj = None
            stored_exp_name = payload.get("experiment_name")
            
            # 1) Try to find experiment by stored name
            if stored_exp_name:
                exp_obj = chisurf.experiment.get(stored_exp_name)
            
            # 2) Special case for global datasets (fallback for older projects)
            if exp_obj is None:
                ds_name_lower = str(name or ds_id).lower()
                if "global" in ds_name_lower:
                    exp_obj = chisurf.experiment.get("Global")
                    if exp_obj is None:
                        # try case variants
                        for en in ("Global", "Global-Fit", "Global fit"):
                            exp_obj = chisurf.experiment.get(en)
                            if exp_obj is not None: break
            
            # 3) Final fallback to current experiment
            if exp_obj is None and cs is not None:
                exp_obj = getattr(cs, "current_experiment", None)
            
            if exp_obj is not None:
                try:
                    dc.experiment = exp_obj
                except Exception as e_exp:
                    log.warning(f"load_project: could not attach experiment to dataset {ds_id}: {e_exp}")

            dataset_objects[ds_id] = dc
        except Exception as exc:
            log.warning(f"load_project: could not reconstruct dataset {ds_id}: {exc}")
            continue

    # Rebuild dataset tree/list (including grouped datasets) if a layout
    # snapshot is available; otherwise fall back to plain flat curves.
    dataset_layout = ui_state.get("dataset_layout")
    restored_datasets: typing.List[typing.Any] = []
    used_dataset_ids: typing.Set[str] = set()

    if isinstance(dataset_layout, list) and dataset_layout:
        for rec in dataset_layout:
            if not isinstance(rec, dict):
                continue
            kind = rec.get("kind")

            if kind == "dataset":
                ds_id = rec.get("dataset_id")
                dc = dataset_objects.get(ds_id)
                if dc is None:
                    continue
                dataset_indices[ds_id] = len(restored_datasets)
                restored_datasets.append(dc)
                used_dataset_ids.add(ds_id)
                continue

            if kind == "group":
                member_ids = rec.get("dataset_ids")
                if not isinstance(member_ids, list):
                    continue
                members = [dataset_objects.get(ds_id) for ds_id in member_ids]
                members = [dc for dc in members if dc is not None]
                if not members:
                    continue

                group_class = chisurf.data.ExperimentDataCurveGroup
                if rec.get("group_type") == "ExperimentDataGroup":
                    group_class = chisurf.data.ExperimentDataGroup
                try:
                    group_obj = group_class(members)
                except Exception:
                    group_obj = chisurf.data.ExperimentDataCurveGroup(members)

                group_name = rec.get("name")
                if isinstance(group_name, str) and group_name:
                    group_obj.__dict__["name"] = group_name

                try:
                    current_idx = int(rec.get("current_dataset_index", 0))
                except Exception:
                    current_idx = 0
                if 0 <= current_idx < len(group_obj):
                    group_obj._current_dataset = current_idx

                dataset_idx = len(restored_datasets)
                restored_datasets.append(group_obj)
                for ds_id in member_ids:
                    if ds_id in dataset_objects:
                        dataset_indices[ds_id] = dataset_idx
                        used_dataset_ids.add(ds_id)

    # Append any datasets missing from layout (backstop for partial records).
    for ds_id, dc in dataset_objects.items():
        if ds_id in used_dataset_ids:
            continue
        dataset_indices[ds_id] = len(restored_datasets)
        restored_datasets.append(dc)

    chisurf.imported_datasets[:] = restored_datasets

    if cs is not None:
        try:
            cs.dataset_selector.update()
        except Exception:
            pass

    # --- Rebuild fit groups and restore their state -----------------------
    fits_list = proj.fits or []
    fits_root = pathlib.Path(project_path)
    for rec in fits_list:
        if not isinstance(rec, dict):
            continue
        key = rec.get("id") or rec.get("name")
        local_fits = rec.get("local_fits") or []
        if not isinstance(local_fits, list) or not local_fits:
            log.warning(f"load_project: fit record {key} has no local_fits; skipping")
            continue

        # Determine dataset indices for this group
        group_indices: typing.List[int] = []
        for lf in local_fits:
            if not isinstance(lf, dict):
                continue
            ds_id = lf.get("dataset_id")
            if not ds_id:
                continue
            idx = dataset_indices.get(ds_id)
            if idx is not None:
                group_indices.append(idx)

        # Keep first occurrence order while removing duplicates; grouped
        # datasets intentionally map multiple local fits to one dataset index.
        deduped_indices: typing.List[int] = []
        seen_indices: typing.Set[int] = set()
        for idx in group_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            deduped_indices.append(idx)
        group_indices = deduped_indices

        if not group_indices:
            log.warning(f"load_project: fit record {key} has no valid datasets; skipping")
            continue

        model_name = rec.get("model_name")
        try:
            add_fit(dataset_indices=group_indices, model_name=model_name)
        except Exception as exc:
            log.warning(f"load_project: add_fit failed for record {key}: {exc}")
            continue

        # Newly created FitGroup is appended to chisurf.fits
        try:
            fit_group = chisurf.fits[-1]
        except Exception:
            continue

        grouped_new = getattr(fit_group, "grouped_fits", [])
        for lf_rec, new_fit in zip(local_fits, grouped_new):
            # Load per-fit state directly from manifest (no external files)
            state = lf_rec.get("fit_state") or {}

            if isinstance(state, dict):
                try:
                    set_state = getattr(new_fit, "set_state", None)
                    if callable(set_state):
                        set_state(state)
                    else:
                        project_fit_state.apply_state_to_fit(new_fit, state)
                except Exception as exc:
                    log.warning(f"load_project: could not restore state for local fit in {key}: {exc}")

            fr = lf_rec.get("fit_range")
            if isinstance(fr, (list, tuple)) and len(fr) == 2:
                try:
                    new_fit.fit_range = (int(fr[0]), int(fr[1]))
                except Exception as exc:
                    log.warning(f"load_project: could not restore fit_range for local fit in {key}: {exc}")

        # Restore global links (if any)
        global_links_state = rec.get("global_links") or {}
        if isinstance(global_links_state, dict):
            global_model = getattr(fit_group, "_model", None)
            if global_model is not None:
                try:
                    project_fit_state.apply_global_links_state(global_model, global_links_state)
                except Exception as exc:
                    log.warning(f"load_project: could not restore global links for {key}: {exc}")

    # --- Restore UI state (current fit and window layout) -----------------
    if cs is not None:
        current_fit_idx = ui_state.get("current_fit_index", 0)
        if 0 <= current_fit_idx < len(chisurf.fits):
            try:
                cs.current_fit = chisurf.fits[current_fit_idx]
            except Exception:
                pass

        try:
            cs.fit_selector.update()
        except Exception:
            pass

        # Restore main-window and MDI geometry/state if present. This should be
        # done only after datasets and fits (and thus subwindows) have been
        # recreated so that Qt has matching widgets to apply the layout to.
        try:
            from chisurf.project.ui_state import set_ui_state
            set_ui_state(cs, ui_state)
        except Exception as exc:
            log.warning(f"load_project: could not restore UI state from dict: {exc}")

        # Trigger a single GUI refresh now that datasets, fits and layout are consistent.
        try:
            cs.update()
        except Exception:
            pass

    _refresh_history_browser()
    _record_history(
        action_type="project_load",
        summary=f"load project from '{project_path}'",
        payload={
            "project_path": str(project_path),
            "history_loaded": bool(history_loaded),
        },
    )

    log.info(f"Project loaded from {project_path}")
