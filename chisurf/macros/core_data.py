from __future__ import annotations

import pathlib
import traceback

import chisurf
import chisurf.base
import chisurf.data
import chisurf.experiments
import chisurf.experiments.modelling
import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets

from chisurf import typing, logging
from chisurf.data import DataGroup, ExperimentDataGroup, ExperimentDataCurveGroup
from chisurf.runtime.actions import record_action


def _record_history(
        action_type: str,
        summary: str,
        payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> None:
    try:
        record_action(action_type=action_type, summary=summary, payload=payload)
    except Exception:
        pass


def _flatten_dataset(dataset: chisurf.base.Data) -> typing.List[chisurf.base.Data]:
    result: typing.List[chisurf.base.Data] = []
    seen: typing.Set[int] = set()
    stack = [dataset]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        uid = id(current)
        if uid in seen:
            continue
        seen.add(uid)
        result.append(current)
        if isinstance(current, (DataGroup, ExperimentDataGroup, ExperimentDataCurveGroup)):
            stack.extend(current)
    return result


def _fit_uses_dataset(fit: chisurf.fitting.fit.Fit, datasets: typing.List[chisurf.base.Data]) -> bool:
    dataset_ids = {id(d) for d in datasets}
    for data_obj in _iter_fit_data(fit):
        if data_obj is None:
            continue
        if id(data_obj) in dataset_ids:
            return True
    return False


def _iter_fit_data(fit: chisurf.fitting.fit.Fit) -> typing.Iterator[chisurf.base.Data | None]:
    grouped = getattr(fit, 'grouped_fits', None)
    if isinstance(grouped, (list, tuple)):
        for member in grouped:
            yield getattr(member, 'data', None)
    yield getattr(fit, 'data', None)


def group_datasets(
        dataset_indices: typing.List[int],
        _from_controller: bool = False,
) -> None:
    if not _from_controller:
        chisurf.action_controller.execute(
            name="dataset.group",
            payload={"dataset_indices": [int(i) for i in dataset_indices]},
        )
        return

    selected_data = [
        chisurf.imported_datasets[i] for i in dataset_indices
    ]
    if isinstance(
            selected_data[0],
            chisurf.data.DataCurve
    ):
        # TODO: check for double names!!!
        dg = chisurf.data.ExperimentDataCurveGroup(
            selected_data,
            name="Data-Group"
        )
    else:
        dg = chisurf.data.ExperimentDataGroup(
            selected_data,
            name="Data-Group"
        )
    dn = list()
    for d in chisurf.imported_datasets:
        if d not in dg:
            dn.append(d)
    dn.append(dg)
    chisurf.imported_datasets = dn
    _record_history(
        action_type="dataset_group",
        summary=f"group datasets into '{getattr(dg, 'name', 'Data-Group')}'",
        payload={
            "dataset_indices": [int(i) for i in dataset_indices],
            "group_size": int(len(dg)),
            "group_name": str(getattr(dg, "name", "Data-Group")),
            "group_uid": str(getattr(dg, "unique_identifier", "")),
            "member_uids": [str(getattr(d, "unique_identifier", "")) for d in list(selected_data)],
        },
    )


def ungroup_datasets(
        dataset_indices: typing.List[int],
        _from_controller: bool = False,
) -> None:
    if not _from_controller:
        chisurf.action_controller.execute(
            name="dataset.ungroup",
            payload={"dataset_indices": [int(i) for i in list(dataset_indices or [])]},
        )
        return

    if not isinstance(dataset_indices, list):
        dataset_indices = [dataset_indices]
    idx_set = {int(i) for i in dataset_indices if isinstance(i, (int, float))}
    if not idx_set:
        return

    new_imported: typing.List[chisurf.base.Data] = []
    group_names: typing.List[str] = []
    group_uids: typing.List[str] = []
    expanded_members = 0
    expanded_names: typing.List[str] = []
    expanded_uids: typing.List[str] = []

    for i, d in enumerate(chisurf.imported_datasets):
        if i in idx_set and isinstance(d, chisurf.data.ExperimentDataGroup):
            try:
                group_names.append(str(getattr(d, "name", f"group_{i}")))
            except Exception:
                group_names.append(f"group_{i}")
            try:
                group_uids.append(str(getattr(d, "unique_identifier", "")))
            except Exception:
                group_uids.append("")
            members = list(d)
            expanded_members += len(members)
            for member in members:
                try:
                    member_name = str(getattr(member, "name", ""))
                except Exception:
                    member_name = ""
                if member_name:
                    expanded_names.append(member_name)
                try:
                    expanded_uids.append(str(getattr(member, "unique_identifier", "")))
                except Exception:
                    expanded_uids.append("")
            new_imported.extend(members)
        else:
            new_imported.append(d)

    if not group_names:
        return

    chisurf.imported_datasets = new_imported
    _record_history(
        action_type="dataset_ungroup",
        summary=f"ungroup {len(group_names)} dataset group(s)",
        payload={
            "dataset_indices": sorted(idx_set),
            "group_names": group_names,
            "group_uids": group_uids,
            "group_count": int(len(group_names)),
            "expanded_member_count": int(expanded_members),
            "expanded_names": expanded_names,
            "expanded_uids": expanded_uids,
        },
    )


def remove_datasets(
        dataset_indices: typing.List[int],
        _from_controller: bool = False,
) -> None:
    if not _from_controller:
        chisurf.action_controller.execute(
            name="dataset.remove",
            payload={"dataset_indices": [int(i) for i in list(dataset_indices or [])]},
        )
        return

    if not isinstance(dataset_indices, list):
        dataset_indices = [dataset_indices]

    dataset_indices = sorted(set(dataset_indices))
    actual_indices: typing.List[int] = []
    removed_names: typing.List[str] = []
    removed_uids: typing.List[str] = []
    to_remove: typing.List[chisurf.base.Data] = []
    for i in dataset_indices:
        if i < 0 or i >= len(chisurf.imported_datasets):
            continue
        d = chisurf.imported_datasets[i]
        if getattr(d, 'name', '') == 'Global Dataset':
            continue
        actual_indices.append(i)
        to_remove.append(d)
        try:
            removed_names.append(str(getattr(d, "name", f"dataset_{i}")))
        except Exception:
            removed_names.append(f"dataset_{i}")
        try:
            removed_uids.append(str(getattr(d, "unique_identifier", "")))
        except Exception:
            removed_uids.append("")

    if not actual_indices:
        return

    datasets_to_remove: typing.List[chisurf.base.Data] = []
    seen_ids: typing.Set[int] = set()
    for dataset in to_remove:
        for entry in _flatten_dataset(dataset):
            uid = id(entry)
            if uid in seen_ids:
                continue
            seen_ids.add(uid)
            datasets_to_remove.append(entry)

    dependent_fit_indices = [
        idx for idx, fit in enumerate(list(chisurf.fits))
        if _fit_uses_dataset(fit, datasets_to_remove)
    ]

    message = (
        f"Remove {len(actual_indices)} dataset(s)?"
        f"\nDependent fits to close: {len(dependent_fit_indices)}."
        f"\nPlease confirm to proceed."
    )
    cs = getattr(chisurf, 'cs', None)
    proceed = True
    if cs is not None:
        reply = chisurf.gui.QtWidgets.QMessageBox.question(
            cs,
            "Remove dataset(s)",
            message,
            chisurf.gui.QtWidgets.QMessageBox.Yes | chisurf.gui.QtWidgets.QMessageBox.No,
            chisurf.gui.QtWidgets.QMessageBox.No,
        )
        proceed = reply == chisurf.gui.QtWidgets.QMessageBox.Yes
    else:
        logging.info("""Removing datasets without GUI confirmation: %s""", message)

    if not proceed:
        return

    if dependent_fit_indices:
        old_confirm = chisurf.settings.gui.get('confirm_close_fit', True)
        chisurf.settings.gui['confirm_close_fit'] = False
        try:
            for idx in sorted(dependent_fit_indices, reverse=True):
                chisurf.action_controller.execute(
                    name="fit.close",
                    payload={"idx": int(idx)},
                )
        finally:
            chisurf.settings.gui['confirm_close_fit'] = old_confirm

    actual_idx_set = set(actual_indices)
    new_imported = []
    for i, d in enumerate(chisurf.imported_datasets):
        if i not in actual_idx_set:
            new_imported.append(d)
    chisurf.imported_datasets = new_imported
    _record_history(
        action_type="dataset_remove",
        summary=f"remove {len(removed_names)} dataset(s)",
        payload={
            "dataset_indices": [int(i) for i in actual_indices],
            "removed_names": removed_names,
            "removed_uids": removed_uids,
            "removed_count": int(len(removed_names)),
        },
    )


def add_dataset(
        experiment_reader: chisurf.experiments.core.reader.ExperimentReader = None,
        dataset: chisurf.base.Data = None,
        _from_controller: bool = False,
        **kwargs
) -> None:
    if not _from_controller:
        payload = dict(kwargs)
        payload["experiment_reader"] = experiment_reader
        payload["dataset"] = dataset
        chisurf.action_controller.execute(
            name="dataset.add",
            payload=payload,
        )
        return

    try:
        cs = getattr(chisurf, 'cs', None)

        # High-level entry trace for PDA crash localization
        try:
            logging.info(
                "PDA TRACE: core_data.add_dataset called (experiment_reader=%s, has_dataset=%s)",
                getattr(experiment_reader, 'name', type(experiment_reader).__name__) if experiment_reader is not None else None,
                dataset is not None,
            )
        except Exception:
            pass

        filename = kwargs.get('filename', None)
        primary_filename = None
        if isinstance(filename, (list, tuple)):
            if filename:
                primary_filename = filename[0]
        elif isinstance(filename, str):
            parts = filename.split('|')
            if len(parts) == 1:
                primary_filename = parts[0]
                filename = parts[0]
            else:
                filename = parts
                primary_filename = parts[0]
        elif filename is not None:
            primary_filename = str(filename)
        kwargs['filename'] = filename

        try:
            logging.info(
                "PDA TRACE: core_data.add_dataset normalized filename=%r (primary=%r)",
                filename,
                primary_filename,
            )
        except Exception:
            pass

        if experiment_reader is None:
            try:
                experiment_reader = getattr(cs, 'current_experiment_reader')
            except Exception:
                experiment_reader = None

        if experiment_reader is None and primary_filename:
            experiment_reader = _auto_reader_from_filename(primary_filename)

        try:
            logging.info(
                "PDA TRACE: core_data.add_dataset using experiment_reader=%s",
                getattr(experiment_reader, 'name', type(experiment_reader).__name__) if experiment_reader is not None else None,
            )
        except Exception:
            pass

        # Obtain dataset if not provided
        if dataset is None and experiment_reader is not None:
            try:
                logging.info(
                    "PDA TRACE: core_data.add_dataset calling experiment_reader.get_data(...)",
                )
            except Exception:
                pass
            dataset = experiment_reader.get_data(**kwargs)
            try:
                logging.info(
                    "PDA TRACE: core_data.add_dataset get_data returned object of type %s",
                    type(dataset).__name__,
                )
            except Exception:
                pass

        # If nothing was read, inform the user and exit safely
        if dataset is None:
            try:
                logging.info(
                    "PDA TRACE: core_data.add_dataset received no dataset (dataset is None); showing error message box.",
                )
            except Exception:
                pass
            chisurf.gui.widgets.msg_box = chisurf.gui.widgets.MyMessageBox(
                label="Error",
                info="No data could be read. Check reading settings and file.",
                details=(
                    "Reader returned no dataset."
                    if experiment_reader is not None
                    else "No experiment reader available for the provided file."
                )
            )
            return

        # Normalize to a group without modifying global state yet.
        #
        # Goal: imported_datasets should contain either plain
        # ExperimentalData instances or ExperimentDataGroup instances, so
        # that chisurf.data.get_data(curve_type='experiment', ...) and the
        # ExperimentalDataSelector behave correctly. Some readers (e.g.
        # TCSPCReader) return DataCurveGroup/DataGroup objects; those need
        # to be converted so that the *elements* become group members,
        # instead of wrapping the group itself as a single element.
        is_experiment_group = isinstance(dataset, chisurf.data.ExperimentDataGroup)
        if is_experiment_group:
            # Already in the expected grouped form
            dataset_group = dataset
        elif isinstance(dataset, (chisurf.data.DataGroup, list, tuple)):
            # Flatten DataGroup/DataCurveGroup or simple sequences into an
            # ExperimentDataCurveGroup of their elements.
            dataset_group = chisurf.data.ExperimentDataCurveGroup(list(dataset))
        else:
            # Single ExperimentalData object
            dataset_group = chisurf.data.ExperimentDataCurveGroup([dataset])

        try:
            logging.info(
                "PDA TRACE: core_data.add_dataset normalized to dataset_group (is_experiment_group=%s, len=%d)",
                is_experiment_group,
                len(dataset_group),
            )
        except Exception:
            pass

        # Guard against empty groups which would break the UI (d[0])
        if len(dataset_group) == 0:
            try:
                logging.info(
                    "PDA TRACE: core_data.add_dataset found empty dataset_group; showing error message box.",
                )
            except Exception:
                pass
            chisurf.gui.widgets.msg_box = chisurf.gui.widgets.MyMessageBox(
                label="Error",
                info="No data entries found in the selected file using the current reader.",
                details=f"Reader: {getattr(experiment_reader, 'name', type(experiment_reader).__name__)}\nFilename: {filename}"
            )
            return

        # Append valid data. Preserve ExperimentDataGroup objects even when
        # they currently hold a single entry so the GUI can still treat them
        # as experiment datasets (e.g. structure modelling results).
        try:
            logging.info(
                "PDA TRACE: core_data.add_dataset appending dataset_group (is_experiment_group=%s, len=%d, imported_before=%d)",
                is_experiment_group,
                len(dataset_group),
                len(getattr(chisurf, 'imported_datasets', [])),
            )
        except Exception:
            pass

        if is_experiment_group:
            chisurf.imported_datasets.append(dataset_group)
        elif len(dataset_group) == 1:
            chisurf.imported_datasets.append(dataset_group[0])
        else:
            chisurf.imported_datasets.append(dataset_group)

        # Update UI only after successful append
        try:
            logging.info(
                "PDA TRACE: core_data.add_dataset calling run_on_gui_thread(cs.update); imported_after=%d",
                len(getattr(chisurf, 'imported_datasets', [])),
            )
        except Exception:
            pass
        chisurf.gui.run_on_gui_thread(cs.update)

        try:
            logging.info("PDA TRACE: core_data.add_dataset finished successfully")
        except Exception:
            pass

        loaded_names: typing.List[str] = []
        loaded_uids: typing.List[str] = []
        try:
            if is_experiment_group:
                loaded_names.append(str(getattr(dataset_group, "name", "ExperimentDataGroup")))
                loaded_uids.append(str(getattr(dataset_group, "unique_identifier", "")))
            elif len(dataset_group) == 1:
                loaded_names.append(str(getattr(dataset_group[0], "name", "dataset")))
                loaded_uids.append(str(getattr(dataset_group[0], "unique_identifier", "")))
            else:
                for d in dataset_group:
                    loaded_names.append(str(getattr(d, "name", "dataset")))
                    loaded_uids.append(str(getattr(d, "unique_identifier", "")))
        except Exception:
            loaded_names = []
            loaded_uids = []

        _record_history(
            action_type="dataset_add",
            summary=f"add dataset(s): {', '.join(loaded_names) if loaded_names else 'unknown'}",
            payload={
                "filename": primary_filename,
                "reader": getattr(experiment_reader, "name", type(experiment_reader).__name__) if experiment_reader is not None else None,
                "loaded_names": loaded_names,
                "loaded_uids": loaded_uids,
                "loaded_count": int(len(loaded_names)),
                "is_experiment_group": bool(is_experiment_group),
            },
        )

    except Exception as e:
        # Capture the full error trace
        error_trace = traceback.format_exc()
        # Show the error popup
        chisurf.gui.widgets.msg_box = chisurf.gui.widgets.MyMessageBox(
            label="Error",
            info="Error reading data. Check Reading settings and file.",
            details=error_trace
        )


def reinitialize_application(
        main_window=None,
        progress_callback=None
) -> None:
    """
    Reinitialize ChiSurf application by clearing all data and resetting state.
    Performs safe cleanup without affecting Python built-ins.
    
    Parameters
    ----------
    main_window : QtWidgets.QMainWindow, optional
        Main window instance for UI operations
    progress_callback : callable, optional
        Callback function for progress updates with signature (step_name, progress_value)
    """
    import gc

    _record_history(
        action_type="app_reinitialize_start",
        summary="start application reinitialize",
        payload={
            "has_main_window": bool(main_window is not None),
        },
    )
    
    log = getattr(chisurf, 'logging', None)

    def _log_exception(step: str) -> None:
        try:
            exc_fn = getattr(log, 'exception', None)
            if callable(exc_fn):
                exc_fn(f"reinitialize: {step} failed")
        except Exception:
            pass

    def _run(step: str, fn, progress_value: int) -> None:
        try:
            if progress_callback:
                progress_callback(step, progress_value)
            fn()
        except Exception as e:
            _log_exception(f"{step}: {str(e)}")

    def _close_subwindows() -> None:
        if main_window and hasattr(main_window, 'mdiarea'):
            for sw in list(main_window.mdiarea.subWindowList()):
                try:
                    sw.close()
                except Exception:
                    pass

    def _clear_imported_datasets_keep_global():
        try:
            global_datasets = [
                d for d in chisurf.imported_datasets
                if hasattr(d, 'name') and d.name == 'Global Dataset'
            ]
        except Exception:
            global_datasets = []
        try:
            chisurf.imported_datasets.clear()
            chisurf.imported_datasets.extend(global_datasets)
        except Exception:
            _log_exception('clear imported_datasets')

    def _clear_fit_windows():
        """Close and clean up all fit windows"""
        try:
            if hasattr(chisurf.gui, 'fit_windows'):
                fit_windows = list(chisurf.gui.fit_windows)
                chisurf.gui.fit_windows.clear()
                
                for fw in fit_windows:
                    try:
                        if hasattr(fw, 'close_confirm'):
                            fw.close_confirm = False
                        fw.close()
                    except Exception:
                        pass
        except Exception:
            _log_exception('clear fit windows')

    def _clear_global_caches():
        """Clear specific chisurf caches safely"""
        try:
            # Only clear specific known caches, don't iterate over all attributes
            cache_modules = [
                ('chisurf.experiments', 'types'),
                ('chisurf.fitting', None)  # None means clear all callable clear methods
            ]
            
            for module_path, attr_name in cache_modules:
                try:
                    module_parts = module_path.split('.')
                    module = chisurf
                    for part in module_parts[:-1]:
                        if hasattr(module, part):
                            module = getattr(module, part)
                        else:
                            break
                    else:
                        if hasattr(module, module_parts[-1]):
                            target = getattr(module, module_parts[-1])
                            if attr_name and hasattr(target, attr_name):
                                attr = getattr(target, attr_name)
                                if hasattr(attr, 'clear_cache') and callable(attr.clear_cache):
                                    attr.clear_cache()
                            elif hasattr(target, 'clear') and callable(target.clear):
                                target.clear()
                except Exception:
                    pass
                            
        except Exception:
            _log_exception('clear global caches')

    def _force_garbage_collection():
        """Perform safe garbage collection"""
        try:
            collected = gc.collect()
            if progress_callback:
                progress_callback(f"Garbage collection (collected {collected} objects)", 10)
        except Exception:
            _log_exception('garbage collection')

    def _cleanup_specific_references():
        """Clean only specific chisurf references safely"""
        try:
            # Only clean specific, known chisurf attributes
            # Keep `cs` alive: it is the main-window anchor used by project
            # save/load and many macros. Removing it breaks close->open cycles.
            refs_to_clean = ['current_dataset', 'current_fit']
            for ref_name in refs_to_clean:
                if hasattr(chisurf, ref_name):
                    try:
                        attr = getattr(chisurf, ref_name)
                        # Only delete if it's a data object, not a function or type
                        if not callable(attr) and not isinstance(attr, type) and not isinstance(attr, (int, float, str, bool, list, dict)):
                            delattr(chisurf, ref_name)
                    except Exception:
                        pass
        except Exception:
            _log_exception('cleanup specific references')

    def _reset_gui_components():
        """Reset GUI components to clean state"""
        if main_window is None:
            return
            
        try:
            # Clear model selector
            if hasattr(main_window, 'comboBox_Model'):
                main_window.comboBox_Model.clear()
                
            # Reset data selectors
            if hasattr(main_window, 'dataset_selector'):
                if hasattr(main_window.dataset_selector, 'clear'):
                    main_window.dataset_selector.clear()
                if hasattr(main_window.dataset_selector, 'update'):
                    main_window.dataset_selector.update()
                    
            # Reset fit selectors
            if hasattr(main_window, 'fit_selector'):
                if hasattr(main_window.fit_selector, 'clear'):
                    main_window.fit_selector.clear()
                if hasattr(main_window.fit_selector, 'update'):
                    main_window.fit_selector.update()
                    
        except Exception:
            _log_exception('reset GUI components')

    # Execute reinitialization steps
    _run('Closing all fits', lambda: main_window.onCloseAllFits() if main_window else None, 1)
    _run('Closing subwindows', _close_subwindows, 2)
    _run('Clearing fit windows', _clear_fit_windows, 3)
    _run('Clearing datasets', _clear_imported_datasets_keep_global, 4)
    _run('Clearing global caches', _clear_global_caches, 5)
    _run('Updating dataset selector', lambda: main_window.dataset_selector.update() if main_window else None, 6)
    _run('Updating fit selector', lambda: main_window.fit_selector.update() if main_window else None, 7)

    def _reset_state():
        if main_window:
            main_window._current_dataset = None
            main_window._current_fit = None
            main_window._fit_idx = 0

    _run('Resetting application state', _reset_state, 8)
    _run('Cleaning up references', _cleanup_specific_references, 9)
    _run('Restoring main window reference', lambda: setattr(chisurf, 'cs', main_window) if main_window is not None else None, 9)
    _run('Force garbage collection', _force_garbage_collection, 10)
    _run('Resetting GUI components', _reset_gui_components, 10)

    _record_history(
        action_type="app_reinitialize_finish",
        summary="finish application reinitialize",
        payload={
            "has_main_window": bool(main_window is not None),
        },
    )


def _auto_reader_from_filename(filename: str):
    """Return a best-effort experiment reader based on the filename."""
    if not filename:
        return None

    suffix = pathlib.Path(filename).suffix.lower()
    structure_ext = {'.pdb', '.cif', '.mmcif', '.gro', '.xyz'}
    if suffix in structure_ext:
        experiment = chisurf.experiment.get('Modelling')
        if experiment is None:
            experiment = chisurf.experiment.get('structure')
        if experiment is None:
            experiment = chisurf.experiments.types.get('structure')
            if experiment is not None:
                chisurf.experiment[experiment.name] = experiment
        reader = chisurf.experiments.modelling.StructureReader(
            name='Structure',
            experiment=experiment
        )
        return reader
    return None
