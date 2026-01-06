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


def group_datasets(dataset_indices: typing.List[int]) -> None:
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


def remove_datasets(dataset_indices: typing.List[int]) -> None:
    if not isinstance(dataset_indices, list):
        dataset_indices = [dataset_indices]

    imported_datasets = list()
    for i, d in enumerate(chisurf.imported_datasets):
        if d.name == 'Global Dataset':
            imported_datasets.append(d)
            continue
        if i not in dataset_indices:
            imported_datasets.append(d)
        else:
            fw = list()
            for fit_window in chisurf.gui.fit_windows:
                if fit_window.fit.data is d:
                    fit_window.close_confirm = False
                    fit_window.close()
                else:
                    fw.append(fit_window)
            chisurf.gui.fit_windows = fw

    chisurf.imported_datasets = imported_datasets


def add_dataset(
        experiment_reader: chisurf.experiments.core.reader.ExperimentReader = None,
        dataset: chisurf.base.Data = None,
        **kwargs
) -> None:
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
            refs_to_clean = ['cs', 'current_dataset', 'current_fit']
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
    _run('Force garbage collection', _force_garbage_collection, 10)
    _run('Resetting GUI components', _reset_gui_components, 10)


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
