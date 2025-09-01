from __future__ import annotations

import os
import gc
import shutil

import chisurf
import chisurf.base
import chisurf.data
import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets

from chisurf import typing
from chisurf import logging


def add_fit(
        dataset_indices: typing.List[int] = None,
        model_name: str = None,
        model_kw: typing.Dict = None
):
    cs = chisurf.cs
    # Process inputs of macro and replace None
    # with more sensible values that are read
    # from the GUI
    if dataset_indices is None:
        dataset_indices = [cs.dataset_selector.selected_curve_index]
    if model_name is None:
        model_name = cs.current_model_name

    # Do nothing of no dataset is selected
    if len(dataset_indices) == 0:
        return

    # create a list of data sets to which a fit with
    # a particular model is added
    data_sets = [cs.dataset_selector.datasets[i] for i in dataset_indices]

    model_names = data_sets[0].experiment.model_names
    model_class = data_sets[0].experiment.model_classes[0]
    for model_idx, mn in enumerate(model_names):
        if mn == model_name:
            model_class = data_sets[0].experiment.model_classes[model_idx]
            break

    for data_set in data_sets:
        if data_set.experiment is data_sets[0].experiment:
            # Make sure the data set is a DataGroup
            if not isinstance(data_set, chisurf.data.DataGroup):
                data_group = chisurf.data.ExperimentDataCurveGroup([data_set])
            else:
                data_group = data_set

            # Create the fit
            fit_group = chisurf.fitting.fit.FitGroup(
                data=data_group,
                model_class=model_class,
                model_kw=model_kw
            )
            chisurf.fits.append(fit_group)

            # Batch UI updates to avoid repeated repaints while constructing widgets
            mdl_parent = getattr(cs.modelLayout, 'parentWidget', lambda: None)()
            plo_parent = getattr(cs.plotOptionsLayout, 'parentWidget', lambda: None)()
            try:
                if mdl_parent: mdl_parent.setUpdatesEnabled(False)
                if plo_parent: plo_parent.setUpdatesEnabled(False)
                cs.mdiarea.setUpdatesEnabled(False)

                fit_control_widget = chisurf.gui.widgets.fitting.FittingControllerWidget(
                    fit=fit_group
                )
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
                # Defer auto-fit range to run after the window is shown to avoid blocking Add Fit
                try:
                    chisurf.gui.QtCore.QTimer.singleShot(0, fit_control_widget.onAutoFitRange)
                except Exception:
                    fit_control_widget.onAutoFitRange()
            finally:
                # Re-enable updates and show
                cs.mdiarea.setUpdatesEnabled(True)
                if mdl_parent: mdl_parent.setUpdatesEnabled(True)
                if plo_parent: plo_parent.setUpdatesEnabled(True)
                fit_window.show()
    cs.update()


def save_fit(target_path: str = None,
             use_complex_name: bool = False,
             fit_window=None):
    log = chisurf.logging
    log.debug("save_fit: start (target_path=%r, use_complex_name=%r)",
              target_path, use_complex_name)

    cs = chisurf.cs
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

    basename = os.path.join(target_path, save_name)
    log.info("Will write files with base %r", basename)

    # 1) dump numeric data
    log.debug("Saving fit CSV and curves to %r.csv", basename)
    fit.save(basename, 'csv', save_curves=True)
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

            fit_name = fit.name
            p2 = os.path.join(target_path, save_name)

            # Ensure per-fit directory handling with overwrite/skip/cancel dialog
            if os.path.exists(p2):
                try:
                    from PyQt5.QtWidgets import QMessageBox
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Question)
                    msg.setWindowTitle("Folder exists")
                    msg.setText(f"The folder '{p2}' already exists.")
                    msg.setInformativeText("Do you want to overwrite it?")
                    overwrite_btn = msg.addButton("Overwrite", QMessageBox.AcceptRole)
                    skip_btn = msg.addButton("Skip", QMessageBox.RejectRole)
                    cancel_btn = msg.addButton("Cancel", QMessageBox.DestructiveRole)
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
            per_fit_docx = os.path.join(p2, f"{save_name}.docx")
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
    cs = chisurf.cs
    if idx is None:
        sub_window = cs.mdiarea.currentSubWindow()
        for i, w in enumerate(chisurf.gui.fit_windows):
            if w is sub_window:
                idx = i
    chisurf.fits.pop(idx)
    sub_window = chisurf.gui.fit_windows.pop(idx)
    sub_window.close()
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
    cs = chisurf.cs
    if csi == 2:
        current_fit = cs.current_fit
        parameter = current_fit.model.parameters_all_dict[fitting_parameter_name]
        for f in cs.current_fit:
            try:
                p = f.model.parameters_all_dict[fitting_parameter_name]
                if p is not parameter:
                    p.link = parameter
            except KeyError:
                chisurf.logging.warning(f"The fit {f.name} has no parameter {fitting_parameter_name}")
    if csi == 0:
        for f in cs.current_fit:
            try:
                p = f.model.parameters_all_dict[fitting_parameter_name]
                p.link = None
            except KeyError:
                pass


def change_selected_fit_of_group(
    selected_fit: int
) -> None:
    """
    Changes the currently selected fit

    :param selected_fit:
    :return:
    """
    cs = chisurf.cs
    cs.current_fit.model.hide()
    cs.current_fit.selected_fit = selected_fit
    cs.current_fit.update()
    cs.current_fit.model.show()


def save_project(target_path: str, project_name: str = "chisurf_project"):
    """
    Save the current state of the application as a project.

    This function saves all fits, their parameters, dependencies (links across fits),
    and the UI state to a project folder.

    Parameters
    ----------
    target_path : str
        The directory where the project folder will be created
    project_name : str
        The name of the project folder (default: "chisurf_project")
    """
    import os
    import pathlib
    import yaml
    import datetime

    cs = chisurf.cs

    # Create project directory
    project_dir = os.path.join(target_path, project_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # Create fits directory
    fits_dir = os.path.join(project_dir, "fits")
    if not os.path.exists(fits_dir):
        os.makedirs(fits_dir)

    # Save project metadata
    metadata = {
        "project_name": project_name,
        "created_date": datetime.datetime.now().isoformat(),
        "chisurf_version": chisurf.info.__version__,
        "fits": []
    }

    # Save all fits
    for i, fit_window in enumerate(chisurf.gui.fit_windows):
        fit = fit_window.fit

        # Create a unique name for the fit
        fit_name = f"fit_{i:03d}"
        if hasattr(fit, 'name') and fit.name:
            fit_name = f"{fit_name}_{chisurf.base.clean_string(fit.name)}"

        # Create fit directory
        fit_dir = os.path.join(fits_dir, fit_name)
        if not os.path.exists(fit_dir):
            os.makedirs(fit_dir)

        # Save fit
        save_fit(target_path=fit_dir, fit_window=fit_window)

        # Add fit metadata
        try:
            model_obj = fit.model
            model_name = type(model_obj).__name__
            model_module = type(model_obj).__module__
        except Exception:
            model_name = None
            model_module = None
        try:
            exp_name = getattr(getattr(fit, 'data', None), 'experiment', None)
            if exp_name is not None:
                exp_name = getattr(exp_name, 'name', None) or str(exp_name)
        except Exception:
            exp_name = None
        fit_metadata = {
            "fit_name": fit_name,
            "original_name": fit.name if hasattr(fit, 'name') else "",
            "fit_index": i,
            "model_name": model_name,
            "model_module": model_module,
            "experiment_name": exp_name,
            "parameters": {}
        }

        # Save parameter links
        if hasattr(fit, 'model') and hasattr(fit.model, 'parameters_all_dict'):
            for param_name, param in fit.model.parameters_all_dict.items():
                param_data = {
                    "value": param.value,
                    "fixed": param.fixed,
                    "bounds": param.bounds,
                    "bounds_on": param.bounds_on,
                    "linked": param.is_linked
                }

                # Save link information
                if param.is_linked and param.link is not None:
                    # Find the fit and parameter that this parameter is linked to
                    for j, other_fit_window in enumerate(chisurf.gui.fit_windows):
                        other_fit = other_fit_window.fit
                        if hasattr(other_fit, 'model') and hasattr(other_fit.model, 'parameters_all_dict'):
                            for other_param_name, other_param in other_fit.model.parameters_all_dict.items():
                                if other_param is param.link:
                                    param_data["linked_to_fit"] = j
                                    param_data["linked_to_param"] = other_param_name
                                    break

                fit_metadata["parameters"][param_name] = param_data

        metadata["fits"].append(fit_metadata)

    # Save UI state
    ui_state = {
        "current_fit_index": cs.fit_idx,
        "current_experiment_idx": cs.current_experiment_idx,
        "current_setup_idx": cs.current_setup_idx
    }
    metadata["ui_state"] = ui_state

    # Save metadata to file
    with open(os.path.join(project_dir, "project.yaml"), "w") as f:
        yaml.dump(metadata, f)

    chisurf.logging.info(f"Project saved to {project_dir}")


def load_project(project_path: str):
    """
    Load a project from a project folder.

    This function loads all fits, their parameters, dependencies (links across fits),
    and the UI state from a project folder.

    Parameters
    ----------
    project_path : str
        The path to the project folder
    """
    import os
    import pickle
    import yaml

    import chisurf
    from chisurf.macros import core_data as core_data_macros

    cs = chisurf.cs

    # Check if project path exists
    if not os.path.exists(project_path):
        chisurf.logging.error(f"Project path {project_path} does not exist")
        return

    # Load project metadata
    project_file = os.path.join(project_path, "project.yaml")
    if not os.path.exists(project_file):
        chisurf.logging.error(f"Project file {project_file} does not exist")
        return

    with open(project_file, "r") as f:
        # Use FullLoader to support python types
        metadata = yaml.load(f, Loader=yaml.FullLoader)
        #metadata = yaml.safe_load(f) or {}

    # Close all existing fits
    cs.onCloseAllFits()

    # Load all fits
    fits_dir = os.path.join(project_path, "fits")
    if not os.path.exists(fits_dir):
        chisurf.logging.error(f"Fits directory {fits_dir} does not exist")
        return

    # First pass: load all fits (data + create fit + load model CSV)
    for fit_metadata in metadata.get("fits", []):
        fit_name = fit_metadata.get("fit_name")
        fit_dir = os.path.join(fits_dir, fit_name)
        if not os.path.isdir(fit_dir):
            chisurf.logging.warning(f"Fit directory missing: {fit_dir}")
            continue

        # Load the data pickle if present
        data_files = [f for f in os.listdir(fit_dir) if f.endswith("_data.pkl")]
        dataset_obj = None
        if data_files:
            data_file = os.path.join(fit_dir, data_files[0])
            try:
                with open(data_file, 'rb') as df:
                    dataset_obj = pickle.load(df)
                # If we loaded a snapshot dict, reconstruct a DataCurve
                if isinstance(dataset_obj, dict) and dataset_obj.get("__chisurf_dataset_snapshot__") == 1:
                    from chisurf.data import DataCurve as _DataCurve
                    arr = dataset_obj.get("arrays", {}) or {}
                    x = arr.get("x")
                    y = arr.get("y")
                    ex = arr.get("ex")
                    ey = arr.get("ey")
                    # Ensure arrays are present
                    import numpy as _np
                    if x is None or y is None:
                        raise ValueError("Snapshot missing x or y arrays")
                    if ex is None:
                        ex = _np.zeros_like(x)
                    if ey is None:
                        ey = _np.ones_like(y)
                    dc = _DataCurve(name=dataset_obj.get("name") or "Dataset")
                    dc.set_data(x=_np.asarray(x), y=_np.asarray(y), ex=_np.asarray(ex), ey=_np.asarray(ey))
                    # Try to preserve filename metadata if available
                    try:
                        dc.filename = dataset_obj.get("filename") or dc.filename
                    except Exception:
                        pass
                    dataset_obj = dc
            except Exception as e:
                chisurf.logging.warning(f"Failed to load dataset pickle {data_file}: {e}")

        # Add dataset to application state
        if dataset_obj is not None:
            core_data_macros.add_dataset(dataset=dataset_obj)
        else:
            # No dataset to add; skip this fit entirely
            chisurf.logging.warning(f"Skipping fit {fit_name}: no dataset available")
            continue

        # Determine the index of the newly added dataset
        dataset_index = len(chisurf.imported_datasets) - 1

        # Create a new fit for this dataset, try to match model by saved name
        model_name = fit_metadata.get("model_name")
        try:
            add_fit(dataset_indices=[dataset_index], model_name=model_name)
        except Exception as e:
            chisurf.logging.warning(f"Failed to create fit for {fit_name}: {e}")
            continue

        # Find the fit result CSV (first non-data CSV)
        fit_files = [f for f in os.listdir(fit_dir) if f.endswith(".csv") and not f.endswith("_data.csv")]
        if not fit_files:
            chisurf.logging.warning(f"No fit file found for fit {fit_name}")
            continue
        fit_file = os.path.join(fit_dir, fit_files[0])

        # Load the fit into the most recently created fit
        fit_index = len(chisurf.fits) - 1
        try:
            load_fit_result(fit_index, fit_file)
        except Exception as e:
            chisurf.logging.warning(f"Failed to load fit result for {fit_name} from {fit_file}: {e}")

    # Second pass: restore parameter links
    for i, fit_metadata in enumerate(metadata.get("fits", [])):
        if i >= len(chisurf.fits):
            chisurf.logging.warning(f"Fit index {i} out of range")
            continue

        fit = chisurf.fits[i]

        # Restore parameter links
        for param_name, param_data in (fit_metadata.get("parameters", {}) or {}).items():
            if param_name not in fit.model.parameters_all_dict:
                chisurf.logging.warning(f"Parameter {param_name} not found in fit {i}")
                continue

            param = fit.model.parameters_all_dict[param_name]

            # Restore fixed state
            param.fixed = param_data.get("fixed", False)

            # Restore bounds
            param.bounds = param_data.get("bounds", (float("-inf"), float("inf")))
            param.bounds_on = param_data.get("bounds_on", False)

            # Restore links
            if param_data.get("linked", False) and "linked_to_fit" in param_data and "linked_to_param" in param_data:
                linked_fit_idx = param_data["linked_to_fit"]
                linked_param_name = param_data["linked_to_param"]

                if linked_fit_idx < len(chisurf.fits):
                    linked_fit = chisurf.fits[linked_fit_idx]
                    if linked_param_name in linked_fit.model.parameters_all_dict:
                        linked_param = linked_fit.model.parameters_all_dict[linked_param_name]
                        param.link = linked_param

    # Restore UI state
    ui_state = metadata.get("ui_state", {}) or {}

    # Set current fit
    current_fit_idx = ui_state.get("current_fit_index", 0)
    if 0 <= current_fit_idx < len(chisurf.fits):
        cs.current_fit = chisurf.fits[current_fit_idx]

    # Set current experiment (via setter to trigger UI updates)
    try:
        current_experiment_idx = ui_state.get("current_experiment_idx", 0)
        total_exp = cs.comboBox_experimentSelect.count()
        if 0 <= current_experiment_idx < total_exp:
            cs.set_current_experiment_idx(current_experiment_idx)
    except Exception:
        pass

    # Set current setup (via setter to trigger UI updates)
    try:
        current_setup_idx = ui_state.get("current_setup_idx", 0)
        total_setup = cs.comboBox_setupSelect.count()
        if 0 <= current_setup_idx < total_setup:
            cs.set_current_setup_idx(current_setup_idx)
    except Exception:
        pass

    chisurf.logging.info(f"Project loaded from {project_path}")
