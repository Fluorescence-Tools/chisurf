from __future__ import annotations

import os
import gc
import shutil
import numpy as np

import chisurf
import chisurf.base
import chisurf.data
import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets

from chisurf import typing
from chisurf import logging
from chisurf.project import Project as CSProject, save_project as project_save_json, load_project as project_load_json
from chisurf.project import fit_state as project_fit_state


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

    # If multiple datasets were requested, build each fit independently
    # using the already-stable single-dataset code path.
    if len(dataset_indices) > 1:
        for idx in dataset_indices:
            try:
                add_fit(dataset_indices=[idx], model_name=model_name, model_kw=model_kw)
            except Exception as e:
                chisurf.logging.warning(f"add_fit: failed for dataset index {idx}: {e}")
        return

    # create a list of data sets to which a fit with
    # a particular model is added
    data_sets = [cs.dataset_selector.datasets[i] for i in dataset_indices]

    # Prefer the experiment attached to the dataset; fall back to the
    # globally selected experiment if necessary (e.g. after project load).
    exp = getattr(data_sets[0], "experiment", None)
    if exp is None:
        exp = getattr(cs, "current_experiment", None)
    if exp is None:
        chisurf.logging.warning("add_fit: no experiment available on dataset or cs.current_experiment; aborting")
        return

    model_names = exp.model_names
    model_class = exp.model_classes[0]
    for model_idx, mn in enumerate(model_names):
        if mn == model_name:
            model_class = exp.model_classes[model_idx]
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
        # Establish a fit-group link: the parameter in the currently
        # selected fit acts as the master, all other fits in the group
        # link their parameter of the same name to this master.
        current_fit = cs.current_fit
        try:
            parameter = current_fit.model.parameters_all_dict[fitting_parameter_name]
        except Exception:
            return

        # Mark master for GUI purposes only; numerical behaviour is still
        # governed by the underlying port links.
        try:
            parameter.is_link_master = True
        except Exception:
            pass

        for f in cs.current_fit:
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
    """Save the current state of the application as a project.

    This implementation writes a JSON ``project.json`` file using
    :class:`chisurf.project.Project` together with a minimal snapshot of all
    datasets, fits and UI state. It no longer creates per‑fit folders or
    YAML/Word reports. Existing callers can keep using this macro unchanged.

    Parameters
    ----------
    target_path : str
        Directory in which the project folder will be created.
    project_name : str
        Name of the project subfolder (default: ``"chisurf_project"``).
    """

    log = chisurf.logging
    cs = chisurf.cs

    base_dir = os.path.abspath(str(target_path))
    project_dir = os.path.join(base_dir, project_name)

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

    # --- Collect fits & global links --------------------------------------
    fits_payload: typing.Dict[str, typing.Dict] = {}

    for i, fit_window in enumerate(chisurf.gui.fit_windows):
        fit_group = getattr(fit_window, "fit", None)
        if fit_group is None:
            continue

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
            # project round-trip restores both parameters and fit limits.
            try:
                get_state = getattr(local_fit, "get_state", None)
                if callable(get_state):
                    fit_state = get_state()
                else:
                    fit_state = project_fit_state.fit_to_state(local_fit)
            except Exception as exc:
                log.warning(f"save_project: could not serialize fit group #{i}: {exc}")
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
        global_links_state: typing.Dict[str, typing.Any] = {}
        global_model = getattr(fit_group, "_model", None)
        if global_model is not None:
            try:
                global_links_state = project_fit_state.global_links_to_state(global_model)
            except Exception as exc:
                log.warning(f"save_project: could not serialize global links for fit group #{i}: {exc}")

        fg_key = f"fitgroup_{i:03d}"
        fits_payload[fg_key] = {
            "type": "fit_group",
            "name": getattr(fit_group, "name", fg_key),
            "model_name": model_name,
            "local_fits": local_fits_state,
            "global_links": global_links_state,
        }

    ui_state = {
        "current_fit_index": getattr(cs, "fit_idx", 0),
        "current_experiment_idx": getattr(cs, "current_experiment_idx", 0),
        "current_setup_idx": getattr(cs, "current_setup_idx", 0),
    }

    # Optionally capture main-window and MDI layout geometry/state in a
    # JSON-serializable form so that a reloaded project can restore the
    # visual arrangement of windows. We keep this best-effort and guard
    # against any Qt/UI issues so that project saving never fails because
    # of GUI state.
    try:
        # Main window geometry/state (toolbars, docks, splitter positions).
        mw_state = {}
        save_geom = getattr(cs, "saveGeometry", None)
        save_state = getattr(cs, "saveState", None)
        if callable(save_geom):
            try:
                ba = save_geom()
                mw_state["geometry"] = bytes(ba).hex()
            except Exception:
                pass
        if callable(save_state):
            try:
                ba = save_state()
                mw_state["state"] = bytes(ba).hex()
            except Exception:
                pass
        if mw_state:
            ui_state["main_window"] = mw_state

        # MDI subwindow layout (tiling/cascading and tabbed layout state).
        mdi = getattr(cs, "mdiarea", None)
        if mdi is not None:
            save_mdi = getattr(mdi, "saveState", None)
            if callable(save_mdi):
                try:
                    ba = save_mdi()
                    ui_state["mdi_area"] = {"state": bytes(ba).hex()}
                except Exception:
                    pass
    except Exception:
        # Geometry is a purely cosmetic extra; never fail project save.
        pass

    proj = CSProject(
        name=project_name,
        description=f"ChiSurf project '{project_name}'",
        chisurf_version=getattr(chisurf.info, "__version__", None),
        datasets=datasets,
        experiments={},  # reserved for future structured experiment state
        fits=fits_payload,
        ui_state=ui_state,
    )

    project_save_json(proj, project_dir)
    log.info(f"Project saved to {project_dir}")


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
    cs = chisurf.cs

    if not os.path.isdir(project_path):
        log.error(f"Project path {project_path} does not exist")
        return

    try:
        proj = project_load_json(project_path)
    except Exception as exc:
        log.error(f"load_project: failed to read project.json from {project_path}: {exc}")
        return

    # --- Restore experiment/setup state early so we can attach it to datasets
    ui_state = proj.ui_state or {}

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

    # Reset current fits and datasets
    try:
        cs.onCloseAllFits()
    except Exception:
        pass

    chisurf.fits = []
    chisurf.gui.fit_windows = []
    chisurf.imported_datasets = []

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

            # Attach a reasonable experiment object so that GUI widgets and
            # macros relying on d.experiment (e.g. dataset selectors, add_fit)
            # continue to work. For now we associate all reloaded datasets with
            # the current experiment, if available.
            try:
                exp_obj = getattr(cs, "current_experiment", None)
            except Exception:
                exp_obj = None
            if exp_obj is not None:
                try:
                    dc.experiment = exp_obj
                except Exception as e_exp:
                    log.warning(f"load_project: could not attach experiment to dataset {ds_id}: {e_exp}")

            # Register datasets directly; we avoid calling core_data.add_dataset
            # here to keep project loading independent of per-reader grouping
            # logic and to prevent creation of tuple-based ExperimentDataGroups.
            dataset_objects[ds_id] = dc
            chisurf.imported_datasets.append(dc)
            dataset_indices[ds_id] = len(chisurf.imported_datasets) - 1
        except Exception as exc:
            log.warning(f"load_project: could not reconstruct dataset {ds_id}: {exc}")
            continue

    # --- Rebuild fit groups and restore their state -----------------------
    fits_map = proj.fits or {}
    for key, rec in fits_map.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("type") != "fit_group":
            continue

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

            # Restore per-fit x-range if stored. This must be applied after
            # creating the FitGroup but before final GUI updates so that
            # model curves and residuals use the intended window.
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
    current_fit_idx = ui_state.get("current_fit_index", 0)
    if 0 <= current_fit_idx < len(chisurf.fits):
        try:
            cs.current_fit = chisurf.fits[current_fit_idx]
        except Exception:
            pass

    # Restore main-window and MDI geometry/state if present. This should be
    # done only after datasets and fits (and thus subwindows) have been
    # recreated so that Qt has matching widgets to apply the layout to.
    try:
        mw_state = ui_state.get("main_window") or {}
        geom_hex = mw_state.get("geometry")
        state_hex = mw_state.get("state")
        if geom_hex:
            try:
                ba = chisurf.gui.QtCore.QByteArray.fromHex(geom_hex.encode("ascii"))
                cs.restoreGeometry(ba)
            except Exception:
                pass
        if state_hex:
            try:
                ba = chisurf.gui.QtCore.QByteArray.fromHex(state_hex.encode("ascii"))
                cs.restoreState(ba)
            except Exception:
                pass

        mdi_info = ui_state.get("mdi_area") or {}
        mdi_hex = mdi_info.get("state")
        mdi = getattr(cs, "mdiarea", None)
        if mdi is not None and mdi_hex:
            restore_mdi = getattr(mdi, "restoreState", None)
            if callable(restore_mdi):
                try:
                    ba = chisurf.gui.QtCore.QByteArray.fromHex(mdi_hex.encode("ascii"))
                    restore_mdi(ba)
                except Exception:
                    pass
    except Exception:
        pass

    # Trigger a single GUI refresh now that datasets, fits and layout are consistent.
    try:
        cs.update()
    except Exception:
        pass

    log.info(f"Project loaded from {project_path}")
