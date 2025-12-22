from __future__ import annotations

import os
import gc
import shutil
import json
import pathlib
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


def save_fit(
        target_path: str = None,
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
            proj = CSProject(
                name=fit.name or save_name,
                description=f"ChiSurf fit '{fit.name or save_name}'",
                chisurf_version=getattr(chisurf.info, "__version__", None),
                datasets=datasets,
                experiments={},
                fits={fg_key: fit_payload},
                ui_state={"current_fit_index": getattr(cs, "fit_idx", 0)},
            )
            # Write fit.json using the base name without the original data extension
            base_no_ext = os.path.join(target_path, os.path.splitext(save_name)[0])
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
    :class:`chisurf.project.Project` together with a snapshot of all datasets,
    fits and UI state. It no longer creates per-fit folders, screenshots or
    DOCX reports; everything is embedded in ``project.json``.

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
    manifest_fits: typing.List[typing.Dict[str, typing.Any]] = []

    for i, fit_window in enumerate(chisurf.gui.fit_windows):
        fit_group = getattr(fit_window, "fit", None)
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

    ui_state = {
        "current_fit_index": getattr(cs, "fit_idx", 0),
        "current_experiment_idx": getattr(cs, "current_experiment_idx", 0),
        "current_setup_idx": getattr(cs, "current_setup_idx", 0),
    }

    # Optionally capture main-window and MDI layout geometry/state
    try:
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
        pass

    proj = CSProject(
        name=project_name,
        description=f"ChiSurf project '{project_name}'",
        chisurf_version=getattr(chisurf.info, "__version__", None),
        datasets=datasets,
        experiments={},  # reserved for future structured experiment state
        fits=manifest_fits,
        ui_state=ui_state,
    )

    project_save_json(proj, project_dir)
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

def _build_fitgroup_payload_from_window(
        fit_window,
        register_datacurve: typing.Callable[[chisurf.data.DataCurve], str],
        log: typing.Any,
        group_index: int = 0,
        include_global_links: bool = True,
) -> typing.Tuple[str, typing.Dict]:
    """Helper to serialize a single FitSubWindow into the Project payload."""
    fit_group = getattr(fit_window, "fit", None)
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


def save_fit_project(target_path: str, fit_window=None, fit_name: str = "chisurf_fit"):
    """Save a single fit (data + model state + window) in the project format.

    The output is a folder containing ``project.json`` so it can be reloaded
    with :func:`load_fit_project` similarly to full projects, but without
    touching other open fits.
    """
    log = chisurf.logging
    cs = chisurf.cs

    if fit_window is None:
        fit_window = getattr(cs.mdiarea, "currentSubWindow", lambda: None)()
    if fit_window is None:
        log.error("save_fit_project: no fit window available")
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

    fg_key, fit_payload = _build_fitgroup_payload_from_window(
        fit_window, register_datacurve, log, group_index=0
    )
    if not fit_payload:
        log.error("save_fit_project: fit payload empty; aborting")
        return

    proj = CSProject(
        name=fit_name,
        description=f"ChiSurf fit '{fit_name}'",
        chisurf_version=getattr(chisurf.info, "__version__", None),
        datasets=datasets,
        experiments={},
        fits={fg_key: fit_payload},
        ui_state={"current_fit_index": getattr(cs, "fit_idx", 0)},
    )

    # Write both project.json (compat) and fit.json (explicit single-fit entry point)
    project_save_json(proj, project_dir)
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
    """
    log = chisurf.logging
    cs = chisurf.cs

    proj = None
    if project_path.endswith(".json") and os.path.isfile(project_path):
        # Accept direct fit.json/project.json paths
        try:
            with open(project_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            proj = CSProject.from_dict(data)
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
        except Exception as exc:
            log.error(f"load_fit_project: failed to read project.json from {base_dir}: {exc}")
            return

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
                exp_obj = getattr(cs, "current_experiment", None)
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

    # Best-effort: bring the newest fit window to front
    try:
        if chisurf.gui.fit_windows:
            win = chisurf.gui.fit_windows[-1]
            win.show()
            win.setFocus()
    except Exception:
        pass


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

    try:
        reinit = getattr(cs, "reinitialize", None)
        if callable(reinit):
            reinit()
        else:
            try:
                cs.onCloseAllFits()
            except Exception:
                pass
            try:
                chisurf.fits.clear()
            except Exception:
                pass
            try:
                chisurf.gui.fit_windows.clear()
            except Exception:
                pass
            try:
                chisurf.imported_datasets.clear()
            except Exception:
                pass
    except Exception:
        pass

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

    try:
        chisurf.fits.clear()
    except Exception:
        pass
    try:
        chisurf.gui.fit_windows.clear()
    except Exception:
        pass
    try:
        chisurf.imported_datasets.clear()
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
