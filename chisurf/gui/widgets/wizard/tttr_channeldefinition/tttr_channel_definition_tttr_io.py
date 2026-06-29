import tempfile
import importlib
import logging

import numpy as np
import tttrlib

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox, QLineEdit

from chisurf.core.fio.fluorescence.bhfiles import BeckerHicklSetReader
from chisurf.core.fio import write_jordi
from .tttr_detector_setups import load_detector_setups, save_detector_setups


def _load_jordi_gfactor_calculator_class():
    mod = importlib.import_module("chisurf.plugins.jordi_g_factor")
    cls = getattr(mod, "JordiGFactorCalculator", None)
    if cls is None:
        raise ImportError("JordiGFactorCalculator not found in chisurf.plugins.jordi_g_factor")
    return cls


def _update_microtime_preview(page, tttr, file_path=None):
    """Feed the page's micro-time preview with the data's decay histogram.

    Extracts both the full (all-channel) histogram and per-routing-channel
    histograms so the preview can show individual routing channel decays
    and persist them separately for downstream detector splitting.

    Parameters
    ----------
    page : DetectorWizardPage
        The wizard page whose preview plot to update.
    tttr : tttrlib.TTTR
        The loaded TTTR data.
    file_path : str or None
        Source file path; stored with the histogram for persistence.
    """
    setter = getattr(page, "set_microtime_data", None)
    if not callable(setter):
        return
    try:
        hist, _ = tttr.get_microtime_histogram(1)
        setter(np.asarray(hist, dtype=float), file_path=file_path)
    except Exception as exc:  # pragma: no cover - preview is best-effort
        logger.debug("Micro-time preview update failed: %s", exc)
        return

    # Extract per-routing-channel microtime histograms
    per_channel_setter = getattr(page, "set_microtime_per_channel_data", None)
    if not callable(per_channel_setter):
        return
    try:
        used_channels = sorted(int(c) for c in tttr.get_used_routing_channels())
        per_channel = {}
        for ch in used_channels:
            ch_hist, _ = tttr.get_microtime_histogram(1, [ch])
            per_channel[ch] = np.asarray(ch_hist, dtype=float)
        per_channel_setter(per_channel, file_path=file_path)
    except Exception as exc:
        logger.debug("Per-channel microtime extraction failed: %s", exc)


def _auto_save_decay_to_setup(page):
    """Persist decay data to the current setup so it survives widget restart."""
    if not page.current_setup_name:
        return
    try:
        data = page.get_settings()
        decay = data.get("_microtime_decay")
        per_ch = data.get("_microtime_per_channel_decay")
        if not decay and not per_ch:
            return
        setups = load_detector_setups(page.current_setups_file)
        setups.setdefault("setups", {})
        if page.current_setup_name in setups["setups"]:
            existing = setups["setups"][page.current_setup_name]
            if decay:
                existing["_microtime_decay"] = decay
            if per_ch:
                existing["_microtime_per_channel_decay"] = per_ch
            setups["setups"][page.current_setup_name] = existing
        else:
            setups["setups"][page.current_setup_name] = data
        save_detector_setups(setups, page.current_setups_file)
    except Exception as exc:
        logger.debug("Auto-save decay to setup failed: %s", exc)


def read_from_tttr_file(page):
    path, _ = QFileDialog.getOpenFileName(
        page,
        "Open TTTR or SPC File",
        "",
        "All Files (*);;TTTR Files (*.ptu *.ht3 *.pt3);;SPC Files (*.spc *.set)"
    )
    if not path:
        return

    try:
        if path.lower().endswith('.set'):
            reader = BeckerHicklSetReader(path)
            micro_time_res = reader.micro_time_resolution
            if micro_time_res is not None:
                page.micro_time_le.setText(str(micro_time_res * 1000.0))
            page._update_effective_resolution()
            logger.info(
                "Successfully read microtime calibration from SET file: %s", path
            )
            _auto_save_decay_to_setup(page)
        elif path.lower().endswith('.spc'):
            tttr = tttrlib.TTTR(path)
            header = tttr.get_header()
            page.macro_time_le.setText(str(header.macro_time_resolution * 1e9))
            page.micro_time_le.setText(str(header.micro_time_resolution * 1e12))
            page._update_effective_resolution()
            _update_microtime_preview(page, tttr, file_path=path)
            logger.info(
                "Successfully read macrotime calibration from SPC file: %s", path
            )
            _auto_save_decay_to_setup(page)
        else:
            tttr = tttrlib.TTTR(path)
            if len(tttr) == 0:
                raise ValueError("File is not a supported TTTR file format or contains no events.")
            header = tttr.get_header()
            page.macro_time_le.setText(str(header.macro_time_resolution * 1e9))
            page.micro_time_le.setText(str(header.micro_time_resolution * 1e12))
            page._update_effective_resolution()
            _update_microtime_preview(page, tttr, file_path=path)
            logger.info(
                "Successfully read calibrations from TTTR file: %s", path
            )
            _auto_save_decay_to_setup(page)

    except Exception as e:
        QMessageBox.critical(
            page,
            "Error",
            f"Failed to read file: {e}"
        )


def on_calc_g_factor(page, row=None):
    if row is None:
        selected_rows = page.detectors_form.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(
                page,
                "Warning",
                "Please select a detector row first."
            )
            return
        row = selected_rows[0].row()

    settings = page.get_settings()
    detectors = settings["detectors"]

    path, _ = QFileDialog.getOpenFileName(
        page,
        "Open TTTR File for G-Factor Calculation",
        "",
        "All Files (*)"
    )
    if not path:
        return

    try:
        tttr = tttrlib.TTTR(path)
        if len(tttr) == 0:
            raise ValueError("File is not a supported TTTR file format or contains no events.")
        micro_time_binning = int(page.micro_binning_combo.currentText())

        selected_detector = page.detectors_form.item(row, 0).text().strip()
        channels_text = page.detectors_form.cellWidget(row, 1).text()
        all_channels = list(map(int, channels_text.split(',')))

        parallel_channels = all_channels[::2]
        perpendicular_channels = all_channels[1::2]

        if len(all_channels) < 2 or len(parallel_channels) == 0 or len(perpendicular_channels) == 0:
            QMessageBox.warning(
                page,
                "Warning",
                "Selected detector must contain at least two routing channels (parallel and perpendicular) to calculate G-Factor."
            )
            return

        page.selected_detector = {
            'row': row,
            'name': selected_detector,
            'parallel_channels': parallel_channels,
            'perpendicular_channels': perpendicular_channels
        }

        parallel_hist, _ = tttr.get_microtime_histogram(micro_time_binning, parallel_channels)
        perpendicular_hist, _ = tttr.get_microtime_histogram(micro_time_binning, perpendicular_channels)

        parallel_nonzero = np.where(parallel_hist > 0)[0]
        perpendicular_nonzero = np.where(perpendicular_hist > 0)[0]

        if len(parallel_nonzero) > 0 and len(perpendicular_nonzero) > 0:
            start_idx = min(parallel_nonzero[0], perpendicular_nonzero[0])
            end_idx = max(parallel_nonzero[-1], perpendicular_nonzero[-1]) + 1
            parallel_hist_trimmed = parallel_hist[start_idx:end_idx]
            perpendicular_hist_trimmed = perpendicular_hist[start_idx:end_idx]
        else:
            parallel_hist_trimmed = parallel_hist
            perpendicular_hist_trimmed = perpendicular_hist

        fd, jordi_file = tempfile.mkstemp(suffix='.dat')
        jordi_data = np.concatenate([parallel_hist_trimmed, perpendicular_hist_trimmed])
        write_jordi(filename=jordi_file, data=jordi_data)

        # Check for headless calculation option
        gf_widget = page.detectors_form.cellWidget(row, 6)
        gf_range_text = gf_widget.text().strip() if gf_widget else ""
        has_valid_range = False
        s, e = 0, 0
        if gf_range_text:
            try:
                parts = gf_range_text.replace(' ', '').split('-')
                if len(parts) == 2:
                    s = int(float(parts[0]))
                    e = int(float(parts[1]))
                    if e < s:
                        s, e = e, s
                    has_valid_range = True
            except Exception:
                pass

        run_headless = False
        if has_valid_range:
            reply = QMessageBox.question(
                page,
                "Headless Calculation",
                f"A G-factor channel range '{gf_range_text}' is already defined.\n"
                "Would you like to run the calculation headlessly without opening the interactive GUI?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Cancel:
                return
            run_headless = (reply == QMessageBox.Yes)

        if run_headless:
            from chisurf.plugins.jordi_g_factor.gui.client import JordiGFactorClient
            client = JordiGFactorClient()
            try:
                res = client.calculate(
                    parallel_data=parallel_hist_trimmed.tolist(),
                    perpendicular_data=perpendicular_hist_trimmed.tolist(),
                    region_bounds=[s, e],
                    decay_shift=0.0,
                    use_bg=False,
                )
                g_factor_val = res.get("g_factor")
                if g_factor_val is not None:
                    g_factor_decay_uuid = None
                    g_factor_calibration_id = None
                    try:
                        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import resolve_active_user_id
                        
                        l1_widget = page.detectors_form.cellWidget(row, 4)
                        l2_widget = page.detectors_form.cellWidget(row, 5)
                        try:
                            l1_val = float(l1_widget.text()) if l1_widget else 0.0
                        except ValueError:
                            l1_val = 0.0
                        try:
                            l2_val = float(l2_widget.text()) if l2_widget else 0.0
                        except ValueError:
                            l2_val = 0.0

                        # Build parameters dict
                        calib_params = {
                            "g_factor": g_factor_val,
                            "g_factor_stddev": res.get("g_factor_stddev_uncorrected"),
                            "g_factor_uncorrected": res.get("g_factor_uncorrected"),
                            "g_factor_corrected": res.get("g_factor_corrected"),
                            "r_inf": res.get("r_inf"),
                            "region_min": float(s),
                            "region_max": float(e),
                            "decay_shift": 0.0,
                            "flip": False,
                            "use_bg": False,
                            "bg_vv": 0.0,
                            "bg_vh": 0.0,
                            "l1": l1_val,
                            "l2": l2_val,
                            "micro_time_resolution": float(page.effective_micro_time_resolution) if hasattr(page, "effective_micro_time_resolution") else None,
                        }
                        logger.info("Archiving G-factor reference decay and calibration to MFDB: %s", jordi_file)
                        archive_res = client.archive_g_factor(
                            file_path=jordi_file,
                            parameters=calib_params,
                            active_user=resolve_active_user_id(),
                        )
                        if isinstance(archive_res, dict):
                            g_factor_calibration_id = archive_res.get("calibration_id")
                            g_factor_decay_uuid = archive_res.get("reference_decay_id")
                    except Exception as e:
                        logger.warning("Failed to archive G-factor to MFDB: %s", e)

                    g_factor_text = f"{g_factor_val:.3f}"
                    page._set_g_factor_programmatically(
                        row, g_factor_text, g_factor_decay_uuid, g_factor_calibration_id,
                        l1=f"{l1_val:.5f}", l2=f"{l2_val:.5f}"
                    )

                    # Update setup
                    if page.current_setup_name:
                        data = page.get_settings()
                        setups = load_detector_setups(page.current_setups_file)
                        setups.setdefault("setups", {})

                        if page.current_setup_name in setups["setups"]:
                            existing_data = setups["setups"][page.current_setup_name]
                            for key in data:
                                if key == 'detectors':
                                    existing_data.setdefault('detectors', {})
                                    for det_name, det_info in data['detectors'].items():
                                        if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                                            existing_data['detectors'][det_name].update(det_info)
                                        else:
                                            existing_data['detectors'][det_name] = det_info
                                else:
                                    existing_data[key] = data[key]
                            setups["setups"][page.current_setup_name] = existing_data
                        else:
                            setups["setups"][page.current_setup_name] = data

                        setups["last_used"] = page.current_setup_name
                        save_detector_setups(setups, page.current_setups_file)

                        QMessageBox.information(
                            page,
                            "Success",
                            f"Headless G-Factor calculation completed successfully!\n"
                            f"Calculated G-Factor: {g_factor_val:.4f}\n"
                            f"Updated detector: {selected_detector}\n"
                            f"Setup '{page.current_setup_name}' saved automatically."
                        )
                    else:
                        QMessageBox.information(
                            page,
                            "Success",
                            f"Headless G-Factor calculation completed successfully!\n"
                            f"Calculated G-Factor: {g_factor_val:.4f}\n"
                            f"Updated detector: {selected_detector}\n"
                            f"Note: No setup was selected, so changes were not saved automatically."
                        )
                else:
                    QMessageBox.critical(page, "Error", "Headless calculation returned None.")
            except Exception as ex:
                QMessageBox.critical(page, "Error", f"Headless calculation failed: {ex}")
            return

        JordiGFactorCalculator = _load_jordi_gfactor_calculator_class()
        g_factor_calculator = JordiGFactorCalculator()
        g_factor_calculator.setWindowModality(Qt.ApplicationModal)

        try:
            setattr(g_factor_calculator, 'parallel_channels', parallel_channels)
            setattr(g_factor_calculator, 'perpendicular_channels', perpendicular_channels)
            setattr(g_factor_calculator, 'micro_time_binning', micro_time_binning)
            setattr(g_factor_calculator, 'detector_name', selected_detector)
            setattr(g_factor_calculator, 'effective_micro_time_resolution_ps', float(page.effective_micro_time_resolution))
            if hasattr(g_factor_calculator, 'fp_dt_spinbox') and g_factor_calculator.fp_dt_spinbox is not None:
                try:
                    g_factor_calculator.fp_dt_spinbox.setValue(float(page.effective_micro_time_resolution) * 1e-3)
                except Exception:
                    pass
        except Exception:
            pass

        page.g_factor_calculator = g_factor_calculator
        page.jordi_file = jordi_file

        original_close_event = g_factor_calculator.closeEvent

        def custom_close_event(event):
            if original_close_event:
                original_close_event(event)

            if hasattr(g_factor_calculator, 'g_factor') and g_factor_calculator.g_factor is not None:
                selected_detector_info = page.selected_detector
                logger.info("Interactive G-factor calculation finished. Selected detector info: %s", selected_detector_info)
                if selected_detector_info:
                    row = selected_detector_info['row']
                    g_factor_value = f"{g_factor_calculator.g_factor:.3f}"
                    logger.info("Updating detectors table row %d with g-factor value %s", row, g_factor_value)

                    # Upload the temporary JORDI file to the Object Store & register G-factor calibration!
                    g_factor_decay_uuid = None
                    g_factor_calibration_id = None
                    try:
                        from chisurf.plugins.jordi_g_factor.gui.client import JordiGFactorClient
                        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import resolve_active_user_id
                        client = JordiGFactorClient()
                        
                        # Determine actual file path (the user might have loaded another one in the widget)
                        file_path = page.jordi_file
                        if hasattr(g_factor_calculator, "file_label") and g_factor_calculator.file_label is not None:
                            fl_txt = g_factor_calculator.file_label.text().strip()
                            if fl_txt and not fl_txt.startswith("Error") and not fl_txt.startswith("No file"):
                                file_path = fl_txt

                        l1_widget = page.detectors_form.cellWidget(row, 4)
                        l2_widget = page.detectors_form.cellWidget(row, 5)
                        try:
                            l1_existing = float(l1_widget.text()) if l1_widget else 0.0
                        except ValueError:
                            l1_existing = 0.0
                        try:
                            l2_existing = float(l2_widget.text()) if l2_widget else 0.0
                        except ValueError:
                            l2_existing = 0.0

                        l1_val = getattr(g_factor_calculator, "l1_estimate", None)
                        l2_val = getattr(g_factor_calculator, "l2_estimate", None)
                        l1_param = float(l1_val) if l1_val is not None and np.isfinite(float(l1_val)) else l1_existing
                        l2_param = float(l2_val) if l2_val is not None and np.isfinite(float(l2_val)) else l2_existing

                        # Prep calibration parameters
                        calib_params = {
                            "g_factor": float(g_factor_calculator.g_factor) if g_factor_calculator.g_factor is not None else None,
                            "g_factor_stddev": float(g_factor_calculator.g_factor_stddev) if getattr(g_factor_calculator, "g_factor_stddev", None) is not None else None,
                            "g_factor_uncorrected": float(g_factor_calculator.g_factor_uncorrected) if getattr(g_factor_calculator, "g_factor_uncorrected", None) is not None else None,
                            "g_factor_corrected": float(g_factor_calculator.g_factor_corrected) if getattr(g_factor_calculator, "g_factor_corrected", None) is not None else None,
                            "region_min": float(min(g_factor_calculator.region_bounds)) if getattr(g_factor_calculator, "region_bounds", None) is not None else None,
                            "region_max": float(max(g_factor_calculator.region_bounds)) if getattr(g_factor_calculator, "region_bounds", None) is not None else None,
                            "decay_shift": float(g_factor_calculator.decay_shift) if getattr(g_factor_calculator, "decay_shift", None) is not None else 0.0,
                            "flip": bool(g_factor_calculator.flip_checkbox.isChecked()) if getattr(g_factor_calculator, "flip_checkbox", None) is not None else False,
                            "use_bg": bool(g_factor_calculator.bg_correction_checkbox.isChecked()) if getattr(g_factor_calculator, "bg_correction_checkbox", None) is not None else False,
                            "bg_vv": float(g_factor_calculator.bg_parallel_value.text()) if getattr(g_factor_calculator, "bg_correction_checkbox", None) is not None and g_factor_calculator.bg_correction_checkbox.isChecked() and hasattr(g_factor_calculator, "bg_parallel_value") else 0.0,
                            "bg_vh": float(g_factor_calculator.bg_perpendicular_value.text()) if getattr(g_factor_calculator, "bg_correction_checkbox", None) is not None and g_factor_calculator.bg_correction_checkbox.isChecked() and hasattr(g_factor_calculator, "bg_perpendicular_value") else 0.0,
                            "l1": l1_param,
                            "l2": l2_param,
                            "micro_time_resolution": float(page.effective_micro_time_resolution) if hasattr(page, "effective_micro_time_resolution") else None,
                        }
                        if hasattr(g_factor_calculator, "bg_region_bounds") and g_factor_calculator.bg_region_bounds is not None:
                            calib_params["bg_region_bounds"] = list(g_factor_calculator.bg_region_bounds)

                        logger.info("Archiving G-factor reference decay and calibration to MFDB: %s", file_path)
                        archive_res = client.archive_g_factor(
                            file_path=file_path,
                            parameters=calib_params,
                            active_user=resolve_active_user_id(),
                        )
                        if isinstance(archive_res, dict):
                            g_factor_calibration_id = archive_res.get("calibration_id")
                            g_factor_decay_uuid = archive_res.get("reference_decay_id")
                    except Exception as e:
                        logger.warning("Failed to archive G-factor to MFDB: %s", e)

                    l1_text = f"{l1_param:.5f}"
                    l2_text = f"{l2_param:.5f}"

                    existing_cell_widget = page.detectors_form.cellWidget(row, 3)
                    if existing_cell_widget:
                        page._set_g_factor_programmatically(
                            row, g_factor_value, g_factor_decay_uuid, g_factor_calibration_id,
                            l1=l1_text, l2=l2_text
                        )
                    else:
                        new_cell_widget = QLineEdit(g_factor_value)
                        page.detectors_form.setCellWidget(row, 3, new_cell_widget)
                        page._wire_g_factor_cell(row, new_cell_widget)
                        
                        le_l1 = QLineEdit(l1_text)
                        page.detectors_form.setCellWidget(row, 4, le_l1)
                        
                        le_l2 = QLineEdit(l2_text)
                        page.detectors_form.setCellWidget(row, 5, le_l2)
                        
                        item = page.detectors_form.item(row, 0)
                        if item:
                            if g_factor_decay_uuid:
                                item.setData(Qt.UserRole + 1, g_factor_decay_uuid)
                            if g_factor_calibration_id:
                                item.setData(Qt.UserRole + 2, g_factor_calibration_id)

                    gf_range_text_new = None
                    try:
                        rng = None
                        if hasattr(g_factor_calculator, 'region') and g_factor_calculator.region is not None:
                            try:
                                rng = g_factor_calculator.region.getRegion()
                            except Exception:
                                rng = None
                        if rng is None and hasattr(g_factor_calculator, 'region_bounds'):
                            rng = getattr(g_factor_calculator, 'region_bounds', None)
                        if isinstance(rng, (list, tuple)) and len(rng) == 2:
                            s_new = int(float(rng[0]))
                            e_new = int(float(rng[1]))
                            if e_new < s_new:
                                s_new, e_new = e_new, s_new
                            gf_range_text_new = f"{s_new}-{e_new}"
                            try:
                                gf_widget_new = page.detectors_form.cellWidget(row, 6)
                                if gf_widget_new is None:
                                    gf_widget_new = QLineEdit(gf_range_text_new)
                                    page.detectors_form.setCellWidget(row, 6, gf_widget_new)
                                else:
                                    gf_widget_new.setText(gf_range_text_new)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    if page.current_setup_name:
                        data = page.get_settings()
                        setups = load_detector_setups(page.current_setups_file)
                        setups.setdefault("setups", {})

                        if page.current_setup_name in setups["setups"]:
                            existing_data = setups["setups"][page.current_setup_name]
                            for key in data:
                                if key == 'detectors':
                                    existing_data.setdefault('detectors', {})
                                    for det_name, det_info in data['detectors'].items():
                                        if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                                            existing_data['detectors'][det_name].update(det_info)
                                        else:
                                            existing_data['detectors'][det_name] = det_info
                                else:
                                    existing_data[key] = data[key]
                            setups["setups"][page.current_setup_name] = existing_data
                        else:
                            setups["setups"][page.current_setup_name] = data

                        setups["last_used"] = page.current_setup_name
                        save_detector_setups(setups, page.current_setups_file)

                        msg = (
                            f"G-Factor calculated: {g_factor_calculator.g_factor:.4f}\n"
                            f"Updated G-Factor for detector: {selected_detector_info['name']}\n"
                        )
                        if gf_range_text_new:
                            msg += f"G-Factor Channels: {gf_range_text_new}\n"
                        msg += f"Setup '{page.current_setup_name}' saved automatically."
                        QMessageBox.information(page, "Success", msg)
                    else:
                        msg = (
                            f"G-Factor calculated: {g_factor_calculator.g_factor:.4f}\n"
                            f"Updated G-Factor for detector: {selected_detector_info['name']}\n"
                        )
                        if gf_range_text_new:
                            msg += f"G-Factor Channels: {gf_range_text_new}\n"
                        msg += "Note: No setup was selected, so changes were not saved automatically."
                        QMessageBox.information(page, "Success", msg)

        g_factor_calculator.closeEvent = custom_close_event
        g_factor_calculator.show()

        try:
            effective_dt = page.effective_micro_time_resolution
            g_factor_calculator.load_jordi_file(jordi_file)

            try:
                gf_widget_new = page.detectors_form.cellWidget(row, 6)
                if gf_widget_new:
                    txt = gf_widget_new.text().strip()
                    if txt:
                        parts = txt.replace(' ', '').split('-')
                        if len(parts) == 2:
                            s_new = int(float(parts[0]))
                            e_new = int(float(parts[1]))
                            if e_new < s_new:
                                s_new, e_new = e_new, s_new
                            if hasattr(g_factor_calculator, 'region'):
                                try:
                                    g_factor_calculator.region.setRegion([s_new, e_new])
                                except Exception:
                                    pass
                            if hasattr(g_factor_calculator, 'region_bounds'):
                                try:
                                    g_factor_calculator.region_bounds = [s_new, e_new]
                                except Exception:
                                    pass
                            try:
                                g_factor_calculator.calculate_g_factor()
                            except Exception:
                                pass
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(
                page,
                "Error",
                f"Failed to load Jordi file: {str(e)}"
            )

    except Exception as e:
        QMessageBox.critical(
            page,
            "Error",
            f"Failed to calculate G-Factor: {str(e)}"
        )
