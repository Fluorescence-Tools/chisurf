import tempfile
import importlib

import numpy as np
import tttrlib

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox, QLineEdit

from chisurf.fio.fluorescence.bhfiles import BeckerHicklSetReader
from chisurf.fio import write_jordi
from .tttr_detector_setups import load_detector_setups, save_detector_setups


def _load_jordi_gfactor_calculator_class():
    mod = importlib.import_module("chisurf.plugins.jordi_g_factor")
    cls = getattr(mod, "JordiGFactorCalculator", None)
    if cls is None:
        raise ImportError("JordiGFactorCalculator not found in chisurf.plugins.jordi_g_factor")
    return cls


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
                page.micro_time_le.setText(str(micro_time_res))
            page._update_effective_resolution()
            QMessageBox.information(
                page,
                "Success",
                f"Successfully read microtime calibration from SET file: {path}"
            )
        elif path.lower().endswith('.spc'):
            tttr = tttrlib.TTTR(path)
            header = tttr.get_header()
            page.macro_time_le.setText(str(header.macro_time_resolution * 1e9))
            page._update_effective_resolution()
            QMessageBox.information(
                page,
                "Success",
                f"Successfully read macrotime calibration from SPC file: {path}"
            )
        else:
            tttr = tttrlib.TTTR(path)
            header = tttr.get_header()
            page.macro_time_le.setText(str(header.macro_time_resolution * 1e9))
            page.micro_time_le.setText(str(header.micro_time_resolution * 1e9))
            page._update_effective_resolution()
            QMessageBox.information(
                page,
                "Success",
                f"Successfully read calibrations from TTTR file: {path}"
            )

    except Exception as e:
        QMessageBox.critical(
            page,
            "Error",
            f"Failed to read file: {e}"
        )


def on_calc_g_factor(page):
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
        micro_time_binning = int(page.micro_binning_combo.currentText())

        selected_rows = page.detectors_form.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(
                page,
                "Warning",
                "Please select a detector row first."
            )
            return

        selected_row = selected_rows[0].row()
        selected_detector = page.detectors_form.item(selected_row, 0).text().strip()
        channels_text = page.detectors_form.cellWidget(selected_row, 1).text()
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
            'row': selected_row,
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
        write_jordi(jordi_data, jordi_file)

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
                print(f"Selected detector info: {selected_detector_info}")
                if selected_detector_info:
                    row = selected_detector_info['row']
                    g_factor_value = f"{g_factor_calculator.g_factor:.3f}"
                    print(f"Updating detectors table row {row} with g-factor value {g_factor_value}")

                    existing_cell_widget = page.detectors_form.cellWidget(row, 3)
                    if existing_cell_widget:
                        page._set_g_factor_programmatically(row, g_factor_value)
                    else:
                        new_cell_widget = QLineEdit(g_factor_value)
                        page.detectors_form.setCellWidget(row, 3, new_cell_widget)
                        page._wire_g_factor_cell(row, new_cell_widget)

                    # Optional l1/l2 estimates from FP calibration mode
                    try:
                        l1_val = getattr(g_factor_calculator, 'l1_estimate', None)
                        l2_val = getattr(g_factor_calculator, 'l2_estimate', None)
                        if l1_val is not None and np.isfinite(float(l1_val)):
                            l1_text = f"{float(l1_val):.5f}"
                            l1_widget = page.detectors_form.cellWidget(row, 4)
                            if l1_widget is None:
                                l1_widget = QLineEdit(l1_text)
                                page.detectors_form.setCellWidget(row, 4, l1_widget)
                            else:
                                l1_widget.setText(l1_text)
                        if l2_val is not None and np.isfinite(float(l2_val)):
                            l2_text = f"{float(l2_val):.5f}"
                            l2_widget = page.detectors_form.cellWidget(row, 5)
                            if l2_widget is None:
                                l2_widget = QLineEdit(l2_text)
                                page.detectors_form.setCellWidget(row, 5, l2_widget)
                            else:
                                l2_widget.setText(l2_text)
                    except Exception:
                        pass

                    gf_range_text = None
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
                            s = int(float(rng[0]))
                            e = int(float(rng[1]))
                            if e < s:
                                s, e = e, s
                            gf_range_text = f"{s}-{e}"
                            try:
                                gf_widget = page.detectors_form.cellWidget(row, 6)
                                if gf_widget is None:
                                    gf_widget = QLineEdit(gf_range_text)
                                    page.detectors_form.setCellWidget(row, 6, gf_widget)
                                else:
                                    gf_widget.setText(gf_range_text)
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
                        if gf_range_text:
                            msg += f"G-Factor Channels: {gf_range_text}\n"
                        msg += f"Setup '{page.current_setup_name}' saved automatically."
                        QMessageBox.information(page, "Success", msg)
                    else:
                        msg = (
                            f"G-Factor calculated: {g_factor_calculator.g_factor:.4f}\n"
                            f"Updated G-Factor for detector: {selected_detector_info['name']}\n"
                        )
                        if gf_range_text:
                            msg += f"G-Factor Channels: {gf_range_text}\n"
                        msg += "Note: No setup was selected, so changes were not saved automatically."
                        QMessageBox.information(page, "Success", msg)

        g_factor_calculator.closeEvent = custom_close_event

        g_factor_calculator.show()

        try:
            effective_dt = page.effective_micro_time_resolution

            g_factor_calculator.load_jordi_file(jordi_file)

            try:
                gf_widget = page.detectors_form.cellWidget(selected_row, 6)
                if gf_widget:
                    txt = gf_widget.text().strip()
                    if txt:
                        parts = txt.replace(' ', '').split('-')
                        if len(parts) == 2:
                            s = int(float(parts[0]))
                            e = int(float(parts[1]))
                            if e < s:
                                s, e = e, s
                            if hasattr(g_factor_calculator, 'region'):
                                try:
                                    g_factor_calculator.region.setRegion([s, e])
                                except Exception:
                                    pass
                            if hasattr(g_factor_calculator, 'region_bounds'):
                                try:
                                    g_factor_calculator.region_bounds = [s, e]
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
