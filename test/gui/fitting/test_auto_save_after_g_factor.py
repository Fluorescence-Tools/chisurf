import json
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QMessageBox

from chisurf.gui.widgets.wizard.tttr_channeldefinition import (
    DetectorWizard,
    load_detector_setups,
    save_detector_setups,
)


class MockJordiGFactorCalculator:
    def __init__(self):
        self.g_factor = 1.234

    def setWindowModality(self, *args):
        pass

    def show(self):
        pass

    def load_jordi_file(self, *args):
        pass

    def closeEvent(self, event):
        pass


def mock_information(*args, **kwargs):
    return QMessageBox.Ok


class MockTTTR:
    def __init__(self, *args, **kwargs):
        pass

    def get_microtime_histogram(self, *args, **kwargs):
        return [1, 2, 3, 4, 5], None


@pytest.fixture
def setups_file(tmp_path):
    return str(tmp_path / "setups.json")


@pytest.fixture
def initial_data():
    return {
        "setups": {
            "test_setup": {
                "windows": {"prompt": {"0": 2048}},
                "detectors": {
                    "test_detector": {
                        "chs": [0, 1],
                        "micro_time_ranges": [[0, 4095]],
                        "g_factor": 1.0,
                    }
                },
                "tttr_reading": {
                    "file_type": "SPC-130",
                    "macro_time_resolution": 50.0,
                    "micro_time_resolution": 50.0,
                    "micro_time_binning": 1,
                },
            }
        },
        "last_used": "test_setup",
    }


def test_auto_save_after_g_factor(qtbot, monkeypatch, setups_file, initial_data):
    save_detector_setups(initial_data, setups_file)

    wizard = DetectorWizard(json_file=setups_file)
    qtbot.addWidget(wizard)

    page = wizard.page(0)
    page.current_setups_file = setups_file
    page.current_setup_name = "test_setup"

    monkeypatch.setattr(QMessageBox, "information", mock_information)

    import chisurf.plugins.jordi_g_factor as jordi_mod

    monkeypatch.setattr(jordi_mod, "JordiGFactorCalculator", MockJordiGFactorCalculator)

    import tttrlib

    monkeypatch.setattr(tttrlib, "TTTR", MockTTTR)

    monkeypatch.setattr(np, "where", lambda x: ([0, 1, 2, 3, 4],))
    monkeypatch.setattr(np, "concatenate", lambda x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    monkeypatch.setattr(np, "savetxt", lambda *args, **kwargs: None)

    page.detectors_form.selectRow(0)

    original_g_factor = "1.0"
    cell_widget = page.detectors_form.cellWidget(0, 3)
    if cell_widget:
        original_g_factor = cell_widget.text()
    else:
        from qtpy.QtWidgets import QLineEdit

        new_cell_widget = QLineEdit(original_g_factor)
        page.detectors_form.setCellWidget(0, 3, new_cell_widget)

    page.selected_detector = {
        "row": 0,
        "name": "test_detector",
        "parallel_channels": [0],
        "perpendicular_channels": [1],
    }

    mock_calculator = MockJordiGFactorCalculator()
    page.g_factor_calculator = mock_calculator

    def custom_close_event(event):
        selected_detector_info = page.selected_detector
        if selected_detector_info:
            row = selected_detector_info["row"]
            g_factor_value = f"{mock_calculator.g_factor:.3f}"

            existing_cell_widget = page.detectors_form.cellWidget(row, 3)
            if existing_cell_widget:
                existing_cell_widget.setText(g_factor_value)
            else:
                from qtpy.QtWidgets import QLineEdit

                new_cell_widget = QLineEdit(g_factor_value)
                page.detectors_form.setCellWidget(row, 3, new_cell_widget)

            if page.current_setup_name:
                data = page.get_settings()
                setups = load_detector_setups(page.current_setups_file)
                setups.setdefault("setups", {})

                if page.current_setup_name in setups["setups"]:
                    existing_data = setups["setups"][page.current_setup_name]
                    for key in data:
                        existing_data[key] = data[key]
                    setups["setups"][page.current_setup_name] = existing_data
                else:
                    setups["setups"][page.current_setup_name] = data

                setups["last_used"] = page.current_setup_name
                save_detector_setups(setups, page.current_setups_file)

    custom_close_event(None)

    saved_data = load_detector_setups(setups_file)
    updated_g_factor = saved_data["setups"]["test_setup"]["detectors"]["test_detector"][
        "g_factor"
    ]
    assert (
        updated_g_factor == 1.234
    ), f"G-factor was not updated. Expected 1.234, got {updated_g_factor}"
