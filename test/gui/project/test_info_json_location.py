import os
import pathlib
import tempfile
import shutil
from qtpy import QtWidgets
import json
from datetime import datetime

from chisurf.gui.widgets.wizard.tttr_photonfilter import WizardTTTRPhotonFilter


def test_info_json_location(qapp, qtbot):
    temp_dir = tempfile.mkdtemp()

    try:
        test_folder = "burstwise_All 0.4000#30"
        test_folder_with_suffix = f"{test_folder}_0"

        original_folder_path = pathlib.Path(temp_dir) / test_folder
        suffixed_folder_path = pathlib.Path(temp_dir) / test_folder_with_suffix

        original_folder_path.mkdir(exist_ok=True)
        suffixed_folder_path.mkdir(exist_ok=True)

        windows = {}
        detectors = {}

        filter_widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
        qtbot.addWidget(filter_widget)

        filter_widget.lineEdit_2.setText(test_folder)

        mock_filename = str(pathlib.Path(temp_dir) / "test_file.ptu")
        filter_widget.settings['tttr_filenames'] = [mock_filename]

        original_dirs = filter_widget.original_directories
        parent_dirs = filter_widget.parent_directories

        assert original_dirs[0].name == test_folder
        assert parent_dirs[0].name.startswith(test_folder)
        assert "_" in parent_dirs[0].name

        info_dir = original_dirs[0] / 'info'
        info_dir.mkdir(exist_ok=True, parents=True)

        parameters = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_param": "test_value"
        }

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        params_filename = info_dir / f"photon_selection_parameters_{timestamp}.json"

        with open(params_filename, 'w') as f:
            json.dump(parameters, f, indent=4)

        assert params_filename.exists()
        assert params_filename.parent.parent.name == test_folder

    finally:
        shutil.rmtree(temp_dir)
