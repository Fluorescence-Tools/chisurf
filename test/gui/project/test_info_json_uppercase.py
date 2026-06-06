import pathlib
import tempfile
import shutil
import json
from datetime import datetime

from chisurf.gui.widgets.wizard.tttr_photonfilter import WizardTTTRPhotonFilter, load_detector_setups


def test_info_json_uppercase(qapp, qtbot):
    temp_dir = tempfile.mkdtemp()

    try:
        test_folder = "burstwise_All 0.4000#30"

        original_folder_path = pathlib.Path(temp_dir) / test_folder
        original_folder_path.mkdir(exist_ok=True)

        windows = {}
        detectors = {}

        filter_widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
        qtbot.addWidget(filter_widget)

        filter_widget.lineEdit_2.setText(test_folder)

        mock_filename = str(pathlib.Path(temp_dir) / "test_file.ptu")
        filter_widget.settings['tttr_filenames'] = [mock_filename]

        mock_setup_name = "Test Setup"
        mock_setup_data = {
            "detectors": {
                "Detector1": {"chs": [0, 1, 2]},
                "Detector2": {"chs": [3, 4, 5]}
            },
            "windows": {
                "Window1": [0, 100],
                "Window2": [200, 300]
            },
            "tttr_reading": {
                "file_type": "PTU",
                "micro_time_binning": 8
            }
        }

        mock_setups = {
            "setups": {
                mock_setup_name: mock_setup_data
            },
            "last_used": mock_setup_name
        }

        original_load_detector_setups = filter_widget.load_detector_setups
        filter_widget.load_detector_setups = lambda: mock_setups
        filter_widget.comboBox.currentText = lambda: mock_setup_name

        original_dirs = filter_widget.original_directories

        info_dir = original_dirs[0] / 'Info'
        info_dir.mkdir(exist_ok=True, parents=True)

        parameters = filter_widget.get_burst_selection_parameters()
        parameters["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parameters["selected_setup"] = mock_setup_name

        setup_data = mock_setups["setups"][mock_setup_name]
        parameters["setup_info"] = setup_data

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        params_filename = info_dir / f"photon_selection_parameters_{timestamp}.json"

        with open(params_filename, 'w') as f:
            json.dump(parameters, f, indent=4)

        assert params_filename.exists()
        assert params_filename.parent.name == "Info"

        with open(params_filename, 'r') as f:
            saved_params = json.load(f)

        assert "setup_info" in saved_params
        assert "detectors" in saved_params["setup_info"]
        assert "windows" in saved_params["setup_info"]
        assert "tttr_reading" in saved_params["setup_info"]

        filter_widget.load_detector_setups = original_load_detector_setups

    finally:
        shutil.rmtree(temp_dir)
