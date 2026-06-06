import pathlib
import tempfile
import shutil
import json
from datetime import datetime

from chisurf.gui.widgets.wizard.tttr_photonfilter import WizardTTTRPhotonFilter


def test_datetime_file(qapp, qtbot):
    temp_dir = tempfile.mkdtemp()

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

    original_dirs = filter_widget.original_directories

    info_dir = original_dirs[0] / 'Info'
    info_dir.mkdir(exist_ok=True, parents=True)

    parameters = filter_widget.get_burst_selection_parameters()
    parameters["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    params_filename = info_dir / "photon_selection_parameters.json"

    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d-%H%M%S")
    datetime_filename = info_dir / "datetime.txt"

    with open(params_filename, 'w') as f:
        json.dump(parameters, f, indent=4)

    with open(datetime_filename, 'w') as f:
        f.write(f"Date: {current_datetime.strftime('%Y-%m-%d')}\n")
        f.write(f"Time: {current_datetime.strftime('%H:%M:%S')}\n")
        f.write(f"Timestamp: {timestamp}\n")

    assert params_filename.exists()
    assert datetime_filename.exists()

    with open(datetime_filename, 'r') as f:
        content = f.read()

    assert "Date: " in content
    assert "Time: " in content
    assert "Timestamp: " in content

    shutil.rmtree(temp_dir)
