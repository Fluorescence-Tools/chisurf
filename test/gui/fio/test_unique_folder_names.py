import pathlib
import tempfile
import shutil

from chisurf.gui.widgets.wizard.tttr_photonfilter import WizardTTTRPhotonFilter


def test_unique_folder_paths(qapp, qtbot):
    temp_dir = tempfile.mkdtemp()

    test_folders = [
        "bocpd_All 1.0000#10",
        "kalman_All 1.0000#10",
        "burstwise_All 1.0000#10"
    ]

    for folder in test_folders:
        folder_path = pathlib.Path(temp_dir) / folder
        folder_path.mkdir(exist_ok=True)

    windows = {}
    detectors = {}

    filter_widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
    qtbot.addWidget(filter_widget)

    for folder in test_folders:
        base_path = pathlib.Path(temp_dir) / folder
        unique_path = filter_widget.get_unique_folder_path(base_path)
        assert unique_path != base_path
        assert str(unique_path).startswith(str(base_path))

    non_existent = pathlib.Path(temp_dir) / "non_existent_folder"
    unique_path = filter_widget.get_unique_folder_path(non_existent)
    assert unique_path == non_existent

    multi_suffix_base = pathlib.Path(temp_dir) / "multi_suffix"
    multi_suffix_base.mkdir(exist_ok=True)

    (multi_suffix_base.parent / f"{multi_suffix_base.name}_0").mkdir(exist_ok=True)
    (multi_suffix_base.parent / f"{multi_suffix_base.name}_1").mkdir(exist_ok=True)

    unique_path = filter_widget.get_unique_folder_path(multi_suffix_base)
    assert unique_path.name == f"{multi_suffix_base.name}_2"

    shutil.rmtree(temp_dir)
