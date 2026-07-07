"""Tests for the migrated PyQt Burst Selection GUI defaults."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from chisurf.core.mfdb.models import SampleDefinition
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.samples.sample_manager import create_sample
from chisurf.gui.widgets.dock_area.dock_area import DockArea
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
    _resolve_active_user_id,
    setup_id_for_name,
)
from chisurf.plugins.burst.burst_selection import USE_LEGACY_GUI
from chisurf.plugins.burst.burst_selection.api.models import BurstFilterMode
from chisurf.plugins.burst.burst_selection.gui import tool as tool_module
from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient
from chisurf.plugins.burst.burst_selection.gui.tool import (
    DEFAULT_CHANNELS,
    DEFAULT_D_T_MAX,
    DEFAULT_D_T_MIN,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_PHOTONS,
    DEFAULT_PHOTON_WINDOW,
    DEFAULT_TIME_WINDOW_MS,
    BurstSelectionTool,
    _raw_artifact_id_for_path,
    _register_raw_input_for_sample,
    _sample_id_for_raw_path,
    default_analysis_settings,
    histogram_data_from_frame,
    make_ui_dataframe,
)


def _bh_spc130_files() -> list[Path]:
    """Return the real BH SPC-130 fixture set used for MFDB preflight tests."""
    fixture_dir = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna"
    return sorted(fixture_dir.glob("*.spc"))


def _bh_spc130_detectors() -> dict[str, dict[str, list[int]]]:
    """Return the BH SPC-130 donor/acceptor detector channel grouping."""
    return {"green": {"chs": [0, 8]}, "red": {"chs": [1, 9]}}


def test_migrated_gui_defaults_use_legacy_thresholding() -> None:
    """The migrated GUI defaults should match the legacy Burst Selection controls."""
    settings = default_analysis_settings()
    assert DEFAULT_CHANNELS == [0, 1, 8, 9]
    assert settings.photon_filter.channels == DEFAULT_CHANNELS
    assert settings.photon_filter.filter_active is True
    assert settings.photon_filter.used_filter == BurstFilterMode.BURST
    assert settings.photon_filter.invert_filter is True
    assert settings.photon_filter.delta_macro_time_filter.dT_min == DEFAULT_D_T_MIN
    assert settings.photon_filter.delta_macro_time_filter.dT_max == DEFAULT_D_T_MAX
    assert settings.photon_filter.delta_macro_time_filter.dT_min_active is False
    assert settings.photon_filter.delta_macro_time_filter.dT_max_active is True
    assert settings.photon_filter.max_gap == DEFAULT_MAX_GAP
    assert settings.photon_filter.use_gap_fill is False
    assert settings.burst_detection.min_photons == DEFAULT_MIN_PHOTONS
    assert settings.burst_detection.photon_window == DEFAULT_PHOTON_WINDOW
    assert settings.burst_detection.time_window == DEFAULT_TIME_WINDOW_MS / 1000.0
    assert BurstSelectionTool.__init__.__kwdefaults__["show_filter_plot"] is False
    assert BurstSelectionTool.__init__.__kwdefaults__["show_burst_plot"] is False


def test_diagnostic_pens_distinguish_all_and_selected_photons() -> None:
    """All-photon and selected-photon diagnostic layers must use different colors."""
    tool = BurstSelectionTool.__new__(BurstSelectionTool)

    assert BurstSelectionTool._diagnostic_pen(tool, 0) != BurstSelectionTool._diagnostic_pen(tool, 0, selected=True)
    assert BurstSelectionTool._diagnostic_pen(tool, 1) != BurstSelectionTool._diagnostic_pen(tool, 1, selected=True)


def test_macro_time_offsets_continue_across_file_boundaries() -> None:
    """Diagnostic macro times should continue at file boundaries instead of restarting."""

    class FakeHeader:
        """TTTR header stand-in."""

        macro_time_resolution = 0.001

    class FakeTTTR:
        """TTTR stand-in."""

        header = FakeHeader()
        macro_times = np.array([10, 15, 30])

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tttr_a = FakeTTTR()
    tttr_b = FakeTTTR()
    tool._last_diagnostics = [
        {"tttr": tttr_a, "selected": np.ones(3, dtype=bool)},
        {"tttr": tttr_b, "selected": np.ones(3, dtype=bool)},
    ]

    offsets = BurstSelectionTool._macro_time_offsets_ms(tool, tool._last_diagnostics)
    delta_b = BurstSelectionTool._delta_macro_time_ms(tool, tttr_b, offsets[1])

    assert offsets == [0.0, 20.0]
    assert delta_b.tolist() == [20.0, 5.0, 15.0]


def test_histogram_data_ignores_interleaved_zero_rows() -> None:
    """Histogram updates should ignore Margarita zero separator rows."""
    frame = pd.DataFrame(
        {
            "Number of Photons": [0, 10, 0, 20],
            "Proximity Ratio": [0.0, 0.1, 0.0, 0.3],
        }
    )
    assert histogram_data_from_frame(frame, "Proximity Ratio").tolist() == [0.1, 0.3]


def test_make_ui_dataframe_computes_proximity_ratio() -> None:
    """The histogram feature list should include computed proximity ratios."""
    frame = pd.DataFrame(
        {
            "Number of Photons (red)": [0, 10, 30],
            "Number of Photons (green)": [0, 40, 70],
        }
    )

    ui_frame = make_ui_dataframe(frame)

    assert "Proximity Ratio" in ui_frame.columns
    assert ui_frame["Proximity Ratio"].round(6).tolist() == [0.2, 3 / 10]
    assert "Proximity Ratio" in list(ui_frame.columns)


def test_histogram_data_uses_computed_proximity_ratio() -> None:
    """Histogram data should use the computed proximity-ratio column."""
    frame = pd.DataFrame(
        {
            "Number of Photons (red)": [10, 30],
            "Number of Photons (green)": [40, 70],
        }
    )

    values = histogram_data_from_frame(frame, "Proximity Ratio")

    assert values.tolist() == [0.2, 3 / 10]


def test_show_selected_file_result_updates_selected_table_and_histogram() -> None:
    """Selecting a cached file should display that file's burst table and histogram."""
    path = Path("selected.spc")
    frame = pd.DataFrame({"Number of Photons": [10, 20]})
    settings = default_analysis_settings()
    calls: list[str] = []

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_frames_by_file = {path.resolve(): frame}
    tool._fill_table = lambda df: calls.append(f"table:{len(df)}") if df is not None else calls.append("table:0")
    tool._populate_feature_combo = lambda df: calls.append(f"features:{len(df.columns)}")
    tool.update_histogram = lambda: calls.append("histogram")

    BurstSelectionTool._show_selected_file_result(tool, path, settings)

    assert tool._last_frame["Number of Photons"].tolist() == [10, 20]
    assert tool._last_bur_frames == [frame]
    assert tool._last_settings is settings
    assert calls == ["table:2", "features:10", "histogram"]


def test_analyze_selected_file_updates_selected_table_and_histogram() -> None:
    """Selecting an uncached file should analyze only that file for the table and histogram."""
    path = Path("selected.spc")
    settings = default_analysis_settings()

    class FakeClient:
        """Client stub that returns one selected-file frame."""

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def analyze_files(self, file_paths: list[Path], **kwargs: object) -> dict[str, object]:
            self.calls.append({"file_paths": file_paths, **kwargs})
            return {
                "dataframes": {str(path): [{"Number of Photons": 12}]},
                "metadata": {"n_files": 1, "n_bursts": 1, "n_photons": 30},
            }

    class FakeWizard:
        """Minimal wizard stand-in for RPC context."""

        windows = {"prompt": [0, 2048]}
        detectors = {"green": {"chs": [0]}}
        decay_coarse = 8

        class ComboBox:
            """Minimal combo-box stand-in."""

            def currentText(self) -> str:
                return "Test setup"

        comboBox = ComboBox()

    client = FakeClient()
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_frames_by_file = {}
    tool._client = client
    tool.wizard = FakeWizard()
    tool._selected_filetype = "SPC-130"
    tool._status_bar = type("FakeStatusBar", (), {"showMessage": lambda self, message: None})()
    tool.summary = type("FakeSummary", (), {"setPlainText": lambda self, text: None})()
    tool._fill_table = lambda df: calls.append(f"table:{len(df)}") if df is not None else calls.append("table:0")
    tool._populate_feature_combo = lambda df: calls.append(f"features:{len(df.columns)}")
    tool.update_histogram = lambda: calls.append("histogram")
    calls: list[str] = []

    BurstSelectionTool._analyze_selected_file(tool, path, settings)

    assert client.calls[0]["file_paths"] == [path]
    assert client.calls[0]["legacy_output"] is False
    assert client.calls[0]["selected_setup"] == "Test setup"
    assert client.calls[0]["legacy_parameters"] == {"decay_coarse": 8}
    assert tool._last_frames_by_file[path.resolve()]["Number of Photons"].tolist() == [12]
    assert tool._last_bur_frames[0]["Number of Photons"].tolist() == [12]
    assert calls == ["table:1", "features:10", "histogram"]


def test_analyze_selected_file_does_not_archive_preview() -> None:
    """Selected-file preview should not emit MFDB context as an output side effect."""
    path = Path("selected.spc")
    settings = default_analysis_settings()

    class FakeClient:
        """Client stub that records MFDB context for preview analysis."""

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def analyze_files(self, file_paths: list[Path], **kwargs: object) -> dict[str, object]:
            self.calls.append({"file_paths": file_paths, **kwargs})
            return {
                "dataframes": {str(path): [{"Number of Photons": 12}]},
                "metadata": {"n_files": 1},
            }

    class FakeWizard:
        """Minimal wizard stand-in for RPC context."""

        windows = {}
        detectors = {}

        class ComboBox:
            """Minimal combo-box stand-in."""

            def currentText(self) -> str:
                return "Test setup"

        comboBox = ComboBox()

    client = FakeClient()
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_frames_by_file = {}
    tool._client = client
    tool.wizard = FakeWizard()
    tool._selected_filetype = "SPC-130"
    tool._status_bar = type("FakeStatusBar", (), {"showMessage": lambda self, message: None})()

    BurstSelectionTool._analyze_file_frame(tool, path, settings)

    assert client.calls[0]["mfdb"] is None


def test_mfdb_raw_registration_binds_content_to_sample(tmp_path: Path) -> None:
    """Registering raw input should make future content-MD5 lookups find the sample."""
    db = MFDatabase(tmp_path / "mfdb.sqlite")
    sample_id = create_sample(db, SampleDefinition(name="DNA burst sample"))
    raw_paths = _bh_spc130_files()

    assert raw_paths
    for raw_path in raw_paths:
        artifact_id = _register_raw_input_for_sample(
            db=db,
            path=raw_path,
            sample_id=sample_id,
            filetype="SPC-130",
            selected_setup="Test setup",
        )

        assert artifact_id
        assert _raw_artifact_id_for_path(db, raw_path) == artifact_id
        assert _sample_id_for_raw_path(db, raw_path) == sample_id


def test_prepare_mfdb_context_prompts_when_raw_sample_is_missing(tmp_path: Path, monkeypatch: object) -> None:
    """MFDB output should open sample registration when raw content has no sample."""
    db = MFDatabase(tmp_path / "mfdb.sqlite")
    sample_id = create_sample(db, SampleDefinition(name="Registered sample"))
    raw_paths = _bh_spc130_files()
    prompts: list[str] = []

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def isChecked(self) -> bool:
            return True

    class FakeWizard:
        """Minimal wizard stand-in exposing the selected setup."""

        detectors = _bh_spc130_detectors()
        windows = {}

        class ComboBox:
            """Minimal combo-box stand-in."""

            def currentText(self) -> str:
                return "BH SPC-130 setup"

        comboBox = ComboBox()

    def fake_sample_picker(*, db: MFDatabase, parent: object | None = None) -> str:
        """Return the sample selected in the registration dialog."""
        prompts.append("shown")
        return sample_id

    monkeypatch.setattr(tool_module, "show_sample_picker_dialog", fake_sample_picker)
    monkeypatch.setattr(tool_module, "_resolve_active_user_id", lambda: "")
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._mfdb_db = db
    tool.mfdb_output_check = FakeCheck()
    tool._selected_filetype = "SPC-130"
    tool.wizard = FakeWizard()
    tool._selected_sample_id = lambda: ""

    context = BurstSelectionTool._prepare_mfdb_context_for_paths(tool, raw_paths)

    assert prompts == ["shown"]
    assert context is not None
    assert context["enabled"] is True
    assert context["sample_id"] == sample_id
    assert set(context["source_artifact_ids"]) == {str(path.resolve()) for path in raw_paths}
    assert context["register_missing_inputs"] is True
    assert context["setup_id"] == setup_id_for_name("BH SPC-130 setup")
    assert db.get_setup(context["setup_id"]) is not None
    assert all(_sample_id_for_raw_path(db, raw_path) == sample_id for raw_path in raw_paths)


def test_mfdb_only_output_runs_batch_analysis(tmp_path: Path, monkeypatch: object) -> None:
    """Batch analysis should accept MFDB as the only selected output mode."""
    paths = _bh_spc130_files()
    settings = default_analysis_settings()
    settings.output_formats = []
    mfdb_context = {"enabled": True, "sample_id": "sample_1", "source_artifact_ids": {}}

    class FakeClient:
        """Client stub that records batch-analysis calls."""

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def analyze_files(self, file_paths: list[Path], **kwargs: object) -> dict[str, object]:
            self.calls.append({"file_paths": file_paths, **kwargs})
            return {
                "dataframes": {str(path): [{"Number of Photons": 12}] for path in paths},
                "metadata": {"n_files": len(paths), "n_bursts": len(paths), "n_photons": 12 * len(paths), "n_selected": 12 * len(paths)},
            }

    class FakeDialog:
        """Progress-dialog stand-in for headless batch tests."""

        def __init__(self, **_kwargs: object) -> None:
            self.finished: list[str] = []

        def show(self) -> None:
            return

        def update_progress(self, *_args: object) -> None:
            return

        def finish(self, final_text: str, **_kwargs: object) -> None:
            self.finished.append(final_text)

    class FakeWizard:
        """Minimal wizard stand-in for batch RPC context."""

        windows = {}
        detectors = _bh_spc130_detectors()
        decay_coarse = 8

        class ComboBox:
            """Minimal combo-box stand-in."""

            def currentText(self) -> str:
                return "Test setup"

        comboBox = ComboBox()

    client = FakeClient()
    summary_text: list[str] = []
    monkeypatch.setattr(tool_module, "EnhancedProgressDialog", FakeDialog)
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._file_paths = paths
    tool._client = client
    tool.wizard = FakeWizard()
    tool._selected_filetype = "SPC-130"
    tool._settings_from_controls = lambda: settings
    tool._mfdb_output_selected = lambda: True
    tool._prepare_mfdb_context_for_paths = lambda paths: mfdb_context
    tool._legacy_parameters = lambda: {}
    tool._selected_file_paths_from_list = lambda: paths
    tool._display_frame_set = lambda frames, current_settings, indices: True
    tool._load_tttr_for_plots = lambda paths, current_settings: None
    tool.update_burst_plots = lambda: None
    tool.summary = type("FakeSummary", (), {"setPlainText": lambda self, text: summary_text.append(text)})()

    BurstSelectionTool.analyze_files(tool)

    assert client.calls[0]["file_paths"] == paths
    assert client.calls[0]["detectors"] == _bh_spc130_detectors()
    assert client.calls[0]["mfdb"] == mfdb_context
    assert client.calls[0]["legacy_output"] is True
    assert "No output format selected." not in summary_text


def test_mfdb_only_output_keeps_zip_controls_disabled() -> None:
    """MFDB-only output should not expose packaging controls for file outputs."""

    class FakeCheck:
        """Minimal checkbox stand-in with mutable state."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked
            self.enabled = True

        def isChecked(self) -> bool:
            return self.checked

        def setChecked(self, checked: bool) -> None:
            self.checked = checked

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool.csv_output_check = FakeCheck(False)
    tool.hdf_output_check = FakeCheck(False)
    tool.mfdb_output_check = FakeCheck(True)
    tool.zip_output_check = FakeCheck(True)
    tool.remove_folder_check = FakeCheck(True)

    BurstSelectionTool._sync_output_format_controls(tool)

    assert tool.zip_output_check.enabled is False
    assert tool.zip_output_check.checked is False
    assert tool.remove_folder_check.enabled is False
    assert tool.remove_folder_check.checked is False


def test_selected_file_paths_support_multiple_selection() -> None:
    """The file list should return all queued selected paths."""
    path_a = Path("a.spc")
    path_b = Path("b.spc")

    class FakeItem:
        """Minimal selected file item stand-in."""

        def __init__(self, text: Path) -> None:
            self._text = str(text)

        def text(self) -> str:
            return self._text

    class FakeFileList:
        """Minimal file-list stand-in."""

        def selectedItems(self) -> list[FakeItem]:
            return [FakeItem(path_a), FakeItem(path_b)]

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool.file_list = FakeFileList()
    tool._file_paths = [path_a.resolve(), path_b.resolve()]

    assert BurstSelectionTool._selected_file_paths_from_list(tool) == [path_a, path_b]


def test_update_selected_files_stacks_cached_results() -> None:
    """Multiple selected files should stack their burst tables and histograms."""
    path_a = Path("a.spc")
    path_b = Path("b.spc")
    frame_a = pd.DataFrame({"Number of Photons": [10]})
    frame_b = pd.DataFrame({"Number of Photons": [20]})
    settings = default_analysis_settings()
    calls: list[str] = []

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_frames_by_file = {path_a.resolve(): frame_a, path_b.resolve(): frame_b}
    tool._fill_table = lambda df: calls.append(f"table:{len(df)}")
    tool._populate_feature_combo = lambda df: calls.append(f"features:{len(df.columns)}")
    tool.update_histogram = lambda: calls.append("histogram")
    tool._load_tttr_for_plots = lambda selected_paths, current_settings: calls.append(f"diagnostics:{selected_paths[0].name}")

    BurstSelectionTool._update_selected_files(tool, [path_a, path_b], settings)

    assert tool._last_frame["Number of Photons"].tolist() == [10, 20]
    assert tool._last_bur_frames == [frame_a, frame_b]
    assert calls == ["table:2", "features:10", "histogram", "diagnostics:a.spc"]


def test_filter_settings_change_updates_selected_file_plots() -> None:
    """Changing filter settings should refresh the selected file results and plots."""
    path = Path("selected.spc")
    selected_item = type("FakeItem", (), {"text": lambda self: str(path)})()

    class FakeFileList:
        """Minimal file-list stand-in."""

        def selectedItems(self) -> list[object]:
            return [selected_item]

    class FakeSettings:
        """Minimal settings stand-in."""

    settings = FakeSettings()
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool.file_list = FakeFileList()
    tool._file_paths = [path]
    tool._status_bar = type("FakeStatusBar", (), {"showMessage": lambda self, message: None})()
    tool._settings_from_controls = lambda: settings
    tool._analyze_selected_files = lambda selected_paths, current_settings: calls.append(("analyze", selected_paths, current_settings))
    tool._load_tttr_for_plots = lambda selected_paths, current_settings: calls.append(("diagnostics", selected_paths, current_settings))
    calls: list[tuple[str, list[Path], FakeSettings]] = []

    BurstSelectionTool._on_filter_settings_changed(tool)

    assert calls == [
        ("analyze", [path], settings),
        ("diagnostics", [path], settings),
    ]


def test_burst_plot_update_refreshes_embedded_filter_settings_plot(tmp_path: Path) -> None:
    """The filter-settings dT plot should mirror the current selected file."""

    class FakeDataItem:
        """Minimal pyqtgraph data item stand-in."""

        def __init__(self) -> None:
            self.args: tuple[object, ...] = ()
            self.kwargs: dict[str, object] = {}

        def setData(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class FakePlot:
        """Minimal plot widget stand-in."""

        def __init__(self) -> None:
            self.plots: list[dict[str, object]] = []
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

        def plot(self, *args: object, **kwargs: object) -> None:
            self.plots.append({"args": args, "kwargs": kwargs})

        def setYRange(self, *args: object, **kwargs: object) -> None:
            self.y_range = (args, kwargs)

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def __init__(self, value: int = 0) -> None:
            self._value = value
            self.maximum = 0

        def value(self) -> int:
            return self._value

        def setValue(self, value: int) -> None:
            self._value = value

        def setMaximum(self, value: int) -> None:
            self.maximum = value

        def blockSignals(self, _blocked: bool) -> None:
            return

    class FakeLineEdit:
        """Minimal line edit stand-in."""

        def __init__(self) -> None:
            self.text = ""

        def setText(self, text: str) -> None:
            self.text = text

    class FakeHeader:
        """TTTR header stand-in with macro-time resolution."""

        macro_time_resolution = 0.001

    class FakeTTTR:
        """TTTR stand-in with macro times."""

        header = FakeHeader()
        macro_times = np.array([10, 15, 30, 31])

    class FakeWizard:
        """Embedded photon-filter widget stand-in."""

        def __init__(self) -> None:
            self.tttr_objects: dict[str, object] = {}
            self.settings: dict[str, object] = {"tttr_filenames": []}
            self.lineEdit = FakeLineEdit()
            self.spinBox_2 = FakeSpinBox()
            self.spinBox_3 = FakeSpinBox()
            self.spinBox_4 = FakeSpinBox()
            self.plot_selected = FakeDataItem()
            self.plot_unselected = FakeDataItem()
            self.plot_select = FakeDataItem()
            self.tttr = None

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_tttr = FakeTTTR()
    tool._last_selected = np.array([True, False, True, False])
    tool._last_start_stop = np.array([[0, 2]])
    tool._last_diagnostic_path = tmp_path / "selected.spc"
    tool.plot_min_spin = FakeSpinBox(0)
    tool.plot_max_spin = FakeSpinBox(3)
    tool.dt_plot = FakePlot()
    tool.filter_plot = FakePlot()
    tool.filter_settings_panel = object()
    tool.wizard = FakeWizard()
    tool._update_mcs_plot = lambda start, stop: None
    tool._update_decay_plot = lambda: None
    tool._update_burst_length_plot = lambda: None
    tool._status_bar = type("FakeStatusBar", (), {"showMessage": lambda self, message: None})()

    BurstSelectionTool.update_burst_plots(tool)

    selected_x = tool.wizard.plot_selected.kwargs["x"]
    selected_y = tool.wizard.plot_selected.kwargs["y"]
    assert selected_x.tolist() == [0, 2]
    assert selected_y.tolist() == [0.0, 15.0]
    assert tool.wizard.plot_select.kwargs["y"].tolist() == [1, 0, 1, 0]
    assert tool.wizard.settings["tttr_filenames"] == [str((tmp_path / "selected.spc").resolve())]
    assert tool.wizard.lineEdit.text == str((tmp_path / "selected.spc").resolve())


def test_plot_range_controls_follow_selected_file_photon_count() -> None:
    """Toolbar photon range should match the current selected file."""

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def __init__(self, value: int = 0) -> None:
            self._value = value
            self.range = (0, 0)
            self.blocked: list[bool] = []

        def value(self) -> int:
            return self._value

        def setValue(self, value: int) -> None:
            self._value = value

        def setRange(self, minimum: int, maximum: int) -> None:
            self.range = (minimum, maximum)

        def blockSignals(self, blocked: bool) -> None:
            self.blocked.append(blocked)

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool.plot_min_spin = FakeSpinBox(10)
    tool.plot_max_spin = FakeSpinBox(100_000)

    BurstSelectionTool._sync_plot_range_controls(tool, 42, reset=True)

    assert tool.plot_min_spin.value() == 0
    assert tool.plot_max_spin.value() == 41
    assert tool.plot_min_spin.range == (0, 41)
    assert tool.plot_max_spin.range == (0, 41)

    tool.plot_min_spin.setValue(12)
    tool.plot_max_spin.setValue(99)
    BurstSelectionTool._sync_plot_range_controls(tool, 30)

    assert tool.plot_min_spin.value() == 12
    assert tool.plot_max_spin.value() == 29
    assert tool.plot_min_spin.range == (0, 29)
    assert tool.plot_max_spin.range == (0, 29)


def test_mcs_plot_offsets_use_seconds_for_time_axis() -> None:
    """MCS time offsets must be converted from macro-time milliseconds to seconds."""

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked

        def isChecked(self) -> bool:
            return self.checked

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def value(self) -> float:
            return 0.25

    class FakePlot:
        """Plot stand-in that records plot calls."""

        def __init__(self) -> None:
            self.plots: list[dict[str, object]] = []

        def clear(self) -> None:
            self.plots.clear()

        def plot(self, *args: object, **kwargs: object) -> None:
            self.plots.append({"args": args, "kwargs": kwargs})

        def setLabel(self, *_args: object, **_kwargs: object) -> None:
            return

    class FakeHeader:
        """TTTR header stand-in."""

        macro_time_resolution = 0.001

    class FakeTTTR:
        """TTTR stand-in."""

        header = FakeHeader()
        macro_times = np.array([0, 1, 2])

        def __getitem__(self, _indices: np.ndarray) -> FakeTTTR:
            return self

        def get_intensity_trace(self, time_window_length: float) -> np.ndarray:
            return np.array([1.0])

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_diagnostics = [
        {"tttr": FakeTTTR(), "selected": np.ones(1, dtype=bool)},
        {"tttr": FakeTTTR(), "selected": np.ones(1, dtype=bool)},
    ]
    tool.mcs_bin_spin = FakeSpinBox()
    tool.mcs_plot = FakePlot()
    tool.mcs_show_all_check = FakeCheck(False)
    tool.mcs_show_selected_check = FakeCheck(True)

    BurstSelectionTool._update_mcs_plot(tool, 0, 2)

    assert tool.mcs_plot.plots[0]["args"][0].tolist() == [0.0]
    assert tool.mcs_plot.plots[1]["args"][0].tolist() == [0.002]


def test_mcs_plot_draws_all_before_selected_and_respects_toggles() -> None:
    """Selected MCS trace should be plotted last and disabled traces should not compute."""

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked

        def isChecked(self) -> bool:
            return self.checked

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def value(self) -> float:
            return 0.25

    class FakePlot:
        """Plot stand-in that records plot calls."""

        def __init__(self) -> None:
            self.plots: list[dict[str, object]] = []

        def clear(self) -> None:
            self.plots.clear()

        def plot(self, *args: object, **kwargs: object) -> None:
            self.plots.append({"args": args, "kwargs": kwargs})

        def setLabel(self, *_args: object, **_kwargs: object) -> None:
            return

    class FakeTTTR:
        """TTTR stand-in that distinguishes full and selected traces."""

        calls: list[list[int]] = []

        def __init__(self, label: str = "full") -> None:
            self.label = label

        def __getitem__(self, indices: np.ndarray) -> FakeTTTR:
            values = np.asarray(indices, dtype=int).tolist()
            FakeTTTR.calls.append(values)
            if values == [1, 2, 3]:
                return FakeTTTR("range")
            return FakeTTTR("selected")

        def get_intensity_trace(self, time_window_length: float) -> np.ndarray:
            assert time_window_length == 0.00025
            if self.label == "range":
                return np.array([1.0, 2.0, 3.0])
            if self.label == "selected":
                return np.array([10.0])
            return np.array([99.0])

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_tttr = FakeTTTR()
    tool._last_selected = np.array([False, False, True, False, True])
    tool.mcs_bin_spin = FakeSpinBox()
    tool.mcs_plot = FakePlot()
    tool.mcs_show_all_check = FakeCheck(True)
    tool.mcs_show_selected_check = FakeCheck(True)

    BurstSelectionTool._update_mcs_plot(tool, 1, 4)

    assert len(tool.mcs_plot.plots) == 2
    assert tool.mcs_plot.plots[0]["args"][1].tolist() == [1.0, 2.0, 3.0]
    assert tool.mcs_plot.plots[1]["args"][1].tolist() == [10.0]
    assert FakeTTTR.calls == [[1, 2, 3], [2]]

    tool.mcs_show_all_check.checked = False
    FakeTTTR.calls.clear()
    BurstSelectionTool._update_mcs_plot(tool, 1, 4)

    assert len(tool.mcs_plot.plots) == 1
    assert tool.mcs_plot.plots[0]["args"][1].tolist() == [10.0]
    assert FakeTTTR.calls == [[2]]


def test_shared_photon_toggles_apply_to_dt_and_filter_plots() -> None:
    """Toolbar photon toggles should control dT and filter diagnostic layers."""

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked

        def isChecked(self) -> bool:
            return self.checked

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def __init__(self, value: int) -> None:
            self._value = value

        def value(self) -> int:
            return self._value

        def setValue(self, value: int) -> None:
            self._value = value

        def setRange(self, _minimum: int, _maximum: int) -> None:
            return

        def blockSignals(self, _blocked: bool) -> None:
            return

    class FakePlot:
        """Plot stand-in that records plot calls."""

        def __init__(self) -> None:
            self.plots: list[dict[str, object]] = []

        def clear(self) -> None:
            self.plots.clear()

        def plot(self, *args: object, **kwargs: object) -> None:
            self.plots.append({"args": args, "kwargs": kwargs})

        def setYRange(self, *_args: object, **_kwargs: object) -> None:
            return

    class FakeHeader:
        """TTTR header stand-in."""

        macro_time_resolution = 0.001

    class FakeTTTR:
        """TTTR stand-in."""

        header = FakeHeader()
        macro_times = np.array([10, 15, 30, 31])

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_tttr = FakeTTTR()
    tool._last_selected = np.array([True, False, True, False])
    tool._last_start_stop = np.array([[0, 2]])
    tool.plot_min_spin = FakeSpinBox(0)
    tool.plot_max_spin = FakeSpinBox(3)
    tool.dt_plot = FakePlot()
    tool.filter_plot = FakePlot()
    tool.filter_settings_panel = None
    tool.show_all_photons_check = FakeCheck(True)
    tool.show_selected_photons_check = FakeCheck(True)
    tool._diagnostic_plot_features = {
        "Filter": {"initial_enabled": True, "check": FakeCheck(True), "widget": tool.filter_plot},
        "MCS": {"initial_enabled": False, "check": FakeCheck(False), "widget": object()},
        "Decay": {"initial_enabled": False, "check": FakeCheck(False), "widget": object()},
        "Burst length": {"initial_enabled": False, "check": FakeCheck(False), "widget": object()},
    }
    tool._closed_diagnostic_plots = set()
    tool._status_bar = type("FakeStatusBar", (), {"showMessage": lambda self, message: None})()

    BurstSelectionTool.update_burst_plots(tool)

    assert len(tool.dt_plot.plots) == 2
    assert tool.dt_plot.plots[0]["args"][0].tolist() == [0, 1, 2, 3]
    assert tool.dt_plot.plots[1]["args"][0].tolist() == [0, 2]
    assert tool.dt_plot.plots[0]["kwargs"]["pen"] != tool.dt_plot.plots[1]["kwargs"]["pen"]
    assert tool.filter_plot.plots[0]["args"][0].tolist() == [0, 1, 2, 3]
    assert tool.filter_plot.plots[1]["args"][0].tolist() == [0, 2]
    assert tool.filter_plot.plots[0]["kwargs"]["pen"] != tool.filter_plot.plots[1]["kwargs"]["pen"]
    assert tool.filter_plot.plots[0]["args"][1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert tool.filter_plot.plots[1]["args"][1].tolist() == [1.0, 1.0]

    tool.show_all_photons_check.checked = True
    tool.show_selected_photons_check.checked = False
    BurstSelectionTool.update_burst_plots(tool)

    assert len(tool.dt_plot.plots) == 1
    assert tool.dt_plot.plots[0]["args"][0].tolist() == [0, 1, 2, 3]
    assert tool.filter_plot.plots[0]["args"][0].tolist() == [0, 1, 2, 3]
    assert tool.filter_plot.plots[0]["args"][1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_shared_photon_toggles_apply_to_decay_plot() -> None:
    """Decay plotting should compute only enabled photon layers."""

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked

        def isChecked(self) -> bool:
            return self.checked

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def value(self) -> int:
            return 2

    class FakePlot:
        """Plot stand-in that records calls."""

        def __init__(self) -> None:
            self.plots: list[dict[str, object]] = []

        def clear(self) -> None:
            self.plots.clear()

        def plot(self, *args: object, **kwargs: object) -> None:
            self.plots.append({"args": args, "kwargs": kwargs})

    class FakeTTTR:
        """TTTR stand-in with separate all/selected histogram calls."""

        calls: list[str] = []

        def __init__(self, label: str = "all") -> None:
            self.label = label

        def __getitem__(self, _indices: np.ndarray) -> FakeTTTR:
            return FakeTTTR("selected")

        def get_microtime_histogram(self, coarse: int) -> tuple[np.ndarray, np.ndarray]:
            assert coarse == 2
            FakeTTTR.calls.append(self.label)
            return np.array([1.0, 0.0]), np.array([1.0, 2.0])

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_tttr = FakeTTTR()
    tool._last_selected = np.array([True, False, True])
    tool.decay_bins_spin = FakeSpinBox()
    tool.decay_plot = FakePlot()
    tool.show_all_photons_check = FakeCheck(False)
    tool.show_selected_photons_check = FakeCheck(True)

    BurstSelectionTool._update_decay_plot(tool)

    assert FakeTTTR.calls == ["selected"]

    tool.show_all_photons_check.checked = True
    tool.show_selected_photons_check.checked = False
    FakeTTTR.calls.clear()
    BurstSelectionTool._update_decay_plot(tool)

    assert FakeTTTR.calls == ["all"]


def test_update_burst_plots_skips_closed_mcs_dock() -> None:
    """Closed MCS docks should not trigger MCS trace computation."""

    class FakeSpinBox:
        """Minimal spin box stand-in."""

        def __init__(self, value: int) -> None:
            self._value = value

        def value(self) -> int:
            return self._value

    class FakePlot:
        """Minimal plot stand-in."""

        def clear(self) -> None:
            return

        def plot(self, *_args: object, **_kwargs: object) -> None:
            return

        def setYRange(self, *_args: object, **_kwargs: object) -> None:
            return

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked

        def isChecked(self) -> bool:
            return self.checked

    class FakeHeader:
        """TTTR header stand-in."""

        macro_time_resolution = 0.001

    class FakeTTTR:
        """TTTR stand-in."""

        header = FakeHeader()
        macro_times = np.array([1, 2, 4])

    calls = {"mcs": 0}
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_tttr = FakeTTTR()
    tool._last_selected = np.array([True, False, True])
    tool.plot_min_spin = FakeSpinBox(0)
    tool.plot_max_spin = FakeSpinBox(2)
    tool.dt_plot = None
    tool.filter_settings_panel = None
    tool.filter_plot = FakePlot()
    tool._diagnostic_plot_features = {
        "Filter": {"initial_enabled": False, "check": FakeCheck(False), "widget": object()},
        "MCS": {"initial_enabled": True, "check": FakeCheck(True), "widget": object(), "dock_widget": object()},
        "Decay": {"initial_enabled": False, "check": FakeCheck(False), "widget": object()},
        "Burst length": {"initial_enabled": False, "check": FakeCheck(False), "widget": object()},
    }
    tool._closed_diagnostic_plots = {"MCS"}
    tool._update_mcs_plot = lambda start, stop: calls.__setitem__("mcs", calls["mcs"] + 1)
    tool._status_bar = type("FakeStatusBar", (), {"showMessage": lambda self, message: None})()

    BurstSelectionTool.update_burst_plots(tool)

    assert calls["mcs"] == 0


def test_dock_context_menu_lists_closed_docks() -> None:
    """Dock context menu should offer closed diagnostic docks for reopening."""

    class FakeCheck:
        """Minimal checkbox stand-in."""

        def __init__(self, checked: bool) -> None:
            self.checked = checked

        def isChecked(self) -> bool:
            return self.checked

    class FakeSignal:
        """Minimal signal stand-in."""

        def connect(self, _slot: object) -> None:
            return

    class FakeAction:
        """Minimal action stand-in."""

        def __init__(self, text: str, submenu: FakeMenu | None = None) -> None:
            self._text = text
            self._submenu = submenu
            self.triggered = FakeSignal()

        def text(self) -> str:
            return self._text

        def menu(self) -> FakeMenu | None:
            return self._submenu

    class FakeMenu:
        """Minimal menu stand-in."""

        def __init__(self, title: str = "") -> None:
            self._title = title
            self._actions: list[FakeAction] = []

        def addSeparator(self) -> None:
            return

        def addMenu(self, title: str) -> FakeMenu:
            submenu = FakeMenu(title)
            self._actions.append(FakeAction(title, submenu))
            return submenu

        def addAction(self, title: str) -> FakeAction:
            action = FakeAction(title)
            self._actions.append(action)
            return action

        def actions(self) -> list[FakeAction]:
            return self._actions

        def title(self) -> str:
            return self._title

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._closed_diagnostic_plots = {"MCS", "Decay"}
    tool._diagnostic_plot_features = {
        "MCS": {"initial_enabled": True, "check": FakeCheck(False), "widget": object()},
        "Decay": {"initial_enabled": True, "check": FakeCheck(False), "widget": object()},
        "Filter": {"initial_enabled": True, "check": FakeCheck(True), "widget": object()},
    }

    menu = FakeMenu()
    BurstSelectionTool._add_dock_context_menu_actions(tool, menu, -1)

    submenus = [action.menu() for action in menu.actions() if action.menu() is not None]
    reopen_menu = next(submenu for submenu in submenus if submenu.title() == "Reopen closed docks")
    assert [action.text() for action in reopen_menu.actions()] == ["MCS", "Decay"]


def test_dock_configuration_enables_close_buttons() -> None:
    """Diagnostic dock tabs should expose close buttons."""

    class FakeDockArea:
        """Minimal dock-area stand-in."""

        def __init__(self) -> None:
            self.context_menu_enabled = False
            self.context_menu_mode = ""
            self.tabs_closable = False
            self.close_callback = None
            self.context_callback = None

        def setContextMenuEnabled(self, enabled: bool) -> None:
            self.context_menu_enabled = enabled

        def setContextMenuMode(self, mode: str) -> None:
            self.context_menu_mode = mode

        def setTabsClosable(self, closable: bool) -> None:
            self.tabs_closable = closable

        def setCloseTabCallback(self, callback: object) -> None:
            self.close_callback = callback

        def setContextMenuCallback(self, callback: object) -> None:
            self.context_callback = callback

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool.dock_area = FakeDockArea()
    tool.filter_plot = object()
    tool.filter_dock_widget = object()
    tool.plot_filter_check = object()
    tool.show_filter_plot = True
    tool.mcs_plot = object()
    tool.mcs_dock_widget = object()
    tool.plot_mcs_check = object()
    tool.show_mcs_plot = True
    tool.decay_plot = object()
    tool.decay_dock_widget = object()
    tool.plot_decay_check = object()
    tool.show_decay_plot = True
    tool.burst_plot = object()
    tool.burst_dock_widget = object()
    tool.plot_burst_check = object()
    tool.show_burst_plot = True

    BurstSelectionTool._configure_dock_context_menu(tool)

    assert tool.dock_area.context_menu_enabled is True
    assert tool.dock_area.context_menu_mode == "basic"
    assert tool.dock_area.tabs_closable is True
    assert tool.dock_area.close_callback is not None
    assert tool.dock_area.context_callback is not None


def test_dock_close_callback_hides_non_file_docks() -> None:
    """Burst dock callback should only remove the files dock."""

    class FakeDockArea:
        """Minimal dock-area stand-in for close callback behavior."""

        def __init__(self, widgets: list[object]) -> None:
            self.widgets = widgets
            self.hidden: list[int] = []
            self.removed: list[int] = []

        def widget(self, index: int) -> object:
            return self.widgets[index]

        def hideTab(self, index: int) -> None:
            self.hidden.append(index)

        def removeTab(self, index: int) -> None:
            self.removed.append(index)

    files_widget = object()
    dt_widget = object()
    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool.files_dock_widget = files_widget
    tool.dock_area = FakeDockArea([files_widget, dt_widget])
    tool._diagnostic_plot_features = {}

    BurstSelectionTool._on_dock_tab_close_requested(tool, 1)
    BurstSelectionTool._on_dock_tab_close_requested(tool, 0)

    assert tool.dock_area.hidden == [1]
    assert tool.dock_area.removed == [0]


def test_dock_visibility_menu_lists_available_docks() -> None:
    """Dock context menu should list all available docks with check states."""
    class FakeSignal:
        """Minimal signal stand-in."""

        def connect(self, slot: object) -> None:
            self.slot = slot

    class FakeAction:
        """Minimal action stand-in."""

        def __init__(self, text: str) -> None:
            self._text = text
            self.checkable = False
            self.checked = False
            self.enabled = True
            self.triggered = FakeSignal()

        def text(self) -> str:
            return self._text

        def setCheckable(self, checkable: bool) -> None:
            self.checkable = checkable

        def setChecked(self, checked: bool) -> None:
            self.checked = checked

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class FakeMenu:
        """Minimal menu stand-in."""

        def __init__(self) -> None:
            self._actions: list[FakeAction] = []
            self.separators = 0

        def actions(self) -> list[FakeAction]:
            return self._actions

        def addSeparator(self) -> None:
            self.separators += 1

        def addAction(self, text: str) -> FakeAction:
            action = FakeAction(text)
            self._actions.append(action)
            return action

    dock_area = DockArea.__new__(DockArea)
    dock_area._all_widgets = [object(), object(), object()]
    dock_area._tab_names = {
        dock_area._all_widgets[0]: "Files",
        dock_area._all_widgets[1]: "MCS",
        dock_area._all_widgets[2]: "Decay",
    }
    dock_area.count = lambda: 3
    dock_area.tabText = lambda index: ["Files", "MCS", "Decay"][index]
    dock_area.isTabVisible = lambda index: index != 1
    dock_area.visibleCount = lambda: 2
    calls: list[tuple[int, bool]] = []
    dock_area._set_dock_visible = lambda index, visible: calls.append((index, visible))

    menu = FakeMenu()
    DockArea._add_dock_visibility_actions(dock_area, menu)

    assert [action.text() for action in menu.actions()] == ["Files", "MCS", "Decay"]
    assert [action.checkable for action in menu.actions()] == [True, True, True]
    assert [action.checked for action in menu.actions()] == [True, False, True]
    menu.actions()[1].triggered.slot(True)
    assert calls == [(1, True)]


def test_burst_durations_use_macro_time_resolution_ms() -> None:
    """Burst duration values should be reported in milliseconds."""

    class FakeHeader:
        """TTTR header stand-in."""

        macro_time_resolution = 0.002

    class FakeTTTR:
        """TTTR stand-in."""

        header = FakeHeader()
        macro_times = np.array([0, 5, 8, 20])

    tool = BurstSelectionTool.__new__(BurstSelectionTool)
    tool._last_tttr = FakeTTTR()
    tool._last_start_stop = np.array([[0, 1], [1, 3]])

    durations = BurstSelectionTool._burst_durations_ms(tool)

    assert durations.tolist() == [10.0, 30.0]


def test_client_analyze_files_passes_detector_setup_context() -> None:
    """Detector setup context must be forwarded to the RPC analyze method."""
    class FakeClient:
        """Minimal client stub that records RPC calls."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((method, params))
            return {"ok": True, "result": {"dataframes": {}}}

    fake = FakeClient()
    client = BurstSelectionClient(fake)
    result = client.analyze_files(
        [Path("example.spc")],
        settings={"photon_filter": {"channels": [0, 8]}},
        windows={"prompt": [0, 2048]},
        detectors={"green": {"chs": [0, 8]}},
        filetype="SPC-130",
        output_dir=Path("burstwise_All 0.2000#60") / "bi4_bur",
        legacy_output=True,
        legacy_output_folder_name="burstwise_All 0.2000#60",
        selected_setup="Test",
        legacy_parameters={"decay_coarse": 8},
        mfdb={
            "enabled": True,
            "sample_id": "sample_1",
            "source_artifact_ids": {"example.spc": "artifact_1"},
        },
    )
    assert result == {"dataframes": {}}
    assert fake.calls[0][0] == "burst_selection.jobs.analyze_files"
    assert fake.calls[0][1]["filetype"] == "SPC-130"
    assert fake.calls[0][1]["windows"] == {"prompt": [0, 2048]}
    assert fake.calls[0][1]["detectors"] == {"green": {"chs": [0, 8]}}
    assert fake.calls[0][1]["output_dir"] == "burstwise_All 0.2000#60/bi4_bur"
    assert fake.calls[0][1]["legacy_output"] is True
    assert fake.calls[0][1]["legacy_output_folder_name"] == "burstwise_All 0.2000#60"
    assert fake.calls[0][1]["selected_setup"] == "Test"
    assert fake.calls[0][1]["legacy_parameters"] == {"decay_coarse": 8}
    assert fake.calls[0][1]["mfdb"] == {
        "enabled": True,
        "sample_id": "sample_1",
        "source_artifact_ids": {"example.spc": "artifact_1"},
    }


def test_client_load_diagnostics_accepts_unset_microtime_ranges() -> None:
    """Diagnostic loading should treat unset microtime ranges as unrestricted."""
    path = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna" / "m000.spc"
    settings = default_analysis_settings()
    settings.photon_filter.channels = []
    settings.photon_filter.microtime_ranges = None

    diag = BurstSelectionClient().load_diagnostics(path, settings=asdict(settings))

    assert set(diag) >= {"tttr", "selected", "start_stop", "settings"}
    assert diag["settings"].photon_filter.microtime_ranges == []
    assert len(diag["selected"]) == len(diag["tttr"])


def test_plugin_uses_migrated_gui_by_default() -> None:
    """The plugin-level switch should select the migrated GUI."""
    assert USE_LEGACY_GUI is False
    assert BurstSelectionTool.__name__ == "BurstSelectionTool"
