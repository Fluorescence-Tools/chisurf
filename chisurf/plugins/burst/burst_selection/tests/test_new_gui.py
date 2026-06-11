"""Tests for the migrated PyQt Burst Selection GUI defaults."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from chisurf.plugins.burst.burst_selection import USE_LEGACY_GUI
from chisurf.plugins.burst.burst_selection.api.models import BurstFilterMode
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
    default_analysis_settings,
    histogram_data_from_frame,
)


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


def test_histogram_data_ignores_interleaved_zero_rows() -> None:
    """Histogram updates should ignore Margarita zero separator rows."""
    frame = pd.DataFrame(
        {
            "Number of Photons": [0, 10, 0, 20],
            "Proximity Ratio": [0.0, 0.1, 0.0, 0.3],
        }
    )
    assert histogram_data_from_frame(frame, "Proximity Ratio").tolist() == [0.1, 0.3]


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

        def __getitem__(self, indices: np.ndarray) -> "FakeTTTR":
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
    tool.show_all_photons_check = FakeCheck(False)
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

    assert len(tool.dt_plot.plots) == 1
    assert tool.dt_plot.plots[0]["args"][0].tolist() == [0, 2]
    assert tool.filter_plot.plots[0]["args"][0].tolist() == [0, 2]
    assert tool.filter_plot.plots[0]["args"][1].tolist() == [1.0, 1.0]

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

        def __getitem__(self, _indices: np.ndarray) -> "FakeTTTR":
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

        def __init__(self, text: str, submenu: "FakeMenu | None" = None) -> None:
            self._text = text
            self._submenu = submenu
            self.triggered = FakeSignal()

        def text(self) -> str:
            return self._text

        def menu(self) -> "FakeMenu | None":
            return self._submenu

    class FakeMenu:
        """Minimal menu stand-in."""

        def __init__(self, title: str = "") -> None:
            self._title = title
            self._actions: list[FakeAction] = []

        def addSeparator(self) -> None:
            return

        def addMenu(self, title: str) -> "FakeMenu":
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
