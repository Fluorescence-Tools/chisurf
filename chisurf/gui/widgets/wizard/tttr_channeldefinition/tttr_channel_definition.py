# This file contains the DetectorWizardPage class which is used for detector and PIE-window definition.
# The UI for this class is now defined in a separate .ui file (detector_wizard_page.ui) instead of
# being created programmatically. This makes it easier to maintain and modify the UI.
# The UI file is loaded in the __init__ method of the DetectorWizardPage class.
#
# A helper method _hide_layout_widgets is used to hide/show all widgets in a layout.
# This method is used instead of trying to access widget containers directly,
# which can cause AttributeError if the widget names don't match between the
# code and the UI file.

import json
import pathlib
import sys

import numpy as np

try:  # pyqtgraph is optional; the preview plot degrades gracefully without it
    import pyqtgraph as pg
except Exception:  # pragma: no cover - environment without pyqtgraph
    pg = None

from qtpy import uic as _uic
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)


def qtpy_loadUi(path, baseinstance=None):
    return _uic.loadUi(path, baseinstance)

import tttrlib

from .tttr_channel_definition_json_dialog import JsonEditorDialog
from .tttr_channel_definition_tttr_io import (
    on_calc_g_factor as _on_calc_g_factor,
)
from .tttr_channel_definition_tttr_io import (
    read_from_tttr_file as _read_from_tttr_file,
)
from .tttr_detector_setups import (
    DETECTOR_SETUPS_FILE,
    _resolve_active_user_id,
    load_detector_setups,
    save_detector_setups,
)
from chisurf.plugins.core.lightpath_simulator.core.workflow import (
    get_probes_info,
    resolve_db_path,
)
from chisurf.plugins.core.lightpath_simulator.gui.easy_mode import (
    LightPathEasyDialog,
)

help_text = """You can either load an existing detector Pulsed-Interleaved Excitation (PIE)
window definition by clicking on the '...' button to define channels, or define your own PIE
and detector settings by editing the tables below. New PIE windows and detector windows can
be added by clicking the "Add" button next to the detector name field. The "Edit" button
displays a JSON file representing the data, and the "Save" button allows you to save your
channel configuration.

You can also select from predefined setups using the Setup dropdown, or save your current
configuration as a new setup.

The TTTR reading routine section allows you to specify the file type and time resolution
parameters used when reading TTTR files. These settings will be saved with your setup.

IMPORTANT — micro-time units: PIE windows ("Start"/"End") and detector "Micro Time Ranges"
are given in RAW micro-time channels of the TTTR file (typically 0 … a few thousand), i.e.
the SAME units as the loaded data. They are NOT divided by the micro-time binning — binning
only changes how the preview decay is displayed, never the stored ranges. A common mistake is
to enter a range like 0-256 that only covers a small fraction of the data and therefore
selects almost no photons (e.g. a proximity ratio that collapses to 0).

Use "Read from file…" to load a dataset: the preview below shows the micro-time decay of your
data with your PIE windows and detector ranges overlaid, so you can confirm the ranges cover
the intended part of the decay before saving the setup.
"""

# Initial PIE-Windows and Detectors
_initial_windows = {
    "prompt": (0, 2048),
    "delayed": (2048, 4095)
}

_initial_detectors = {
    "green":  {"chs": [8, 0, 3], "micro_time_ranges": [(0, 4095)], "g_factor": 1, "l1": 0, "l2": 0},
    "red":    {"chs": [9, 1, 2], "micro_time_ranges": [(0, 2048)], "g_factor": 1, "l1": 0, "l2": 0},
    "yellow": {"chs": [9, 1, 2], "micro_time_ranges": [(2048, 4095)], "g_factor": 1, "l1": 0, "l2": 0},
}

# Initial TTTR reading routine settings
_initial_tttr_reading = {
    "file_type": "SPC-130",
    "macro_time_resolution": 50.0,  # in nanoseconds
    "micro_time_resolution": 50.0,  # in picoseconds
    "micro_time_binning": 1,
    "excitation_period": 13.6,  # in nanoseconds
    "g_factor": 1.08316,
    "l1": 0.03080,
    "l2": 0.03680
}


class DetectorWizardPage(QWizardPage):
    detectorsChanged = Signal()

    def event(self, event):
        # On macOS, QWizardPage C++ event implementation can cause the main window
        # to lose focus when the page is embedded as a standard widget outside a QWizard.
        # To prevent this focus-loss bug, we bypass QWizardPage.event and delegate
        # directly to QWidget.event if we are not hosted inside a QWizard.
        if self.wizard() is None:
            return QWidget.event(self, event)
        return super().event(event)


    def __init__(self, json_file=None, *args, show_edit_json=False, show_save=False,
                 show_setups_file=True, show_setup_selection=True, show_help=True,
                 show_tttr_reading=True, show_tables=True, show_add_inputs=True,
                 allow_finish=True, **kwargs):
        """Initialize the DetectorWizardPage.
        
        This class uses a UI file (detector_wizard_page.ui) for its layout and widgets.
        The UI file is loaded in the __init__ method and all signals are connected to their
        respective slots.

        Args:
            json_file (str, optional): Path to a JSON file to load. Defaults to None.
            show_edit_json (bool, optional): Whether to show the "Edit JSON" button. Defaults to False.
            show_save (bool, optional): Whether to show the "Save" button. Defaults to False.
            show_setups_file (bool, optional): Whether to show the setups file section. Defaults to True.
            show_setup_selection (bool, optional): Whether to show the setup selection section. Defaults to True.
            show_help (bool, optional): Whether to show the help button and text. Defaults to True.
            show_tttr_reading (bool, optional): Whether to show the TTTR reading routine section. Defaults to True.
            show_tables (bool, optional): Whether to show the PIE-Windows and Detectors tables. Defaults to True.
            show_add_inputs (bool, optional): Whether to show the controls for adding windows and detectors. Defaults to True.
            *args: Additional positional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.setTitle("Detectors and PIE-window definition")
        self.current_setup_name = None
        self.current_setups_file = str(DETECTOR_SETUPS_FILE)
        self._selected_detector_info = None
        self._optical_config = None
        self.show_edit_json = show_edit_json
        self.show_save = show_save
        self.show_setups_file = show_setups_file
        self.show_setup_selection = show_setup_selection
        self.show_help = show_help
        self.show_tttr_reading = show_tttr_reading
        self.show_tables = show_tables
        self.show_add_inputs = show_add_inputs

        # Protection flags/state for G-Factor edits
        # Only direct user edits or internal calculator/data loading may change g-factor fields
        self._allow_g_update = False  # internal whitelist for programmatic updates
        self._g_user_editing = {}     # row -> bool, True while the user is actively editing
        self._g_last_valid = {}       # row -> last accepted string value

        # Load the UI file
        ui_file_path = pathlib.Path(__file__).parent / "detector_wizard_page.ui"
        qtpy_loadUi(str(ui_file_path), self)

        # Set initial values
        
        # Connect signals
        self.setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        self.save_setup_button.clicked.connect(self._on_save_setup)
        self.rename_setup_button.clicked.connect(self._on_rename_setup)
        self.delete_setup_button.clicked.connect(self._on_delete_setup)
        self.help_button.clicked.connect(self._toggle_help)
        # Public visibility checkbox — only the owner can toggle it
        self.public_checkbox = QCheckBox("Public")
        self.public_checkbox.setChecked(False)
        self.public_checkbox.setToolTip(
            "When checked, this setup is visible to all users in "
            "the MFDB. Only the owner can change this setting."
        )
        # Disabled by default; enabled when an owned setup is selected
        self.public_checkbox.setEnabled(False)
        self.setup_layout.insertWidget(self.setup_layout.count() - 1, self.public_checkbox)
        self.setup_layout.insertSpacing(self.setup_layout.count() - 1, 15)

        # Calibration date snapshot combobox
        self.calibration_label = QLabel("Calibration:")
        self.calibration_label.setToolTip(
            "Select a calibration date snapshot. "
            "'Latest' uses the most recent calibration values."
        )
        self.calibration_label.setVisible(self.show_setup_selection)
        self.calibration_combo = QComboBox()
        self.calibration_combo.setToolTip(
            "Select a calibration date snapshot. "
            "'Latest' uses the most recent calibration values."
        )
        self.calibration_combo.setVisible(self.show_setup_selection)
        self.calibration_combo.addItem("Latest")
        self.setup_layout.insertWidget(self.setup_layout.count() - 1, self.calibration_label)
        self.setup_layout.insertWidget(self.setup_layout.count() - 1, self.calibration_combo)
        self.calibration_combo.currentIndexChanged.connect(self._on_calibration_changed)
        self.read_tttr_button.clicked.connect(self._read_from_tttr_file)
        self.micro_time_le.textChanged.connect(self._update_effective_resolution)
        self.micro_binning_combo.currentTextChanged.connect(self._update_effective_resolution)
        self.add_window_button.clicked.connect(self._add_window)
        self.add_detector_button.clicked.connect(self._add_detector)
        self.edit_json_button.clicked.connect(self._edit_json)
        self.save_button.clicked.connect(self._on_save)
        try:
            self.toolButton_calc_g_factor.setVisible(False)
        except Exception:
            pass
        
        # Set help text
        self.help_text.setText(help_text)
        self.help_text.setVisible(False)
        
        # Set visibility based on parameters
        # Use the helper method to hide/show widgets in layouts
        self._hide_layout_widgets(self.setup_layout, self.show_setup_selection)
        self._hide_layout_widgets(self.tttr_layout, self.show_tttr_reading)
        self._hide_layout_widgets(self.gridLayout_3, self.show_tables)
        self._hide_layout_widgets(self.gridLayout_2, self.show_tables)
        self._hide_layout_widgets(self.controls, self.show_add_inputs)
        
        # For widgets that are directly accessible, we can use setVisible directly
        if not self.show_help:
            self.help_text.setVisible(False)
        self.edit_json_button.setVisible(self.show_edit_json)
        self.save_button.setVisible(self.show_save)

        # Optical Setup button — opens the Light Path easy mode dialog
        self.optical_setup_button = QPushButton("Optical Setup…")
        self.optical_setup_button.setToolTip(
            "Open Light Path easy mode to configure filters, "
            "dyes, and compute Förster radii / cross-talk"
        )
        self.optical_setup_button.clicked.connect(self._on_optical_setup)
        try:
            self.controls.addWidget(self.optical_setup_button)
        except Exception:
            pass
        # Initialize file type combo
        self.file_type_combo.addItem("Auto")
        self.file_type_combo.addItems(list(tttrlib.TTTR.get_supported_container_names()))
        
        # Initialize micro binning combo
        self.micro_binning_combo.addItems(["1", "2", "4", "8", "16", "32", "64", "128"])
        
        # Set initial values for TTTR reading
        self.macro_time_le.setText(str(_initial_tttr_reading["macro_time_resolution"]))
        self.micro_time_le.setText(str(_initial_tttr_reading["micro_time_resolution"]))
        self.micro_binning_combo.setCurrentText(str(_initial_tttr_reading["micro_time_binning"]))

        # Plot toggle button in the TTTR section
        self.plot_toggle_button = QToolButton()
        self.plot_toggle_button.setText("📊 Plot")
        self.plot_toggle_button.setCheckable(True)
        self.plot_toggle_button.setToolTip("Show/hide micro-time decay preview plot")
        self.plot_toggle_button.toggled.connect(self._toggle_plot_visibility)
        self.tttr_layout.addWidget(self.plot_toggle_button, 4, 0, 1, 4)
        
        # Set table headers
        self.windows_form.setColumnCount(4)
        self.windows_form.setHorizontalHeaderLabels(["Window Name", "Start", "End", ""])
        try:
            self.detectors_form.setColumnCount(9)
        except Exception:
            pass
        self.detectors_form.setHorizontalHeaderLabels(["Detector Name", "Channels", "Micro Time Ranges", "G-Factor", "l1", "l2", "G-Factor Channels", "", ""])

        # Improve table space usage: adaptive column widths and stretch
        try:
            from qtpy.QtWidgets import QHeaderView, QSizePolicy
            # Windows table: fixed vertical size, horizontal expanding
            self.windows_form.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.windows_form.setWordWrap(False)
            self.windows_form.horizontalHeader().setHighlightSections(False)
            self.windows_form.horizontalHeader().setMinimumSectionSize(60)
            self.windows_form.verticalHeader().setVisible(False)
            self.windows_form.setAlternatingRowColors(False)

            # Detectors table: expanding both directions to fill available space
            self.detectors_form.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.detectors_form.setWordWrap(False)
            self.detectors_form.horizontalHeader().setHighlightSections(False)
            self.detectors_form.horizontalHeader().setMinimumSectionSize(60)
            self.detectors_form.verticalHeader().setVisible(False)
            self.detectors_form.setAlternatingRowColors(False)

            # Set maximum height for windows_form to 3 lines (approximately)
            # Use a reasonable estimate: 25px per row + header height + margin
            self.windows_form.setMaximumHeight(3 * 25 + 25 + 8)  # ~108px total

            # Windows table: Name stretches, Start/End resize to contents but user-resizable
            wh = self.windows_form.horizontalHeader()
            wh.setSectionResizeMode(0, QHeaderView.Stretch)
            wh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            wh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            wh.setSectionResizeMode(3, QHeaderView.ResizeToContents)

            # Detectors table: allocate space sensibly across 9 columns
            dh = self.detectors_form.horizontalHeader()
            # Name, Channels, Micro Time Ranges should stretch
            dh.setSectionResizeMode(0, QHeaderView.Stretch)  # Detector Name
            dh.setSectionResizeMode(1, QHeaderView.Stretch)  # Channels
            dh.setSectionResizeMode(2, QHeaderView.Stretch)  # Micro Time Ranges
            # Numeric fields: size to contents but allow user to expand
            for col in (3, 4, 5):  # G-Factor, l1, l2
                dh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            # G-Factor Channels: stretch (often a short range but can use leftover)
            dh.setSectionResizeMode(6, QHeaderView.Stretch)
            dh.setSectionResizeMode(7, QHeaderView.ResizeToContents)
            dh.setSectionResizeMode(8, QHeaderView.ResizeToContents)

            # Enable interactive resizing by the user
            for col in range(0, 9):
                # Start with Interactive so user can drag; the above modes define initial behavior
                dh.setSectionResizeMode(col, dh.sectionResizeMode(col))
            dh.setCascadingSectionResizes(True)
        except Exception:
            pass

        # Micro-time preview: a decay histogram of a loaded dataset with the PIE
        # windows and detector micro-time ranges overlaid, so users can see that
        # their ranges actually cover the data (raw micro-time channel units).
        # MUST be initialized BEFORE _load_data so that _load_data can restore
        # persisted decays from the saved setup (otherwise they get overwritten).
        self._microtime_counts = None
        self._microtime_decay_file_path = None
        self._microtime_per_channel_counts = {}
        self._microtime_region_items = []
        try:
            self._setup_microtime_preview()
        except Exception:  # pragma: no cover - preview must never break the wizard
            pass

        # Load available setups
        self._load_available_setups()

        # Load initial or file
        if json_file:
            with open(json_file, "r") as f:
                data = json.load(f)
            self._load_data(data)
            if isinstance(data, dict):
                self.public_checkbox.setChecked(bool(data.get("_is_public", False)))
                setup_owner = data.get("_owner")
                if setup_owner is None:
                    self.public_checkbox.setEnabled(True)
        else:
            # If no file specified, try to load the last used setup or use defaults
            setups = load_detector_setups(self.current_setups_file)
            if setups.get("last_used") and setups["last_used"] in setups["setups"]:
                self.current_setup_name = setups["last_used"]
                self.setup_combo.setCurrentText(self.current_setup_name)
                data = setups["setups"][self.current_setup_name]
            else:
                data = {
                    "windows": _initial_windows, 
                    "detectors": _initial_detectors,
                    "tttr_reading": _initial_tttr_reading
                }
            self._load_data(data)
            
        # Initialize the effective micro time resolution
        self._update_effective_resolution()

        # Initialize finish state: disable Finish until user explicitly saves
        self._allow_finish = allow_finish

    def isComplete(self):
        """Only allow finishing the wizard after the user saved settings."""
        # QWizard queries this to enable/disable the Finish button
        return bool(getattr(self, "_allow_finish", False))

    # ------------------------------------------------------------------
    # Micro-time preview (data decay with PIE windows + detector ranges)
    # ------------------------------------------------------------------
    #: Distinct colors for detectors keyed by common names; others cycle.
    _DETECTOR_COLORS = {
        "green": (0, 200, 0),
        "red": (220, 40, 40),
        "yellow": (220, 200, 0),
        "blue": (60, 120, 230),
    }
    _DETECTOR_CYCLE = [
        (0, 200, 0), (220, 40, 40), (220, 200, 0), (60, 120, 230),
        (200, 120, 0), (160, 60, 200), (0, 180, 180),
    ]

    def _setup_microtime_preview(self):
        """Create the micro-time decay preview in a separate window."""
        if pg is None:
            self._microtime_plot = None
            self._microtime_plot_window = None
            return
        self._microtime_plot_window = QDialog(self, Qt.Window)
        self._microtime_plot_window.setWindowTitle("Micro-time Decay Preview")
        self._microtime_plot_window.resize(700, 450)
        vbox = QVBoxLayout(self._microtime_plot_window)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)

        self._microtime_plot = pg.PlotWidget(self._microtime_plot_window)
        self._microtime_plot.setLabel("bottom", "Micro-time channel")
        self._microtime_plot.setLabel("left", "Counts")
        self._microtime_plot.setMinimumHeight(300)
        self._microtime_plot.getPlotItem().setLogMode(False, True)
        self._microtime_plot.addLegend(offset=(-10, 10))
        vbox.addWidget(self._microtime_plot)

        # Sync the toggle button when the window is closed by the user.
        self._microtime_plot_window.finished.connect(
            lambda: self._sync_plot_btn()
        )

        # Refresh overlays whenever the detector table changes.
        try:
            self.detectorsChanged.connect(self._refresh_microtime_preview)
        except Exception:
            pass

    def _sync_plot_btn(self):
        btn = getattr(self, "plot_toggle_button", None)
        if btn is not None:
            win = getattr(self, "_microtime_plot_window", None)
            btn.setChecked(win is not None and win.isVisible())

    def _toggle_plot_visibility(self, visible):
        win = getattr(self, "_microtime_plot_window", None)
        if win is None:
            return
        if visible:
            win.show()
            win.raise_()
        else:
            win.hide()

    def set_microtime_data(self, counts, file_path=None):
        """Set the micro-time histogram (counts per raw channel) to preview.

        Parameters
        ----------
        counts : array-like or None
            Histogram counts per raw micro-time channel.
        file_path : str or None
            Path of the TTTR file from which the histogram was derived.
            Stored so it can be persisted with the setup.
        """
        try:
            arr = np.asarray(counts, dtype=float).ravel()
            self._microtime_counts = arr if arr.size else None
        except Exception:
            self._microtime_counts = None
        self._microtime_decay_file_path = file_path
        self._refresh_microtime_preview()

    def set_microtime_per_channel_data(self, per_channel_counts, file_path=None):
        """Store per-routing-channel microtime histograms.

        Parameters
        ----------
        per_channel_counts : dict of int -> array-like
            Routing channel number mapped to its microtime histogram.
        file_path : str or None
            Source TTTR file path.
        """
        if per_channel_counts:
            self._microtime_per_channel_counts = {
                k: np.asarray(v, dtype=float).ravel()
                for k, v in per_channel_counts.items()
            }
        else:
            self._microtime_per_channel_counts = {}
        if file_path is not None:
            self._microtime_decay_file_path = file_path
        self._refresh_microtime_preview()

    def _detector_color(self, name, index):
        return self._DETECTOR_COLORS.get(
            str(name).strip().lower(),
            self._DETECTOR_CYCLE[index % len(self._DETECTOR_CYCLE)],
        )

    def _refresh_microtime_preview(self):
        """Redraw the decay curve and the window/detector range overlays."""
        plot = getattr(self, "_microtime_plot", None)
        if plot is None:
            return
        plot_item = plot.getPlotItem()
        # Clear previous region overlays (keep nothing stale).
        for item in getattr(self, "_microtime_region_items", []):
            try:
                plot_item.removeItem(item)
            except Exception:
                pass
        self._microtime_region_items = []
        plot_item.clear()

        counts = self._microtime_counts
        if counts is not None and counts.size:
            x = np.arange(counts.size + 1, dtype=float)
            y = np.clip(counts, 0, None)
            plot_item.plot(
                x, y, stepMode=True, fillLevel=0,
                brush=(120, 120, 120, 80), pen=pg.mkPen((180, 180, 180), width=1),
                name="data",
            )
            x_max = float(counts.size)

            # Overlay per-detector combined traces when detectors are defined,
            # otherwise fall back to individual routing channel traces.
            try:
                settings = self.get_settings()
            except Exception:
                settings = {"windows": {}, "detectors": {}}

            per_ch = getattr(self, "_microtime_per_channel_counts", {})
            detectors = (settings.get("detectors", {}) or {})
            if detectors and per_ch:
                for idx, (name, info) in enumerate(detectors.items()):
                    chs = (info or {}).get("chs", [])
                    color = self._detector_color(name, idx)
                    combined = None
                    for ch in chs:
                        ch_counts = per_ch.get(int(ch))
                        if ch_counts is not None and ch_counts.size:
                            if combined is None:
                                combined = ch_counts.copy()
                            else:
                                combined += ch_counts
                    if combined is not None:
                        plot_item.plot(
                            x, np.clip(combined, 0, None),
                            stepMode=True, fillLevel=0,
                            brush=(*color, 50), pen=pg.mkPen(color, width=1.5),
                            name=name,
                        )
            elif per_ch:
                ch_colors = [
                    (200, 50, 50), (50, 150, 50), (50, 80, 200),
                    (200, 150, 50), (150, 50, 150), (50, 180, 180),
                    (200, 100, 50), (100, 100, 100),
                ]
                for idx, (ch, ch_counts) in enumerate(sorted(per_ch.items())):
                    if ch_counts is not None and ch_counts.size:
                        color = ch_colors[idx % len(ch_colors)]
                        plot_item.plot(
                            x, np.clip(ch_counts, 0, None),
                            stepMode=True, fillLevel=0,
                            brush=(*color, 40), pen=pg.mkPen(color, width=1),
                            name=f"ch {ch}",
                        )
        else:
            x_max = None
            settings = {"windows": {}, "detectors": {}}

        # PIE windows: movable bands that update the table on drag.
        for name, (start, end) in (settings.get("windows", {}) or {}).items():
            def _make_window_cb(wname=name):
                def _cb():
                    region = self.sender()
                    r0, r1 = region.getRegion()
                    s, e = int(round(r0)), int(round(r1))
                    for r in range(self.windows_form.rowCount()):
                        if self.windows_form.item(r, 0).text().strip() == wname:
                            self.windows_form.cellWidget(r, 1).setText(str(s))
                            self.windows_form.cellWidget(r, 2).setText(str(e))
                            break
                return _cb
            self._add_preview_region(
                start, end, (150, 150, 150), f"PIE: {name}",
                alpha=40, on_changed=_make_window_cb(),
            )

        # Detector micro-time ranges: movable, color-coded per detector.
        det_items = {}  # name -> list of (region, range_index)
        for idx, (name, info) in enumerate((settings.get("detectors", {}) or {}).items()):
            color = self._detector_color(name, idx)
            ranges = (info or {}).get("micro_time_ranges", []) or []
            det_items[name] = det_regions = []
            for ri, (start, end) in enumerate(ranges):
                def _make_det_cb(dname=name):
                    def _cb():
                        for row in range(self.detectors_form.rowCount()):
                            if self.detectors_form.item(row, 0).text().strip() == dname:
                                parts = []
                                for reg, _ in det_items.get(dname, []):
                                    rr0, rr1 = reg.getRegion()
                                    parts.append(f"{int(round(rr0))}:{int(round(rr1))}")
                                self.detectors_form.cellWidget(row, 2).setText(", ".join(parts))
                                break
                    return _cb
                region = self._add_preview_region(
                    start, end, color, str(name),
                    alpha=70, on_changed=_make_det_cb(),
                )
                if region is not None:
                    det_regions.append((region, ri))

        if x_max is not None:
            plot_item.setXRange(0, x_max, padding=0.02)

    def _add_preview_region(self, start, end, color, label, alpha=60, on_changed=None):
        plot = getattr(self, "_microtime_plot", None)
        if plot is None:
            return
        try:
            r0, r1 = float(start), float(end)
        except (TypeError, ValueError):
            return
        if r1 < r0:
            r0, r1 = r1, r0
        brush = pg.mkBrush(color[0], color[1], color[2], alpha)
        movable = on_changed is not None
        region = pg.LinearRegionItem(
            values=(r0, r1), movable=movable, brush=brush,
            pen=pg.mkPen(color[0], color[1], color[2], width=1),
        )
        if on_changed is not None:
            region.sigRegionChangeFinished.connect(on_changed)
        region.setZValue(-10)
        plot.addItem(region)
        self._microtime_region_items.append(region)
        text = pg.TextItem(label, color=color, anchor=(0, 1))
        text.setPos(r0, 0)
        plot.addItem(text)
        self._microtime_region_items.append(text)
        return region

    # The _with_label method is no longer needed as the UI file already includes labels for widgets
    # This method is kept for backward compatibility but is not used in the new implementation
    def _with_label(self, text, widget):
        """Helper to wrap a widget with a label above (legacy method, not used with UI file)."""
        v = QVBoxLayout()
        v.addWidget(QLabel(text))
        v.addWidget(widget)
        w = QWidget()
        w.setLayout(v)
        return w
        
    def _hide_layout_widgets(self, layout, visible):
        """Helper method to hide/show all widgets in a layout.
        
        This method is used instead of trying to access widget containers directly,
        which can cause AttributeError if the widget names don't match between the
        code and the UI file. It iterates through all widgets in the given layout
        and sets their visibility based on the provided flag.
        
        Args:
            layout: The layout containing widgets to hide/show
            visible: Boolean indicating whether widgets should be visible
        """
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)

    def _toggle_help(self):
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("Help — Detector Setup")
        dlg.resize(600, 400)
        layout = QVBoxLayout(dlg)
        text = QTextEdit(dlg)
        text.setReadOnly(True)
        text.setHtml(self.help_text.toHtml())
        layout.addWidget(text)
        btn = QPushButton("Close", dlg)
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec_()

    def _on_load_setups_file(self):
        """Open a file dialog to select a different detector setups file."""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Detector Setups File", 
            "", 
            "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            # Load setups from the selected file
            setups = load_detector_setups(path)

            # Update the UI to display the new file path
            self.setups_file_le.setText(path)

            # Store the current file path as an instance variable
            self.current_setups_file = path

            # Update the setup combo box with the setups from the new file
            self.setup_combo.blockSignals(True)
            self.setup_combo.clear()

            # Add a blank item for "custom" setup
            self.setup_combo.addItem("")

            # Add setups from the loaded file
            for setup_name in setups.get("setups", {}).keys():
                self.setup_combo.addItem(setup_name)

            # If there's a last used setup, select it
            if setups.get("last_used") and setups["last_used"] in setups["setups"]:
                self.current_setup_name = setups["last_used"]
                index = self.setup_combo.findText(self.current_setup_name)
                if index >= 0:
                    self.setup_combo.setCurrentIndex(index)

                    # Load the selected setup
                    data = setups["setups"][self.current_setup_name]
                    self._load_data(data)

            self.setup_combo.blockSignals(False)

            QMessageBox.information(
                self, 
                "Success", 
                f"Loaded detector setups from {path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to load detector setups file: {e}"
            )


    def _update_effective_resolution(self):
        """
        Calculate and update the effective micro time resolution based on the current
        micro time resolution and binning factor.
        """
        try:
            micro_time_res = float(self.micro_time_le.text())
            binning = int(self.micro_binning_combo.currentText())
            effective_res = micro_time_res * binning
            self.effective_micro_time_le.setText(f"{effective_res:.6f}")
        except (ValueError, TypeError):
            # Handle case where inputs are not valid numbers
            self.effective_micro_time_le.setText("N/A")

    def _load_data(self, data):
        # block updates/signals
        self.windows_form.setUpdatesEnabled(False)
        self.detectors_form.setUpdatesEnabled(False)
        self.windows_form.blockSignals(True)
        self.detectors_form.blockSignals(True)

        # reset g-factor protection state for fresh rows
        self._g_user_editing.clear()
        self._g_last_valid.clear()

        # clear
        self.windows_form.setRowCount(0)
        self.detectors_form.setRowCount(0)

        # During programmatic population, allow g-factor text changes
        prev_allow = self._allow_g_update
        self._allow_g_update = True
        try:
            # populate windows
            for name, (start, end) in data.get("windows", {}).items():
                self._add_window_row(name, str(start), str(end))

            # populate detectors
            for name, props in data.get("detectors", {}).items():
                chs = ", ".join(map(str, props["chs"]))
                mtr = self._format_microtime_ranges(props.get("micro_time_ranges", []))
                g_factor = str(props.get("g_factor", 1.00))
                l1 = str(props.get("l1", 0.00))
                l2 = str(props.get("l2", 0.00))
                # New: g_factor_channels supports [start, end] or "start-end"; anything else -> empty
                gfch = props.get("g_factor_channels")
                if isinstance(gfch, (list, tuple)) and len(gfch) == 2:
                    gf_channels_text = f"{int(gfch[0])}-{int(gfch[1])}"
                elif isinstance(gfch, str):
                    gf_channels_text = gfch
                else:
                    gf_channels_text = ""
                g_factor_decay_uuid = props.get("g_factor_decay_uuid", "")
                g_factor_calibration_id = props.get("g_factor_calibration_id", "")
                self._add_detector_row(name, chs, mtr, g_factor, l1, l2, gf_channels_text, g_factor_decay_uuid, g_factor_calibration_id)
        finally:
            self._allow_g_update = prev_allow

        # populate TTTR reading routine settings
        tttr_reading = data.get("tttr_reading", _initial_tttr_reading)
        self.file_type_combo.setCurrentText(tttr_reading.get("file_type", "SPC-130"))
        self.macro_time_le.setText(str(tttr_reading.get("macro_time_resolution", 50.0)))
        self.micro_time_le.setText(str(tttr_reading.get("micro_time_resolution", 50.0)))
        self.micro_binning_combo.setCurrentText(str(tttr_reading.get("micro_time_binning", 1)))

        # Update the effective resolution
        self._update_effective_resolution()

        # Restore the microtime decay histogram if it was saved with the setup.
        decay = data.get("_microtime_decay")
        if isinstance(decay, dict) and "counts" in decay:
            try:
                counts = np.asarray(decay["counts"], dtype=float).ravel()
                file_path = decay.get("file_path")
                self._microtime_counts = counts if counts.size else None
                self._microtime_decay_file_path = file_path
            except Exception:
                self._microtime_counts = None
                self._microtime_decay_file_path = None
        else:
            self._microtime_counts = None
            self._microtime_decay_file_path = None

        # Restore per-routing-channel microtime histograms
        per_ch_decay = data.get("_microtime_per_channel_decay")
        if isinstance(per_ch_decay, dict) and "channels" in per_ch_decay:
            try:
                per_ch = {}
                for ch_str, ch_counts in per_ch_decay["channels"].items():
                    arr = np.asarray(ch_counts, dtype=float).ravel()
                    if arr.size:
                        per_ch[int(ch_str)] = arr
                self._microtime_per_channel_counts = per_ch
            except Exception:
                self._microtime_per_channel_counts = {}
        else:
            self._microtime_per_channel_counts = {}

        self._refresh_microtime_preview()

        # Restore optical config (from easy mode dialog)
        self._optical_config = data.get("optical_config")

        # re-enable
        self.windows_form.blockSignals(False)
        self.detectors_form.blockSignals(False)
        self.windows_form.setUpdatesEnabled(True)
        self.detectors_form.setUpdatesEnabled(True)
        self.detectorsChanged.emit()

    def _add_window_row(self, name, start, end):
        row = self.windows_form.rowCount()
        self.windows_form.insertRow(row)
        self.windows_form.setItem(row, 0, QTableWidgetItem(name))
        start_le = QLineEdit(start)
        end_le = QLineEdit(end)
        self.windows_form.setCellWidget(row, 1, start_le)
        self.windows_form.setCellWidget(row, 2, end_le)
        for _le in (start_le, end_le):
            _le.editingFinished.connect(self._refresh_microtime_preview)

        btn = QPushButton("🗑️")
        btn.setMaximumWidth(30)
        btn.setToolTip("Delete window")
        btn.clicked.connect(lambda _, b=btn: self._remove_window_by_button(b))
        self.windows_form.setCellWidget(row, 3, btn)

    def _remove_window_by_button(self, button):
        for r in range(self.windows_form.rowCount()):
            if self.windows_form.cellWidget(r, 3) == button:
                self.windows_form.removeRow(r)
                self.detectorsChanged.emit()
                break

    def _add_detector_row(self, name, ch_text, mtr_text, g_factor="1.00", l1="0.00", l2="0.00", gf_channels_text: str = "", g_factor_decay_uuid: str = "", g_factor_calibration_id: str = ""):
        row = self.detectors_form.rowCount()
        self.detectors_form.insertRow(row)
        item = QTableWidgetItem(name)
        if g_factor_decay_uuid:
            item.setData(Qt.UserRole + 1, g_factor_decay_uuid)
        if g_factor_calibration_id:
            item.setData(Qt.UserRole + 2, g_factor_calibration_id)
        self.detectors_form.setItem(row, 0, item)
        self.detectors_form.setCellWidget(row, 1, QLineEdit(ch_text))
        mtr_le = QLineEdit(mtr_text)
        self.detectors_form.setCellWidget(row, 2, mtr_le)
        mtr_le.editingFinished.connect(self._refresh_microtime_preview)
        g_le = QLineEdit(g_factor)
        self.detectors_form.setCellWidget(row, 3, g_le)
        self._wire_g_factor_cell(row, g_le)
        self.detectors_form.setCellWidget(row, 4, QLineEdit(l1))
        self.detectors_form.setCellWidget(row, 5, QLineEdit(l2))
        try:
            self.detectors_form.setCellWidget(row, 6, QLineEdit(gf_channels_text))
        except Exception:
            pass

        # G-factor calculator button
        gf_btn = QPushButton("🧮")
        gf_btn.setMaximumWidth(30)
        gf_btn.setToolTip("Calculate G-Factor for this detector")
        gf_btn.clicked.connect(lambda _, b=gf_btn: self._on_calc_g_factor_for_row_button(b))
        try:
            self.detectors_form.setCellWidget(row, 7, gf_btn)
        except Exception:
            pass

        # Delete button
        btn = QPushButton("🗑️")
        btn.setMaximumWidth(30)
        btn.setToolTip("Delete detector")
        btn.clicked.connect(lambda _, b=btn: self._remove_detector_by_button(b))
        try:
            self.detectors_form.setCellWidget(row, 8, btn)
        except Exception:
            pass

    def _remove_detector_by_button(self, button):
        for r in range(self.detectors_form.rowCount()):
            if self.detectors_form.cellWidget(r, 8) == button:
                self.detectors_form.removeRow(r)
                self.detectorsChanged.emit()
                break

    def _on_calc_g_factor_for_row_button(self, button):
        for r in range(self.detectors_form.rowCount()):
            if self.detectors_form.cellWidget(r, 7) == button:
                self._on_calc_g_factor_for_row(r)
                break

    def _add_window(self):
        base_name = "New Window"
        name = base_name
        counter = 1
        while any(self.windows_form.item(r,0).text() == name for r in range(self.windows_form.rowCount())):
            name = f"{base_name} {counter}"
            counter += 1
        self._add_window_row(name, "0", "2048")
        self.detectorsChanged.emit()

    def _add_detector(self):
        base_name = "New Detector"
        name = base_name
        counter = 1
        while any(self.detectors_form.item(r,0).text() == name for r in range(self.detectors_form.rowCount())):
            name = f"{base_name} {counter}"
            counter += 1
        self._add_detector_row(name, "0, 1", "0:2048", gf_channels_text="")
        self.detectorsChanged.emit()

    def _edit_json(self):
        data = self.get_settings()
        dlg = JsonEditorDialog(data, self)
        if dlg.exec_():
            edited = dlg.get_edited_data()
            if edited:
                self._load_data(edited)

    def _parse_microtime_ranges_text(self, text: str):
        if not isinstance(text, str):
            return []
        txt = text.strip()
        if not txt:
            return []
        segs = []
        for item in txt.replace(";", ",").split(","):
            item = item.strip()
            if item:
                segs.append(item)
        ranges = []
        for seg in segs:
            if ":" in seg:
                a_txt, b_txt = seg.split(":", 1)
            else:
                pos = seg.rfind("-")
                if pos <= 0:
                    a_txt = seg
                    b_txt = seg
                else:
                    a_txt = seg[:pos]
                    b_txt = seg[pos + 1 :]
            try:
                a = int(a_txt.strip())
                b = int(b_txt.strip())
            except Exception:
                continue
            if a <= b:
                ranges.append((a, b))
            else:
                ranges.append((b, a))
        return ranges

    def _format_microtime_ranges(self, ranges):
        r = []
        try:
            for s, e in ranges or []:
                r.append(f"{int(s)}:{int(e)}")
        except Exception:
            pass
        return ", ".join(r)

    def get_settings(self):
        # windows
        wins = {}
        for r in range(self.windows_form.rowCount()):
            name = self.windows_form.item(r,0).text().strip()
            start = int(self.windows_form.cellWidget(r,1).text())
            end   = int(self.windows_form.cellWidget(r,2).text())
            wins[name] = (start, end)

        # detectors
        dets = {}
        for r in range(self.detectors_form.rowCount()):
            name = self.detectors_form.item(r,0).text().strip()
            ch_text = self.detectors_form.cellWidget(r,1).text()
            mtr_text = self.detectors_form.cellWidget(r,2).text()
            chs = []
            try:
                parts = [p for p in str(ch_text).replace(";", ",").split(",")]
                for p in parts:
                    p = p.strip()
                    if p:
                        chs.append(int(p))
            except Exception:
                chs = []
            mtr = self._parse_microtime_ranges_text(mtr_text)
            
            # Get the G-factor cell widget and its text
            g_factor_widget = self.detectors_form.cellWidget(r,3)
            g_factor_text = g_factor_widget.text() if g_factor_widget else "1.00"
            
            # Convert to float with fallback to default value
            try:
                g_factor = float(g_factor_text)
            except ValueError:
                g_factor = 1.00
                
            l1 = float(self.detectors_form.cellWidget(r,4).text())
            l2 = float(self.detectors_form.cellWidget(r,5).text())

            # Optional: G-Factor Channels from column 6 as "start-end"
            gf_channels = None
            try:
                gf_widget = self.detectors_form.cellWidget(r,6)
                if gf_widget:
                    txt = gf_widget.text().strip()
                    if txt:
                        parts = txt.replace(' ', '').split('-')
                        if len(parts) == 2:
                            gf_start = int(parts[0])
                            gf_end = int(parts[1])
                            gf_channels = [gf_start, gf_end]
            except Exception:
                gf_channels = None

            item = self.detectors_form.item(r, 0)
            g_factor_decay_uuid = item.data(Qt.UserRole + 1) if item else None
            g_factor_calibration_id = item.data(Qt.UserRole + 2) if item else None

            det_entry = {
                "chs": chs, 
                "micro_time_ranges": mtr,
                "g_factor": g_factor,
                "l1": l1,
                "l2": l2
            }
            if gf_channels is not None:
                det_entry["g_factor_channels"] = gf_channels
            if g_factor_decay_uuid:
                det_entry["g_factor_decay_uuid"] = g_factor_decay_uuid
            if g_factor_calibration_id:
                det_entry["g_factor_calibration_id"] = g_factor_calibration_id
            dets[name] = det_entry

        # TTTR reading routine
        tttr_reading = {
            "file_type": self.file_type_combo.currentText(),
            "macro_time_resolution": float(self.macro_time_le.text()),
            "micro_time_resolution": float(self.micro_time_le.text()),
            "micro_time_binning": int(self.micro_binning_combo.currentText()),
            "effective_micro_time_resolution": self.effective_micro_time_resolution,
            "excitation_period": self.excitation_period
        }

        # Return the result
        result = {"windows": wins, "detectors": dets, "tttr_reading": tttr_reading}

        # Include optical config from the easy mode dialog (if set)
        if self._optical_config is not None:
            result["optical_config"] = self._optical_config

        # Persist the microtime decay histogram (if loaded) so it survives restarts.
        counts = self._microtime_counts
        fpath = self._microtime_decay_file_path
        if counts is not None and fpath is not None:
            result["_microtime_decay"] = {
                "file_path": str(fpath),
                "counts": counts.tolist(),
            }

        # Persist per-routing-channel microtime histograms
        per_ch = getattr(self, "_microtime_per_channel_counts", {})
        if per_ch and fpath is not None:
            result["_microtime_per_channel_decay"] = {
                "file_path": str(fpath),
                "channels": {str(k): v.tolist() for k, v in per_ch.items()},
            }

        return result

    def channels(self):
        chs = {}
        settings = self.get_settings()
        for wname, wrange in settings["windows"].items():
            for dname, dinfo in settings["detectors"].items():
                cname = f"{wname}_{dname}"
                chs[cname] = []
                for mtr in dinfo["micro_time_ranges"]:
                    chs[cname].append({
                        "window_range": wrange,
                        "detector_chs": dinfo["chs"],
                        "micro_time_range": mtr
                    })
        return chs

    def _on_save(self):
        data = self.get_settings()
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            # If we have a current setup, update it as well
            if self.current_setup_name:
                setups = load_detector_setups(self.current_setups_file)
                setups.setdefault("setups", {})
                
                # If the setup already exists, preserve any additional fields
                if self.current_setup_name in setups["setups"]:
                    existing_data = setups["setups"][self.current_setup_name]
                    # Update fields while preserving unknown nested data (e.g., per-detector mle_settings)
                    for key in data:
                        if key == 'detectors':
                            existing_data.setdefault('detectors', {})
                            # Merge per-detector entries
                            for det_name, det_info in data['detectors'].items():
                                if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                                    # Update known fields only, preserve anything else
                                    existing_data['detectors'][det_name].update(det_info)
                                else:
                                    existing_data['detectors'][det_name] = det_info
                            # Keep detectors present in existing_data but not in new data as-is
                        else:
                            existing_data[key] = data[key]
                    # Use the updated existing data
                    setups["setups"][self.current_setup_name] = existing_data
                else:
                    # New setup, just use the data as is
                    setups["setups"][self.current_setup_name] = data
                    
                save_detector_setups(setups, self.current_setups_file)

            QMessageBox.information(self, "Success", f"Settings saved to {path}")

            # Mark page as complete and notify wizard so Finish becomes enabled
            self._allow_finish = True
            try:
                self.completeChanged.emit()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")


    @property
    def detectors(self):
        """
        Legacy accessor for external code:
        returns the same dict you’re saving as JSON.
        """
        return self.get_settings()['detectors']
        
    @detectors.setter
    def detectors(self, new_detectors):
        """
        Setter for detectors property. Updates the detectors in the UI.
        
        Args:
            new_detectors (dict): Dictionary of detector configurations
        """
        # Get current settings
        current_settings = self.get_settings()
        
        # Update detectors in settings
        current_settings['detectors'] = new_detectors
        
        # Load updated settings into UI
        self._load_data(current_settings)

    @property
    def windows(self):
        """
        Legacy accessor for external code: returns the same dict
        you're saving as JSON under "windows".
        """
        return self.get_settings()['windows']
        
    @windows.setter
    def windows(self, new_windows):
        """
        Setter for windows property. Updates the windows in the UI.
        
        Args:
            new_windows (dict): Dictionary of window name -> (start, end) tuples
        """
        # Get current settings
        current_settings = self.get_settings()
        
        # Update windows in settings
        current_settings['windows'] = new_windows
        
        # Load updated settings into UI
        self._load_data(current_settings)

    @property
    def filetype(self) -> str | None:
        """
        Returns the selected file type, handling the "Auto" option by trying to infer
        the file type from a file if available.

        Returns:
            str | None: The file type name, or None if "Auto" is selected and no file is available
                        to infer the type from.
        """
        txt = self.file_type_combo.currentText()
        if txt == 'Auto':
            # In this context, we don't have a specific file to infer from
            # External code should handle this by using tttrlib's auto-detection
            return None
        return txt

    @property
    def effective_micro_time_resolution(self):
        """
        Calculate and return the effective micro time resolution based on the current
        micro time resolution and binning factor.

        Returns:
            float: The effective micro time resolution in picoseconds.
        """
        try:
            micro_time_res = float(self.micro_time_le.text())
            binning = int(self.micro_binning_combo.currentText())
            return micro_time_res * binning
        except (ValueError, TypeError):
            # Return default value if inputs are not valid numbers
            return 50.0 * int(self.micro_binning_combo.currentText())

    @property
    def tttr_reading(self):
        """
        Accessor for external code: returns the TTTR reading routine settings
        as a dict with file_type, macro_time_resolution, micro_time_resolution,
        and micro_time_binning.
        """
        return self.get_settings()['tttr_reading']
        
    @property
    def excitation_period(self):
        """
        Returns the excitation period in nanoseconds.
        
        Returns:
            float: The excitation period in nanoseconds.
        """
        return float(self.macro_time_le.text()) #self.excitation_period_spin.value()
        
    @property
    def selected_detector(self):
        """
        Get the currently selected detector information.
        
        Returns:
            dict: A dictionary containing information about the selected detector,
                  or None if no detector is selected.
        """
        return self._selected_detector_info
        
    @selected_detector.setter
    def selected_detector(self, info):
        """
        Set the currently selected detector information.
        
        Args:
            info (dict): A dictionary containing information about the selected detector.
        """
        self._selected_detector_info = info

    def _load_available_setups(self):
        """Load available setups into the combobox."""
        self.setup_combo.blockSignals(True)
        self.setup_combo.clear()

        # Add a blank item for "custom" setup
        self.setup_combo.addItem("")

        # Load setups from the current setups file
        setups = load_detector_setups(self.current_setups_file)
        for setup_name in setups.get("setups", {}).keys():
            self.setup_combo.addItem(setup_name)

        # If we have a current setup, select it
        if self.current_setup_name:
            index = self.setup_combo.findText(self.current_setup_name)
            if index >= 0:
                self.setup_combo.setCurrentIndex(index)
                # Update checkbox state from the loaded data
                data = setups.get("setups", {}).get(self.current_setup_name, {})
                if isinstance(data, dict):
                    setup_public = bool(data.get("_is_public", False))
                    setup_owner = data.get("_owner")
                    active_user = _resolve_active_user_id()
                    is_owner = (setup_owner is None) or (setup_owner == active_user)
                    self.public_checkbox.setChecked(setup_public)
                    self.public_checkbox.setEnabled(is_owner)

        self.setup_combo.blockSignals(False)

    def _on_setup_changed(self, index):
        """Handle setup selection changes."""
        if index <= 0:  # Empty or custom setup
            self.current_setup_name = None
            self.public_checkbox.setChecked(False)
            self.public_checkbox.setEnabled(False)
            self.calibration_combo.blockSignals(True)
            self.calibration_combo.clear()
            self.calibration_combo.addItem("Latest")
            self.calibration_combo.blockSignals(False)
            return

        setup_name = self.setup_combo.currentText()
        if not setup_name:
            return

        # Load the selected setup from the current setups file
        setups = load_detector_setups(self.current_setups_file)
        if setup_name in setups.get("setups", {}):
            self.current_setup_name = setup_name
            data = setups["setups"][setup_name]
            self._load_data(data)

            # Reflect visibility & ownership for the checkbox
            setup_public = bool(data.get("_is_public", False))
            setup_owner = data.get("_owner")
            active_user = _resolve_active_user_id()
            is_owner = (setup_owner is None) or (setup_owner == active_user)
            self.public_checkbox.setChecked(setup_public)
            self.public_checkbox.setEnabled(is_owner)
            if not is_owner:
                self.public_checkbox.setToolTip(
                    "Only the owner can change visibility for this setup."
                )
            else:
                self.public_checkbox.setToolTip(
                    "When checked, this setup is visible to all users."
                )

            # Update last used setup
            setups["last_used"] = setup_name
            save_detector_setups(setups, self.current_setups_file)

        # Populate calibration date combobox from MFDB
        self._populate_calibration_combo(setup_name)

    def _populate_calibration_combo(self, setup_name: str) -> None:
        """Populate the calibration date combobox from MFDB calibration history.

        Parameters
        ----------
        setup_name : str
            The setup name to look up calibration snapshots for.
        """
        try:
            from chisurf.core.mfdb.repository import MFDatabase
            from chisurf.core.mfdb.database_resolver import resolve_database_path
            from .tttr_setup_utils import setup_id_for_name

            setup_id = setup_id_for_name(setup_name, _resolve_active_user_id())
            self.calibration_combo.blockSignals(True)
            self.calibration_combo.clear()
            self.calibration_combo.addItem("Latest")
            with MFDatabase(resolve_database_path()) as db:
                dates = db.list_setup_calibration_dates(setup_id)
                for dt in dates:
                    self.calibration_combo.addItem(dt)
            self.calibration_combo.blockSignals(False)
        except Exception:
            self.calibration_combo.blockSignals(True)
            self.calibration_combo.clear()
            self.calibration_combo.addItem("Latest")
            self.calibration_combo.blockSignals(False)

    def _on_calibration_changed(self, index: int) -> None:
        """Handle calibration date selection changes.

        Loads the selected calibration snapshot values into the detector
        table's G-factor, l1, l2 fields.
        """
        if index < 0 or not self.current_setup_name:
            return

        selected = self.calibration_combo.currentText()
        if not selected or selected == "Latest":
            return

        try:
            from chisurf.core.mfdb.repository import MFDatabase
            from chisurf.core.mfdb.database_resolver import resolve_database_path
            from .tttr_setup_utils import setup_id_for_name

            setup_id = setup_id_for_name(self.current_setup_name, _resolve_active_user_id())
            with MFDatabase(resolve_database_path()) as db:
                snapshots = db.get_setup_calibration(
                    setup_id, calibrated_at=selected
                )

            cal_by_channel: dict[str, dict] = {}
            for snap in snapshots:
                ch_name = snap.get("channel_name", "")
                cal_by_channel[ch_name] = snap

            # Update the detector table rows
            prev_allow = self._allow_g_update
            self._allow_g_update = True
            try:
                for r in range(self.detectors_form.rowCount()):
                    item = self.detectors_form.item(r, 0)
                    if item is None:
                        continue
                    det_name = item.text().strip()
                    snap = cal_by_channel.get(det_name)
                    if snap is None:
                        continue
                    g_val = snap.get("g_factor")
                    l1_val = snap.get("l1")
                    l2_val = snap.get("l2")
                    g_widget = self.detectors_form.cellWidget(r, 3)
                    if g_widget and g_val is not None:
                        g_widget.setText(str(g_val))
                    l1_widget = self.detectors_form.cellWidget(r, 4)
                    if l1_widget and l1_val is not None:
                        l1_widget.setText(str(l1_val))
                    l2_widget = self.detectors_form.cellWidget(r, 5)
                    if l2_widget and l2_val is not None:
                        l2_widget.setText(str(l2_val))
            finally:
                self._allow_g_update = prev_allow

            self.detectorsChanged.emit()
        except Exception:
            pass

    def _on_save_setup(self):
        """Save the current settings as a setup."""
        # Get current settings
        data = self.get_settings()

        # Ask for a setup name
        setup_name, ok = QInputDialog.getText(
            self, "Save Setup", "Enter a name for this setup:",
            text=self.current_setup_name or ""
        )

        if not ok or not setup_name:
            return

        # Attach visibility flag from the checkbox
        data["_is_public"] = self.public_checkbox.isChecked()

        # Save to the current setups file
        setups = load_detector_setups(self.current_setups_file)
        setups.setdefault("setups", {})
        
        # If the setup already exists, preserve any additional fields that aren't in the current settings
        if setup_name in setups["setups"]:
            existing_data = setups["setups"][setup_name]
            # Update fields while preserving unknown nested data (e.g., per-detector mle_settings)
            for key in data:
                if key == 'detectors':
                    existing_data.setdefault('detectors', {})
                    # Merge per-detector entries
                    for det_name, det_info in data['detectors'].items():
                        if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                            # Update known fields only, preserve anything else
                            existing_data['detectors'][det_name].update(det_info)
                        else:
                            existing_data['detectors'][det_name] = det_info
                    # Keep detectors present in existing_data but not in new data as-is
                else:
                    existing_data[key] = data[key]
            # Use the updated existing data
            setups["setups"][setup_name] = existing_data
        else:
            # New setup, just use the data as is
            setups["setups"][setup_name] = data
            
        setups["last_used"] = setup_name

        if save_detector_setups(setups, self.current_setups_file):
            self.current_setup_name = setup_name
            QMessageBox.information(self, "Success", f"Setup '{setup_name}' saved successfully.")

            # Refresh the combobox and select the new setup
            self._load_available_setups()
            index = self.setup_combo.findText(setup_name)
            if index >= 0:
                self.setup_combo.setCurrentIndex(index)
        else:
            QMessageBox.critical(self, "Error", f"Failed to save setup '{setup_name}'.")

    def _on_optical_setup(self):
        """Open the Light Path easy mode dialog for optical configuration."""
        # Fetch probes synchronously from MFDB
        try:
            probes_result = get_probes_info(resolve_db_path())
            probes = probes_result.get("probes", [])
        except Exception as exc:
            QMessageBox.critical(
                self, "MFDB Error",
                f"Could not load probe catalogue:\n{exc}"
            )
            return

        # Get detector names from the current wizard table
        det_names = []
        for r in range(self.detectors_form.rowCount()):
            item = self.detectors_form.item(r, 0)
            if item is not None:
                name = item.text().strip()
                if name:
                    det_names.append(name)

        dlg = LightPathEasyDialog(
            probes,
            parent=self,
            detector_names=det_names if det_names else None,
            optical_config=self._optical_config,
            db_path=resolve_db_path(),
        )
        if dlg.exec_():
            self._optical_config = dlg.get_optical_config()

    def _on_delete_setup(self):
        """Delete the current setup."""
        setup_name = self.setup_combo.currentText()
        if not setup_name:
            QMessageBox.warning(self, "Warning", "No setup selected.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete the setup '{setup_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete from the current setups file
        setups = load_detector_setups(self.current_setups_file)
        if setup_name in setups.get("setups", {}):
            del setups["setups"][setup_name]
            if setups.get("last_used") == setup_name:
                setups["last_used"] = ""

            if save_detector_setups(setups, self.current_setups_file, replace=True):
                QMessageBox.information(self, "Success", f"Setup '{setup_name}' deleted successfully.")

                # Refresh the combobox
                self.current_setup_name = None
                self._load_available_setups()
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete setup '{setup_name}'.")

    def _on_rename_setup(self):
        """Rename the current setup."""
        old_name = self.setup_combo.currentText()
        if not old_name:
            QMessageBox.warning(self, "Warning", "No setup selected.")
            return

        # Ask for a new setup name
        new_name, ok = QInputDialog.getText(
            self, "Rename Setup", "Enter a new name for this setup:",
            text=old_name
        )

        if not ok or not new_name or new_name == old_name:
            return

        # Check if the new name already exists
        setups = load_detector_setups(self.current_setups_file)
        if new_name in setups.get("setups", {}):
            reply = QMessageBox.question(
                self, "Setup Exists", 
                f"A setup with the name '{new_name}' already exists. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

        # Rename the setup in the current setups file
        if old_name in setups.get("setups", {}):
            # Get the current setup data
            setup_data = setups["setups"][old_name]

            # Remove the old setup and add with the new name
            del setups["setups"][old_name]
            setups["setups"][new_name] = setup_data

            # Update last_used if it was the renamed setup
            if setups.get("last_used") == old_name:
                setups["last_used"] = new_name

            if save_detector_setups(setups, self.current_setups_file, replace=True):
                self.current_setup_name = new_name
                QMessageBox.information(self, "Success", f"Setup renamed from '{old_name}' to '{new_name}' successfully.")

                # Refresh the combobox and select the renamed setup
                self._load_available_setups()
                index = self.setup_combo.findText(new_name)
                if index >= 0:
                    self.setup_combo.setCurrentIndex(index)
            else:
                QMessageBox.critical(self, "Error", f"Failed to rename setup from '{old_name}' to '{new_name}'.")

    def _read_from_tttr_file(self):
        _read_from_tttr_file(self)

    def _on_calc_g_factor(self):
        _on_calc_g_factor(self)

    def _on_calc_g_factor_for_row(self, row: int):
        _on_calc_g_factor(self, row)
    
    def load_data_into_tables(self, data):
        """
        Legacy alias for external callers.
        """
        # reuse our internal loader
        self._load_data(data)


    # --- G-Factor protection helpers ---
    def _wire_g_factor_cell(self, row, line_edit: QLineEdit):
        """Protect a row's G-Factor QLineEdit so only user edits or internal allowed updates can change it."""
        # Initialize tracking for this row
        self._g_user_editing[row] = False
        self._g_last_valid[row] = line_edit.text()

        def on_text_edited(_):
            # Fired only by user typing
            self._g_user_editing[row] = True

        def on_editing_finished():
            try:
                txt = line_edit.text().strip()
                # Accept empty as default 1.0
                val = float(txt) if txt else 1.0
                # Normalize formatting
                new_txt = f"{val:.3f}"
                # Allow internal write for normalization
                prev = self._allow_g_update
                self._allow_g_update = True
                try:
                    if line_edit.text() != new_txt:
                        line_edit.setText(new_txt)
                finally:
                    self._allow_g_update = prev
                # Commit last valid
                self._g_last_valid[row] = new_txt
            except Exception:
                # Revert to last valid on invalid input
                prev = self._allow_g_update
                self._allow_g_update = True
                try:
                    line_edit.setText(self._g_last_valid.get(row, "1.000"))
                finally:
                    self._allow_g_update = prev
            finally:
                self._g_user_editing[row] = False

        def on_text_changed(_):
            # Reject programmatic changes unless explicitly allowed
            if self._allow_g_update:
                # Keep last_valid in sync during allowed writes
                self._g_last_valid[row] = line_edit.text()
                return
            if self._g_user_editing.get(row, False):
                # User typing: allow
                return
            # Unauthorised programmatic change: revert
            prev = self._allow_g_update
            self._allow_g_update = True
            try:
                line_edit.setText(self._g_last_valid.get(row, line_edit.text()))
            finally:
                self._allow_g_update = prev

        # Connect signals
        try:
            line_edit.textEdited.connect(on_text_edited)
        except Exception:
            pass
        line_edit.editingFinished.connect(on_editing_finished)
        line_edit.textChanged.connect(on_text_changed)

    def _set_g_factor_programmatically(self, row: int, value_text: str, g_factor_decay_uuid: str = None, g_factor_calibration_id: str = None, l1: str = None, l2: str = None):
        """Safely set a row's G-Factor, l1, and l2 from internal code (calculator/data load)."""
        le = self.detectors_form.cellWidget(row, 3)
        if not isinstance(le, QLineEdit):
            return
        prev = self._allow_g_update
        self._allow_g_update = True
        try:
            le.setText(value_text)
            self._g_last_valid[row] = value_text

            if l1 is not None:
                le_l1 = self.detectors_form.cellWidget(row, 4)
                if isinstance(le_l1, QLineEdit):
                    le_l1.setText(l1)
            if l2 is not None:
                le_l2 = self.detectors_form.cellWidget(row, 5)
                if isinstance(le_l2, QLineEdit):
                    le_l2.setText(l2)

            item = self.detectors_form.item(row, 0)
            if item:
                if g_factor_decay_uuid:
                    item.setData(Qt.UserRole + 1, g_factor_decay_uuid)
                if g_factor_calibration_id:
                    item.setData(Qt.UserRole + 2, g_factor_calibration_id)
        finally:
            self._allow_g_update = prev

class DetectorWizard(QWizard):
    def __init__(self, json_file=None, show_edit_json=True, show_save=True, 
                 show_setups_file=True, show_setup_selection=True, show_help=True,
                 show_tttr_reading=True, show_tables=True, show_add_inputs=True, **kwargs):
        """Initialize the DetectorWizard.

        Args:
            json_file (str, optional): Path to a JSON file to load. Defaults to None.
            show_edit_json (bool, optional): Whether to show the "Edit JSON" button. Defaults to True.
            show_save (bool, optional): Whether to show the "Save" button. Defaults to True.
            show_setups_file (bool, optional): Whether to show the setups file section. Defaults to True.
            show_setup_selection (bool, optional): Whether to show the setup selection section. Defaults to True.
            show_help (bool, optional): Whether to show the help button and text. Defaults to True.
            show_tttr_reading (bool, optional): Whether to show the TTTR reading routine section. Defaults to True.
            show_tables (bool, optional): Whether to show the PIE-Windows and Detectors tables. Defaults to True.
            show_add_inputs (bool, optional): Whether to show the controls for adding windows and detectors. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the DetectorWizardPage.
        """
        super().__init__()
        self.addPage(DetectorWizardPage(
            json_file=json_file,
            show_edit_json=show_edit_json,
            show_save=show_save,
            show_setups_file=show_setups_file,
            show_setup_selection=show_setup_selection,
            show_help=show_help,
            show_tttr_reading=show_tttr_reading,
            show_tables=show_tables,
            show_add_inputs=show_add_inputs,
            **kwargs
        ))
        self.setWindowTitle("Detector Configuration Wizard")


if __name__ == "__main__":
    json_arg = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    wiz = DetectorWizard(json_file=json_arg)
    wiz.show()
    sys.exit(app.exec_())
