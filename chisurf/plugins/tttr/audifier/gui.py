"""
TTTR Audifier GUI

Qt widget for TTTR to audio conversion with waterfall plot.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional
import time

import numpy as np
import tempfile

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox,
    QScrollArea, QFormLayout, QFileDialog, QMessageBox,
    QSplitter, QFrame, QProgressBar, QTabWidget, QColorDialog,
    QRadioButton, QButtonGroup
)
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor

try:
    from qtpy.QtMultimedia import QSound
except ImportError:
    QSound = None

import pyqtgraph as pg

from chisurf import logging

try:
    import tttrlib
except ImportError:
    tttrlib = None

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

from .core import (
    TTTRData, ChannelConfig, load_tttr_with_tttrlib,
    compute_microtime_waterfall, make_default_channel_cfg,
    tttr_to_wav, CHORD_TYPE_NAMES
)
from .waterfall_plot import WaterfallPlotWidget
from .sound_playback import SoundPlayer, create_tttr_audio

# Import lifetime analysis functions
try:
    from .lifetime_analysis import (
        compute_lifetime_waterfall,
        plot_lifetime_waterfall,
        plot_lifetime_waterfall_multichannel
    )
except ImportError:
    # Fallback if lifetime analysis not available
    compute_lifetime_waterfall = None
    plot_lifetime_waterfall = None
    plot_lifetime_waterfall_multichannel = None

# Import dynamic icon utilities
try:
    from .dynamic_icons import get_icon_manager, create_audifier_icon_from_state
    _icon_manager = get_icon_manager()
except ImportError:
    _icon_manager = None

def _format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS string"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


class ChannelControl(QWidget):
    """Widget to control one channel's parameters."""

    def __init__(self, channel: int, cfg: ChannelConfig, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.chk_enable = QCheckBox(f"Ch {channel}", self)
        self.chk_enable.setChecked(True)
        self.layout.addWidget(self.chk_enable)

        self.combo_chord = QComboBox(self)
        self.combo_chord.addItems(CHORD_TYPE_NAMES)
        if cfg.chord_type in CHORD_TYPE_NAMES:
            self.combo_chord.setCurrentText(cfg.chord_type)
        self.layout.addWidget(QLabel("Chord:", self))
        self.layout.addWidget(self.combo_chord)

        self.spin_pitch = QDoubleSpinBox(self)
        self.spin_pitch.setRange(-24, 24)
        self.spin_pitch.setValue(cfg.pitch_semitones)
        self.layout.addWidget(QLabel("Pitch:", self))
        self.layout.addWidget(self.spin_pitch)

        self.spin_gain = QDoubleSpinBox(self)
        self.spin_gain.setRange(0, 10)
        self.spin_gain.setValue(cfg.gain)
        self.layout.addWidget(QLabel("Gain:", self))
        self.layout.addWidget(self.spin_gain)

    def get_config(self) -> Optional[ChannelConfig]:
        if not self.chk_enable.isChecked():
            return None
        return ChannelConfig(
            note_hz=261.63,
            chord_type=self.combo_chord.currentText(),
            pitch_semitones=self.spin_pitch.value(),
            micro_min=0,
            micro_max=2**31 - 1,
            gain=self.spin_gain.value()
        )


class DetectorControl(QWidget):
    """Widget to control one detector's parameters."""

    colorChanged = Signal(QColor)

    def __init__(self, name: str, color: QColor, enabled: bool, parent=None):
        super().__init__(parent)
        self.name = name
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.chk = QCheckBox(name, self)
        self.chk.setChecked(enabled)
        self.layout.addWidget(self.chk)

        self.color_btn = QPushButton("Color", self)
        self.color_btn.setStyleSheet(f"background-color: {color.name()}; color: white; font-weight: bold;")
        self.color_btn.clicked.connect(self._select_color)
        self.layout.addWidget(self.color_btn)

        self._color = color

    def _select_color(self):
        color = QColorDialog.getColor(self._color, self)
        if color.isValid():
            self._color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()}; color: white; font-weight: bold;")
            self.colorChanged.emit(color)

    @property
    def enabled(self):
        return self.chk.isChecked()

    @property
    def color(self):
        return self._color


@persist_plugin_state("tttr_audifier")
class TTTRAudifierWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTTR Audifier")
        self.resize(800, 600)

        self.data: Optional[TTTRData] = None
        self.channels: List[int] = []
        self.channel_controls: Dict[int, ChannelControl] = {}
        self.waterfall_plot: Optional[WaterfallPlotWidget] = None
        
        # Sound playback system
        self.sound_player = SoundPlayer(self)
        
        self.detectors: List[Dict] = []
        self.detector_controls: Dict[str, 'DetectorControl'] = {}

        self._setup_ui()
        self._connect_signals()
        
        # Initialize icon state
        self._update_icon_state()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignTop)

        # File loading
        file_group = QGroupBox("Load TTTR File", self)
        file_group.setContentsMargins(0, 0, 0, 0)
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(0)
        self.btn_load = QPushButton("Load TTTR...", self)
        file_layout.addWidget(self.btn_load)
        self.lbl_file = QLabel("No file loaded", self)
        file_layout.addWidget(self.lbl_file, 1)
        layout.addWidget(file_group)

        # Tabs
        self.tabs = QTabWidget(self)
        self.tabs.setContentsMargins(0, 0, 0, 0)

        # Setup tab
        setup_tab = QWidget(self)
        setup_layout = QVBoxLayout(setup_tab)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(0)
        self.detector_page = DetectorWizardPage(show_help=False, show_setups_file=True, show_setup_selection=True, show_tttr_reading=True, show_tables=True, show_add_inputs=True)
        setup_layout.addWidget(self.detector_page)

        # Audio tab
        audio_tab = QWidget(self)
        audio_layout = QVBoxLayout(audio_tab)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.setSpacing(0)

        # Audio parameters
        param_group = QGroupBox("Audio Parameters", self)
        param_group.setContentsMargins(0, 0, 0, 0)
        param_layout = QFormLayout(param_group)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(0)
        self.spin_bin_width = QDoubleSpinBox(self)
        self.spin_bin_width.setRange(0.001, 1.0)
        self.spin_bin_width.setValue(0.02)
        self.spin_bin_width.setSuffix(" s")
        param_layout.addRow("Bin Width:", self.spin_bin_width)

        self.combo_env_mode = QComboBox(self)
        self.combo_env_mode.addItems(["linear", "sqrt", "log"])
        self.combo_env_mode.setCurrentText("log")
        param_layout.addRow("Envelope Mode:", self.combo_env_mode)

        self.spin_sample_rate = QSpinBox(self)
        self.spin_sample_rate.setRange(8000, 192000)
        self.spin_sample_rate.setValue(44100)
        param_layout.addRow("Sample Rate:", self.spin_sample_rate)

        self.spin_master_gain = QDoubleSpinBox(self)
        self.spin_master_gain.setRange(0, 2)
        self.spin_master_gain.setValue(0.8)
        param_layout.addRow("Master Gain:", self.spin_master_gain)

        self.spin_env_floor = QDoubleSpinBox(self)
        self.spin_env_floor.setRange(0.0, 0.99)
        self.spin_env_floor.setValue(0.0)
        self.spin_env_floor.setSingleStep(0.05)
        param_layout.addRow("Env Floor:", self.spin_env_floor)

        self.spin_env_scale = QDoubleSpinBox(self)
        self.spin_env_scale.setRange(0.1, 10.0)
        self.spin_env_scale.setValue(1.0)
        param_layout.addRow("Env Scale:", self.spin_env_scale)

        self.spin_attack_frames = QSpinBox(self)
        self.spin_attack_frames.setRange(0, 100)
        self.spin_attack_frames.setValue(2)
        param_layout.addRow("Attack:", self.spin_attack_frames)

        self.spin_release_frames = QSpinBox(self)
        self.spin_release_frames.setRange(0, 200)
        self.spin_release_frames.setValue(6)
        param_layout.addRow("Release:", self.spin_release_frames)
        audio_layout.addWidget(param_group)

        # Waterfall parameters
        wf_group = QGroupBox("Waterfall Parameters", self)
        wf_group.setContentsMargins(0, 0, 0, 0)
        wf_layout = QVBoxLayout(wf_group)
        wf_layout.setContentsMargins(0, 0, 0, 0)
        wf_layout.setSpacing(0)
        
        # Waterfall mode selection
        mode_group = QGroupBox("Display Mode", self)
        mode_group.setContentsMargins(0, 0, 0, 0)
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(0)
        
        self.waterfall_mode_group = QButtonGroup(self)
        self.radio_microtime = QRadioButton("Microtime Waterfall", self)
        self.radio_microtime.setChecked(True)
        self.radio_lifetime = QRadioButton("Lifetime Waterfall", self)
        
        self.waterfall_mode_group.addButton(self.radio_microtime, 0)
        self.waterfall_mode_group.addButton(self.radio_lifetime, 1)
        
        mode_layout.addWidget(self.radio_microtime)
        mode_layout.addWidget(self.radio_lifetime)
        wf_layout.addWidget(mode_group)
        
        # Microtime parameters (for microtime mode)
        self.microtime_params_group = QGroupBox("Microtime Parameters", self)
        self.microtime_params_group.setContentsMargins(0, 0, 0, 0)
        microtime_layout = QFormLayout(self.microtime_params_group)
        microtime_layout.setContentsMargins(0, 0, 0, 0)
        microtime_layout.setSpacing(0)
        
        self.spin_wf_bin_width = QDoubleSpinBox(self)
        self.spin_wf_bin_width.setRange(0.001, 1.0)
        self.spin_wf_bin_width.setValue(0.05)
        self.spin_wf_bin_width.setSuffix(" s")
        microtime_layout.addRow("Bin Width:", self.spin_wf_bin_width)

        self.spin_wf_micro_bins = QSpinBox(self)
        self.spin_wf_micro_bins.setRange(32, 1024)
        self.spin_wf_micro_bins.setValue(256)
        microtime_layout.addRow("Micro Bins:", self.spin_wf_micro_bins)

        self.chk_wf_log = QCheckBox("Log Scale", self)
        self.chk_wf_log.setChecked(True)
        microtime_layout.addRow(self.chk_wf_log)
        
        wf_layout.addWidget(self.microtime_params_group)
        
        # Lifetime parameters (for lifetime mode)
        self.lifetime_params_group = QGroupBox("Lifetime Parameters", self)
        self.lifetime_params_group.setContentsMargins(0, 0, 0, 0)
        self.lifetime_params_group.setEnabled(False)  # Disabled by default
        lifetime_layout = QFormLayout(self.lifetime_params_group)
        lifetime_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_layout.setSpacing(0)
        
        self.spin_lifetime_bin_width = QDoubleSpinBox(self)
        self.spin_lifetime_bin_width.setRange(0.001, 1.0)
        self.spin_lifetime_bin_width.setValue(0.05)
        self.spin_lifetime_bin_width.setSuffix(" s")
        lifetime_layout.addRow("Bin Width:", self.spin_lifetime_bin_width)
        
        self.spin_lifetime_tau_min = QDoubleSpinBox(self)
        self.spin_lifetime_tau_min.setRange(0.1, 100.0)
        self.spin_lifetime_tau_min.setValue(0.5)
        self.spin_lifetime_tau_min.setSuffix(" ns")
        self.spin_lifetime_tau_min.setDecimals(2)
        lifetime_layout.addRow("τ_min:", self.spin_lifetime_tau_min)
        
        self.spin_lifetime_tau_max = QDoubleSpinBox(self)
        self.spin_lifetime_tau_max.setRange(0.1, 100.0)
        self.spin_lifetime_tau_max.setValue(15.0)
        self.spin_lifetime_tau_max.setSuffix(" ns")
        self.spin_lifetime_tau_max.setDecimals(2)
        lifetime_layout.addRow("τ_max:", self.spin_lifetime_tau_max)
        
        self.spin_lifetime_n_tau = QSpinBox(self)
        self.spin_lifetime_n_tau.setRange(20, 500)
        self.spin_lifetime_n_tau.setValue(100)
        lifetime_layout.addRow("N τ points:", self.spin_lifetime_n_tau)
        
        self.spin_lifetime_reg = QDoubleSpinBox(self)
        self.spin_lifetime_reg.setRange(1e-6, 1.0)
        self.spin_lifetime_reg.setValue(1e-2)
        self.spin_lifetime_reg.setDecimals(4)
        self.spin_lifetime_reg.setSingleStep(1e-3)
        lifetime_layout.addRow("Regularization:", self.spin_lifetime_reg)
        
        self.chk_lifetime_log = QCheckBox("Log Lifetime Scale", self)
        self.chk_lifetime_log.setChecked(True)
        lifetime_layout.addRow(self.chk_lifetime_log)
        
        self.chk_lifetime_log_amp = QCheckBox("Log Amplitude", self)
        self.chk_lifetime_log_amp.setChecked(True)
        lifetime_layout.addRow(self.chk_lifetime_log_amp)
        
        wf_layout.addWidget(self.lifetime_params_group)
        
        audio_layout.addWidget(wf_group)

        # Channel notes
        self.channel_group = QGroupBox("Channel Notes", self)
        self.channel_group.setContentsMargins(0, 0, 0, 0)
        channel_layout = QVBoxLayout(self.channel_group)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(0)
        self.btn_update_channels = QPushButton("Update Channels from Setup", self)
        channel_layout.addWidget(self.btn_update_channels)
        self.channel_scroll = QScrollArea(self)
        self.channel_scroll.setContentsMargins(0, 0, 0, 0)
        self.channel_scroll.setWidgetResizable(True)
        self.channel_container = QWidget(self)
        self.channel_layout_inner = QVBoxLayout(self.channel_container)
        self.channel_layout_inner.setContentsMargins(0, 0, 0, 0)
        self.channel_layout_inner.setSpacing(0)
        self.channel_scroll.setWidget(self.channel_container)
        channel_layout.addWidget(self.channel_scroll)
        audio_layout.addWidget(self.channel_group)

        # Detector control
        detector_group = QGroupBox("Detector Control", self)
        detector_group.setContentsMargins(0, 0, 0, 0)
        detector_layout = QVBoxLayout(detector_group)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.setSpacing(0)
        self.detector_scroll = QScrollArea(self)
        self.detector_scroll.setContentsMargins(0, 0, 0, 0)
        self.detector_scroll.setWidgetResizable(True)
        self.detector_container = QWidget(self)
        self.detector_layout_inner = QVBoxLayout(self.detector_container)
        self.detector_layout_inner.setContentsMargins(0, 0, 0, 0)
        self.detector_layout_inner.setSpacing(0)
        self.detector_scroll.setWidget(self.detector_container)
        detector_layout.addWidget(self.detector_scroll)
        audio_layout.addWidget(detector_group)

        self.tabs.addTab(setup_tab, "Setup")
        self.tabs.addTab(audio_tab, "Audio")

        # Plot tab
        plot_tab = QWidget(self)
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.setAlignment(Qt.AlignTop)

        # Create waterfall plot widget
        self.waterfall_plot = WaterfallPlotWidget(self)
        plot_layout.addWidget(self.waterfall_plot)

        # Cassette player controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(0)
        ctrl_layout.setAlignment(Qt.AlignTop)
        
        self.btn_update_plot = QPushButton("🔄", self)
        self.btn_update_plot.setToolTip("Update Waterfall")
        self.btn_update_plot.setFixedSize(40, 30)
        ctrl_layout.addWidget(self.btn_update_plot)
        
        # Add separator
        separator1 = QFrame(self)
        separator1.setFrameShape(QFrame.VLine)
        separator1.setLineWidth(1)
        ctrl_layout.addWidget(separator1)
        
        self.btn_play = QPushButton("▶️", self)
        self.btn_play.setToolTip("Play")
        self.btn_play.setFixedSize(40, 30)
        ctrl_layout.addWidget(self.btn_play)
        
        self.btn_pause = QPushButton("⏸️", self)
        self.btn_pause.setToolTip("Pause")
        self.btn_pause.setFixedSize(40, 30)
        self.btn_pause.setEnabled(False)
        ctrl_layout.addWidget(self.btn_pause)
        
        self.btn_stop = QPushButton("⏹️", self)
        self.btn_stop.setToolTip("Stop")
        self.btn_stop.setFixedSize(40, 30)
        self.btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self.btn_stop)
        
        self.btn_revert = QPushButton("⏪", self)
        self.btn_revert.setToolTip("Revert to Start")
        self.btn_revert.setFixedSize(40, 30)
        ctrl_layout.addWidget(self.btn_revert)
        
        # Add separator
        separator = QFrame(self)
        separator.setFrameShape(QFrame.VLine)
        separator.setLineWidth(1)
        ctrl_layout.addWidget(separator)
        
        self.btn_save_wav = QPushButton("💾", self)
        self.btn_save_wav.setToolTip("Save WAV")
        self.btn_save_wav.setFixedSize(40, 30)
        ctrl_layout.addWidget(self.btn_save_wav)
        
        self.lbl_sound_pos = QLabel("00:00 / 00:00", self)
        self.lbl_sound_pos.setMinimumWidth(100)
        ctrl_layout.addWidget(self.lbl_sound_pos)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        ctrl_layout.addWidget(self.progress)
        plot_layout.addLayout(ctrl_layout)

        # Info
        self.lbl_info = QLabel("", self)
        plot_layout.addWidget(self.lbl_info)

        self.tabs.addTab(plot_tab, "Plot")

        layout.addWidget(self.tabs, 1)
        layout.addStretch()

    def _connect_signals(self):
        self.btn_load.clicked.connect(self._load_tttr)
        self.btn_update_channels.clicked.connect(self._update_channels_from_wizard)
        self.btn_update_plot.clicked.connect(self._update_waterfall)
        self.btn_play.clicked.connect(self._play_sound)
        self.btn_pause.clicked.connect(self._pause_sound)
        self.btn_stop.clicked.connect(self._stop_sound)
        self.btn_revert.clicked.connect(self._revert_sound)
        self.btn_save_wav.clicked.connect(self._save_wav)
        
        # Connect waterfall mode radio buttons
        self.waterfall_mode_group.buttonClicked.connect(self._on_waterfall_mode_changed)
        
        # Connect sound player signals
        self.sound_player.position_changed.connect(self._on_sound_position_changed)
        self.sound_player.state_changed.connect(self._on_sound_state_changed)
        self.sound_player.error_occurred.connect(self._on_sound_error)

    def _load_tttr(self):
        if tttrlib is None:
            QMessageBox.warning(self, "Error", "tttrlib not available.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Load TTTR File")
        if not path:
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.lbl_file.setText(path)
            self.data = load_tttr_with_tttrlib(path)
            self._update_channels_from_wizard()
            self._update_waterfall()
            self._update_icon_state()  # Update icon after loading
        except Exception as e:
            logging.exception(f"Failed to load TTTR: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load: {e}")
            self._update_icon_state()  # Update icon for error state
        finally:
            self.progress.setVisible(False)

    def _setup_channels(self):
        # Clear existing
        for ctrl in self.channel_controls.values():
            ctrl.setParent(None)
            ctrl.deleteLater()
        self.channel_controls.clear()

        if not self.channels:
            return

        cfg = make_default_channel_cfg(self.channels)
        for ch in self.channels:
            ctrl = ChannelControl(ch, cfg[ch])
            self.channel_layout_inner.addWidget(ctrl)
            self.channel_controls[ch] = ctrl

    def _setup_detectors(self):
        for ctrl in self.detector_controls.values():
            ctrl.setParent(None)
            ctrl.deleteLater()
        self.detector_controls.clear()

        for det in self.detectors:
            ctrl = DetectorControl(det['name'], det['color'], det['enabled'])
            ctrl.colorChanged.connect(lambda c, name=det['name']: self._on_detector_color_changed(name, c))
            self.detector_layout_inner.addWidget(ctrl)
            self.detector_controls[det['name']] = ctrl

    def _on_detector_color_changed(self, name, color):
        det = next(d for d in self.detectors if d['name'] == name)
        det['color'] = color

    def _get_selected_channels(self) -> List[int]:
        return [ch for ch, ctrl in self.channel_controls.items() if ctrl.chk_enable.isChecked()]

    def _get_channel_configs(self) -> Dict[int, ChannelConfig]:
        cfg = {}
        for ch, ctrl in self.channel_controls.items():
            c = ctrl.get_config()
            if c is not None:
                default_cfg = make_default_channel_cfg([ch])
                cfg[ch] = ChannelConfig(
                    note_hz=default_cfg[ch].note_hz,
                    chord_type=c.chord_type,
                    pitch_semitones=c.pitch_semitones,
                    micro_min=c.micro_min,
                    micro_max=c.micro_max,
                    gain=c.gain
                )
        return cfg

    def _on_waterfall_mode_changed(self, button):
        """Handle waterfall mode radio button change."""
        is_lifetime = button == self.radio_lifetime
        
        # Enable/disable parameter groups
        self.microtime_params_group.setEnabled(not is_lifetime)
        self.lifetime_params_group.setEnabled(is_lifetime)
        
        # Update plot title
        if self.waterfall_plot:
            title = "Lifetime Waterfall" if is_lifetime else "Microtime Waterfall"
            self.waterfall_plot.set_title(title)
        
        # Auto-update waterfall if data is loaded
        if self.data is not None:
            self._update_waterfall()

    def _update_waterfall(self):
        if self.data is None:
            return

        enabled_dets = [d for d in self.detectors if d['enabled']]
        if not enabled_dets:
            self.lbl_info.setText("No detectors enabled")
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)
            
            # Check which mode is selected
            is_lifetime = self.radio_lifetime.isChecked()
            
            if is_lifetime:
                self._update_lifetime_waterfall(enabled_dets)
            else:
                self._update_microtime_waterfall(enabled_dets)

        except Exception as e:
            logging.exception(f"Failed to update waterfall: {e}")
            QMessageBox.warning(self, "Error", f"Failed to update: {e}")
        finally:
            self.progress.setVisible(False)

    def _update_microtime_waterfall(self, enabled_dets):
        """Update microtime waterfall (original functionality)."""
        # Get dimensions from first enabled detector
        W, macro_t_s, micro_centers = compute_microtime_waterfall(
            data=self.data,
            channels=enabled_dets[0]['channels'],
            macro_bin_width_s=self.spin_wf_bin_width.value(),
            n_micro_bins=self.spin_wf_micro_bins.value()
        )
        n_macro, n_micro = W.shape
        rgb = np.zeros((n_macro, n_micro, 3), dtype=np.float32)
        total = np.zeros((n_macro, n_micro), dtype=np.float32)

        for det in enabled_dets:
            W_det, _, _ = compute_microtime_waterfall(
                data=self.data,
                channels=det['channels'],
                macro_bin_width_s=self.spin_wf_bin_width.value(),
                n_micro_bins=self.spin_wf_micro_bins.value()
            )
            if self.chk_wf_log.isChecked():
                W_det = np.log1p(W_det)
            total += W_det
            r, g, b, _ = det['color'].getRgbF()
            rgb[:, :, 0] += W_det * r
            rgb[:, :, 1] += W_det * g
            rgb[:, :, 2] += W_det * b

        mask = total > 0
        rgb[mask] /= total[mask, np.newaxis]

        # Transpose the waterfall matrix for display
        # Original: W.shape = (n_macro, n_micro), display as (micro_time vs macro_time)
        rgb_transposed = rgb.transpose(1, 0, 2)  # Shape: (n_micro, n_macro, 3)

        # Set waterfall data using the widget
        self.waterfall_plot.set_waterfall_data(
            rgb_data=rgb_transposed,
            macro_t_s=macro_t_s,
            micro_centers=micro_centers,
            n_macro_bins=n_macro,
            n_micro_bins=n_micro
        )

        self.lbl_info.setText(f"Microtime Waterfall: {n_macro} macro bins, {n_micro} micro bins")

    def _update_lifetime_waterfall(self, enabled_dets):
        """Update lifetime waterfall (new functionality)."""
        if compute_lifetime_waterfall is None:
            QMessageBox.warning(self, "Error", "Lifetime analysis not available.")
            return
            
        # Get parameters from GUI
        bin_width_s = self.spin_lifetime_bin_width.value()
        tau_min = self.spin_lifetime_tau_min.value() * 1e-9  # Convert ns to s
        tau_max = self.spin_lifetime_tau_max.value() * 1e-9  # Convert ns to s
        n_tau = self.spin_lifetime_n_tau.value()
        lam = self.spin_lifetime_reg.value()
        log_tau = self.chk_lifetime_log.isChecked()
        log_amp = self.chk_lifetime_log_amp.isChecked()
        
        # Collect all channels from enabled detectors
        all_channels = []
        channel_to_detector = {}
        for det in enabled_dets:
            for ch in det['channels']:
                all_channels.append(ch)
                channel_to_detector[ch] = det
        
        if not all_channels:
            self.lbl_info.setText("No channels in enabled detectors")
            return
        
        # Compute lifetime waterfalls for each channel
        channel_data = {}
        for ch in all_channels:
            try:
                # Use detector's microtime gate if available, otherwise full range
                micro_gate = None  # Could be enhanced to use detector-specific gates
                
                A, macro_t_s, tau = compute_lifetime_waterfall(
                    data=self.data,
                    channel=ch,
                    macro_bin_width_s=bin_width_s,
                    micro_gate=micro_gate,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    n_tau=n_tau,
                    lam=lam
                )
                channel_data[ch] = (A, macro_t_s, tau)
                
            except Exception as e:
                logging.warning(f"Failed to compute lifetime waterfall for channel {ch}: {e}")
                continue
        
        if not channel_data:
            self.lbl_info.setText("Failed to compute lifetime waterfalls")
            return
        
        # Create RGB composite similar to microtime waterfall
        # Use the first channel as reference for dimensions
        first_ch = list(channel_data.keys())[0]
        A_ref, macro_t_s, tau = channel_data[first_ch]
        n_macro, n_tau = A_ref.shape
        
        # Create RGB matrix
        rgb = np.zeros((n_macro, n_tau, 3), dtype=np.float32)
        total = np.zeros((n_macro, n_tau), dtype=np.float32)
        
        for ch, (A, _, _) in channel_data.items():
            det = channel_to_detector[ch]
            A_plot = A.copy()
            
            if log_amp:
                A_plot = np.log1p(A_plot)
            
            total += A_plot
            r, g, b, _ = det['color'].getRgbF()
            rgb[:, :, 0] += A_plot * r
            rgb[:, :, 1] += A_plot * g
            rgb[:, :, 2] += A_plot * b
        
        # Normalize
        mask = total > 0
        rgb[mask] /= total[mask, np.newaxis]
        
        # Transpose for display (lifetime vs macro time)
        rgb_transposed = rgb.transpose(1, 0, 2)  # Shape: (n_tau, n_macro, 3)
        
        # Set waterfall data
        self.waterfall_plot.set_waterfall_data(
            rgb_data=rgb_transposed,
            macro_t_s=macro_t_s,
            micro_centers=tau,  # Use lifetime axis as "micro" axis for display
            n_macro_bins=n_macro,
            n_micro_bins=n_tau
        )
        
        # Update plot title and info
        self.waterfall_plot.set_title("Lifetime Waterfall")
        self.lbl_info.setText(f"Lifetime Waterfall: {n_macro} macro bins, {n_tau} lifetime points")
        
        # Update axis labels for lifetime display
        plot_widget = self.waterfall_plot.get_plot_widget()
        plot_widget.setLabel('bottom', 'Lifetime τ (s)')
        if log_tau:
            plot_widget.setLogMode(x=True, y=False)
        else:
            plot_widget.setLogMode(x=False, y=False)

    def _update_channels_from_wizard(self):
        settings = self.detector_page.get_settings()
        dets = settings.get("detectors", {})
        self.detectors = []
        for i, (name, det) in enumerate(dets.items()):
            color = QColor.fromHsv((i * 60) % 360, 200, 255)
            self.detectors.append({'name': name, 'channels': det.get('chs', []), 'color': color, 'enabled': True})
        self._setup_detectors()
        self.channels = sorted(set(sum((d['channels'] for d in self.detectors), [])))
        self._setup_channels()

    def _play_sound(self):
        """Play or resume sound playback"""
        logging.info("Play sound requested")
        
        if self.data is None:
            logging.warning("No data loaded")
            QMessageBox.warning(self, "Error", "No data loaded.")
            return

        channels = self._get_selected_channels()
        logging.info(f"Selected channels: {channels}")
        if not channels:
            logging.warning("No channels selected")
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        cfg = self._get_channel_configs()
        logging.info(f"Channel configs: {cfg}")
        if not cfg:
            logging.warning("No channels configured")
            QMessageBox.warning(self, "Error", "No channels configured.")
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)
            self._update_icon_state()  # Set processing state
            
            # Create audio using the new sound playback module
            wav_data, duration = create_tttr_audio(
                data=self.data,
                channels=channels,
                channel_cfg=cfg,
                bin_width_s=self.spin_bin_width.value(),
                sample_rate=self.spin_sample_rate.value(),
                env_mode=self.combo_env_mode.currentText(),
                master_gain=self.spin_master_gain.value()
            )
            
            logging.info(f"Audio created successfully, duration: {duration:.2f}s")
            
            # Load audio into sound player
            if self.sound_player.load_audio(wav_data, self.spin_sample_rate.value()):
                # Start playback
                if self.sound_player.play():
                    logging.info("Sound playback started")
                    self._update_icon_state()  # Set playing state
                else:
                    logging.error("Failed to start playback")
                    QMessageBox.warning(self, "Error", "Failed to start playback.")
            else:
                logging.error("Failed to load audio")
                QMessageBox.warning(self, "Error", "Failed to load audio.")

        except Exception as e:
            logging.exception(f"Failed to play sound: {e}")
            QMessageBox.warning(self, "Error", f"Failed to play: {e}")
        finally:
            self.progress.setVisible(False)

    def _pause_sound(self):
        """Pause sound playback"""
        if self.sound_player.pause():
            logging.info("Sound paused")
        else:
            logging.warning("Failed to pause sound")

    def _stop_sound(self):
        """Stop sound playback"""
        if self.sound_player.stop():
            logging.info("Sound stopped")
        else:
            logging.warning("Failed to stop sound")

    def _revert_sound(self):
        """Revert playback to start"""
        if self.sound_player.revert():
            logging.info("Sound reverted to start")
        else:
            logging.warning("Failed to revert sound")

    def _on_sound_position_changed(self, current: float, duration: float):
        """Handle sound position updates from SoundPlayer."""
        # Update position label
        self.lbl_sound_pos.setText(f"{_format_time(current)} / {_format_time(duration)}")
        
        # Update waterfall position indicator
        if self.waterfall_plot is not None:
            n_macro_bins, n_micro_bins = self.waterfall_plot.get_bin_count()
            if n_macro_bins > 0 and duration > 0:
                progress = current / duration
                bin_pos = progress * n_macro_bins
                self.waterfall_plot.set_position(bin_pos)

    def _on_sound_state_changed(self, state: str):
        """Handle sound state changes from SoundPlayer."""
        if state == "playing":
            # Update button states for playing
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_revert.setEnabled(True)
            self.btn_play.setText("▶️")
            
            # Show position indicator
            if self.waterfall_plot is not None:
                self.waterfall_plot.show_position_indicator(True)
                self.waterfall_plot.reset_position()
            
            self._update_icon_state()  # Set playing state
            
        elif state == "paused":
            # Update button states for paused
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_play.setText("▶️ Resume")
            
            self._update_icon_state()  # Set paused state
            
        elif state == "stopped":
            # Reset button states
            self._reset_playback_buttons()
            
            # Hide position indicator
            if self.waterfall_plot is not None:
                self.waterfall_plot.show_position_indicator(False)
            
            self._update_icon_state()  # Return to loaded/idle state
            
        elif state == "finished":
            # Playback finished naturally
            self._reset_playback_buttons()
            
            # Hide position indicator
            if self.waterfall_plot is not None:
                self.waterfall_plot.show_position_indicator(False)
            
            self._update_icon_state()  # Return to loaded/idle state

    def _on_sound_error(self, error_message: str):
        """Handle sound playback errors from SoundPlayer."""
        logging.error(f"Sound playback error: {error_message}")
        QMessageBox.warning(self, "Playback Error", error_message)
        self._reset_playback_buttons()

    def _reset_playback_buttons(self):
        """Reset playback buttons to initial state"""
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_revert.setEnabled(False)
        self.btn_play.setText("▶️")  # Reset play button text

    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        # Clean up sound player
        if hasattr(self, 'sound_player'):
            self.sound_player.cleanup()
        
        # Clean up waterfall plot
        if hasattr(self, 'waterfall_plot') and self.waterfall_plot is not None:
            self.waterfall_plot.clear_plot()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction

    def _save_wav(self):
        logging.info("Save WAV requested")
        if self.data is None:
            logging.warning("No data loaded")
            QMessageBox.warning(self, "Error", "No data loaded.")
            return

        channels = self._get_selected_channels()
        logging.info(f"Selected channels: {channels}")
        if not channels:
            logging.warning("No channels selected")
            QMessageBox.warning(self, "Error", "No channels selected.")
            return

        cfg = self._get_channel_configs()
        logging.info(f"Channel configs: {cfg}")
        if not cfg:
            logging.warning("No channels configured")
            QMessageBox.warning(self, "Error", "No channels configured.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save WAV File", "", "WAV Files (*.wav)")
        if not path:
            logging.info("Save dialog cancelled")
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)
            logging.info(f"Saving WAV to: {path}")

            tttr_to_wav(
                data=self.data,
                out_wav_path=path,
                channels=channels,
                channel_cfg=cfg,
                bin_width_s=self.spin_bin_width.value(),
                sample_rate=self.spin_sample_rate.value(),
                env_mode=self.combo_env_mode.currentText(),
                env_floor=self.spin_env_floor.value(),
                env_scale=self.spin_env_scale.value(),
                attack_frames=self.spin_attack_frames.value(),
                release_frames=self.spin_release_frames.value(),
                master_gain=self.spin_master_gain.value()
            )
            logging.info("WAV file saved successfully")
            QMessageBox.information(self, "Success", f"WAV saved to {path}")

        except Exception as e:
            logging.exception(f"Failed to save WAV: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save: {e}")
        finally:
            self.progress.setVisible(False)

    def _update_icon_state(self):
        """Update the plugin icon based on current state."""
        if _icon_manager is None:
            return
        
        # Determine current state from sound_player
        if self.data is None:
            state = "idle"
        elif self.sound_player.state.is_playing and not self.sound_player.state.is_paused:
            state = "playing"
        elif self.sound_player.state.is_paused:
            state = "paused"
        elif self.progress.isVisible():
            state = "processing"
        else:
            state = "loaded"
        
        # Update icon manager state
        _icon_manager.set_state(state)
