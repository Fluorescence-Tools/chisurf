"""
Enhanced Simulation Setup Dialog

Based on Burbulator's comprehensive parameter structure with support for:
- Multiple species with individual brightness/dynamics controls
- CW vs Pulsed excitation modes
- Anisotropy parameters
- TAC/IRF parameters
- Background and scattering controls
- Species interconversion rates
"""

import json
import os
import numpy as np
import numba as nb
from typing import Dict, Any, Optional
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QSplitter,
    QWidget, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QCheckBox,
    QPushButton, QToolButton, QGroupBox, QLabel, QFileDialog, QMessageBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QButtonGroup, QRadioButton, QFrame,
    QSizePolicy,
)
from qtpy.QtCore import Qt, QTimer
import math
import json


def load_channel_settings():
    """Load channel conversion settings from JSON file."""
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    settings_file = os.path.join(plugin_dir, 'channel_settings.json')
    
    default_settings = {
        "channel_conversion": {
            "default": [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5],
            "green_p": 0,
            "green_s": 1,
            "red_p": 2,
            "red_s": 3,
            "yellow_p": 4,
            "yellow_s": 5
        },
        "detector_channels": {
            "green_p": 8,
            "green_s": 9,
            "red_p": 10,
            "red_s": 11,
            "yellow_p": 12,
            "yellow_s": 13
        }
    }
    
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
        else:
            # Create default settings file
            save_channel_settings(default_settings)
            return default_settings
    except Exception:
        return default_settings


@nb.jit(nopython=True, nogil=True)
def fast_conv_burbulator(lifetime_spectrum, irf, n_channels, tac_dt, laser_period):
    """
    Exact FastConv implementation from C# Burbulator.
    Fast convolution for high repetition rate TCSPC.

    Parameters:
    lifetime_spectrum: array of [amplitude1, lifetime1, amplitude2, lifetime2, ...]
    irf: instrument response function array
    n_channels: number of TAC channels
    tac_dt: time per channel (ns)
    laser_period: laser repetition period (ns)

    Returns:
    convolved pattern
    """
    # Calculate period in channels (same as C#)
    period_n = int(np.ceil(laser_period / tac_dt - 0.5))

    # Initialize output
    p_conv = np.zeros(n_channels)

    # Auto TAC range (same as C#)
    start = 1
    stop = min(n_channels - 1, period_n + start)

    num_exp = len(lifetime_spectrum) // 2
    delta_half = tac_dt * 0.5

    # Convolution loop (exact C# implementation)
    for ne in range(num_exp):
        amplitude = lifetime_spectrum[2 * ne]
        lifetime = lifetime_spectrum[2 * ne + 1]

        if lifetime == 0.0:
            continue

        exp_curr = np.exp(-tac_dt / lifetime)
        tail_a = 1.0 / (1.0 - np.exp(-laser_period / lifetime))

        fit_curr = 0.0

        # First phase: regular convolution (C# uses i <= stop, which is 1 to stop inclusive)
        for i in range(1, min(stop + 1, n_channels)):
            # Ensure we don't access out of bounds
            if i > 0 and i < n_channels and i-1 < len(irf) and i < len(irf):
                fit_curr = (fit_curr + delta_half * irf[i - 1]) * exp_curr + delta_half * irf[i]
                p_conv[i] += fit_curr * amplitude

        # Second phase: periodic tail correction (C# uses i <= stop)
        fit_curr *= np.exp(-(period_n - stop + start) * tac_dt / lifetime)
        for i in range(start, min(stop + 1, n_channels)):
            if i < n_channels:
                fit_curr *= exp_curr
                p_conv[i] += fit_curr * amplitude * tail_a

    return p_conv


class MoleculeControls(QWidget):
    """Controls for a single molecular species - mirrors C# Burbulator MoleculeCW."""

    def __init__(self, species_idx, params, parent=None):
        super().__init__(parent)
        self.species_idx = species_idx
        self.params = params
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        # Species header with enable checkbox
        header_layout = QHBoxLayout()
        self.enabled_check = QCheckBox(f"Species {self.species_idx + 1}")
        self.enabled_check.setChecked(True)
        header_layout.addWidget(self.enabled_check)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Quantum yields (brightness) - multi-channel like C# Burbulator
        bright_group = QGroupBox("Quantum Yields (kHz)")
        bright_layout = QGridLayout(bright_group)
        
        # Headers
        bright_layout.addWidget(QLabel("Channel"), 0, 0)
        bright_layout.addWidget(QLabel("P"), 0, 1)
        bright_layout.addWidget(QLabel("S"), 0, 2)
        bright_layout.addWidget(QLabel("Total"), 0, 3)

        # Green channel (always enabled)
        bright_layout.addWidget(QLabel("Green"), 1, 0)
        self.q_green_p = QDoubleSpinBox()
        self.q_green_p.setRange(0, 1000)
        self.q_green_p.setValue(50.0)
        bright_layout.addWidget(self.q_green_p, 1, 1)
        
        self.q_green_s = QDoubleSpinBox()
        self.q_green_s.setRange(0, 1000)
        self.q_green_s.setValue(50.0)
        bright_layout.addWidget(self.q_green_s, 1, 2)
        
        self.s_green_total = QLabel("100")
        bright_layout.addWidget(self.s_green_total, 1, 3)

        # Red channel
        bright_layout.addWidget(QLabel("Red"), 2, 0)
        self.q_red_p = QDoubleSpinBox()
        self.q_red_p.setRange(0, 1000)
        self.q_red_p.setValue(0.0)
        bright_layout.addWidget(self.q_red_p, 2, 1)
        
        self.q_red_s = QDoubleSpinBox()
        self.q_red_s.setRange(0, 1000)
        self.q_red_s.setValue(0.0)
        bright_layout.addWidget(self.q_red_s, 2, 2)
        
        self.s_red_total = QLabel("0")
        bright_layout.addWidget(self.s_red_total, 2, 3)

        # Yellow channel
        bright_layout.addWidget(QLabel("Yellow"), 3, 0)
        self.q_yellow_p = QDoubleSpinBox()
        self.q_yellow_p.setRange(0, 1000)
        self.q_yellow_p.setValue(0.0)
        bright_layout.addWidget(self.q_yellow_p, 3, 1)
        
        self.q_yellow_s = QDoubleSpinBox()
        self.q_yellow_s.setRange(0, 1000)
        self.q_yellow_s.setValue(0.0)
        bright_layout.addWidget(self.q_yellow_s, 3, 2)
        
        self.s_yellow_total = QLabel("0")
        bright_layout.addWidget(self.s_yellow_total, 3, 3)

        layout.addWidget(bright_group)

        # Diffusion parameters
        diff_group = QGroupBox("Diffusion")
        diff_layout = QFormLayout(diff_group)

        self.n_molecules_spin = QDoubleSpinBox()
        self.n_molecules_spin.setRange(0, 10000)
        self.n_molecules_spin.setValue(self.params['M'][self.species_idx] if self.species_idx < len(self.params['M']) else 5.0)
        diff_layout.addRow("N molecules:", self.n_molecules_spin)

        self.diffusion_spin = QDoubleSpinBox()
        self.diffusion_spin.setRange(0, 100)
        self.diffusion_spin.setValue(self.params['D'][self.species_idx] if self.species_idx < len(self.params['D']) else 3.0)
        diff_layout.addRow("D coefficient (μm²/s):", self.diffusion_spin)

        # Tau D (calculated from D and w0)
        self.tau_d_spin = QDoubleSpinBox()
        self.tau_d_spin.setRange(0, 1000)
        self.tau_d_spin.setValue(0.01)
        self.tau_d_spin.setReadOnly(True)
        diff_layout.addRow("τD (ms):", self.tau_d_spin)

        layout.addWidget(diff_group)

        # Scattering/Dark counts
        scatter_group = QGroupBox("Scattering & Dark")
        scatter_layout = QFormLayout(scatter_group)

        self.scatter_green_spin = QDoubleSpinBox()
        self.scatter_green_spin.setRange(0, 100)
        self.scatter_green_spin.setValue(0.0)
        scatter_layout.addRow("Green scatter:", self.scatter_green_spin)

        self.scatter_red_spin = QDoubleSpinBox()
        self.scatter_red_spin.setRange(0, 100)
        self.scatter_red_spin.setValue(0.0)
        scatter_layout.addRow("Red scatter:", self.scatter_red_spin)

        self.scatter_yellow_spin = QDoubleSpinBox()
        self.scatter_yellow_spin.setRange(0, 100)
        self.scatter_yellow_spin.setValue(0.0)
        scatter_layout.addRow("Yellow scatter:", self.scatter_yellow_spin)

        self.dark_green_spin = QDoubleSpinBox()
        self.dark_green_spin.setRange(0, 100)
        self.dark_green_spin.setValue(0.0)
        scatter_layout.addRow("Green dark:", self.dark_green_spin)

        self.dark_red_spin = QDoubleSpinBox()
        self.dark_red_spin.setRange(0, 100)
        self.dark_red_spin.setValue(0.0)
        scatter_layout.addRow("Red dark:", self.dark_red_spin)

        self.dark_yellow_spin = QDoubleSpinBox()
        self.dark_yellow_spin.setRange(0, 100)
        self.dark_yellow_spin.setValue(0.0)
        scatter_layout.addRow("Yellow dark:", self.dark_yellow_spin)

        layout.addWidget(scatter_group)

    def connect_signals(self):
        """Connect signals for automatic updates."""
        self.q_green_p.valueChanged.connect(self.update_green_total)
        self.q_green_s.valueChanged.connect(self.update_green_total)
        self.q_red_p.valueChanged.connect(self.update_red_total)
        self.q_red_s.valueChanged.connect(self.update_red_total)
        self.q_yellow_p.valueChanged.connect(self.update_yellow_total)
        self.q_yellow_s.valueChanged.connect(self.update_yellow_total)

    def update_green_total(self):
        total = self.q_green_p.value() + self.q_green_s.value()
        self.s_green_total.setText(f"{total:.1f}")

    def update_red_total(self):
        total = self.q_red_p.value() + self.q_red_s.value()
        self.s_red_total.setText(f"{total:.1f}")

    def update_yellow_total(self):
        total = self.q_yellow_p.value() + self.q_yellow_s.value()
        self.s_yellow_total.setText(f"{total:.1f}")


class BackgroundControls(QWidget):
    """Controls for background parameters."""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        bg_group = QGroupBox("Background")
        bg_layout = QFormLayout(bg_group)

        n_channels = self.params.get('N_channels', 2)

        # Background counts per channel
        self.bg_parallel_spin = QDoubleSpinBox()
        self.bg_parallel_spin.setRange(0, 10)
        self.bg_parallel_spin.setSingleStep(0.001)
        val = self.params['q_bg'][0] if len(self.params['q_bg']) > 0 else 0.001
        self.bg_parallel_spin.setValue(val)
        bg_layout.addRow("Parallel bg:", self.bg_parallel_spin)

        self.bg_perp_spin = QDoubleSpinBox()
        self.bg_perp_spin.setRange(0, 10)
        self.bg_perp_spin.setSingleStep(0.001)
        val = self.params['q_bg'][1] if len(self.params['q_bg']) > 1 else 0.001
        self.bg_perp_spin.setValue(val)
        bg_layout.addRow("Perp bg:", self.bg_perp_spin)

        # Dark counts
        self.dark_parallel_spin = QDoubleSpinBox()
        self.dark_parallel_spin.setRange(0, 10)
        self.dark_parallel_spin.setSingleStep(0.001)
        self.dark_parallel_spin.setValue(0.0)
        bg_layout.addRow("Parallel dark:", self.dark_parallel_spin)

        self.dark_perp_spin = QDoubleSpinBox()
        self.dark_perp_spin.setRange(0, 10)
        self.dark_perp_spin.setSingleStep(0.001)
        self.dark_perp_spin.setValue(0.0)
        bg_layout.addRow("Perp dark:", self.dark_perp_spin)

        layout.addWidget(bg_group)


class AnisotropyControls(QWidget):
    """Controls for anisotropy parameters."""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        aniso_group = QGroupBox("Anisotropy Parameters")
        aniso_layout = QFormLayout(aniso_group)

        # Fundamental anisotropy
        self.r0_spin = QDoubleSpinBox()
        self.r0_spin.setRange(0, 1)
        self.r0_spin.setDecimals(4)
        self.r0_spin.setValue(0.38)
        aniso_layout.addRow("r0:", self.r0_spin)

        # G-factor
        self.g_factor_spin = QDoubleSpinBox()
        self.g_factor_spin.setRange(0.1, 2.0)
        self.g_factor_spin.setDecimals(4)
        self.g_factor_spin.setValue(1.0)
        aniso_layout.addRow("G-factor:", self.g_factor_spin)

        # Instrumental factors
        self.l1_spin = QDoubleSpinBox()
        self.l1_spin.setRange(0, 1)
        self.l1_spin.setDecimals(4)
        self.l1_spin.setValue(0.0308)
        aniso_layout.addRow("l1:", self.l1_spin)

        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0, 1)
        self.l2_spin.setDecimals(4)
        self.l2_spin.setValue(0.0368)
        aniso_layout.addRow("l2:", self.l2_spin)

        layout.addWidget(aniso_group)


class TACControls(QWidget):
    """Controls for TAC/IRF parameters."""

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        tac_group = QGroupBox("TAC Parameters")
        tac_layout = QFormLayout(tac_group)

        # TAC channels
        self.n_channels_spin = QSpinBox()
        self.n_channels_spin.setRange(256, 65536)
        self.n_channels_spin.setValue(self.params.get('N_tac_channels', 4096))
        tac_layout.addRow("TAC channels:", self.n_channels_spin)

        # TAC bin width
        self.tac_dt_spin = QDoubleSpinBox()
        self.tac_dt_spin.setRange(0.0001, 1.0)
        self.tac_dt_spin.setDecimals(6)
        self.tac_dt_spin.setValue(self.params.get('tac_dt', 0.004069))
        tac_layout.addRow("TAC dt (ns):", self.tac_dt_spin)

        # Laser period (for pulsed)
        self.laser_period_spin = QDoubleSpinBox()
        self.laser_period_spin.setRange(1.0, 100.0)
        self.laser_period_spin.setValue(self.params.get('laser_period', 13.596))
        tac_layout.addRow("Laser period (ns):", self.laser_period_spin)

        layout.addWidget(tac_group)

        # IRF section
        irf_group = QGroupBox("Instrument Response Function")
        irf_layout = QFormLayout(irf_group)

        self.use_gaussian_irf_check = QCheckBox("Use Gaussian IRF")
        self.use_gaussian_irf_check.setChecked(self.params.get('use_gaussian_irf', True))
        irf_layout.addRow(self.use_gaussian_irf_check)

        self.gaussian_fwhm_spin = QDoubleSpinBox()
        self.gaussian_fwhm_spin.setRange(0, 10)
        self.gaussian_fwhm_spin.setValue(self.params.get('gaussian_irf_fwhm', 0.11))
        self.gaussian_fwhm_spin.setEnabled(True)  # Enabled by default since IRF is enabled
        self.use_gaussian_irf_check.toggled.connect(lambda: self.gaussian_fwhm_spin.setEnabled(self.use_gaussian_irf_check.isChecked()))
        irf_layout.addRow("Gaussian FWHM (ns):", self.gaussian_fwhm_spin)

        # Gaussian IRF position (mean)
        self.gaussian_mean_spin = QDoubleSpinBox()
        self.gaussian_mean_spin.setRange(-100, 100)
        self.gaussian_mean_spin.setDecimals(3)
        self.gaussian_mean_spin.setValue(self.params.get('gaussian_irf_mean', 1.5))
        self.gaussian_mean_spin.setEnabled(True)  # Enabled by default since IRF is enabled
        self.use_gaussian_irf_check.toggled.connect(lambda: self.gaussian_mean_spin.setEnabled(self.use_gaussian_irf_check.isChecked()))
        irf_layout.addRow("Gaussian mean (ns):", self.gaussian_mean_spin)

        # Note: Gaussian σ is automatically calculated from FWHM internally

        self.irf_file_edit = QLineEdit()
        self.irf_file_edit.setPlaceholderText("Select IRF file...")
        self.irf_file_edit.setEnabled(False)
        irf_layout.addRow("IRF file:", self.irf_file_edit)

        self.browse_irf_button = QPushButton("Browse...")
        self.browse_irf_button.clicked.connect(self.browse_irf)
        self.browse_irf_button.setEnabled(False)
        irf_layout.addRow("", self.browse_irf_button)

        # Connect IRF options
        def update_irf_mode():
            use_file = not self.use_gaussian_irf_check.isChecked()
            self.irf_file_edit.setEnabled(use_file)
            self.browse_irf_button.setEnabled(use_file)
            if use_file and not self.irf_file_edit.text():
                self.browse_irf()

        self.use_gaussian_irf_check.toggled.connect(update_irf_mode)

        layout.addWidget(irf_group)

    def browse_irf(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select IRF file", "", "Text files (*.txt);;All files (*)"
        )
        if filename:
            self.irf_file_edit.setText(filename)


class EnhancedSimulationSetupDialog(QDialog):
    """Enhanced setup dialog based on Burbulator's comprehensive parameter structure."""

    # Focus type mappings (name -> integer value for DLL)
    FOCUS_TYPES = {
        "3D Gaussian, uniform CEF": 0,
        "3D Gaussian excitation and CEF": 1,
        "Rectangular excitation, uniform CEF": 2,
        "Cylindrical excitation, uniform CEF": 3,
        "Gaussian-Lorentzian excitation, pinhole CEF": 4,
        "Gaussian-Lorentzian excitation, cylindrical CEF": 5,
        "Flow excitation": 6
    }

    def __init__(self, device=None, parent=None):
        super().__init__(parent)
        self.device = device
        self.setWindowTitle("Enhanced Simulation Setup (Burbulator-style)")
        self.setModal(True)
        self.resize(800, 600)

        self._load_initial_params()
        self.setup_ui()

    def _load_initial_params(self):
        """Load initial parameters with comprehensive defaults."""
        # Load channel settings from JSON
        channel_settings = load_channel_settings()
        
        default_params = {
            # Species parameters
            'N_species': 1,
            'M': [5.0],        # Molecules per species
            'D': [3.0],         # Diffusion coefficients

            # Detection parameters - 6 channels for green/red/yellow P/S
            'N_channels': 6,    # Green P/S, Red P/S, Yellow P/S
            'q': [50.0, 50.0, 0.0, 0.0, 0.0, 0.0], # Brightness per channel
            'q_bg': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],  # Background per channel

            # Species transitions (radiative/non-radiative) - 2D arrays for NxN transitions
            'k_rad': [[0.0]],   # Radiative rates (NxN matrix)
            'k_nrad': [[0.0]],  # Non-radiative rates (NxN matrix)

            # Geometry
            'box_xy': 2.0,      # Lateral box size (μm)
            'box_z': 4.0,       # Axial box size (μm)
            'focus_type': 0,    # Focus type
            'focus_param': [0.3, 2.0],  # Focus parameters

            # Simulation timing
            'dt': 0.01,         # Time step (μs)
            'N_ph_max': 50000,  # Max photons to generate

            # Excitation mode
            'excitation_mode': 'CW',  # 'CW' or 'Pulsed'

            # TAC/IRF parameters
            'N_tac_channels': 4096,
            'tac_dt': 0.004069,
            'laser_period': 13.596,
            'use_gaussian_irf': True,      # Enable Gaussian IRF by default
            'gaussian_irf_fwhm': 0.11,     # 0.11 ns FWHM
            'gaussian_irf_mean': 1.5,      # 1.5 ns mean position
            'gaussian_irf_sigma': 0.0467,  # Calculated from FWHM=0.11: 0.11/(2*sqrt(2*ln(2)))
            'irf_file': '',

            # Anisotropy
            'r0': 0.38,
            'g_factor': 1.0,
            'l1': 0.0308,
            'l2': 0.0368,

            # Background/scattering
            'parallel_scatter': 0.0,
            'perp_scatter': 0.0,
            'parallel_dark': 0.0,
            'perp_dark': 0.0,

            # Output
            'spc_output_path': '',
            'N_ph_per_file': 50000,

            # BH_SPC conversion - 6 channel mapping from settings
            'pulsed_exc': 0,
            'ch_conversion': channel_settings['channel_conversion']['default'],
            
            # RNG parameters
            'rng_mode': 0,  # 0 = start from new state, 1 = continue from last state
            'rmt1seed': 12345,
            'rmt2seed': 67890,
            
            # Channel enable flags (like C# Burbulator)
            'green_enabled': True,
            'red_enabled': False,
            'yellow_enabled': False,
            
            # Fluorescence decay parameters (critical for data2spc_tac)
            'decay_lifetimes': [[4.0, 4.0, 4.0]],  # Default 4ns lifetime per species [Green, Red, Yellow]
            'decay_patterns': [''],                 # Custom decay pattern files per species
            'rotational_correlation_times': [[0.4, 0.4, 0.4]],  # Rotational correlation times per species [Green, Red, Yellow] (ns)
            
            # Dark state parameters
            'k_bd': [0.0],     # Bright → Dark transition rates per species
            'k_bb': [0.0],     # Bright → Bleached transition rates per species  
            'k_db': [0.0],     # Dark → Bright transition rates per species
            'darkstate_interconvert': False,
            
            # Focus-specific parameters (extracted from focus_param based on focus_type)
            'w0': 0.3,         # Beam waist (μm)
            'z0': 2.0,         # Rayleigh length (μm)
            'Rph': 0.15,       # Pinhole radius (μm) - for pinhole CEF types
            'z0_CEF': 1.0,     # CEF axial extent (μm) - for cylindrical CEF
            
            # Additional simulation parameters
            'tw': 0.01,        # Time window (same as dt, for compatibility)
            'pulsed_exc': 0,   # Pulsed excitation flag (0=CW, 1=Pulsed)
        }

        params = dict(default_params)
        dev_params = None
        if self.device is not None:
            if hasattr(self.device, 'simulation_params'):
                dev_params = getattr(self.device, 'simulation_params')
            elif hasattr(self.device, 'device') and hasattr(self.device.device, 'simulation_params'):
                dev_params = getattr(self.device.device, 'simulation_params')
        if isinstance(dev_params, dict):
            params.update(dev_params)
        self.params = params

    def _validate_initial_channel_state(self):
        """Ensure at least one channel is enabled during initialization."""
        green_enabled = self.green_enabled_check.isChecked()
        red_enabled = self.red_enabled_check.isChecked()
        yellow_enabled = self.yellow_enabled_check.isChecked()
        
        # If all channels are disabled, enable Green by default
        if not green_enabled and not red_enabled and not yellow_enabled:
            self.green_enabled_check.setChecked(True)
            self.params['green_enabled'] = True

    def _validate_channel_toggle(self):
        """Ensure at least one color channel is always enabled."""
        # Check if any channels are still enabled
        green_enabled = self.green_enabled_check.isChecked()
        red_enabled = self.red_enabled_check.isChecked()
        yellow_enabled = self.yellow_enabled_check.isChecked()
        
        # If all channels would be disabled, prevent the last one from being turned off
        if not green_enabled and not red_enabled and not yellow_enabled:
            # Find which checkbox was just toggled and re-enable it
            sender = self.sender()
            if sender == self.green_enabled_check:
                self.green_enabled_check.setChecked(True)
            elif sender == self.red_enabled_check:
                self.red_enabled_check.setChecked(True)
            elif sender == self.yellow_enabled_check:
                self.yellow_enabled_check.setChecked(True)
            
            # Show message to user
            QMessageBox.information(self, "Channel Requirement", 
                                   "At least one color channel (Green, Red, or Yellow) must be enabled.")
            return
        
        # If validation passes, update the channel enable state
        self._update_channel_enable_state()

    def _update_channel_enable_state(self):
        """Gray out CW and Pulsed controls for disabled Green/Red/Yellow channels based on the top bar checkboxes."""
        ge = bool(self.green_enabled_check.isChecked()) if hasattr(self, 'green_enabled_check') else True
        re = bool(self.red_enabled_check.isChecked()) if hasattr(self, 'red_enabled_check') else False
        ye = bool(self.yellow_enabled_check.isChecked()) if hasattr(self, 'yellow_enabled_check') else False

        # CW table
        if hasattr(self, 'cw_row_widgets'):
            for rw in self.cw_row_widgets:
                self._set_enabled(rw.get('G_P'), ge)
                self._set_enabled(rw.get('G_S'), ge)
                self._set_enabled(rw.get('R_P'), re)
                self._set_enabled(rw.get('R_S'), re)
                self._set_enabled(rw.get('Y_P'), ye)
                self._set_enabled(rw.get('Y_S'), ye)
                # Totals as labels: dim via style when disabled
                if 'Sg' in rw:
                    rw['Sg'].setEnabled(ge)
                if 'Sr' in rw:
                    rw['Sr'].setEnabled(re)
                if 'Sy' in rw:
                    rw['Sy'].setEnabled(ye)

        # Pulsed table
        if hasattr(self, 'pulsed_row_widgets'):
            for rw in self.pulsed_row_widgets:
                self._set_enabled(rw.get('tau_g'), ge)
                self._set_enabled(rw.get('rho_g'), ge)
                self._set_enabled(rw.get('tau_r'), re)
                self._set_enabled(rw.get('rho_r'), re)
                self._set_enabled(rw.get('tau_y'), ye)
                self._set_enabled(rw.get('rho_y'), ye)

    def _set_enabled(self, widget, enabled: bool):
        try:
            widget.setEnabled(enabled)
        except Exception:
            pass

    def update_mode_visibility(self):
        """Update tab visibility based on excitation mode."""
        # Enable/disable pulsed tab content based on pulsed radio button
        is_pulsed = self.pulsed_radio.isChecked()
        
        # All tabs are always enabled, but pulsed tab content is disabled when not in pulsed mode
        self.tab_widget.setTabEnabled(0, True)   # CW tab
        self.tab_widget.setTabEnabled(1, True)   # Pulsed tab - always accessible
        self.tab_widget.setTabEnabled(2, True)   # Kinetics tab
        self.tab_widget.setTabEnabled(3, True)   # Acquisition tab
        
        # Enable/disable the pulsed tab contents (gray out when not in pulsed mode)
        self._set_pulsed_tab_enabled(is_pulsed)
    
    def _set_pulsed_tab_enabled(self, enabled: bool):
        """Enable or disable all widgets in the pulsed tab."""
        if hasattr(self, 'pulsed_row_widgets'):
            for rw in self.pulsed_row_widgets:
                for widget_name, widget in rw.items():
                    if widget is not None and hasattr(widget, 'setEnabled'):
                        widget.setEnabled(enabled)

    def create_kinetics_tab(self):
        """Create the kinetics tab for state conversions."""
        kinetics_widget = QWidget()
        kinetics_layout = QVBoxLayout(kinetics_widget)
        kinetics_layout.setContentsMargins(0, 0, 0, 0)
        kinetics_layout.setSpacing(2)

        # Title and description
        title_label = QLabel("State Conversion Kinetics")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        kinetics_layout.addWidget(title_label)
        
        desc_label = QLabel("Configure transition rates between molecular states (k values)")
        desc_label.setStyleSheet("font-size: 10px; color: gray;")
        kinetics_layout.addWidget(desc_label)

        # Create kinetics tables organized by species as shown in screenshots
        n_species = max(1, int(self.params.get('N_species', 1)))
        
        # Radiative transitions table
        rad_group = QGroupBox("Radiative Transitions (k_rad)")
        rad_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand horizontally and vertically
        rad_layout = QVBoxLayout(rad_group)
        rad_layout.setContentsMargins(2, 2, 2, 2)  # Reduced margins
        
        # Create radiative table with no headers
        n_rows = n_species  # No background row
        n_cols = n_species  # One column per target species
        self.rad_table = QTableWidget(n_rows, n_cols)
        self.rad_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Table expands
        self.rad_table.horizontalHeader().setVisible(False)
        self.rad_table.verticalHeader().setVisible(True)  # Keep row labels for first table
        self.rad_table.setHorizontalHeaderLabels([])  # No column labels
        self.rad_table.setVerticalHeaderLabels([f"Species {i+1}" for i in range(n_species)])
        self.rad_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.rad_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._populate_rad_table()
        rad_layout.addWidget(self.rad_table, 1)
        
        # Add "All to 0" toolbutton inside radiative groupbox
        rad_zero_btn = QToolButton()
        rad_zero_btn.setText("All to 0")
        rad_zero_btn.clicked.connect(lambda: self._set_all_to_zero('rad'))
        rad_zero_btn.setToolTip("Set all radiative rates to 0")
        rad_zero_btn.setAutoRaise(True)
        rad_layout.addWidget(rad_zero_btn, 0)

        # Non-radiative transitions table
        nrad_group = QGroupBox("Non-Radiative Transitions (k_nrad)")
        nrad_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand horizontally and vertically
        nrad_layout = QVBoxLayout(nrad_group)
        nrad_layout.setContentsMargins(2, 2, 2, 2)  # Reduced margins
        
        # Create non-radiative table with no row labels
        self.nrad_table = QTableWidget(n_rows, n_cols)
        self.nrad_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Table expands
        self.nrad_table.horizontalHeader().setVisible(False)
        self.nrad_table.verticalHeader().setVisible(False)  # No row labels
        self.nrad_table.setHorizontalHeaderLabels([])  # No column labels
        self.nrad_table.setVerticalHeaderLabels([])  # No row labels
        self.nrad_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.nrad_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._populate_nrad_table()
        nrad_layout.addWidget(self.nrad_table, 1)
        
        # Add "All to 0" toolbutton inside non-radiative groupbox
        nrad_zero_btn = QToolButton()
        nrad_zero_btn.setText("All to 0")
        nrad_zero_btn.clicked.connect(lambda: self._set_all_to_zero('nrad'))
        nrad_zero_btn.setToolTip("Set all non-radiative rates to 0")
        nrad_zero_btn.setAutoRaise(True)
        nrad_layout.addWidget(nrad_zero_btn, 0)

        # Individual state conversion rates table
        state_group = QGroupBox("Dark State / Bleached Conversions")
        state_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand both directions
        state_layout = QVBoxLayout(state_group)
        state_layout.setContentsMargins(2, 2, 2, 2)  # Reduced margins
        state_layout.setSpacing(2)  # Reduced spacing
        
        # Create state conversion table with no row labels
        self.state_table = QTableWidget(n_species, 3)  # 3 columns: k_bd, k_bb, k_db
        self.state_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand both directions
        self.state_table.horizontalHeader().setVisible(False)
        self.state_table.verticalHeader().setVisible(False)  # No row labels
        self.state_table.setHorizontalHeaderLabels([])  # No column labels
        self.state_table.setVerticalHeaderLabels([])  # No row labels
        self.state_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.state_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._populate_state_table()
        state_layout.addWidget(self.state_table, 1)
        
        # Add "All to 0" toolbutton
        state_zero_btn = QToolButton()
        state_zero_btn.setText("All to 0")
        state_zero_btn.clicked.connect(lambda: self._set_all_to_zero('state'))
        state_zero_btn.setToolTip("Set all state conversion rates to 0")
        state_zero_btn.setAutoRaise(True)
        state_layout.addWidget(state_zero_btn, 0)

        # Create horizontal layout for all three tables
        tables_layout = QHBoxLayout()
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(4)  # Add some spacing between tables
        
        # Add all groups to horizontal layout with proper stretching
        tables_layout.addWidget(rad_group, 3)  # Give radiative table more stretch
        tables_layout.addWidget(nrad_group, 3)  # Give non-radiative table more stretch
        tables_layout.addWidget(state_group, 1)  # Give state table less stretch
        
        # Create controls layout for darkstate interconvert
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        
        # Add darkstate interconvert checkbox
        self.darkstate_interconvert = QCheckBox("Darkstate interconvert")
        self.darkstate_interconvert.setChecked(False)
        self.darkstate_interconvert.setToolTip("Enable dark state interconversion")
        controls_layout.addWidget(self.darkstate_interconvert)
        controls_layout.addStretch()  # Add stretch to push checkbox to left
        
        # Add horizontal layout to main kinetics layout with proper expansion
        kinetics_layout.addLayout(tables_layout, 1)
        kinetics_layout.addLayout(controls_layout)

        self.tab_widget.addTab(kinetics_widget, "Kinetics")
        
        # Store reference for later updates
        self.rad_row_widgets = []
        self.nrad_row_widgets = []
        self.state_row_widgets = []

    def _populate_rad_table(self):
        """Populate the radiative transitions table."""
        n_rows = self.rad_table.rowCount()
        n_cols = self.rad_table.columnCount()
        n_species = n_cols  # Calculate species from column count
        
        self.rad_row_widgets = []
        
        for row in range(n_rows):
            row_widgets = {}
            from_species = row + 1  # Species number (1-based)
            
            # k_rad columns for each target species
            for target_species in range(n_species):
                k_rad_spin = QDoubleSpinBox()
                k_rad_spin.setRange(0, 1e6)
                k_rad_spin.setDecimals(2)
                k_rad_spin.setValue(0.0)
                k_rad_spin.setToolTip(f"Species {from_species} → Species {target_species + 1} radiative rate")
                self.rad_table.setCellWidget(row, target_species, k_rad_spin)
                row_widgets[f'k_rad_{target_species + 1}'] = k_rad_spin
            
            self.rad_row_widgets.append(row_widgets)

    def _populate_nrad_table(self):
        """Populate the non-radiative transitions table."""
        n_rows = self.nrad_table.rowCount()
        n_cols = self.nrad_table.columnCount()
        n_species = n_cols  # Calculate species from column count
        
        self.nrad_row_widgets = []
        
        for row in range(n_rows):
            row_widgets = {}
            from_species = row + 1  # Species number (1-based)
            
            # k_nrad columns for each target species
            for target_species in range(n_species):
                k_nrad_spin = QDoubleSpinBox()
                k_nrad_spin.setRange(0, 1e6)
                k_nrad_spin.setDecimals(2)
                k_nrad_spin.setValue(0.0)
                k_nrad_spin.setToolTip(f"Species {from_species} → Species {target_species + 1} non-radiative rate")
                self.nrad_table.setCellWidget(row, target_species, k_nrad_spin)
                row_widgets[f'k_nrad_{target_species + 1}'] = k_nrad_spin
            
            self.nrad_row_widgets.append(row_widgets)

    def _populate_state_table(self):
        """Populate the individual state conversion rates table."""
        n_rows = self.state_table.rowCount()
        n_species = n_rows  # Each row represents one species
        
        self.state_row_widgets = []
        
        for row in range(n_rows):
            row_widgets = {}
            species_num = row + 1
            
            # Bright → Dark rate
            k_bd_spin = QDoubleSpinBox()
            k_bd_spin.setRange(0, 1e6)
            k_bd_spin.setDecimals(2)
            k_bd_spin.setValue(0.0)
            k_bd_spin.setToolTip(f"Species {species_num} Bright → Dark transition rate")
            self.state_table.setCellWidget(row, 0, k_bd_spin)
            row_widgets['k_bd'] = k_bd_spin
            
            # Bright → Bleached rate
            k_bb_spin = QDoubleSpinBox()
            k_bb_spin.setRange(0, 1e6)
            k_bb_spin.setDecimals(2)
            k_bb_spin.setValue(0.0)
            k_bb_spin.setToolTip(f"Species {species_num} Bright → Bleached transition rate")
            self.state_table.setCellWidget(row, 1, k_bb_spin)
            row_widgets['k_bb'] = k_bb_spin
            
            # Dark → Bright rate
            k_db_spin = QDoubleSpinBox()
            k_db_spin.setRange(0, 1e6)
            k_db_spin.setDecimals(2)
            k_db_spin.setValue(0.0)
            k_db_spin.setToolTip(f"Species {species_num} Dark → Bright transition rate")
            self.state_table.setCellWidget(row, 2, k_db_spin)
            row_widgets['k_db'] = k_db_spin
            
            self.state_row_widgets.append(row_widgets)

    def _set_all_to_zero(self, table_type: str):
        """Set all rates in specified table to 0."""
        if table_type == 'rad' and hasattr(self, 'rad_row_widgets'):
            for row_widgets in self.rad_row_widgets:
                for widget in row_widgets.values():
                    if hasattr(widget, 'setValue'):
                        widget.setValue(0.0)
        elif table_type == 'nrad' and hasattr(self, 'nrad_row_widgets'):
            for row_widgets in self.nrad_row_widgets:
                for widget in row_widgets.values():
                    if hasattr(widget, 'setValue'):
                        widget.setValue(0.0)
        elif table_type == 'state' and hasattr(self, 'state_row_widgets'):
            for row_widgets in self.state_row_widgets:
                for widget in row_widgets.values():
                    if hasattr(widget, 'setValue'):
                        widget.setValue(0.0)

    def _recreate_kinetics_table(self, n_species: int):
        """Recreate all three kinetics tables with new structure based on species count."""
        # Recreate radiative table
        n_rows = n_species  # No background row
        n_cols = n_species
        self.rad_table.setRowCount(n_rows)
        self.rad_table.setColumnCount(n_cols)
        self.rad_table.setHorizontalHeaderLabels([])  # No column labels
        self.rad_table.setVerticalHeaderLabels([f"Species {i+1}" for i in range(n_species)])
        self._populate_rad_table()
        
        # Recreate non-radiative table
        self.nrad_table.setRowCount(n_rows)
        self.nrad_table.setColumnCount(n_cols)
        self.nrad_table.setHorizontalHeaderLabels([])  # No column labels
        self.nrad_table.setVerticalHeaderLabels([])  # No row labels
        self._populate_nrad_table()
        
        # Recreate state conversion table
        self.state_table.setRowCount(n_species)
        self.state_table.setColumnCount(3)  # 3 columns: k_bd, k_bb, k_db
        self.state_table.setHorizontalHeaderLabels([])  # No column labels
        self.state_table.setVerticalHeaderLabels([])  # No row labels
        self._populate_state_table()

    def _populate_kinetics_table(self, table, param_prefix):
        """Populate kinetics table with rate values and enable only valid transitions."""
        n_rows = table.rowCount()
        n_cols = table.columnCount()
        
        # Get parameter values
        rates = self.params.get(param_prefix, [])
        
        for i in range(n_rows):
            for j in range(n_cols):
                spin = QDoubleSpinBox()
                spin.setRange(0, 1e6)
                spin.setDecimals(6)
                spin.setValue(0.0)
                
                # Only enable transitions between different species (off-diagonal)
                # Diagonal elements (i == j) should be disabled
                enabled = (i != j)
                spin.setEnabled(enabled)
                
                # Set value from parameters if available
                try:
                    if i < len(rates) and isinstance(rates[i], (list, tuple)) and j < len(rates[i]):
                        spin.setValue(float(rates[i][j]))
                except (TypeError, IndexError, ValueError):
                    # Use default value if parameters are not properly formatted
                    pass
                
                # Set tooltip
                if enabled:
                    spin.setToolTip(f"Transition rate from species {i+1} to species {j+1}")
                else:
                    spin.setToolTip("Self-transitions are not allowed (diagonal)")
                    spin.setStyleSheet("background-color: #f0f0f0;")
                
                table.setCellWidget(i, j, spin)

    def setup_ui(self):
        """Set up the comprehensive user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tab widget for different parameter categories
        self.tab_widget = QTabWidget()
        self.tab_widget.setContentsMargins(0, 0, 0, 0)

        # CW/Pulsed mode selection + Channel toggles + Options button
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(0)
        mode_layout.addWidget(QLabel("Excitation Mode:"))

        self.excitation_mode_group = QButtonGroup(self)
        self.cw_radio = QRadioButton("CW")
        self.pulsed_radio = QRadioButton("Pulsed")
        self.cw_radio.setChecked(self.params.get('excitation_mode', 'CW') == 'CW')
        self.pulsed_radio.setChecked(self.params.get('excitation_mode', 'CW') == 'Pulsed')

        self.excitation_mode_group.addButton(self.cw_radio)
        self.excitation_mode_group.addButton(self.pulsed_radio)

        mode_layout.addWidget(self.cw_radio)
        mode_layout.addWidget(self.pulsed_radio)

        # Species count
        mode_layout.addSpacing(12)
        mode_layout.addWidget(QLabel("Species:"))
        self.species_spin = QSpinBox()
        self.species_spin.setRange(1, 999)  # Allow up to 999 species
        self.species_spin.setValue(int(self.params.get('N_species', 1)))
        self.species_spin.valueChanged.connect(self._species_count_changed)
        mode_layout.addWidget(self.species_spin)

        # Channel toggles moved to the top bar
        self.green_enabled_check = QCheckBox("Green")
        self.green_enabled_check.setChecked(self.params.get('green_enabled', True))
        self.red_enabled_check = QCheckBox("Red")
        self.red_enabled_check.setChecked(self.params.get('red_enabled', False))
        self.yellow_enabled_check = QCheckBox("Yellow")
        self.yellow_enabled_check.setChecked(self.params.get('yellow_enabled', False))
        self.green_enabled_check.toggled.connect(self._validate_channel_toggle)
        self.green_enabled_check.toggled.connect(self._on_channel_changed)
        self.red_enabled_check.toggled.connect(self._validate_channel_toggle)
        self.red_enabled_check.toggled.connect(self._on_channel_changed)
        self.yellow_enabled_check.toggled.connect(self._validate_channel_toggle)
        self.yellow_enabled_check.toggled.connect(self._on_channel_changed)

        # Ensure at least one channel is enabled initially
        self._validate_initial_channel_state()

        mode_layout.addSpacing(12)
        mode_layout.addWidget(QLabel("Channels:"))
        mode_layout.addWidget(self.green_enabled_check)
        mode_layout.addWidget(self.red_enabled_check)
        mode_layout.addWidget(self.yellow_enabled_check)

        # Floating options dialog button
        mode_layout.addSpacing(12)
        self.options_btn = QPushButton("Options…")
        self.options_btn.clicked.connect(self._open_options_dialog)
        mode_layout.addWidget(self.options_btn)
        
        # Decays visualization button
        self.decays_btn = QPushButton("Decays")
        self.decays_btn.clicked.connect(self._open_decays_dialog)
        mode_layout.addWidget(self.decays_btn)
        mode_layout.addStretch()

        # Add to main layout
        layout.addLayout(mode_layout)
        layout.addWidget(self.tab_widget)

        # Create tabs (Acquisition last)
        self.create_cw_tab()
        self.create_pulsed_tab()
        self.create_kinetics_tab()
        self.create_geometry_tab()
        # self.create_tac_irf_tab()  # Hidden - IRF controls moved to Pulsed Excitation tab

        # Update tab visibility based on mode (safe during construction)
        getattr(self, 'update_mode_visibility', lambda: None)()
        self.cw_radio.toggled.connect(self.update_mode_visibility)
        self.pulsed_radio.toggled.connect(self.update_mode_visibility)

        # Note: Anisotropy controls are now in the Options dialog

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)

        view_json_button = QPushButton("View JSON")
        view_json_button.clicked.connect(self.view_json)
        load_json_button = QPushButton("Load JSON")
        load_json_button.clicked.connect(self.load_json)
        save_json_button = QPushButton("Save JSON")
        save_json_button.clicked.connect(self.save_json)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(view_json_button)
        button_layout.addWidget(load_json_button)
        button_layout.addWidget(save_json_button)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def create_cw_tab(self):
        """Create the CW excitation tab in a table style (C# grid)."""
        cw_widget = QWidget()
        cw_layout = QVBoxLayout(cw_widget)
        cw_layout.setContentsMargins(0, 0, 0, 0)
        cw_layout.setSpacing(0)

        # Species count is now in the top bar next to excitation mode

        # Table columns
        # Abbreviated headers to save width
        self.cw_columns = [
            "ON", "M", "D",
            "GP", "GS", "RP", "RS", "YP", "YS",
            "τD", "Nfcs", "Sg", "Sr", "Sy"
        ]

        n_rows = max(1, int(self.params.get('N_species', 1)))
        self.cw_table = QTableWidget(n_rows, len(self.cw_columns))
        self.cw_table.setHorizontalHeaderLabels(self.cw_columns)
        self.cw_table.verticalHeader().setVisible(False)
        # Make columns fit the available width and avoid horizontal scroll
        hh = self.cw_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        self.cw_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cw_table.setAlternatingRowColors(True)
        self.cw_table.setFrameShape(QFrame.NoFrame)
        self.cw_table.setShowGrid(False)
        vh = self.cw_table.verticalHeader()
        vh.setDefaultSectionSize(22)
        vh.setMinimumSectionSize(18)
        self.cw_table.setContentsMargins(0, 0, 0, 0)
        self.cw_table.setStyleSheet("QTableWidget{padding:0;margin:0;} QTableWidget::item{padding:0;margin:0;}")

        # Populate rows
        self._populate_cw_table()

        cw_layout.addWidget(self.cw_table)

        # Apply current channel enable state immediately
        try:
            getattr(self, '_update_channel_enable_state', lambda: None)()
        except Exception:
            pass

        # Background table (replaces simple background controls)
        bg_group = QGroupBox("Background")
        bg_group_layout = QVBoxLayout(bg_group)
        bg_group_layout.setContentsMargins(2, 2, 2, 2)  # Reduced spacing
        
        # Create background table with no row label
        self.bg_table = QTableWidget()
        self.bg_table.setColumnCount(len(self.cw_columns))
        self.bg_table.setHorizontalHeaderLabels([])  # No column headers like species table
        self.bg_table.horizontalHeader().setVisible(False)  # Hide column headers
        self.bg_table.verticalHeader().setVisible(False)  # Hide row header
        self.bg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bg_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Set initial row count (always 1 for background)
        self.bg_table.setRowCount(1)
        self.bg_table.setVerticalHeaderLabels([])
        
        # Populate background table
        self._populate_background_table()
        bg_group_layout.addWidget(self.bg_table)
        
        cw_layout.addWidget(bg_group)

        self.tab_widget.addTab(cw_widget, "CW Excitation")

    
    def _populate_background_table(self):
        """Populate the background table with disabled M, D, tauD, and Nfcs columns."""
        self.bg_row_widgets = []
        
        # Single background row
        row_widgets = {}
        
        # On checkbox (enabled)
        on_check = QCheckBox()
        on_check.setChecked(True)
        on_check.setToolTip("Background enabled")
        self.bg_table.setCellWidget(0, 0, on_check)
        row_widgets['on'] = on_check
        
        # M (disabled)
        m_spin = QDoubleSpinBox()
        m_spin.setRange(0, 1e6)
        m_spin.setDecimals(2)
        m_spin.setValue(0.0)
        m_spin.setEnabled(False)  # Disabled as requested
        m_spin.setStyleSheet("QDoubleSpinBox:disabled { color: gray; }")
        m_spin.setToolTip("Background brightness (disabled)")
        self.bg_table.setCellWidget(0, 1, m_spin)
        row_widgets['M'] = m_spin
        
        # D (disabled)
        d_spin = QDoubleSpinBox()
        d_spin.setRange(0, 1e6)
        d_spin.setDecimals(2)
        d_spin.setValue(0.0)
        d_spin.setEnabled(False)  # Disabled as requested
        d_spin.setStyleSheet("QDoubleSpinBox:disabled { color: gray; }")
        d_spin.setToolTip("Background diffusion (disabled)")
        self.bg_table.setCellWidget(0, 2, d_spin)
        row_widgets['D'] = d_spin
        
        # Channel brightness controls (enabled)
        col_idx = 3
        for channel in ['G', 'R', 'Y']:
            for pol in ['P', 'S']:
                spin = QDoubleSpinBox()
                spin.setRange(0, 1e6)
                spin.setDecimals(2)
                spin.setValue(0.0)
                spin.setToolTip(f"Background {channel}{pol} brightness")
                self.bg_table.setCellWidget(0, col_idx, spin)
                row_widgets[f'{channel}{pol}'] = spin
                col_idx += 1
        
        # tauD (disabled)
        taud_spin = QDoubleSpinBox()
        taud_spin.setRange(0, 1000)
        taud_spin.setDecimals(2)
        taud_spin.setValue(0.0)
        taud_spin.setEnabled(False)  # Disabled as requested
        taud_spin.setStyleSheet("QDoubleSpinBox:disabled { color: gray; }")
        taud_spin.setToolTip("Background tauD (disabled)")
        self.bg_table.setCellWidget(0, col_idx, taud_spin)
        row_widgets['tauD'] = taud_spin
        col_idx += 1
        
        # Nfcs (disabled)
        nfcs_spin = QDoubleSpinBox()
        nfcs_spin.setRange(0, 1e6)
        nfcs_spin.setDecimals(0)
        nfcs_spin.setValue(0.0)
        nfcs_spin.setEnabled(False)  # Disabled as requested
        nfcs_spin.setStyleSheet("QDoubleSpinBox:disabled { color: gray; }")
        nfcs_spin.setToolTip("Background Nfcs (disabled)")
        self.bg_table.setCellWidget(0, col_idx, nfcs_spin)
        row_widgets['Nfcs'] = nfcs_spin
        
        self.bg_row_widgets.append(row_widgets)

    def _species_count_changed(self, value):
        """Resize CW and Pulsed tables when species count changes, preserving existing values."""
        # Ensure value is always an integer
        value = int(float(value))
        value = max(1, value)  # Ensure at least 1 species, no upper limit
        # Snapshot CW
        old_cw = getattr(self, 'cw_row_widgets', [])
        saved_cw = []
        for rw in old_cw:
            saved_cw.append({
                'on': bool(rw['on'].isChecked()),
                'M': float(rw['M'].value()),
                'D': float(rw['D'].value()),
                'GP': float(rw['G_P'].value()), 'GS': float(rw['G_S'].value()),
                'RP': float(rw['R_P'].value()), 'RS': float(rw['R_S'].value()),
                'YP': float(rw['Y_P'].value()), 'YS': float(rw['Y_S'].value()),
                'Nfcs': float(rw['Nfcs'].value()),
            })

        # Snapshot Pulsed
        old_p = getattr(self, 'pulsed_row_widgets', [])
        saved_p = []
        for rw in old_p:
            saved_p.append({
                'tau_g': float(rw['tau_g'].value()), 'rho_g': float(rw['rho_g'].value()),
                'tau_r': float(rw['tau_r'].value()), 'rho_r': float(rw['rho_r'].value()),
                'tau_y': float(rw['tau_y'].value()), 'rho_y': float(rw['rho_y'].value()),
                'pattern': rw['pattern'].text() if 'pattern' in rw else "",
            })

        # Snapshot Kinetics - both tables
        old_rad = getattr(self, 'rad_row_widgets', [])
        saved_rad = []
        for rw in old_rad:
            row_data = {}
            for key, widget in rw.items():
                if hasattr(widget, 'value'):  # SpinBox
                    row_data[key] = float(widget.value())
            saved_rad.append(row_data)
        
        old_nrad = getattr(self, 'nrad_row_widgets', [])
        saved_nrad = []
        for rw in old_nrad:
            row_data = {}
            for key, widget in rw.items():
                if hasattr(widget, 'value'):  # SpinBox
                    row_data[key] = float(widget.value())
            saved_nrad.append(row_data)
        
        # Snapshot state conversion table
        old_state = getattr(self, 'state_row_widgets', [])
        saved_state = []
        for rw in old_state:
            row_data = {}
            for key, widget in rw.items():
                if hasattr(widget, 'value'):  # SpinBox
                    row_data[key] = float(widget.value())
            saved_state.append(row_data)

        # Apply new row count to CW
        if hasattr(self, 'cw_table'):
            self.cw_table.setRowCount(value)
            self._populate_cw_table()
            for i in range(min(int(value), len(saved_cw))):
                rw = self.cw_row_widgets[i]
                sv = saved_cw[i]
                rw['on'].setChecked(sv['on'])
                rw['M'].setValue(sv['M']); rw['D'].setValue(sv['D'])
                rw['G_P'].setValue(sv['GP']); rw['G_S'].setValue(sv['GS'])
                rw['R_P'].setValue(sv['RP']); rw['R_S'].setValue(sv['RS'])
                rw['Y_P'].setValue(sv['YP']); rw['Y_S'].setValue(sv['YS'])
                rw['Nfcs'].setValue(sv['Nfcs'])
                # Update derived values
                self._update_tauD_row(i)
                self._update_S_totals_row(i)

        # Apply new row count to Pulsed
        if hasattr(self, 'pulsed_table'):
            self.pulsed_table.setRowCount(value)
            self._populate_pulsed_table()
            for i in range(min(int(value), len(saved_p))):
                rw = self.pulsed_row_widgets[i]
                sv = saved_p[i]
                rw['tau_g'].setValue(sv['tau_g']); rw['rho_g'].setValue(sv['rho_g'])
                rw['tau_r'].setValue(sv['tau_r']); rw['rho_r'].setValue(sv['rho_r'])
                rw['tau_y'].setValue(sv['tau_y']); rw['rho_y'].setValue(sv['rho_y'])
                if 'pattern' in rw:
                    rw['pattern'].setText(sv['pattern'])

        # Background table always has 1 row, no need to resize

        # Apply new structure to Kinetics tables
        if hasattr(self, 'rad_table') and hasattr(self, 'nrad_table') and hasattr(self, 'state_table'):
            self._recreate_kinetics_table(value)
            # Restore radiative values where possible
            for i in range(min(int(value), len(saved_rad))):  # No background row
                if i < len(self.rad_row_widgets):
                    rw = self.rad_row_widgets[i]
                    sv = saved_rad[i]
                    for key, val in sv.items():
                        if key in rw and hasattr(rw[key], 'value'):
                            rw[key].setValue(float(val))
            # Restore non-radiative values where possible
            for i in range(min(int(value), len(saved_nrad))):  # No background row
                if i < len(self.nrad_row_widgets):
                    rw = self.nrad_row_widgets[i]
                    sv = saved_nrad[i]
                    for key, val in sv.items():
                        if key in rw and hasattr(rw[key], 'value'):
                            rw[key].setValue(float(val))
            # Restore state conversion values where possible
            for i in range(min(int(value), len(saved_state))):  # No background row for state table
                if i < len(self.state_row_widgets):
                    rw = self.state_row_widgets[i]
                    sv = saved_state[i]
                    for key, val in sv.items():
                        if key in rw and hasattr(rw[key], 'value'):
                            rw[key].setValue(float(val))

        # Update cached and keep in sync
        self.params['N_species'] = value
        # Re-apply channel enable state after resizing
        try:
            self._update_channel_enable_state()
        except Exception:
            pass

    def _populate_cw_table(self):
        """Fill the CW table with controls and bind updates."""
        self.cw_row_widgets = []
        n_rows = self.cw_table.rowCount()

        # helper to get brightness defaults from params.q vector for 6 channels
        q = list(self.params.get('q', []))
        def q_at(si, ch_idx):
            idx = si * 6 + ch_idx
            return q[idx] if idx < len(q) else 0.0

        for row in range(n_rows):
            row_widgets = {}

            # ON
            on = QCheckBox()
            on.setChecked(True)
            self.cw_table.setCellWidget(row, 0, on)
            row_widgets['on'] = on

            # Number of molecules (M)
            m = QDoubleSpinBox()
            m.setRange(0, 10000)
            m.setValue(self.params['M'][row] if row < len(self.params['M']) else 5.0)
            self.cw_table.setCellWidget(row, 1, m)
            row_widgets['M'] = m

            # Diffusion coef (D)
            d = QDoubleSpinBox()
            d.setRange(0, 100)
            d.setDecimals(4)
            d.setValue(self.params['D'][row] if row < len(self.params['D']) else 3.0)
            self.cw_table.setCellWidget(row, 2, d)
            row_widgets['D'] = d

            # Brightness G/R/Y P,S (6 channels)
            g_p = QDoubleSpinBox(); g_p.setRange(0, 1000); g_p.setValue(q_at(row, 0))
            g_s = QDoubleSpinBox(); g_s.setRange(0, 1000); g_s.setValue(q_at(row, 1))
            r_p = QDoubleSpinBox(); r_p.setRange(0, 1000); r_p.setValue(q_at(row, 2))
            r_s = QDoubleSpinBox(); r_s.setRange(0, 1000); r_s.setValue(q_at(row, 3))
            y_p = QDoubleSpinBox(); y_p.setRange(0, 1000); y_p.setValue(q_at(row, 4))
            y_s = QDoubleSpinBox(); y_s.setRange(0, 1000); y_s.setValue(q_at(row, 5))
            for col, w in enumerate([g_p, g_s, r_p, r_s, y_p, y_s], start=3):
                self.cw_table.setCellWidget(row, col, w)
            row_widgets.update({'G_P': g_p, 'G_S': g_s, 'R_P': r_p, 'R_S': r_s, 'Y_P': y_p, 'Y_S': y_s})

            # tauD (editable spin), Nfcs (spin), S totals (labels)
            tau = QDoubleSpinBox(); tau.setRange(0.0, 1e9); tau.setDecimals(3)
            tau.setToolTip("tauD = w0^2 / (4D)")
            self.cw_table.setCellWidget(row, 9, tau)
            row_widgets['tau'] = tau

            nfcs = QDoubleSpinBox(); nfcs.setRange(0, 1e6); nfcs.setDecimals(3); nfcs.setValue(1.0)
            self.cw_table.setCellWidget(row, 10, nfcs)
            row_widgets['Nfcs'] = nfcs

            sg = QLabel("0"); sr = QLabel("0"); sy = QLabel("0")
            self.cw_table.setCellWidget(row, 11, sg)
            self.cw_table.setCellWidget(row, 12, sr)
            self.cw_table.setCellWidget(row, 13, sy)
            row_widgets.update({'Sg': sg, 'Sr': sr, 'Sy': sy})

            # Bind updates
            d.valueChanged.connect(lambda _=0, r=row: self._update_tauD_row(r))
            tau.valueChanged.connect(lambda _=0, r=row: self._update_D_from_tau_row(r))
            # Couple M <-> Nfcs using volume factor
            m.valueChanged.connect(lambda _=0, r=row: self._update_Nfcs_from_M_row(r))
            nfcs.valueChanged.connect(lambda _=0, r=row: self._update_M_from_Nfcs_row(r))
            for w in [g_p, g_s]:
                w.valueChanged.connect(lambda _=0, r=row: self._update_S_totals_row(r))
            for w in [r_p, r_s, y_p, y_s]:
                w.valueChanged.connect(lambda _=0, r=row: self._update_S_totals_row(r))

            self.cw_row_widgets.append(row_widgets)

            # Initial calc
            self._update_tauD_row(row)
            self._update_Nfcs_from_M_row(row)
            self._update_S_totals_row(row)

    def _update_tauD_row(self, row: int):
        if row < 0 or row >= len(self.cw_row_widgets):
            return
        w0 = 0.0
        try:
            if self.focus_param_spins:
                w0 = float(self.focus_param_spins[0].value())
        except Exception:
            w0 = 0.0
        D = max(1e-12, float(self.cw_row_widgets[row]['D'].value()))
        tau_val = (w0 * w0) / (4.0 * D) if w0 > 0 else 0.0
        sp: QDoubleSpinBox = self.cw_row_widgets[row]['tau']
        sp.blockSignals(True)
        sp.setValue(tau_val)
        sp.blockSignals(False)

    def _update_D_from_tau_row(self, row: int):
        if row < 0 or row >= len(self.cw_row_widgets):
            return
        try:
            w0 = float(self.focus_param_spins[0].value()) if self.focus_param_spins else 0.0
        except Exception:
            w0 = 0.0
        tau_val = float(self.cw_row_widgets[row]['tau'].value())
        if w0 <= 0 or tau_val <= 0:
            return
        D = (w0 * w0) / (4.0 * tau_val)
        dspin: QDoubleSpinBox = self.cw_row_widgets[row]['D']
        dspin.blockSignals(True)
        dspin.setValue(D)
        dspin.blockSignals(False)

    def _volume_factor(self) -> float:
        """Compute conversion factor between M and Nfcs as in C# approximation.
        Nfcs = M / C, where C ~ 4 * box_size^3 / (3 * sqrt(pi)).
        Here we use box_xy as the size proxy to match the single-size model.
        """
        try:
            box_size = float(self.box_xy_spin.value()) if hasattr(self, 'box_xy_spin') else 1.0
        except Exception:
            box_size = 1.0
        return 4.0 * (box_size ** 3) / 3.0 / (math.pi ** 0.5)

    def _update_Nfcs_from_M_row(self, row: int):
        if row < 0 or row >= len(self.cw_row_widgets):
            return
        C = max(1e-12, self._volume_factor())
        mval = float(self.cw_row_widgets[row]['M'].value())
        nf = mval / C
        spin: QDoubleSpinBox = self.cw_row_widgets[row]['Nfcs']
        spin.blockSignals(True)
        spin.setValue(nf)
        spin.blockSignals(False)

    def _update_M_from_Nfcs_row(self, row: int):
        if row < 0 or row >= len(self.cw_row_widgets):
            return
        C = max(1e-12, self._volume_factor())
        nf = float(self.cw_row_widgets[row]['Nfcs'].value())
        mval = nf * C
        spin: QDoubleSpinBox = self.cw_row_widgets[row]['M']
        spin.blockSignals(True)
        spin.setValue(mval)
        spin.blockSignals(False)

    def _update_S_totals_row(self, row: int):
        if row < 0 or row >= len(self.cw_row_widgets):
            return
        rw = self.cw_row_widgets[row]
        sg = rw['G_P'].value() + rw['G_S'].value()
        sr = rw['R_P'].value() + rw['R_S'].value()
        sy = rw['Y_P'].value() + rw['Y_S'].value()
        rw['Sg'].setText(f"{sg:.1f}")
        rw['Sr'].setText(f"{sr:.1f}")
        rw['Sy'].setText(f"{sy:.1f}")

    def create_pulsed_tab(self):
        """Create the Pulsed excitation tab."""
        pulsed_widget = QWidget()
        pulsed_layout = QHBoxLayout(pulsed_widget)
        pulsed_layout.setContentsMargins(0, 0, 0, 0)
        pulsed_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(True)

        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)

        # Pulsed species table (tau, rho per color OR pattern file)
        # tau = lifetime, rho = rotational correlation time
        self.pulsed_columns = [
            "TauG", "RhoG", "TauR", "RhoR", "TauY", "RhoY", "Pattern", "..."
        ]
        n_rows = max(1, int(self.params.get('N_species', 1)))
        self.pulsed_table = QTableWidget(n_rows, len(self.pulsed_columns))
        self.pulsed_table.setHorizontalHeaderLabels(self.pulsed_columns)
        self.pulsed_table.verticalHeader().setVisible(False)
        self.pulsed_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pulsed_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._populate_pulsed_table()
        left.addWidget(self.pulsed_table)

        # Apply current channel enable state immediately
        try:
            self._update_channel_enable_state()
        except Exception:
            pass

        # Pulsed options on the right
        right_widget = QWidget()
        right_widget.setMinimumWidth(260)
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(4, 0, 0, 0)
        right.setSpacing(2)

        opt_group = QGroupBox("Pulsed Options")
        opt_form = QFormLayout(opt_group)
        opt_form.setContentsMargins(4, 4, 4, 4)
        opt_form.setSpacing(2)

        self.back_dark_check = QCheckBox("Background: Dark counts")
        self.back_dark_check.setChecked(self.params.get('background_dark', True))
        opt_form.addRow(self.back_dark_check)

        self.convolve_loaded_check = QCheckBox("Convolute loaded patterns")
        self.convolve_loaded_check.setChecked(self.params.get('convolve_loaded', True))
        opt_form.addRow(self.convolve_loaded_check)

        self.tau_rho_override_check = QCheckBox("tau and rho override brightness")
        self.tau_rho_override_check.setChecked(self.params.get('tau_rho_override', False))
        opt_form.addRow(self.tau_rho_override_check)

        right.addWidget(opt_group)
        
        # Add IRF groupbox to pulsed excitation tab
        self.create_irf_groupbox()
        right.addWidget(self.irf_group)
        
        right.addStretch()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        pulsed_layout.addWidget(splitter)

        self.tab_widget.addTab(pulsed_widget, "Pulsed Excitation")

    def _populate_pulsed_table(self):
        self.pulsed_row_widgets = []
        n_rows = self.pulsed_table.rowCount()
        lifetime = self.params.get('decay_lifetimes', [])  # per species [tauG, tauR, tauY]
        rho_rot = self.params.get('rotational_correlation_times', [])  # per species [rhoG, rhoR, rhoY] - rotational times
        patt = self.params.get('decay_patterns', [])      # per species pattern path or ''

        # Get r0 from anisotropy options for amplitude
        r0 = self.params.get('r0', 0.38)
        
        for i in range(n_rows):
            rw = {}
            
            # tau/rho per color
            def lif(i, j):
                try:
                    return float(lifetime[i][j])
                except Exception:
                    return 5.0
            def rho_rot_val(i, j):
                try:
                    return float(rho_rot[i][j])  # rho values are rotational correlation times
                except Exception:
                    return 0.4  # Default 0.4 ns

            tau_g = QDoubleSpinBox(); tau_g.setRange(0.001, 1e6); tau_g.setDecimals(3); tau_g.setValue(lif(i, 0))
            rho_g = QDoubleSpinBox(); rho_g.setRange(0.001, 1e6); rho_g.setDecimals(3); rho_g.setValue(rho_rot_val(i, 0))  # rotational time
            tau_r = QDoubleSpinBox(); tau_r.setRange(0.001, 1e6); tau_r.setDecimals(3); tau_r.setValue(lif(i, 1))
            rho_r = QDoubleSpinBox(); rho_r.setRange(0.001, 1e6); rho_r.setDecimals(3); rho_r.setValue(rho_rot_val(i, 1))  # rotational time
            tau_y = QDoubleSpinBox(); tau_y.setRange(0.001, 1e6); tau_y.setDecimals(3); tau_y.setValue(lif(i, 2))
            rho_y = QDoubleSpinBox(); rho_y.setRange(0.001, 1e6); rho_y.setDecimals(3); rho_y.setValue(rho_rot_val(i, 2))  # rotational time

            for col, w in enumerate([tau_g, rho_g, tau_r, rho_r, tau_y, rho_y], start=0):
                self.pulsed_table.setCellWidget(i, col, w)

            rw.update({'tau_g': tau_g, 'rho_g': rho_g, 'tau_r': tau_r, 'rho_r': rho_r, 'tau_y': tau_y, 'rho_y': rho_y})

            # Connect valueChanged signals to trigger decay dialog refresh
            for widget in [tau_g, rho_g, tau_r, rho_r, tau_y, rho_y]:
                widget.valueChanged.connect(self._on_pulsed_table_changed)

            # Pattern path + browse
            le = QLineEdit(); le.setText(patt[i] if i < len(patt) else "")
            self.pulsed_table.setCellWidget(i, 6, le); rw['pattern'] = le
            btn = QPushButton("…")
            def mk_browse(row=i):
                return lambda: self._browse_pattern_for_row(row)
            btn.clicked.connect(mk_browse(i))
            self.pulsed_table.setCellWidget(i, 7, btn); rw['browse'] = btn

            self.pulsed_row_widgets.append(rw)

    def _browse_pattern_for_row(self, row: int):
        filename, _ = QFileDialog.getOpenFileName(self, "Select decay pattern", "", "Text files (*.txt *.dat);;All files (*)")
        if filename and 0 <= row < len(self.pulsed_row_widgets):
            self.pulsed_row_widgets[row]['pattern'].setText(filename)

    def create_geometry_tab(self):
        """Create the acquisition/focus tab."""
        geom_widget = QWidget()
        geom_layout = QVBoxLayout(geom_widget)

        # Focus parameters
        focus_group = QGroupBox("Focus Parameters")
        focus_layout = QFormLayout(focus_group)

        self.focus_type_combo = QComboBox()
        for name in self.FOCUS_TYPES.keys():
            self.focus_type_combo.addItem(name)
        self.focus_type_combo.setCurrentText("3D Gaussian, uniform CEF")
        focus_layout.addRow("Focus type:", self.focus_type_combo)
        self.focus_type_combo.currentIndexChanged.connect(self._update_focus_param_visibility)

        # Focus parameters (dynamic based on type)
        self.focus_param_widget = QWidget()
        self.focus_param_layout = QFormLayout(self.focus_param_widget)
        self.focus_param_spins = []
        self.focus_param_labels = []
        
        # Create all possible parameter controls (will show/hide based on focus type)
        focus_params = self.params.get('focus_param', [0.3, 2.0, 0.1, 0.5, 0.05, 1.0])
        
        # w0 - beam waist (all focus types)
        self.w0_spin = QDoubleSpinBox()
        self.w0_spin.setRange(0.0, 10.0)
        self.w0_spin.setDecimals(4)
        self.w0_spin.setValue(focus_params[0] if len(focus_params) > 0 else 0.3)
        self.focus_param_spins.append(self.w0_spin)
        
        # z0 - axial parameter (most focus types)
        self.z0_spin = QDoubleSpinBox()
        self.z0_spin.setRange(0.0, 50.0)
        self.z0_spin.setDecimals(4)
        self.z0_spin.setValue(focus_params[1] if len(focus_params) > 1 else 2.0)
        self.focus_param_spins.append(self.z0_spin)
        
        # res0 - lens resolution (Gaussian-Lorentzian types)
        self.res0_spin = QDoubleSpinBox()
        self.res0_spin.setRange(0.0, 1.0)
        self.res0_spin.setDecimals(4)
        self.res0_spin.setValue(focus_params[2] if len(focus_params) > 2 else 0.1)
        self.focus_param_spins.append(self.res0_spin)
        
        # tgalpha - tan of aperture half-angle (Gaussian-Lorentzian types)
        self.tgalpha_spin = QDoubleSpinBox()
        self.tgalpha_spin.setRange(0.0, 2.0)
        self.tgalpha_spin.setDecimals(4)
        self.tgalpha_spin.setValue(focus_params[3] if len(focus_params) > 3 else 0.5)
        self.focus_param_spins.append(self.tgalpha_spin)
        
        # Rph - pinhole radius (Gaussian-Lorentzian types)
        self.rph_spin = QDoubleSpinBox()
        self.rph_spin.setRange(0.0, 1.0)
        self.rph_spin.setDecimals(4)
        self.rph_spin.setValue(focus_params[4] if len(focus_params) > 4 else 0.05)
        self.focus_param_spins.append(self.rph_spin)
        
        # z0_CEF - cutoff for cylindrical CEF (Gaussian-Lorentzian cylindrical)
        self.z0_cef_spin = QDoubleSpinBox()
        self.z0_cef_spin.setRange(0.0, 50.0)
        self.z0_cef_spin.setDecimals(4)
        self.z0_cef_spin.setValue(focus_params[5] if len(focus_params) > 5 else 1.0)
        self.focus_param_spins.append(self.z0_cef_spin)
        
        # Add all controls to layout (visibility will be managed by _update_focus_param_visibility)
        self.focus_param_layout.addRow("w0 (μm):", self.w0_spin)
        self.focus_param_layout.addRow("z0 (μm):", self.z0_spin)
        self.focus_param_layout.addRow("res0 (μm):", self.res0_spin)
        self.focus_param_layout.addRow("tan(α):", self.tgalpha_spin)
        self.focus_param_layout.addRow("Rph (μm):", self.rph_spin)
        self.focus_param_layout.addRow("z0_CEF (μm):", self.z0_cef_spin)

        focus_layout.addRow(self.focus_param_widget)
        
        # Initialize focus parameter visibility
        self._update_focus_param_visibility()
        
        geom_layout.addWidget(focus_group)

        # Simulation box
        box_group = QGroupBox("Simulation Volume")
        box_layout = QFormLayout(box_group)

        self.box_xy_spin = QDoubleSpinBox()
        self.box_xy_spin.setRange(0.1, 10.0)
        self.box_xy_spin.setValue(self.params['box_xy'])
        box_layout.addRow("Lateral size (μm):", self.box_xy_spin)

        self.box_z_spin = QDoubleSpinBox()
        self.box_z_spin.setRange(0.1, 10.0)
        self.box_z_spin.setValue(self.params['box_z'])
        box_layout.addRow("Axial size (μm):", self.box_z_spin)

        geom_layout.addWidget(box_group)

        # Simulation parameters
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QFormLayout(sim_group)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 10.0)
        self.dt_spin.setValue(self.params['dt'])
        sim_layout.addRow("Time step (μs):", self.dt_spin)

        self.n_ph_max_spin = QSpinBox()
        self.n_ph_max_spin.setRange(1000, 10000000)
        self.n_ph_max_spin.setValue(self.params['N_ph_max'])
        sim_layout.addRow("Max photons:", self.n_ph_max_spin)

        geom_layout.addWidget(sim_group)

        # Hidden widgets that are controlled by Options dialog
        self.output_path_edit = QLineEdit(self.params.get('spc_output_path', ''))
        self.output_path_edit.setVisible(False)  # Hidden since it's controlled in Options
        
        # TAC controls for Options dialog to reference
        self.tac_controls = TACControls(self.params)
        # Note: TAC controls are now visible in their own tab

        geom_layout.addStretch()
        self.tab_widget.addTab(geom_widget, "Acquisition")


    def _on_irf_changed(self):
        """Handle IRF control changes and trigger decay dialog refresh."""
        try:
            # Refresh the decay dialog if it exists and is visible
            if hasattr(self, 'decay_dialog') and self.decay_dialog and self.decay_dialog.isVisible():
                self.decay_dialog.refresh_data()
        except Exception as e:
            # Don't show error dialogs for background refresh issues
            print(f"Warning: Could not refresh decay dialog: {e}")

    def _on_anisotropy_changed(self):
        """Handle anisotropy control changes and trigger decay dialog refresh."""
        try:
            # Refresh the decay dialog if it exists and is visible
            if hasattr(self, 'decay_dialog') and self.decay_dialog and self.decay_dialog.isVisible():
                self.decay_dialog.refresh_data()
        except Exception as e:
            # Don't show error dialogs for background refresh issues
            print(f"Warning: Could not refresh decay dialog: {e}")

    def create_irf_groupbox(self):
        """Create the IRF groupbox for the pulsed excitation tab."""
        self.irf_group = QGroupBox("Instrument Response Function")
        irf_layout = QFormLayout(self.irf_group)

        self.use_gaussian_irf_check = QCheckBox("Use Gaussian IRF")
        self.use_gaussian_irf_check.setChecked(self.params.get('use_gaussian_irf', True))
        irf_layout.addRow(self.use_gaussian_irf_check)

        self.gaussian_fwhm_spin = QDoubleSpinBox()
        self.gaussian_fwhm_spin.setRange(0, 10)
        self.gaussian_fwhm_spin.setValue(self.params.get('gaussian_irf_fwhm', 0.11))
        self.gaussian_fwhm_spin.setEnabled(True)  # Enabled by default since IRF is enabled
        self.use_gaussian_irf_check.toggled.connect(lambda: self.gaussian_fwhm_spin.setEnabled(self.use_gaussian_irf_check.isChecked()))
        irf_layout.addRow("Gaussian FWHM (ns):", self.gaussian_fwhm_spin)

        # Gaussian IRF position (mean)
        self.gaussian_mean_spin = QDoubleSpinBox()
        self.gaussian_mean_spin.setRange(-100, 100)
        self.gaussian_mean_spin.setDecimals(3)
        self.gaussian_mean_spin.setValue(self.params.get('gaussian_irf_mean', 1.5))
        self.gaussian_mean_spin.setEnabled(True)  # Enabled by default since IRF is enabled
        self.use_gaussian_irf_check.toggled.connect(lambda: self.gaussian_mean_spin.setEnabled(self.use_gaussian_irf_check.isChecked()))
        irf_layout.addRow("Gaussian mean (ns):", self.gaussian_mean_spin)

        # Note: Gaussian σ is automatically calculated from FWHM internally

        # IRF file selection with label and browse button
        irf_file_widget = QWidget()
        irf_file_layout = QHBoxLayout(irf_file_widget)
        irf_file_layout.setContentsMargins(0, 0, 0, 0)
        
        self.irf_file_label = QLabel("No file selected")
        irf_file = self.params.get('irf_file', '')
        self._irf_file_path = irf_file  # Store full path
        if irf_file:
            import os
            self.irf_file_label.setText(f"IRF: {os.path.basename(irf_file)}")
        self.irf_file_label.setEnabled(False)
        
        self.browse_irf_button = QToolButton()
        self.browse_irf_button.setText("...")
        self.browse_irf_button.setToolTip("Select IRF file")
        self.browse_irf_button.clicked.connect(self.browse_irf)
        self.browse_irf_button.setEnabled(False)
        
        irf_file_layout.addWidget(self.irf_file_label)
        irf_file_layout.addWidget(self.browse_irf_button)
        irf_file_layout.addStretch()
        
        irf_layout.addRow("IRF file:", irf_file_widget)

        # Connect IRF options
        def update_irf_mode():
            use_file = not self.use_gaussian_irf_check.isChecked()
            self.irf_file_label.setEnabled(use_file)
            self.browse_irf_button.setEnabled(use_file)
            # Don't auto-browse for IRF file on initialization

        self.use_gaussian_irf_check.toggled.connect(update_irf_mode)
        update_irf_mode()  # Initialize state
        
        # Connect IRF controls to decay dialog refresh
        self.use_gaussian_irf_check.toggled.connect(self._on_irf_changed)
        self.gaussian_fwhm_spin.valueChanged.connect(self._on_irf_changed)
        self.gaussian_mean_spin.valueChanged.connect(self._on_irf_changed)

    def browse_irf(self):
        """Browse for IRF file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select IRF file", "", "Text files (*.txt);;All files (*)"
        )
        if filename:
            import os
            self.irf_file_label.setText(f"IRF: {os.path.basename(filename)}")
            # Store the full path for parameter collection
            self._irf_file_path = filename
        else:
            self.irf_file_label.setText("No file selected")
            self._irf_file_path = ""

    def create_tac_irf_tab(self):
        """Create the TAC/IRF tab."""
        self.tab_widget.addTab(self.tac_controls, "TAC/IRF")

    def _update_focus_param_visibility(self):
        """Show/hide focus params depending on focus type selection."""
        if not hasattr(self, 'focus_type_combo'):
            return
            
        focus_type_name = self.focus_type_combo.currentText()
        focus_type_id = self.FOCUS_TYPES.get(focus_type_name, 0)
        
        # Hide all parameters and their labels first
        controls_and_labels = [
            (self.w0_spin, "w0 (μm):"),
            (self.z0_spin, "z0 (μm):"),
            (self.res0_spin, "res0 (μm):"),
            (self.tgalpha_spin, "tan(α):"),
            (self.rph_spin, "Rph (μm):"),
            (self.z0_cef_spin, "z0_CEF (μm):")
        ]
        
        for control, _ in controls_and_labels:
            control.setVisible(False)
            label = self.focus_param_layout.labelForField(control)
            if label:
                label.setVisible(False)
        
        # Show parameters based on focus type
        if focus_type_id == 0:  # "3D Gaussian, uniform CEF"
            self._show_focus_param(self.w0_spin, "w0 (μm):")
            self._show_focus_param(self.z0_spin, "z0 (μm):")
            
        elif focus_type_id == 1:  # "3D Gaussian excitation and CEF"
            self._show_focus_param(self.w0_spin, "w0 (μm):")
            self._show_focus_param(self.z0_spin, "z0 (μm):")
            
        elif focus_type_id == 2:  # "Rectangular excitation, uniform CEF"
            self._show_focus_param(self.w0_spin, "Half-width X,Y (μm):")
            self._show_focus_param(self.z0_spin, "Half-width Z (μm):")
            
        elif focus_type_id == 3:  # "Cylindrical excitation, uniform CEF"
            self._show_focus_param(self.w0_spin, "Radius (μm):")
            self._show_focus_param(self.z0_spin, "Half-height Z (μm):")
            
        elif focus_type_id == 4:  # "Gaussian-Lorentzian excitation, pinhole CEF"
            self._show_focus_param(self.w0_spin, "w0 (μm):")
            self._show_focus_param(self.z0_spin, "z0 (μm):")
            self._show_focus_param(self.res0_spin, "Lens resolution (μm):")
            self._show_focus_param(self.tgalpha_spin, "tan(aperture α):")
            self._show_focus_param(self.rph_spin, "Pinhole radius (μm):")
            
        elif focus_type_id == 5:  # "Gaussian-Lorentzian excitation, cylindrical CEF"
            self._show_focus_param(self.w0_spin, "w0 (μm):")
            self._show_focus_param(self.z0_spin, "z0 (μm):")
            self._show_focus_param(self.res0_spin, "Lens resolution (μm):")
            self._show_focus_param(self.tgalpha_spin, "tan(aperture α):")
            self._show_focus_param(self.rph_spin, "CEF radius (μm):")
            self._show_focus_param(self.z0_cef_spin, "CEF z-cutoff (μm):")
            
        elif focus_type_id == 6:  # "Flow excitation"
            self._show_focus_param(self.w0_spin, "Flow width (μm):")
            self._show_focus_param(self.z0_spin, "Flow height (μm):")

    def _show_focus_param(self, control, label_text):
        """Helper method to show a focus parameter control and update its label."""
        control.setVisible(True)
        label = self.focus_param_layout.labelForField(control)
        if label:
            label.setText(label_text)
            label.setVisible(True)

    def browse_output(self):
        """Browse for output directory and set it to output_path_edit."""
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path and hasattr(self, 'output_path_edit'):
            self.output_path_edit.setText(path)

    # Floating Options dialog
    def _open_options_dialog(self):
        dlg = OptionsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            # Values already pushed via Apply; nothing else to do
            pass
    
    # Decays visualization dialog
    def _open_decays_dialog(self):
        """Open the fluorescence decay visualization dialog."""
        try:
            self.decay_dialog = DecayVisualizationDialog(self)
            self.decay_dialog.show()  # Use show() instead of exec_() for non-modal
        except Exception as e:
            QMessageBox.warning(self, "Decay Visualization Error", 
                              f"Could not open decay visualization: {e}")

    def _on_pulsed_table_changed(self):
        """Handle pulsed table value changes and trigger decay dialog refresh."""
        try:
            # Refresh the decay dialog if it exists and is visible
            if hasattr(self, 'decay_dialog') and self.decay_dialog and self.decay_dialog.isVisible():
                self.decay_dialog.refresh_data()
        except Exception as e:
            # Don't show error dialogs for background refresh issues
            print(f"Warning: Could not refresh decay dialog: {e}")

    def _on_channel_changed(self):
        """Handle channel checkbox changes and trigger decay dialog refresh."""
        try:
            # Refresh the decay dialog if it exists and is visible
            if hasattr(self, 'decay_dialog') and self.decay_dialog and self.decay_dialog.isVisible():
                self.decay_dialog.refresh_data()
        except Exception as e:
            # Don't show error dialogs for background refresh issues
            print(f"Warning: Could not refresh decay dialog: {e}")

    def view_json(self):
        """Show current parameters as formatted JSON for quick inspection."""
        try:
            json_str = self.to_json()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to serialize parameters: {e}")
            return

        preview_dialog = QDialog()
        preview_dialog.setWindowTitle("Current Simulation Parameters (JSON)")
        preview_dialog.resize(700, 500)
        preview_dialog.setModal(True)

        layout = QVBoxLayout(preview_dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(json_str)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(preview_dialog.accept)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        preview_dialog.exec_()

    def to_json(self):
        """Serialize current parameters to JSON string."""
        params = self.get_parameters()
        return json.dumps(params, indent=2)

    def load_json(self):
        """Load parameters from JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Simulation Parameters", "", "JSON files (*.json);;All files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    json_str = f.read()
                self.from_json(json_str)
                QMessageBox.information(self, "Load Successful", "Parameters loaded from JSON file.")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load JSON file: {e}")

    def save_json(self):
        """Save current parameters to JSON file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation Parameters", "", "JSON files (*.json);;All files (*)"
        )
        if filename:
            try:
                json_str = self.to_json()
                with open(filename, 'w') as f:
                    f.write(json_str)
                QMessageBox.information(self, "Save Successful", "Parameters saved to JSON file.")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save JSON file: {e}")

    def from_json(self, json_str):
        """Deserialize parameters from JSON string and update dialog."""
        try:
            params = json.loads(json_str)
            self._apply_parameters(params)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "JSON Error", f"Invalid JSON format: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Error loading parameters: {e}")

    def get_parameters(self):
        """Get the current parameter values."""
        params = self.params.copy()

        # Update from controls
        params['excitation_mode'] = 'CW' if self.cw_radio.isChecked() else 'Pulsed'

        # Species parameters - collect exactly species_spin rows, include OFF rows as zeros
        M = []
        D = []
        q = []
        n_spec = int(self.species_spin.value()) if hasattr(self, 'species_spin') else len(getattr(self, 'cw_row_widgets', []))

        for row in range(n_spec):
            rw = self.cw_row_widgets[row] if hasattr(self, 'cw_row_widgets') and row < len(self.cw_row_widgets) else None
            if rw is None:
                # Fallback default row
                M.append(0.0)
                D.append(3.0)
                q.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            if rw['on'].isChecked():
                M.append(float(rw['M'].value()))
                D.append(float(rw['D'].value()))
                q.extend([
                    float(rw['G_P'].value()), float(rw['G_S'].value()),
                    float(rw['R_P'].value()), float(rw['R_S'].value()),
                    float(rw['Y_P'].value()), float(rw['Y_S'].value())
                ])
            else:
                M.append(0.0)
                D.append(float(rw['D'].value()))
                q.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        params['M'] = M
        params['D'] = D
        params['q'] = q
        params['N_species'] = n_spec
        params['N_channels'] = 6  # 6 channels for green/red/yellow P/S

        # Kinetics parameters - collect from kinetics tables
        if hasattr(self, 'rad_row_widgets') and self.rad_row_widgets:
            k_rad = []
            for row in range(n_spec):
                k_rad_row = []
                for col in range(n_spec):
                    if row < len(self.rad_row_widgets):
                        widget_key = f'k_rad_{col + 1}'
                        if widget_key in self.rad_row_widgets[row]:
                            k_rad_row.append(float(self.rad_row_widgets[row][widget_key].value()))
                        else:
                            k_rad_row.append(0.0)
                    else:
                        k_rad_row.append(0.0)
                k_rad.append(k_rad_row)
            params['k_rad'] = k_rad

        if hasattr(self, 'nrad_row_widgets') and self.nrad_row_widgets:
            k_nrad = []
            for row in range(n_spec):
                k_nrad_row = []
                for col in range(n_spec):
                    if row < len(self.nrad_row_widgets):
                        widget_key = f'k_nrad_{col + 1}'
                        if widget_key in self.nrad_row_widgets[row]:
                            k_nrad_row.append(float(self.nrad_row_widgets[row][widget_key].value()))
                        else:
                            k_nrad_row.append(0.0)
                    else:
                        k_nrad_row.append(0.0)
                k_nrad.append(k_nrad_row)
            params['k_nrad'] = k_nrad

        if hasattr(self, 'state_row_widgets') and self.state_row_widgets:
            k_bd = []
            k_bb = []
            k_db = []
            for row in range(n_spec):
                if row < len(self.state_row_widgets):
                    k_bd.append(float(self.state_row_widgets[row]['k_bd'].value()))
                    k_bb.append(float(self.state_row_widgets[row]['k_bb'].value()))
                    k_db.append(float(self.state_row_widgets[row]['k_db'].value()))
                else:
                    k_bd.append(0.0)
                    k_bb.append(0.0)
                    k_db.append(0.0)
            params['k_bd'] = k_bd
            params['k_bb'] = k_bb
            params['k_db'] = k_db

        # Background parameters
        if hasattr(self, 'bg_row_widgets') and self.bg_row_widgets:
            bg_q = []
            for channel in ['GP', 'GS', 'RP', 'RS', 'YP', 'YS']:
                if channel in self.bg_row_widgets[0]:
                    bg_q.append(float(self.bg_row_widgets[0][channel].value()))
                else:
                    bg_q.append(0.0)
            params['q_bg'] = bg_q
        elif hasattr(self, 'background_controls'):
            params['q_bg'] = [
                self.background_controls.bg_parallel_spin.value(),
                self.background_controls.bg_perp_spin.value()
            ]

        # Geometry parameters
        if hasattr(self, 'box_xy_spin'):
            params['box_xy'] = self.box_xy_spin.value()
        if hasattr(self, 'box_z_spin'):
            params['box_z'] = self.box_z_spin.value()
        if hasattr(self, 'focus_type_combo'):
            params['focus_type'] = self.FOCUS_TYPES[self.focus_type_combo.currentText()]
        if hasattr(self, 'focus_param_spins'):
            params['focus_param'] = [spin.value() for spin in self.focus_param_spins]
        if hasattr(self, 'dt_spin'):
            params['dt'] = self.dt_spin.value()
        if hasattr(self, 'n_ph_max_spin'):
            params['N_ph_max'] = self.n_ph_max_spin.value()

        # Anisotropy parameters (now handled in Options dialog)
        params['r0'] = self.params.get('r0', 0.38)
        params['g_factor'] = self.params.get('g_factor', 1.0)
        params['l1'] = self.params.get('l1', 0.0308)
        params['l2'] = self.params.get('l2', 0.0368)

        # TAC parameters (from tac_controls)
        if hasattr(self, 'tac_controls'):
            params['N_tac_channels'] = self.tac_controls.n_channels_spin.value()
            params['tac_dt'] = self.tac_controls.tac_dt_spin.value()
            params['laser_period'] = self.tac_controls.laser_period_spin.value()
        
        # IRF parameters (from main dialog)
        if hasattr(self, 'use_gaussian_irf_check'):
            params['use_gaussian_irf'] = self.use_gaussian_irf_check.isChecked()
            fwhm = self.gaussian_fwhm_spin.value()
            params['gaussian_irf_fwhm'] = fwhm
            params['gaussian_irf_mean'] = self.gaussian_mean_spin.value()
            # Calculate sigma from FWHM: FWHM = 2*sqrt(2*ln(2))*sigma
            params['gaussian_irf_sigma'] = fwhm / (2 * (2 * 0.693147180559945)**0.5)
            params['irf_file'] = getattr(self, '_irf_file_path', '')

        # Output parameters
        if hasattr(self, 'output_path_edit'):
            params['spc_output_path'] = self.output_path_edit.text()
        if hasattr(self, 'photons_per_file_spin'):
            params['N_ph_per_file'] = self.photons_per_file_spin.value()
        if hasattr(self, 'ch_conversion_edit'):
            try:
                params['ch_conversion'] = [int(x.strip()) for x in self.ch_conversion_edit.text().split(',')]
            except ValueError:
                params['ch_conversion'] = [8, 0, 9, 1, 10, 2]

        # Channel toggles and RNG parameters
        params['green_enabled'] = bool(self.green_enabled_check.isChecked()) if hasattr(self, 'green_enabled_check') else True
        params['red_enabled'] = bool(self.red_enabled_check.isChecked()) if hasattr(self, 'red_enabled_check') else False
        params['yellow_enabled'] = bool(self.yellow_enabled_check.isChecked()) if hasattr(self, 'yellow_enabled_check') else False
        
        if hasattr(self, 'rng_mode_combo'):
            params['rng_mode'] = int(self.rng_mode_combo.currentIndex())
        if hasattr(self, 'rmt1seed_spin'):
            params['rmt1seed'] = int(self.rmt1seed_spin.value())
        if hasattr(self, 'rmt2seed_spin'):
            params['rmt2seed'] = int(self.rmt2seed_spin.value())

        # Dark state interconvert
        if hasattr(self, 'darkstate_interconvert'):
            params['darkstate_interconvert'] = self.darkstate_interconvert.isChecked()

        # Fluorescence decay parameters (critical for data2spc_tac)
        if hasattr(self, 'pulsed_row_widgets') and self.pulsed_row_widgets:
            decay_lifetimes = []
            decay_patterns = []
            rotational_correlation_times = []
            
            for row_widgets in self.pulsed_row_widgets:
                # Collect lifetimes for each channel (Green, Red, Yellow)
                lifetimes_row = []
                rho_rot_row = []  # rotational correlation times per channel
                
                if 'tau_g' in row_widgets:
                    lifetimes_row.append(float(row_widgets['tau_g'].value()))
                    rho_rot_row.append(float(row_widgets['rho_g'].value()) if 'rho_g' in row_widgets else 0.4)
                else:
                    lifetimes_row.append(4.0)  # Default lifetime
                    rho_rot_row.append(0.4)  # Default rotational time
                    
                if 'tau_r' in row_widgets:
                    lifetimes_row.append(float(row_widgets['tau_r'].value()))
                    rho_rot_row.append(float(row_widgets['rho_r'].value()) if 'rho_r' in row_widgets else 0.4)
                else:
                    lifetimes_row.append(4.0)
                    rho_rot_row.append(0.4)
                    
                if 'tau_y' in row_widgets:
                    lifetimes_row.append(float(row_widgets['tau_y'].value()))
                    rho_rot_row.append(float(row_widgets['rho_y'].value()) if 'rho_y' in row_widgets else 0.4)
                else:
                    lifetimes_row.append(4.0)
                    rho_rot_row.append(0.4)
                
                decay_lifetimes.append(lifetimes_row)
                rotational_correlation_times.append(rho_rot_row)  # Now per channel, not per species
                
                # Pattern file path
                if 'pattern' in row_widgets:
                    decay_patterns.append(row_widgets['pattern'].text())
                else:
                    decay_patterns.append('')
            
            params['decay_lifetimes'] = decay_lifetimes
            params['decay_patterns'] = decay_patterns
            params['rotational_correlation_times'] = rotational_correlation_times
        else:
            # Default decay parameters for all species
            n_species = params.get('N_species', 1)
            params['decay_lifetimes'] = [[4.0, 4.0, 4.0] for _ in range(n_species)]  # Default 4ns lifetime
            params['rotational_correlation_times'] = [[0.4, 0.4, 0.4] for _ in range(n_species)]  # Default rotational times
            params['decay_patterns'] = ['' for _ in range(n_species)]               # No custom patterns

        # Pulsed excitation flag
        params['pulsed_exc'] = 1 if params.get('excitation_mode') == 'Pulsed' else 0

        # Focus-specific parameters (needed for focus functions in focus.cpp)
        focus_type = params.get('focus_type', 0)
        focus_param = params.get('focus_param', [0.3, 2.0])
        
        if focus_type == 0:  # 3D Gaussian, uniform CEF
            params['w0'] = focus_param[0] if len(focus_param) > 0 else 0.3  # beam waist
            params['z0'] = focus_param[1] if len(focus_param) > 1 else 2.0  # Rayleigh length
        elif focus_type == 1:  # 3D Gaussian excitation and CEF
            params['w0'] = focus_param[0] if len(focus_param) > 0 else 0.3
            params['z0'] = focus_param[1] if len(focus_param) > 1 else 2.0
            params['Rph'] = focus_param[2] if len(focus_param) > 2 else 0.15  # pinhole radius
        elif focus_type == 2:  # Gaussian-Lorentzian, pinhole CEF
            params['w0'] = focus_param[0] if len(focus_param) > 0 else 0.3
            params['z0'] = focus_param[1] if len(focus_param) > 1 else 2.0
            params['Rph'] = focus_param[2] if len(focus_param) > 2 else 0.15
        elif focus_type == 3:  # Gaussian-Lorentzian, cylindrical CEF
            params['w0'] = focus_param[0] if len(focus_param) > 0 else 0.3
            params['z0'] = focus_param[1] if len(focus_param) > 1 else 2.0
            params['Rph'] = focus_param[2] if len(focus_param) > 2 else 0.15
            params['z0_CEF'] = focus_param[3] if len(focus_param) > 3 else 1.0

        # Compute decay patterns only for pulsed excitation
        if params['pulsed_exc'] == 1:
            decay_patterns_computed = self.compute_decay_patterns_for_simulation()
            params['decay_patterns_computed'] = decay_patterns_computed

        return params

    def compute_decay_patterns_for_simulation(self):
        """Compute decay patterns for all species and channels for simulation.
        
        This method separates decay computation from visualization, providing
        computed patterns that can be used by both the dialog display and
        the data2spc_tac simulation engine.
        
        Returns:
            dict: Dictionary with computed decay patterns, F (integrated), and lookup tables
        """
        import numpy as np
        
        # Get TAC parameters
        n_tac_channels = self.params.get('N_tac_channels', 4096)
        tac_dt = self.params.get('tac_dt', 0.004069)  # ns
        
        # Create time axis
        time_axis = np.arange(n_tac_channels) * tac_dt
        
        # Get anisotropy parameters
        r0 = self.params.get('r0', 0.38)
        g_factor = self.params.get('g_factor', 1.0)
        l1 = self.params.get('l1', 0.0)
        l2 = self.params.get('l2', 0.0)
        
        # Get number of species and channels
        n_species = self.params.get('N_species', 1)
        n_channels = 3  # Green, Red, Yellow
        
        # Initialize results
        decay_patterns = {}
        integrated_patterns = {}  # F arrays for TAC
        lookup_tables = {}  # lookup tables for TAC
        
        for species_idx in range(n_species):
            species_key = f"species_{species_idx}"
            decay_patterns[species_key] = {}
            integrated_patterns[species_key] = {}
            lookup_tables[species_key] = {}
            
            # Get lifetimes and rotational times for this species
            lifetimes = self.get_species_lifetimes_for_simulation(species_idx)
            rotational_times = self.get_species_rotational_times_for_simulation(species_idx)
            pattern_file = self.get_species_pattern_for_simulation(species_idx)
            
            for channel_idx, (name, color) in enumerate([('Green', 'green'), ('Red', 'red'), ('Yellow', 'orange')]):
                channel_key = f"{name.lower()}_channel"
                
                # Check if this channel is enabled
                if name.lower() == 'green':
                    channel_enabled = self.params.get('green_enabled', True)
                elif name.lower() == 'red':
                    channel_enabled = self.params.get('red_enabled', False)
                elif name.lower() == 'yellow':
                    channel_enabled = self.params.get('yellow_enabled', False)
                else:
                    channel_enabled = True
                
                if not channel_enabled:
                    continue
                
                # Get lifetime and rotational correlation time for this channel
                lifetime = lifetimes[channel_idx]
                tau_rot = rotational_times[channel_idx]
                
                # Compute base decay pattern
                if pattern_file and os.path.exists(pattern_file):
                    # Load custom decay pattern
                    try:
                        pattern_data = np.loadtxt(pattern_file)
                        if len(pattern_data) >= n_tac_channels:
                            base_decay = pattern_data[:n_tac_channels]
                        else:
                            # Pad with zeros if pattern is shorter
                            base_decay = np.zeros(n_tac_channels)
                            base_decay[:len(pattern_data)] = pattern_data
                        # Normalize
                        if np.max(base_decay) > 0:
                            base_decay = base_decay / np.max(base_decay)
                    except Exception as e:
                        # Fall back to computed decay if pattern loading fails
                        base_decay = self.compute_exponential_decay(time_axis, lifetime, 1.0)
                else:
                    # Compute exponential decay
                    base_decay = self.compute_exponential_decay(time_axis, lifetime, 1.0)
                
                # Apply IRF convolution if enabled
                if self.params.get('use_gaussian_irf', False):
                    base_decay = self.convolve_with_gaussian_irf_for_simulation(base_decay, time_axis, lifetime)
                
                # Compute anisotropy components (parallel + perpendicular combined for TAC)
                parallel_decay, perp_decay = self.compute_anisotropy_decays(
                    time_axis, base_decay, lifetime, r0, g_factor, l1, l2, tau_rot
                )
                
                # For TAC simulation, we typically use the parallel component or a combination
                # For now, use parallel component (can be adjusted based on requirements)
                final_decay = parallel_decay
                
                # Store the computed decay pattern (convert to list for JSON serialization)
                decay_patterns[species_key][channel_key] = final_decay.tolist()
                
                # Compute integrated pattern F and lookup table for TAC
                # F is the cumulative distribution (integrated p(t))
                F = np.cumsum(final_decay)
                if np.max(F) > 0:
                    F = F / np.max(F)  # Normalize to [0, 1]
                
                # Create lookup table for TAC generation
                lookup = np.zeros(n_tac_channels, dtype=np.int32)
                for i in range(n_tac_channels):
                    # Find the TAC bin where F exceeds the random threshold
                    threshold = (i + 0.5) / n_tac_channels
                    bin_idx = np.searchsorted(F, threshold, side='left')
                    lookup[i] = min(bin_idx, n_tac_channels - 1)
                
                integrated_patterns[species_key][channel_key] = F.tolist()
                lookup_tables[species_key][channel_key] = lookup.tolist()
        
        return {
            'decay_patterns': decay_patterns,
            'integrated_patterns': integrated_patterns,  # F arrays
            'lookup_tables': lookup_tables,  # lookup tables
            'n_tac_channels': n_tac_channels,
            'tac_dt': tac_dt,
            'time_axis': time_axis.tolist()
        }

    def get_species_lifetimes_for_simulation(self, species_idx):
        """Get lifetimes for a species for simulation (similar to DecayVisualizationDialog method)."""
        # Try to get from pulsed row widgets first
        if hasattr(self, 'pulsed_row_widgets') and self.pulsed_row_widgets:
            if species_idx < len(self.pulsed_row_widgets):
                row_widgets = self.pulsed_row_widgets[species_idx]
                lifetimes = []
                for tau_key in ['tau_g', 'tau_r', 'tau_y']:
                    if tau_key in row_widgets:
                        lifetimes.append(float(row_widgets[tau_key].value()))
                    else:
                        lifetimes.append(4.0)  # Default
                return lifetimes
        
        # Fall back to decay_lifetimes parameter
        decay_lifetimes = self.params.get('decay_lifetimes', [[4.0, 4.0, 4.0]])
        if species_idx < len(decay_lifetimes):
            return decay_lifetimes[species_idx]
        else:
            return [4.0, 4.0, 4.0]

    def get_species_rotational_times_for_simulation(self, species_idx):
        """Get rotational correlation times for a species for simulation."""
        # Try to get from pulsed row widgets first
        if hasattr(self, 'pulsed_row_widgets') and self.pulsed_row_widgets:
            if species_idx < len(self.pulsed_row_widgets):
                row_widgets = self.pulsed_row_widgets[species_idx]
                rho_times = []
                for rho_key in ['rho_g', 'rho_r', 'rho_y']:
                    if rho_key in row_widgets:
                        rho_times.append(float(row_widgets[rho_key].value()))
                    else:
                        rho_times.append(0.4)  # Default
                return rho_times
        
        # Fall back to rotational_correlation_times parameter
        rot_times = self.params.get('rotational_correlation_times', [[0.4, 0.4, 0.4]])
        if species_idx < len(rot_times):
            return rot_times[species_idx]
        else:
            return [0.4, 0.4, 0.4]

    def get_species_pattern_for_simulation(self, species_idx):
        """Get pattern file for a species for simulation."""
        # Try to get from pulsed row widgets first
        if hasattr(self, 'pulsed_row_widgets') and self.pulsed_row_widgets:
            if species_idx < len(self.pulsed_row_widgets):
                row_widgets = self.pulsed_row_widgets[species_idx]
                if 'pattern' in row_widgets:
                    return row_widgets['pattern'].text()
        
        # Fall back to decay_patterns parameter
        decay_patterns = self.params.get('decay_patterns', [''])
        if species_idx < len(decay_patterns):
            return decay_patterns[species_idx]
        else:
            return ''

    def convolve_with_gaussian_irf_for_simulation(self, decay_pattern, time_axis, lifetime):
        """Simplified IRF convolution for simulation (without dialog dependencies)."""
        # Get IRF parameters
        irf_mean = self.params.get('gaussian_irf_mean', 0.0)
        irf_sigma = self.params.get('gaussian_irf_sigma', 0.0425)

        # Create Gaussian IRF
        irf = np.exp(-((time_axis - irf_mean) ** 2) / (2 * irf_sigma ** 2))
        
        # Normalize IRF
        irf_sum = np.sum(irf)
        if irf_sum > 0:
            irf = irf / irf_sum

        # Simple convolution (for simulation, we can use a basic approach)
        # For production, this should use the same fast_conv_burbulator as the dialog
        try:
            from scipy.signal import convolve
            convolved = convolve(decay_pattern, irf, mode='same')
            # Normalize
            if np.max(convolved) > 0:
                convolved = convolved / np.max(convolved)
            return convolved
        except ImportError:
            # Fallback: no convolution if scipy not available
            return decay_pattern

    def compute_anisotropy_decays(self, time_axis, base_decay, lifetime, r0, g_factor, l1, l2, tau_rot):
        """Compute parallel and perpendicular decay components with anisotropy."""
        # Anisotropy decay: r(t) = r0 * exp(-t/tau_rot)
        anisotropy_decay = r0 * np.exp(-time_axis / tau_rot)
        
        # Parallel and perpendicular intensities
        parallel_intensity = base_decay * (1 + 2 * anisotropy_decay) / 3
        perp_intensity = base_decay * (1 - anisotropy_decay) / 3
        
        # Apply G-factor correction (affects perpendicular channel)
        perp_intensity = perp_intensity * g_factor
        
        # Apply instrumental factors l1, l2 (simplified)
        parallel_intensity = parallel_intensity * (1 + l1)
        perp_intensity = perp_intensity * (1 + l2)
        
        # Normalize
        max_val = max(np.max(parallel_intensity), np.max(perp_intensity))
        if max_val > 0:
            parallel_intensity = parallel_intensity / max_val
            perp_intensity = perp_intensity / max_val
        
        return parallel_intensity, perp_intensity

    def compute_exponential_decay(self, time_axis, lifetime, beta):
        """Compute exponential decay pattern."""
        # Single exponential decay: I(t) = β * exp(-t/τ)
        decay = beta * np.exp(-time_axis / lifetime)
        # Normalize to peak = 1
        if np.max(decay) > 0:
            decay = decay / np.max(decay)
        return decay

    def _apply_parameters(self, params):
        """Apply loaded parameters to the dialog controls."""
        # Update basic parameters
        self.params.update(params)

        # Update excitation mode
        excitation_mode = params.get('excitation_mode', 'CW')
        if hasattr(self, 'cw_radio'):
            self.cw_radio.setChecked(excitation_mode == 'CW')
        if hasattr(self, 'pulsed_radio'):
            self.pulsed_radio.setChecked(excitation_mode == 'Pulsed')

        # Update species count
        n_species = int(params.get('N_species', 1))
        if hasattr(self, 'species_spin'):
            self.species_spin.blockSignals(True)
            self.species_spin.setValue(max(1, n_species))
            self.species_spin.blockSignals(False)

        # Update channel toggles
        if hasattr(self, 'green_enabled_check'):
            self.green_enabled_check.setChecked(params.get('green_enabled', True))
        if hasattr(self, 'red_enabled_check'):
            self.red_enabled_check.setChecked(params.get('red_enabled', False))
        if hasattr(self, 'yellow_enabled_check'):
            self.yellow_enabled_check.setChecked(params.get('yellow_enabled', False))

        # Update TAC parameters
        if hasattr(self, 'tac_controls'):
            if hasattr(self.tac_controls, 'n_channels_spin'):
                self.tac_controls.n_channels_spin.setValue(params.get('N_tac_channels', 4096))
            if hasattr(self.tac_controls, 'tac_dt_spin'):
                self.tac_controls.tac_dt_spin.setValue(params.get('tac_dt', 0.004069))
            if hasattr(self.tac_controls, 'laser_period_spin'):
                self.tac_controls.laser_period_spin.setValue(params.get('laser_period', 13.596))
        
        # Update IRF parameters (from main dialog)
        if hasattr(self, 'use_gaussian_irf_check'):
            self.use_gaussian_irf_check.setChecked(params.get('use_gaussian_irf', False))
            self.gaussian_fwhm_spin.setValue(params.get('gaussian_irf_fwhm', 0.1))
            self.gaussian_mean_spin.setValue(params.get('gaussian_irf_mean', 0.0))
            # Note: sigma is calculated automatically from FWHM
            
            # Update IRF file label and path
            irf_file = params.get('irf_file', '')
            self._irf_file_path = irf_file
            if irf_file:
                import os
                self.irf_file_label.setText(f"IRF: {os.path.basename(irf_file)}")
            else:
                self.irf_file_label.setText("No file selected")

class DecayVisualizationDialog(QDialog):
    """Dialog for visualizing fluorescence decay patterns."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_dialog = parent
        self.setWindowTitle("Fluorescence Decay Visualization")
        self.setModal(False)  # Non-blocking window
        self.resize(900, 600)
        
        # Get current parameters from parent
        self.params = parent.get_parameters()
        
        self.setup_ui()
        self.compute_and_plot_decays()
        
        # Set up auto-refresh timer for real-time updates
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        self.refresh_timer.start(500)  # Refresh every 500ms
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Info label
        info_label = QLabel("Fluorescence decay patterns for each species and channel")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Species selection
        species_layout = QHBoxLayout()
        species_layout.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        n_species = self.params.get('N_species', 1)
        for i in range(n_species):
            self.species_combo.addItem(f"Species {i+1}")
        self.species_combo.currentIndexChanged.connect(self.update_plots)
        species_layout.addWidget(self.species_combo)
        species_layout.addStretch()
        layout.addLayout(species_layout)
        
        # Channel selection controls - removed, will auto-detect from main dialog
        channel_layout = QHBoxLayout()
        # channel_layout.addWidget(QLabel("Show Channels:"))  # Removed
        
        # No more individual channel checkboxes - auto-detect from main dialog
        
        channel_layout.addSpacing(20)
        
        # Polarization selection
        channel_layout.addWidget(QLabel("Show:"))
        self.parallel_check = QCheckBox("Parallel (∥)")
        self.parallel_check.setChecked(True)
        self.parallel_check.toggled.connect(self.update_plots)
        channel_layout.addWidget(self.parallel_check)
        
        self.perp_check = QCheckBox("Perpendicular (⊥)")
        self.perp_check.setChecked(True)
        self.perp_check.toggled.connect(self.update_plots)
        channel_layout.addWidget(self.perp_check)
        
        channel_layout.addSpacing(20)
        
        # Y-axis scale control
        channel_layout.addWidget(QLabel("Y-axis:"))
        self.log_y_check = QCheckBox("Logarithmic")
        self.log_y_check.setChecked(True)
        self.log_y_check.toggled.connect(self.toggle_y_scale)
        channel_layout.addWidget(self.log_y_check)
        
        channel_layout.addStretch()
        layout.addLayout(channel_layout)
        
        # Initial update of display checkboxes after all checkboxes are created
        self.update_display_checkboxes()
        
        # Plot area
        try:
            import pyqtgraph as pg
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel('left', 'Intensity (log)', units='counts')
            self.plot_widget.setLabel('bottom', 'Time', units='ns')
            self.plot_widget.setTitle('Fluorescence Decay Patterns')
            self.plot_widget.showGrid(x=True, y=True)
            # Set logarithmic Y axis
            self.plot_widget.setLogMode(x=False, y=True)
            layout.addWidget(self.plot_widget)
        except ImportError:
            # Fallback to matplotlib if pyqtgraph is not available
            from matplotlib.backends.qt_compat import QtWidgets
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_yscale('log')  # Set logarithmic Y axis
            layout.addWidget(self.canvas)
        
        # Parameters display
        params_group = QGroupBox("Current Parameters")
        params_layout = QVBoxLayout(params_group)
        self.params_text = QTextEdit()
        self.params_text.setMaximumHeight(150)
        self.params_text.setReadOnly(True)
        params_layout.addWidget(self.params_text)
        layout.addWidget(params_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
    
    def compute_and_plot_decays(self):
        """Compute and plot fluorescence decay patterns."""
        self.update_plots()
        self.update_params_display()
    
    def toggle_y_scale(self):
        """Toggle between linear and logarithmic Y-axis."""
        use_log = self.log_y_check.isChecked()
        
        if hasattr(self, 'plot_widget'):
            # PyQtGraph
            self.plot_widget.setLogMode(x=False, y=use_log)
            label = 'Intensity (log)' if use_log else 'Intensity'
            self.plot_widget.setLabel('left', label, units='counts')
        elif hasattr(self, 'ax'):
            # Matplotlib
            if use_log:
                self.ax.set_yscale('log')
            else:
                self.ax.set_yscale('linear')
            self.canvas.draw()
    
    def update_plots(self):
        """Update the decay plots for the selected species."""
        # Safety check: ensure dialog is fully initialized
        if not hasattr(self, 'plot_widget') and not hasattr(self, 'ax'):
            return  # Dialog not fully initialized yet
            
        species_idx = self.species_combo.currentIndex()
        
        # Get TAC parameters
        n_tac_channels = self.params.get('N_tac_channels', 4096)
        tac_dt = self.params.get('tac_dt', 0.004069)  # ns
        
        # Create time axis
        time_axis = np.arange(n_tac_channels) * tac_dt
        
        # Get decay parameters from pulsed table (if available) or defaults
        lifetimes = self.get_species_lifetimes(species_idx)
        rotational_times = self.get_species_rotational_times(species_idx)
        pattern_file = self.get_species_pattern(species_idx)
        
        # Get anisotropy parameters
        r0 = self.params.get('r0', 0.38)
        g_factor = self.params.get('g_factor', 1.0)
        l1 = self.params.get('l1', 0.0)
        l2 = self.params.get('l2', 0.0)
        
        try:
            # Clear previous plots
            if hasattr(self, 'plot_widget'):
                self.plot_widget.clear()
            elif hasattr(self, 'ax'):
                self.ax.clear()
                self.ax.set_yscale('log')  # Restore log scale after clear
            
            # Get lifetimes and rotational times for this species
            lifetimes = self.get_species_lifetimes(species_idx)
            rotational_times = self.get_species_rotational_times(species_idx)
            
            # Track maximum decay value for y-axis range
            max_decay_value = 0.001  # Minimum floor
            
            # Plot each enabled channel
            for i, (name, color) in enumerate([('Green', 'green'), ('Red', 'red'), ('Yellow', 'orange')]):
                # Check if this channel is enabled
                if name.lower() == 'green':
                    channel_enabled = self.params.get('green_enabled', True)
                elif name.lower() == 'red':
                    channel_enabled = self.params.get('red_enabled', False)
                elif name.lower() == 'yellow':
                    channel_enabled = self.params.get('yellow_enabled', False)
                else:
                    channel_enabled = True
                
                if not channel_enabled:
                    continue
                
                # Get lifetime and rotational correlation time for this channel
                lifetime = lifetimes[i]
                tau_rot = rotational_times[i]  # This is guaranteed to be from the pulsed table
                
                # Check if custom pattern file exists and is specified
                if pattern_file and os.path.exists(pattern_file):
                    # Load custom decay pattern
                    try:
                        pattern_data = np.loadtxt(pattern_file)
                        if len(pattern_data) >= n_tac_channels:
                            base_decay = pattern_data[:n_tac_channels]
                        else:
                            # Pad with zeros if pattern is shorter
                            base_decay = np.zeros(n_tac_channels)
                            base_decay[:len(pattern_data)] = pattern_data
                        # Normalize
                        if np.max(base_decay) > 0:
                            base_decay = base_decay / np.max(base_decay)
                        
                        # Apply IRF convolution to custom pattern if enabled
                        if self.params.get('use_gaussian_irf', False):
                            base_decay = self.convolve_with_gaussian_irf(base_decay, time_axis, lifetime)
                            
                    except Exception as e:
                        # Fall back to computed decay if pattern loading fails
                        base_decay = self.compute_exponential_decay(time_axis, lifetime, 1.0)  # Use amplitude = 1.0
                        
                        # Apply IRF convolution to fallback decay if enabled
                        if self.params.get('use_gaussian_irf', False):
                            base_decay = self.convolve_with_gaussian_irf(base_decay, time_axis, lifetime)
                else:
                    # Compute exponential decay
                    base_decay = self.compute_exponential_decay(time_axis, lifetime, 1.0)  # Use amplitude = 1.0
                    
                    # Apply IRF convolution to computed decay if enabled
                    if self.params.get('use_gaussian_irf', False):
                        print(f"DEBUG: Applying IRF convolution to {name} channel")
                        base_decay_before = np.max(base_decay)
                        base_decay = self.convolve_with_gaussian_irf(base_decay, time_axis, lifetime)
                        base_decay_after = np.max(base_decay)
                        print(f"DEBUG: Peak before IRF: {base_decay_before:.6f}, after IRF: {base_decay_after:.6f}")
                
                # Set IRF suffix for labeling
                if self.params.get('use_gaussian_irf', False):
                    irf_suffix = " + IRF"
                else:
                    irf_suffix = ""
                
                # Compute parallel and perpendicular components with anisotropy
                parallel_decay, perp_decay = self.compute_anisotropy_decays(
                    time_axis, base_decay, lifetime, r0, g_factor, l1, l2, tau_rot
                )
                
                # Track maximum values for y-axis range
                max_decay_value = max(max_decay_value, np.max(parallel_decay), np.max(perp_decay))
                
                # Plot parallel and perpendicular based on checkboxes
                if self.parallel_check.isChecked():
                    par_label = f"{name} ∥ (τ={lifetime:.2f}ns){irf_suffix}"
                    if hasattr(self, 'plot_widget'):
                        self.plot_widget.plot(time_axis, parallel_decay, pen={'color': color, 'width': 2}, name=par_label)
                    elif hasattr(self, 'ax'):
                        self.ax.plot(time_axis, parallel_decay, color=color, linestyle='-', linewidth=2, label=par_label)
                
                if self.perp_check.isChecked():
                    perp_label = f"{name} ⊥ (τ={lifetime:.2f}ns){irf_suffix}"
                    if hasattr(self, 'plot_widget'):
                        # Use dashed line for perpendicular
                        try:
                            import pyqtgraph as pg
                            pen = pg.mkPen(color=color, width=2, style=Qt.DashLine)
                            self.plot_widget.plot(time_axis, perp_decay, pen=pen, name=perp_label)
                        except:
                            # Fallback if pyqtgraph pen styling fails
                            self.plot_widget.plot(time_axis, perp_decay, pen={'color': color, 'width': 1}, name=perp_label)
                    elif hasattr(self, 'ax'):
                        self.ax.plot(time_axis, perp_decay, color=color, linestyle='--', linewidth=2, label=perp_label)
            
            # Plot IRF if Gaussian IRF is enabled
            if self.params.get('use_gaussian_irf', False):
                self.plot_irf(time_axis)
            
            # Plot background components if enabled
            self.plot_background(time_axis)
            
            # Set plot properties and y-axis range
            plot_range = np.log10(0.01), np.log10(max_decay_value * 1.1)
            if hasattr(self, 'plot_widget'):
                self.plot_widget.setTitle(f'Species {species_idx + 1} Anisotropy Decay Patterns')
                self.plot_widget.addLegend()
                
                # Set reasonable y-axis range for log scale (0.001 to max of decays)
                if self.log_y_check.isChecked():
                    # Set y-range from 0.001 to max_decay_value
                    self.plot_widget.setYRange(*plot_range) # Add 10% margin
            elif hasattr(self, 'ax'):
                self.ax.set_title(f'Species {species_idx + 1} Anisotropy Decay Patterns')
                self.ax.set_xlabel('Time (ns)')
                self.ax.set_ylabel('Intensity (normalized)')
                self.ax.legend()
                self.ax.grid(True)
                
                # Set reasonable y-axis range for log scale (0.001 to max of decays)
                if self.log_y_check.isChecked():
                    # Set y-range from 0.001 to max_decay_value
                    self.ax.set_ylim(*plot_range)  # Add 10% margin
                
                self.canvas.draw()
                
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error plotting decays: {e}")
    
    def get_species_lifetimes(self, species_idx):
        """Get lifetimes for a species from pulsed table or defaults."""
        # Try to get from pulsed row widgets first
        if hasattr(self.parent_dialog, 'pulsed_row_widgets') and self.parent_dialog.pulsed_row_widgets:
            if species_idx < len(self.parent_dialog.pulsed_row_widgets):
                row_widgets = self.parent_dialog.pulsed_row_widgets[species_idx]
                lifetimes = []
                for tau_key in ['tau_g', 'tau_r', 'tau_y']:
                    if tau_key in row_widgets:
                        lifetimes.append(float(row_widgets[tau_key].value()))
                    else:
                        lifetimes.append(4.0)  # Default
                return lifetimes
        
        # Fall back to decay_lifetimes parameter
        decay_lifetimes = self.params.get('decay_lifetimes', [[4.0, 4.0, 4.0]])
        if species_idx < len(decay_lifetimes):
            return decay_lifetimes[species_idx]
        else:
            return [4.0, 4.0, 4.0]
    
    def get_species_rotational_times(self, species_idx):
        """Get rotational correlation times for a species from pulsed table or defaults."""
        # Try to get from pulsed row widgets first
        if hasattr(self.parent_dialog, 'pulsed_row_widgets') and self.parent_dialog.pulsed_row_widgets:
            if species_idx < len(self.parent_dialog.pulsed_row_widgets):
                row_widgets = self.parent_dialog.pulsed_row_widgets[species_idx]
                rho_times = []
                for rho_key in ['rho_g', 'rho_r', 'rho_y']:
                    if rho_key in row_widgets:
                        rho_times.append(float(row_widgets[rho_key].value()))
                    else:
                        rho_times.append(0.4)  # Default
                return rho_times
        
        # Fall back to rotational_correlation_times parameter
        rot_times = self.params.get('rotational_correlation_times', [[0.4, 0.4, 0.4]])
        if species_idx < len(rot_times):
            return rot_times[species_idx]
        else:
            return [0.4, 0.4, 0.4]
    
    def get_species_pattern(self, species_idx):
        """Get pattern file for a species."""
        # Try to get from pulsed row widgets first
        if hasattr(self.parent_dialog, 'pulsed_row_widgets') and self.parent_dialog.pulsed_row_widgets:
            if species_idx < len(self.parent_dialog.pulsed_row_widgets):
                row_widgets = self.parent_dialog.pulsed_row_widgets[species_idx]
                if 'pattern' in row_widgets:
                    return row_widgets['pattern'].text()
        
        # Fall back to decay_patterns parameter
        decay_patterns = self.params.get('decay_patterns', [''])
        if species_idx < len(decay_patterns):
            return decay_patterns[species_idx]
        else:
            return ''
    
    
    def compute_anisotropy_decays(self, time_axis, base_decay, lifetime, r0, g_factor, l1, l2, tau_rot):
        """Compute parallel and perpendicular decay components with anisotropy."""
        # Use user-specified rotational correlation time from pulsed table
        # tau_rot is now guaranteed to be the value from the pulsed table
        
        # Anisotropy decay: r(t) = r0 * exp(-t/tau_rot)
        anisotropy_decay = r0 * np.exp(-time_axis / tau_rot)
        
        # Parallel and perpendicular intensities
        # I_par(t) = I(t) * (1 + 2*r(t)) / 3
        # I_perp(t) = I(t) * (1 - r(t)) / 3
        # But we need to account for G-factor and instrumental factors
        
        # Apply instrumental corrections
        # Simplified model: I_obs = I_true * (1 + l1*P + l2*P^2) where P is polarization factor
        
        parallel_intensity = base_decay * (1 + 2 * anisotropy_decay) / 3
        perp_intensity = base_decay * (1 - anisotropy_decay) / 3
        
        # Apply G-factor correction (affects perpendicular channel)
        perp_intensity = perp_intensity * g_factor
        
        # Apply instrumental factors l1, l2 (simplified)
        parallel_intensity = parallel_intensity * (1 + l1)
        perp_intensity = perp_intensity * (1 + l2)
        
        # Normalize
        max_val = max(np.max(parallel_intensity), np.max(perp_intensity))
        if max_val > 0:
            parallel_intensity = parallel_intensity / max_val
            perp_intensity = perp_intensity / max_val
        
        return parallel_intensity, perp_intensity
    
    def compute_exponential_decay(self, time_axis, lifetime, beta):
        """Compute exponential decay pattern."""
        # Single exponential decay: I(t) = β * exp(-t/τ)
        decay = beta * np.exp(-time_axis / lifetime)
        # Normalize to peak = 1
        if np.max(decay) > 0:
            decay = decay / np.max(decay)
        return decay
    
    def convolve_with_gaussian_irf(self, decay_pattern, time_axis, lifetime=None):
        """Convolve decay with Gaussian IRF using Burbulator FastConv algorithm."""
        # Get IRF parameters
        irf_mean = self.params.get('gaussian_irf_mean', 0.0)
        irf_sigma = self.params.get('gaussian_irf_sigma', 0.0425)

        # Create Gaussian IRF (same as C# Burbulator)
        irf = np.exp(-((time_axis - irf_mean) ** 2) / (2 * irf_sigma ** 2))

        # Normalize IRF to sum = 1 (like C# NormPattern function)
        irf_sum = np.sum(irf)
        if irf_sum > 0:
            irf = irf / irf_sum

        # For Burbulator FastConv, we need TAC parameters
        n_channels = len(time_axis)
        tac_dt = self.params.get('tac_dt', 0.004069)
        laser_period = self.params.get('laser_period', 13.596)

        # Get TAC parameters from actual controls if available
        if hasattr(self.parent_dialog, 'tac_controls'):
            try:
                n_channels = self.parent_dialog.tac_controls.n_channels_spin.value()
                tac_dt = self.parent_dialog.tac_controls.tac_dt_spin.value()
                laser_period = self.parent_dialog.tac_controls.laser_period_spin.value()
            except:
                pass

        # Since we already have the decay pattern, we need to extract lifetime information
        # For a single exponential decay, we can estimate the lifetime by fitting
        # But for simplicity, let's assume a representative lifetime (this is an approximation)
        # In a full implementation, we'd fit the decay pattern to extract lifetime spectrum

        # Use the passed lifetime parameter, or fall back to a default
        representative_lifetime = lifetime if lifetime is not None else 4.0
        
        # For now, use a single exponential component with amplitude 1.0 and the representative lifetime
        # This is a simplification - ideally we'd fit the actual decay pattern
        lifetime_spectrum = np.array([1.0, representative_lifetime])  # [amplitude, lifetime]

        # Use Burbulator FastConv algorithm
        convolved = fast_conv_burbulator(
            lifetime_spectrum=lifetime_spectrum,
            irf=irf,
            n_channels=n_channels,
            tac_dt=tac_dt,
            laser_period=laser_period
        )

        # Scale the result to match the amplitude of the input decay pattern
        if np.max(convolved) > 0:
            scale_factor = np.max(decay_pattern) / np.max(convolved)
            convolved = convolved * scale_factor

        # Add background (like C# Burbulator ScatterDark2PS)
        convolved_with_bg = self.add_background_to_decay(convolved, irf, time_axis, tac_dt, laser_period)

        return convolved_with_bg
    
    def add_background_to_decay(self, decay_pattern, irf, time_axis, tac_dt, laser_period):
        """
        Add background counts to decay pattern (like C# Burbulator ScatterDark2PS).
        
        Background consists of:
        1. Dark counts: Uniform distribution across active TAC range
        2. Scatter counts: IRF-shaped background
        """
        # Get background parameters
        q_dark = self.params.get('q_bg', [0.001, 0.001])[0] if self.params.get('q_bg') else 0.001
        q_scatter = self.params.get('parallel_scatter', 0.0)
        
        # Get from actual controls if available
        if hasattr(self.parent_dialog, 'background_controls'):
            try:
                q_dark = self.parent_dialog.background_controls.bg_parallel_spin.value()
                # Note: scatter is typically handled separately in the UI
            except:
                pass
        
        # Calculate TAC range (same as C# auto TAC range)
        period_n = int(np.ceil(laser_period / tac_dt - 0.5))
        start = 1
        stop = min(len(time_axis) - 1, period_n + start)
        
        # Initialize background pattern
        background = np.zeros_like(decay_pattern)
        
        # Add dark counts: uniform distribution across active TAC range
        # C#: q_dark_perchannel = q_dark / 2.0 / (stop - start)
        if stop > start:
            q_dark_perchannel = q_dark / (stop - start)
            background[start:stop+1] += q_dark_perchannel
        
        # Add scatter counts: IRF-shaped background
        # C#: irf_vv[i] * aniso.ParallelScatterFraction
        if q_scatter > 0:
            # Scale IRF by scatter rate
            scatter_pattern = irf * q_scatter
            background += scatter_pattern
        
        # Add background to decay
        result = decay_pattern + background
        
        return result
    
    def update_display_checkboxes(self):
        """Update parallel/perp checkboxes based on active channels in main dialog."""
        # Safety check: ensure dialog is fully initialized
        if not hasattr(self, 'parallel_check') or not hasattr(self, 'perp_check'):
            return  # Dialog not fully initialized yet
            
        # Check which channels are enabled in the main dialog
        any_channel_active = (
            self.params.get('green_enabled', True) or
            self.params.get('red_enabled', False) or
            self.params.get('yellow_enabled', False)
        )
        
        # Enable/disable parallel and perp checkboxes based on active channels
        self.parallel_check.setEnabled(any_channel_active)
        self.perp_check.setEnabled(any_channel_active)
        
        # If no channels are active, uncheck the display checkboxes
        if not any_channel_active:
            self.parallel_check.setChecked(False)
            self.perp_check.setChecked(False)
        
        # Trigger plot update
        self.update_plots()
    
    def plot_background(self, time_axis):
        """Plot background components (dark counts and scatter)."""
        try:
            # Get background parameters
            q_dark = self.params.get('q_bg', [0.001, 0.001])[0] if self.params.get('q_bg') else 0.001
            q_scatter = self.params.get('parallel_scatter', 0.0)
            
            # Get from actual controls if available
            if hasattr(self.parent_dialog, 'background_controls'):
                try:
                    q_dark = self.parent_dialog.background_controls.bg_parallel_spin.value()
                except:
                    pass
            
            # Only plot if background is significant
            if q_dark <= 0.0001 and q_scatter <= 0.0001:
                return
                
            # Get TAC parameters
            tac_dt = self.params.get('tac_dt', 0.004069)
            laser_period = self.params.get('laser_period', 13.596)
            
            if hasattr(self.parent_dialog, 'tac_controls'):
                try:
                    tac_dt = self.parent_dialog.tac_controls.tac_dt_spin.value()
                    laser_period = self.parent_dialog.tac_controls.laser_period_spin.value()
                except:
                    pass
            
            # Calculate TAC range
            period_n = int(np.ceil(laser_period / tac_dt - 0.5))
            start = 1
            stop = min(len(time_axis) - 1, period_n + start)
            
            # Create background components
            dark_bg = np.zeros_like(time_axis)
            scatter_bg = np.zeros_like(time_axis)
            
            # Dark counts: uniform across active TAC range
            if q_dark > 0 and stop > start:
                q_dark_perchannel = q_dark / (stop - start)
                dark_bg[start:stop+1] = q_dark_perchannel
            
            # Scatter counts: IRF-shaped
            if q_scatter > 0:
                # Get IRF parameters
                irf_mean = self.params.get('gaussian_irf_mean', 0.0)
                irf_sigma = self.params.get('gaussian_irf_sigma', 0.0425)
                
                if hasattr(self.parent_dialog, 'use_gaussian_irf_check'):
                    try:
                        irf_mean = self.parent_dialog.gaussian_mean_spin.value()
                        irf_fwhm = self.parent_dialog.gaussian_fwhm_spin.value()
                        irf_sigma = irf_fwhm / (2 * (2 * 0.693147180559945)**0.5)
                    except:
                        pass
                
                # Create and normalize IRF
                irf = np.exp(-((time_axis - irf_mean) ** 2) / (2 * irf_sigma ** 2))
                irf_sum = np.sum(irf)
                if irf_sum > 0:
                    irf = irf / irf_sum
                
                scatter_bg = irf * q_scatter
            
            # Plot background components
            if np.max(dark_bg) > 0:
                dark_label = f"Dark BG ({q_dark:.4f} kHz)"
                if hasattr(self, 'plot_widget'):
                    self.plot_widget.plot(time_axis, dark_bg, pen={'color': 'brown', 'width': 1}, name=dark_label)
                elif hasattr(self, 'ax'):
                    self.ax.plot(time_axis, dark_bg, color='brown', linestyle='-', linewidth=1, label=dark_label)
            
            if np.max(scatter_bg) > 0:
                scatter_label = f"Scatter BG ({q_scatter:.4f} kHz)"
                if hasattr(self, 'plot_widget'):
                    self.plot_widget.plot(time_axis, scatter_bg, pen={'color': 'orange', 'width': 1}, name=scatter_label)
                elif hasattr(self, 'ax'):
                    self.ax.plot(time_axis, scatter_bg, color='orange', linestyle=':', linewidth=1, label=scatter_label)
                    
        except Exception as e:
            print(f"Warning: Could not plot background: {e}")
    
    def plot_irf(self, time_axis):
        """Plot the Instrument Response Function."""
        try:
            # Get IRF parameters
            irf_mean = self.params.get('gaussian_irf_mean', 0.0)
            irf_sigma = self.params.get('gaussian_irf_sigma', 0.0425)
            
            # Get from actual controls if available
            if hasattr(self.parent_dialog, 'use_gaussian_irf_check'):
                try:
                    irf_mean = self.parent_dialog.gaussian_mean_spin.value()
                    # Calculate sigma from FWHM
                    irf_fwhm = self.parent_dialog.gaussian_fwhm_spin.value()
                    irf_sigma = irf_fwhm / (2 * (2 * 0.693147180559945)**0.5)
                except:
                    pass
            
            # Create Gaussian IRF with same processing as convolution
            irf = np.exp(-((time_axis - irf_mean) ** 2) / (2 * irf_sigma ** 2))
            
            # Normalize IRF to sum = 1 (same as convolution, like C# NormPattern)
            irf_sum = np.sum(irf)
            if irf_sum > 0:
                irf = irf / irf_sum
            
            # Scale for display visibility (optional)
            if np.max(irf) > 0:
                irf = irf / np.max(irf)
            
            # Plot IRF
            irf_label = f"IRF (σ={irf_sigma:.4f}ns)"
            if hasattr(self, 'plot_widget'):
                # Use gray color for IRF
                self.plot_widget.plot(time_axis, irf, pen={'color': 'gray', 'width': 1}, name=irf_label)
            elif hasattr(self, 'ax'):
                self.ax.plot(time_axis, irf, color='gray', linestyle=':', linewidth=1, label=irf_label)
                
        except Exception as e:
            print(f"Warning: Could not plot IRF: {e}")
    
    def update_params_display(self):
        """Update the parameters display."""
        species_idx = self.species_combo.currentIndex()
        
        # Get actual values from dialog widgets
        lifetimes = self.get_species_lifetimes(species_idx)
        rotational_times = self.get_species_rotational_times(species_idx)
        pattern_file = self.get_species_pattern(species_idx)
        
        # Get anisotropy parameters from dialog controls
        r0 = self.params.get('r0', 0.38)
        g_factor = self.params.get('g_factor', 1.0)
        l1 = self.params.get('l1', 0.0)
        l2 = self.params.get('l2', 0.0)
        
        # Get anisotropy from parent dialog parameters
        r0 = self.parent_dialog.params.get('r0', r0)
        g_factor = self.parent_dialog.params.get('g_factor', g_factor)
        l1 = self.parent_dialog.params.get('l1', l1)
        l2 = self.parent_dialog.params.get('l2', l2)
        
        # Get TAC parameters from actual controls if available
        n_tac_channels = self.params.get('N_tac_channels', 4096)
        tac_dt = self.params.get('tac_dt', 0.004069)
        if hasattr(self.parent_dialog, 'tac_controls') and self.parent_dialog.tac_controls:
            try:
                n_tac_channels = self.parent_dialog.tac_controls.n_channels_spin.value()
                tac_dt = self.parent_dialog.tac_controls.tac_dt_spin.value()
            except:
                pass
        
        # Get IRF parameters from actual controls if available
        use_gaussian_irf = self.params.get('use_gaussian_irf', False)
        irf_mean = self.params.get('gaussian_irf_mean', 0.0)
        irf_sigma = self.params.get('gaussian_irf_sigma', 0.0425)
        if hasattr(self.parent_dialog, 'use_gaussian_irf_check'):
            try:
                use_gaussian_irf = self.parent_dialog.use_gaussian_irf_check.isChecked()
                irf_mean = self.parent_dialog.gaussian_mean_spin.value()
                # Calculate sigma from FWHM since gaussian_sigma_spin was removed
                irf_fwhm = self.parent_dialog.gaussian_fwhm_spin.value()
                irf_sigma = irf_fwhm / (2 * (2 * 0.693147180559945)**0.5)
            except:
                pass
        
        # Format parameters text
        params_text = f"""Species {species_idx + 1} Parameters:

Fluorescence Lifetimes (τ):
  Green: {lifetimes[0]:.3f} ns
  Red: {lifetimes[1]:.3f} ns  
  Yellow: {lifetimes[2]:.3f} ns

Rotational Correlation Times (ρ):
  Green: {rotational_times[0]:.3f} ns
  Red: {rotational_times[1]:.3f} ns
  Yellow: {rotational_times[2]:.3f} ns

Custom Pattern: {pattern_file if pattern_file else 'None'}

Anisotropy Parameters:
  r0 (fundamental anisotropy): {r0:.4f}
  G-factor: {g_factor:.4f}
  l1 (instrumental): {l1:.4f}
  l2 (instrumental): {l2:.4f}

TAC Settings:
  Channels: {n_tac_channels}
  Time/Channel: {tac_dt:.6f} ns
  Total Time: {n_tac_channels * tac_dt:.2f} ns
  
IRF Settings:
  Use Gaussian IRF: {use_gaussian_irf} {'✓ ENABLED' if use_gaussian_irf else '✗ DISABLED'}
  IRF Mean: {irf_mean:.3f} ns
  IRF Sigma: {irf_sigma:.4f} ns
  Note: Enable 'Use Gaussian IRF' in TAC/IRF tab to see convolution effect

Channel Status:
  Green: {'Enabled' if self.params.get('green_enabled', True) else 'Disabled'}
  Red: {'Enabled' if self.params.get('red_enabled', False) else 'Disabled'}
  Yellow: {'Enabled' if self.params.get('yellow_enabled', False) else 'Disabled'}

Note: Parallel (∥) = solid line, Perpendicular (⊥) = dashed line

DEBUG - Widget Values:"""
        
        # Add debug information to show actual widget values
        if hasattr(self.parent_dialog, 'pulsed_row_widgets') and self.parent_dialog.pulsed_row_widgets:
            if species_idx < len(self.parent_dialog.pulsed_row_widgets):
                row_widgets = self.parent_dialog.pulsed_row_widgets[species_idx]
                params_text += f"""
  tau_g widget (lifetime): {row_widgets.get('tau_g', {}).value() if 'tau_g' in row_widgets else 'N/A'}
  rho_g widget (rot. time): {row_widgets.get('rho_g', {}).value() if 'rho_g' in row_widgets else 'N/A'}
  tau_r widget (lifetime): {row_widgets.get('tau_r', {}).value() if 'tau_r' in row_widgets else 'N/A'}
  rho_r widget (rot. time): {row_widgets.get('rho_r', {}).value() if 'rho_r' in row_widgets else 'N/A'}
  tau_y widget (lifetime): {row_widgets.get('tau_y', {}).value() if 'tau_y' in row_widgets else 'N/A'}
  rho_y widget (rot. time): {row_widgets.get('rho_y', {}).value() if 'rho_y' in row_widgets else 'N/A'}"""
        
        params_text += "\n"
        
        self.params_text.setPlainText(params_text)
    
    def auto_refresh(self):
        """Auto-refresh decay plots when parameters change."""
        try:
            # Get fresh parameters from parent dialog
            new_params = self.parent_dialog.get_parameters()
            
            # Check if basic parameters have changed
            basic_changed = (
                new_params.get('N_species') != self.params.get('N_species') or
                new_params.get('r0') != self.params.get('r0') or
                new_params.get('g_factor') != self.params.get('g_factor') or
                new_params.get('l1') != self.params.get('l1') or
                new_params.get('l2') != self.params.get('l2') or
                new_params.get('green_enabled') != self.params.get('green_enabled') or
                new_params.get('red_enabled') != self.params.get('red_enabled') or
                new_params.get('yellow_enabled') != self.params.get('yellow_enabled')
            )
            
            # Check if pulsed table values have changed
            species_idx = self.species_combo.currentIndex()
            current_lifetimes = self.get_species_lifetimes(species_idx)
            current_rot_times = self.get_species_rotational_times(species_idx)
            
            # Store previous values for comparison (initialize if not exists)
            if not hasattr(self, '_prev_lifetimes'):
                self._prev_lifetimes = []
                self._prev_rot_times = []
            
            pulsed_changed = (
                current_lifetimes != self._prev_lifetimes or
                current_rot_times != self._prev_rot_times
            )
            
            # If any parameters changed, update plots
            if basic_changed or pulsed_changed:
                
                # Update cached values
                self._prev_lifetimes = current_lifetimes.copy()
                self._prev_rot_times = current_rot_times.copy()
                
                # Parameters changed, refresh the plots
                self.params = new_params
                self.update_plots()
                self.update_params_display()
                
                # Update display checkboxes based on new channel settings
                self.update_display_checkboxes()
                
                # Update species combo if needed
                n_species = self.params.get('N_species', 1)
                current_count = self.species_combo.count()
                if n_species != current_count:
                    current_selection = self.species_combo.currentIndex()
                    self.species_combo.clear()
                    for i in range(n_species):
                        self.species_combo.addItem(f"Species {i+1}")
                    # Restore selection if still valid
                    if current_selection < n_species:
                        self.species_combo.setCurrentIndex(current_selection)
        except Exception:
            # Ignore errors during auto-refresh to avoid disrupting user interaction
            pass
    
    def refresh_data(self):
        """Refresh data from parent dialog."""
        self.params = self.parent_dialog.get_parameters()
        
        # Cache current pulsed table values for change detection
        species_idx = self.species_combo.currentIndex()
        self._prev_lifetimes = self.get_species_lifetimes(species_idx)
        self._prev_rot_times = self.get_species_rotational_times(species_idx)
        
        # Update display checkboxes based on new channel settings
        self.update_display_checkboxes()
        
        # Update species combo if needed
        n_species = self.params.get('N_species', 1)
        current_count = self.species_combo.count()
        if n_species != current_count:
            current_selection = self.species_combo.currentIndex()
            self.species_combo.clear()
            for i in range(n_species):
                self.species_combo.addItem(f"Species {i+1}")
            # Restore selection if still valid
            if current_selection < n_species:
                self.species_combo.setCurrentIndex(current_selection)
        
        # Update the plots and parameter display
        self.update_plots()
        self.update_params_display()
    def closeEvent(self, event):
        """Clean up timer when dialog is closed."""
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()
        event.accept()


class OptionsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setModal(True)
        # Make dialog wide enough so all tabs fit without scrolling
        self.resize(720, 420)
        self.setMinimumWidth(700)

        lay = QVBoxLayout(self)
        # Tighter margins/spacing for a more compact layout
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabBarAutoHide(False)
        lay.addWidget(self.tabs)

        self._init_anisotropy(parent)
        self._init_saveas(parent)
        self._init_output(parent)
        self._init_tac(parent)
        self._init_rng(parent)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("Apply")
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        apply_btn.clicked.connect(lambda: self._apply(parent))
        ok_btn.clicked.connect(lambda: (self._apply(parent), self.accept()))
        cancel_btn.clicked.connect(self.reject)

    def _init_anisotropy(self, parent):
        """Initialize anisotropy tab with safe attribute access - VERSION 2."""
        print("DEBUG: _init_anisotropy called - FIXED VERSION")  # Debug print
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 4, 4, 4)
        f.setSpacing(4)
        self.r0 = QDoubleSpinBox(); self.r0.setRange(0,1); self.r0.setDecimals(4)
        self.g = QDoubleSpinBox(); self.g.setRange(0.1,2.0); self.g.setDecimals(4)
        self.l1 = QDoubleSpinBox(); self.l1.setRange(0,1); self.l1.setDecimals(4)
        self.l2 = QDoubleSpinBox(); self.l2.setRange(0,1); self.l2.setDecimals(4)
        
        # Always use default values to avoid any attribute access issues
        print("DEBUG: Setting default anisotropy values")  # Debug print
        self.r0.setValue(0.38)
        self.g.setValue(1.0)
        self.l1.setValue(0.0)
        self.l2.setValue(0.0)
            
        f.addRow("r0", self.r0)
        f.addRow("GFactor", self.g)
        f.addRow("l1", self.l1)
        f.addRow("l2", self.l2)
        self.tabs.addTab(w, "Anisotropy")
        print("DEBUG: _init_anisotropy completed successfully")  # Debug print

    def _init_saveas(self, parent):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 4, 4, 4)
        f.setSpacing(4)
        self.photons_per_file = QSpinBox(); self.photons_per_file.setRange(1000, 10000000)
        # Set value safely with fallback
        if hasattr(parent, 'photons_per_file_spin'):
            self.photons_per_file.setValue(parent.photons_per_file_spin.value())
        else:
            self.photons_per_file.setValue(1000000)  # Default 1M photons
        self.file_prefix = QLineEdit(getattr(parent, 'file_name_prefix', 'm'))
        self.file_ext = QLineEdit(getattr(parent, 'file_extension', 'spc'))
        self.file_numfmt = QLineEdit(getattr(parent, 'file_number_format', '000'))
        f.addRow("PhotonsPerFile", self.photons_per_file)
        f.addRow("FileNamePrefix", self.file_prefix)
        f.addRow("FileExtension", self.file_ext)
        f.addRow("FileNumberFormat", self.file_numfmt)
        self.tabs.addTab(w, "Save as")

    def _init_output(self, parent):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 4, 4, 4)
        f.setSpacing(4)
        
        # Output path
        self.output_path = QLineEdit()
        self.output_path.setText(getattr(parent, 'output_path_edit', QLineEdit()).text() if hasattr(parent, 'output_path_edit') else '')
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(lambda: self._browse_output())
        
        # Path row layout
        path_row = QHBoxLayout()
        path_row.addWidget(self.output_path)
        path_row.addWidget(self.browse_btn)
        f.addRow("Output folder:", path_row)
        
        # Channel mapping
        self.ch_conversion = QLineEdit()
        self.ch_conversion.setText(','.join(map(str, parent.params.get('ch_conversion', [8, 0, 9, 1, 10, 2]))))
        f.addRow("Channel mapping:", self.ch_conversion)
        
        self.tabs.addTab(w, "Output")

    def _browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_path.setText(path)

    def _init_tac(self, parent):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 4, 4, 4)
        f.setSpacing(4)
        # Create local copies to avoid re-parenting main widgets
        self.tac_nchannels = QSpinBox(); self.tac_nchannels.setRange(1, 65536)
        self.tac_nchannels.setValue(parent.tac_controls.n_channels_spin.value())
        self.tac_dt = QDoubleSpinBox(); self.tac_dt.setRange(1e-12, 1e6); self.tac_dt.setDecimals(6)
        self.tac_dt.setValue(parent.tac_controls.tac_dt_spin.value())
        self.laser_period = QDoubleSpinBox(); self.laser_period.setRange(1e-12, 1e6); self.laser_period.setDecimals(6)
        self.laser_period.setValue(parent.tac_controls.laser_period_spin.value())
        self.irf_file = QLineEdit(parent.tac_controls.irf_file_edit.text())
        self.bg_file = QLineEdit("(not a path)")
        
        # Color-wise channel mapping
        ch_conversion = parent.params.get('ch_conversion', [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5])
        self.green_p_ch = QSpinBox(); self.green_p_ch.setRange(0, 15); self.green_p_ch.setValue(ch_conversion[1] if len(ch_conversion) > 1 else 0)
        self.green_s_ch = QSpinBox(); self.green_s_ch.setRange(0, 15); self.green_s_ch.setValue(ch_conversion[3] if len(ch_conversion) > 3 else 1)
        self.red_p_ch = QSpinBox(); self.red_p_ch.setRange(0, 15); self.red_p_ch.setValue(ch_conversion[5] if len(ch_conversion) > 5 else 2)
        self.red_s_ch = QSpinBox(); self.red_s_ch.setRange(0, 15); self.red_s_ch.setValue(ch_conversion[7] if len(ch_conversion) > 7 else 3)
        self.yellow_p_ch = QSpinBox(); self.yellow_p_ch.setRange(0, 15); self.yellow_p_ch.setValue(ch_conversion[9] if len(ch_conversion) > 9 else 4)
        self.yellow_s_ch = QSpinBox(); self.yellow_s_ch.setRange(0, 15); self.yellow_s_ch.setValue(ch_conversion[11] if len(ch_conversion) > 11 else 5)
        
        f.addRow("NChannels", self.tac_nchannels)
        f.addRow("TACdt", self.tac_dt)
        f.addRow("LaserPeriod", self.laser_period)
        f.addRow("IRFFullName", self.irf_file)
        f.addRow("BgFullName", self.bg_file)
        f.addRow("GreenPChannel", self.green_p_ch)
        f.addRow("GreenSChannel", self.green_s_ch)
        f.addRow("RedPChannel", self.red_p_ch)
        f.addRow("RedSChannel", self.red_s_ch)
        f.addRow("YellowPChannel", self.yellow_p_ch)
        f.addRow("YellowSChannel", self.yellow_s_ch)
        self.tabs.addTab(w, "TAC parameters")

    def _init_rng(self, parent):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 4, 4, 4)
        f.setSpacing(4)
        
        # RNG mode (start from new state vs continue from last state)
        self.rng_mode = QComboBox()
        self.rng_mode.addItems(["Start from a new state", "Continue from last state"])
        self.rng_mode.setCurrentIndex(parent.params.get('rng_mode', 0))
        f.addRow("RNG Mode:", self.rng_mode)
        
        # Local copies to avoid re-parenting
        self.rnd1 = QSpinBox(); self.rnd1.setRange(0, 2_147_483_647)
        self.rnd2 = QSpinBox(); self.rnd2.setRange(0, 2_147_483_647)
        
        # Set values safely with fallbacks
        if hasattr(parent, 'rmt1seed_spin'):
            self.rnd1.setValue(parent.rmt1seed_spin.value())
        else:
            self.rnd1.setValue(12345)  # Default seed
            
        if hasattr(parent, 'rmt2seed_spin'):
            self.rnd2.setValue(parent.rmt2seed_spin.value())
        else:
            self.rnd2.setValue(67890)  # Default seed
        f.addRow("RND1Seed", self.rnd1)
        f.addRow("RND2Seed", self.rnd2)
        self.tabs.addTab(w, "RNG")

    def _apply(self, parent):
        # Push values back to parent parameters
        parent.params['r0'] = self.r0.value()
        parent.params['g_factor'] = self.g.value()
        parent.params['l1'] = self.l1.value()
        parent.params['l2'] = self.l2.value()
        
        if hasattr(parent, 'photons_per_file_spin'):
            parent.photons_per_file_spin.setValue(self.photons_per_file.value())
        parent.file_name_prefix = self.file_prefix.text()
        parent.file_extension = self.file_ext.text()
        parent.file_number_format = self.file_numfmt.text()
        # Output params
        if hasattr(parent, 'output_path_edit'):
            parent.output_path_edit.setText(self.output_path.text())
        if hasattr(parent, 'ch_conversion_edit'):
            parent.ch_conversion_edit.setText(self.ch_conversion.text())
        # TAC params
        if hasattr(parent, 'tac_controls'):
            parent.tac_controls.n_channels_spin.setValue(self.tac_nchannels.value())
            parent.tac_controls.tac_dt_spin.setValue(self.tac_dt.value())
            parent.tac_controls.laser_period_spin.setValue(self.laser_period.value())
            parent.tac_controls.irf_file_edit.setText(self.irf_file.text())
        # Build ch_conversion from color-wise channel mapping
        ch_conversion = [
            8, self.green_p_ch.value(),  # Green P
            9, self.green_s_ch.value(),  # Green S
            10, self.red_p_ch.value(),   # Red P
            11, self.red_s_ch.value(),   # Red S
            12, self.yellow_p_ch.value(), # Yellow P
            13, self.yellow_s_ch.value()  # Yellow S
        ]
        if hasattr(parent, 'ch_conversion_edit'):
            parent.ch_conversion_edit.setText(','.join(map(str, ch_conversion)))
        
        # Save channel settings to JSON file
        channel_settings = load_channel_settings()
        channel_settings['channel_conversion']['default'] = ch_conversion
        channel_settings['channel_conversion']['green_p'] = self.green_p_ch.value()
        channel_settings['channel_conversion']['green_s'] = self.green_s_ch.value()
        channel_settings['channel_conversion']['red_p'] = self.red_p_ch.value()
        channel_settings['channel_conversion']['red_s'] = self.red_s_ch.value()
        channel_settings['channel_conversion']['yellow_p'] = self.yellow_p_ch.value()
        channel_settings['channel_conversion']['yellow_s'] = self.yellow_s_ch.value()
        save_channel_settings(channel_settings)
        # RNG params
        parent.params['rng_mode'] = self.rng_mode.currentIndex()
        if hasattr(parent, 'rmt1seed_spin'):
            parent.rmt1seed_spin.setValue(self.rnd1.value())
        if hasattr(parent, 'rmt2seed_spin'):
            parent.rmt2seed_spin.setValue(self.rnd2.value())

    def create_output_tab(self):
        """Create the output tab."""
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)

        # Output path
        path_group = QGroupBox("Output Settings")
        path_layout = QFormLayout(path_group)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(self.params.get('spc_output_path', ''))
        path_layout.addRow("Output path:", self.output_path_edit)

        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.clicked.connect(self.browse_output)
        path_layout.addRow("", self.browse_output_button)

        self.photons_per_file_spin = QSpinBox()
        self.photons_per_file_spin.setRange(1000, 1000000)
        n_ph_per_file = self.params.get('N_ph_per_file', self.params.get('photons_per_file', 100000))
        self.photons_per_file_spin.setValue(n_ph_per_file)
        path_layout.addRow("Photons per file:", self.photons_per_file_spin)

        output_layout.addWidget(path_group)

        # BH_SPC conversion (advanced)
        bhspec_group = QGroupBox("BH_SPC Conversion (Advanced)")
        bhspec_layout = QFormLayout(bhspec_group)

        self.ch_conversion_edit = QLineEdit()
        self.ch_conversion_edit.setText(','.join(map(str, self.params['ch_conversion'])))
        bhspec_layout.addRow("Channel mapping:", self.ch_conversion_edit)

        output_layout.addWidget(bhspec_group)

        output_layout.addStretch()
        # Output tab removed; content moved to Acquisition tab

    def browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_path_edit.setText(path)

    def reset_defaults(self):
        """Reset parameters to defaults."""
        self._load_initial_params()
        QMessageBox.information(self, "Reset", "Parameters reset to defaults.")

    def get_parameters(self):
        """Get the current parameter values."""
        params = self.params.copy()

        # Update from controls
        params['excitation_mode'] = 'CW' if self.cw_radio.isChecked() else 'Pulsed'

        # Species parameters - collect exactly species_spin rows, include OFF rows as zeros
        M = []
        D = []
        q = []
        n_spec = int(self.species_spin.value()) if hasattr(self, 'species_spin') else len(getattr(self, 'cw_row_widgets', []))

        for row in range(n_spec):
            rw = self.cw_row_widgets[row] if hasattr(self, 'cw_row_widgets') and row < len(self.cw_row_widgets) else None
            if rw is None:
                # Fallback default row
                M.append(0.0)
                D.append(3.0)
                q.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            if rw['on'].isChecked():
                M.append(float(rw['M'].value()))
                D.append(float(rw['D'].value()))
                q.extend([
                    float(rw['G_P'].value()), float(rw['G_S'].value()),
                    float(rw['R_P'].value()), float(rw['R_S'].value()),
                    float(rw['Y_P'].value()), float(rw['Y_S'].value())
                ])
            else:
                M.append(0.0)
                D.append(float(rw['D'].value()))
                q.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        params['M'] = M
        params['D'] = D
        params['q'] = q
        params['N_species'] = n_spec
        params['N_channels'] = 6  # 6 channels for green/red/yellow P/S

        # Background
        if hasattr(self, 'background_controls'):
            params['q_bg'] = [
                self.background_controls.bg_parallel_spin.value(),
                self.background_controls.bg_perp_spin.value()
            ]

        # Geometry
        params['box_xy'] = self.box_xy_spin.value()
        params['box_z'] = self.box_z_spin.value()
        params['focus_type'] = self.FOCUS_TYPES[self.focus_type_combo.currentText()]
        params['focus_param'] = [spin.value() for spin in self.focus_param_spins]
        params['dt'] = self.dt_spin.value()
        params['N_ph_max'] = self.n_ph_max_spin.value()

        # Anisotropy (now handled in Options dialog)
        params['r0'] = self.params.get('r0', 0.38)
        params['g_factor'] = self.params.get('g_factor', 1.0)
        params['l1'] = self.params.get('l1', 0.0308)
        params['l2'] = self.params.get('l2', 0.0368)

        # TAC/IRF
        if hasattr(self, 'tac_controls'):
            params['N_tac_channels'] = self.tac_controls.n_channels_spin.value()
            params['tac_dt'] = self.tac_controls.tac_dt_spin.value()
            params['laser_period'] = self.tac_controls.laser_period_spin.value()
            params['use_gaussian_irf'] = self.tac_controls.use_gaussian_irf_check.isChecked()
            params['gaussian_irf_fwhm'] = self.tac_controls.gaussian_fwhm_spin.value()
            params['irf_file'] = self.tac_controls.irf_file_edit.text()

        # Output
        params['spc_output_path'] = self.output_path_edit.text()
        params['N_ph_per_file'] = self.photons_per_file_spin.value()
        try:
            params['ch_conversion'] = [int(x.strip()) for x in self.ch_conversion_edit.text().split(',')]
        except ValueError:
            params['ch_conversion'] = [8, 0, 9, 1, 10, 2]

        # Channel toggles and RNG
        params['green_enabled'] = bool(self.green_enabled_check.isChecked())
        params['red_enabled'] = bool(self.red_enabled_check.isChecked())
        params['yellow_enabled'] = bool(self.yellow_enabled_check.isChecked())
        params['rng_mode'] = int(self.rng_mode_combo.currentIndex())
        params['rmt1seed'] = int(self.rmt1seed_spin.value())
        params['rmt2seed'] = int(self.rmt2seed_spin.value())

        return params

    def to_json(self):
        """Serialize current parameters to JSON string."""
        params = self.get_parameters()
        return json.dumps(params, indent=2)

    def from_json(self, json_str):
        """Deserialize parameters from JSON string and update dialog."""
        try:
            params = json.loads(json_str)
            self._apply_parameters(params)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "JSON Error", f"Invalid JSON format: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Error loading parameters: {e}")

    def _apply_parameters(self, params):
        """Apply loaded parameters to the dialog controls."""
        # Update basic parameters
        self.params.update(params)

        # Update excitation mode
        excitation_mode = params.get('excitation_mode', 'CW')
        self.cw_radio.setChecked(excitation_mode == 'CW')
        self.pulsed_radio.setChecked(excitation_mode == 'Pulsed')
        self.update_mode_visibility()

        # Update species count globally and recreate both tables if needed
        n_species = int(params.get('N_species', 1))
        if hasattr(self, 'species_spin'):
            self.species_spin.blockSignals(True)
            self.species_spin.setValue(max(1, n_species))
            self.species_spin.blockSignals(False)
        
        if hasattr(self, 'cw_table'):
            self.cw_table.setRowCount(max(1, n_species))
            self._populate_cw_table()
        if hasattr(self, 'pulsed_table'):
            self.pulsed_table.setRowCount(max(1, n_species))
            self._populate_pulsed_table()

        # Update molecule controls
        M = params.get('M', [])
        D = params.get('D', [])
        q = params.get('q', [])
        for i, control in enumerate(self.molecule_controls):
            if i < len(M):
                control.n_molecules_spin.setValue(M[i])
            if i < len(D):
                control.diffusion_spin.setValue(D[i])

            # Update brightness
            idx_parallel = i * 2
            idx_perp = i * 2 + 1
            if idx_parallel < len(q):
                control.q_parallel_spin.setValue(q[idx_parallel])
            if idx_perp < len(q):
                control.q_perp_spin.setValue(q[idx_perp])

        # Update background controls
        if hasattr(self, 'background_controls'):
            q_bg = params.get('q_bg', [0.001, 0.001])
            if len(q_bg) > 0:
                self.background_controls.bg_parallel_spin.setValue(q_bg[0])
            if len(q_bg) > 1:
                self.background_controls.bg_perp_spin.setValue(q_bg[1])

        # Update geometry
        self.box_xy_spin.setValue(params.get('box_xy', 2.0))
        self.box_z_spin.setValue(params.get('box_z', 4.0))

        # Update focus
        focus_type = params.get('focus_type', 0)
        focus_type_names = list(self.FOCUS_TYPES.keys())
        if focus_type < len(focus_type_names):
            self.focus_type_combo.setCurrentText(focus_type_names[focus_type])

        focus_param = params.get('focus_param', [0.3, 2.0, 0.1, 0.5, 0.05, 1.0])
        for i, spin in enumerate(self.focus_param_spins):
            if i < len(focus_param):
                spin.setValue(focus_param[i])
        
        # Update focus parameter visibility after setting focus type
        self._update_focus_param_visibility()

        # Update simulation parameters
        self.dt_spin.setValue(params.get('dt', 0.01))
        self.n_ph_max_spin.setValue(params.get('N_ph_max', 50000))

        # Update anisotropy (stored in params, controlled via Options dialog)
        self.params['r0'] = params.get('r0', 0.38)
        self.params['g_factor'] = params.get('g_factor', 1.0)
        self.params['l1'] = params.get('l1', 0.0308)
        self.params['l2'] = params.get('l2', 0.0368)

        # Update TAC/IRF
        if hasattr(self, 'tac_controls'):
            self.tac_controls.n_channels_spin.setValue(params.get('N_tac_channels', 4096))
            self.tac_controls.tac_dt_spin.setValue(params.get('tac_dt', 0.004069))
            self.tac_controls.laser_period_spin.setValue(params.get('laser_period', 13.596))
            self.tac_controls.use_gaussian_irf_check.setChecked(params.get('use_gaussian_irf', False))
            self.tac_controls.gaussian_fwhm_spin.setValue(params.get('gaussian_irf_fwhm', 0.1))
            self.tac_controls.irf_file_edit.setText(params.get('irf_file', ''))

        # Update output
        self.output_path_edit.setText(params.get('spc_output_path', ''))
        n_ph_per_file = params.get('N_ph_per_file', params.get('photons_per_file', 100000))
        self.photons_per_file_spin.setValue(n_ph_per_file)
        ch_conversion = params.get('ch_conversion', [8, 0, 9, 1, 10, 2])
        self.ch_conversion_edit.setText(','.join(map(str, ch_conversion)))

        # Channel toggles and RNG
        self.green_enabled_check.setChecked(params.get('green_enabled', True))
        self.red_enabled_check.setChecked(params.get('red_enabled', False))
        self.yellow_enabled_check.setChecked(params.get('yellow_enabled', False))
        self.rng_mode_combo.setCurrentIndex(params.get('rng_mode', 0))
        self.rmt1seed_spin.setValue(params.get('rmt1seed', 12345))
        self.rmt2seed_spin.setValue(params.get('rmt2seed', 67890))

        # Apply channel enable state to tables
        self._update_channel_enable_state()

        # Couple tauD to focus w0: whenever w0 changes, update species tauD
        if self.focus_param_spins:
            try:
                self.focus_param_spins[0].valueChanged.connect(self._update_tauD_all)
            except Exception:
                pass

    def _update_tauD_all(self):
        """Recompute tauD for all CW table rows."""
        if not hasattr(self, 'cw_row_widgets'):
            return
        for row in range(len(self.cw_row_widgets)):
            self._update_tauD_row(row)


    def _apply_channel_state_safe(self):
        """Safely apply channel enable state during construction and later."""
        try:
            self._update_channel_enable_state()
        except Exception:
            pass

    def load_json(self):
        """Load parameters from JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Simulation Parameters", "", "JSON files (*.json);;All files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    json_str = f.read()
                self.from_json(json_str)
                QMessageBox.information(self, "Load Successful", "Parameters loaded from JSON file.")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load JSON file: {e}")

    def save_json(self):
        """Save current parameters to JSON file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation Parameters", "", "JSON files (*.json);;All files (*)"
        )
        if filename:
            try:
                json_str = self.to_json()
                with open(filename, 'w') as f:
                    f.write(json_str)
                QMessageBox.information(self, "Save Successful", "Parameters saved to JSON file.")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save JSON file: {e}")

    def view_json(self):
        """Show current parameters as formatted JSON for quick inspection."""
        try:
            json_str = self.to_json()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to serialize parameters: {e}")
            return

        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Current Simulation Parameters (JSON)")
        preview_dialog.resize(700, 500)
        preview_dialog.setModal(True)

        layout = QVBoxLayout(preview_dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(json_str)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(preview_dialog.accept)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        preview_dialog.exec_()
