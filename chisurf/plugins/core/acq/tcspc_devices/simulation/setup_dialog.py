"""AutoForm setup dialog for the tttrlib acquisition photon simulator."""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Any

from qtpy import QtWidgets

from chisurf.core import dataspec as ds
from chisurf.gui.autoform import AutoForm


@functools.lru_cache(maxsize=1)
def _schema_help() -> dict:
    """Per-parameter help from the acq plugin ``manifest.json`` params_schema.

    Single source of truth for the parameter tooltips — the same text the RPC
    method (``acq.simulation.run``) and CLI document — so the GUI never
    duplicates it.
    """
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    try:
        with open(os.path.join(plugin_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for method in manifest.get("rpc_methods", []):
            if method.get("name") == "acq.simulation.run":
                props = method.get("params_schema", {}).get("properties", {})
                return {k: str(v.get("description", "")) for k, v in props.items()}
    except Exception:
        pass
    return {}


def _help(schema_key: str) -> str:
    """Shared tooltip text for a ``params_schema`` key (``''`` if absent)."""
    return _schema_help().get(schema_key, "")


def load_channel_settings():
    """Load detector channel conversion settings.

    Returns
    -------
    dict
        Channel conversion settings with a default six-channel mapping.
    """
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    settings_file = os.path.join(plugin_dir, "channel_settings.json")
    default_settings = {
        "channel_conversion": {
            "default": [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5],
            "green_p": 0,
            "green_s": 1,
            "red_p": 2,
            "red_s": 3,
            "yellow_p": 4,
            "yellow_s": 5,
        },
        "detector_channels": {
            "green_p": 8,
            "green_s": 9,
            "red_p": 10,
            "red_s": 11,
            "yellow_p": 12,
            "yellow_s": 13,
        },
    }
    try:
        if os.path.exists(settings_file):
            with open(settings_file, encoding="utf-8") as handle:
                return json.load(handle)
        with open(settings_file, "w", encoding="utf-8") as handle:
            json.dump(default_settings, handle, indent=2)
    except Exception:
        pass
    return default_settings


def _flatten(values: Any) -> list[float]:
    """Flatten nested numeric settings.

    Parameters
    ----------
    values : object
        Scalar, nested sequence, or NumPy-like value.

    Returns
    -------
    list of float
        Flat numeric list.
    """
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        out = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [float(values)]


def _list_get(values: list[float], index: int, default: float) -> float:
    """Return a list value with a fallback.

    Parameters
    ----------
    values : list of float
        Source values.
    index : int
        Desired index.
    default : float
        Fallback value.

    Returns
    -------
    float
        Selected or fallback value.
    """
    return float(values[index]) if index < len(values) else float(default)


@dataclass
class SimulationSettingsModel:
    """AutoForm view model for tttrlib photon simulation settings."""

    n_species: int = 1
    molecules: float = 50.0
    diffusion: float = 3.0
    excitation_mode: str = "CW"
    green_enabled: bool = True
    red_enabled: bool = False
    yellow_enabled: bool = False
    q_green_p: float = 50.0
    q_green_s: float = 50.0
    q_red_p: float = 0.0
    q_red_s: float = 0.0
    q_yellow_p: float = 0.0
    q_yellow_s: float = 0.0
    bg_green_p: float = 0.001
    bg_green_s: float = 0.001
    bg_red_p: float = 0.001
    bg_red_s: float = 0.001
    bg_yellow_p: float = 0.001
    bg_yellow_s: float = 0.001
    box_xy: float = 2.0
    box_z: float = 4.0
    focus_w0: float = 0.3
    focus_z0: float = 2.0
    dt: float = 0.01
    n_ph_max: int = 1_000_000
    n_ph_per_file: int = 100_000
    output_path: str = ""
    n_tac_channels: int = 4096
    tac_dt: float = 0.004069
    laser_period: float = 13.596
    seed_diffusion: int = 12345
    seed_emission: int = 54321
    # --- throughput / performance knobs (native tttrlib Sim* engine; see manifest) ---
    max_windows: int = 0
    analytic_excitation: bool = False
    per_molecule_skip: bool = False
    fast_grid_bbox: bool = False
    independent_molecules: bool = False
    active_margin: float = 0.0
    coast_safety: float = 3.0
    min_coast_windows: int = 8
    focus_threshold: float = 0.001
    # --- point-spread / focus model (see manifest params_schema) ---
    psf_type: str = "gaussian3d"
    psf_zR: float = 1.0
    psf_file: str = ""
    psf_r_step: float = 0.05
    psf_z_step: float = 0.05

    @classmethod
    def from_parameters(cls, params: dict[str, Any] | None):
        """Create a model from legacy acquisition simulation parameters.

        Parameters
        ----------
        params : dict, optional
            Existing ``simulation_params`` dictionary.

        Returns
        -------
        SimulationSettingsModel
            Populated view model.
        """
        params = dict(params or {})
        q = _flatten(params.get("q", [50.0, 50.0]))
        q_bg = _flatten(params.get("q_bg", [0.001, 0.001]))
        focus = _flatten(params.get("focus_param", [0.3, 2.0]))
        molecules = _flatten(params.get("M", [50.0]))
        diffusion = _flatten(params.get("D", [3.0]))
        n_channels = int(params.get("N_channels", max(2, min(6, len(q) or 2))))
        red_enabled = bool(params.get("red_enabled", n_channels >= 4))
        yellow_enabled = bool(params.get("yellow_enabled", n_channels >= 6))
        green_enabled = bool(params.get("green_enabled", True))
        if not (green_enabled or red_enabled or yellow_enabled):
            green_enabled = True
        excitation_mode = params.get(
            "excitation_mode",
            "Pulsed" if params.get("pulsed_exc") else "CW",
        )
        n_ph_per_file = params.get(
            "N_ph_per_file",
            params.get("photons_per_file", 100_000),
        )
        return cls(
            n_species=max(1, int(params.get("N_species", 1))),
            molecules=_list_get(molecules, 0, 50.0),
            diffusion=_list_get(diffusion, 0, 3.0),
            excitation_mode=str(excitation_mode),
            green_enabled=green_enabled,
            red_enabled=red_enabled,
            yellow_enabled=yellow_enabled,
            q_green_p=_list_get(q, 0, 50.0),
            q_green_s=_list_get(q, 1, 50.0),
            q_red_p=_list_get(q, 2, 0.0),
            q_red_s=_list_get(q, 3, 0.0),
            q_yellow_p=_list_get(q, 4, 0.0),
            q_yellow_s=_list_get(q, 5, 0.0),
            bg_green_p=_list_get(q_bg, 0, 0.001),
            bg_green_s=_list_get(q_bg, 1, 0.001),
            bg_red_p=_list_get(q_bg, 2, 0.001),
            bg_red_s=_list_get(q_bg, 3, 0.001),
            bg_yellow_p=_list_get(q_bg, 4, 0.001),
            bg_yellow_s=_list_get(q_bg, 5, 0.001),
            box_xy=float(params.get("box_xy", 2.0)),
            box_z=float(params.get("box_z", 4.0)),
            focus_w0=_list_get(focus, 0, 0.3),
            focus_z0=_list_get(focus, 1, 2.0),
            dt=float(params.get("dt", params.get("tw", 0.01))),
            n_ph_max=max(1, int(params.get("N_ph_max", 1_000_000))),
            n_ph_per_file=max(1, int(n_ph_per_file)),
            output_path=str(params.get("spc_output_path", "")),
            n_tac_channels=max(1, int(params.get("N_tac_channels", 4096))),
            tac_dt=float(params.get("tac_dt", 0.004069)),
            laser_period=float(params.get("laser_period", 13.596)),
            seed_diffusion=int(params.get("rmt1seed", params.get("seed_diffusion", 12345))),
            seed_emission=int(params.get("rmt2seed", params.get("seed_emission", 54321))),
            max_windows=int(params.get("max_windows", 0)),
            analytic_excitation=bool(params.get("analytic_excitation", False)),
            per_molecule_skip=bool(params.get("per_molecule_skip", False)),
            fast_grid_bbox=bool(params.get("fast_grid_bbox", False)),
            independent_molecules=bool(params.get("independent_molecules", False)),
            active_margin=float(params.get("active_margin", 0.0)),
            coast_safety=float(params.get("coast_safety", 3.0)),
            min_coast_windows=int(params.get("min_coast_windows", 8)),
            focus_threshold=float(params.get("focus_threshold", 0.001)),
            psf_type=str(params.get("psf_type", "gaussian3d")),
            psf_zR=float(params.get("psf_zR", 1.0)),
            psf_file=str(params.get("psf_file", "")),
            psf_r_step=float(params.get("psf_r_step", 0.05)),
            psf_z_step=float(params.get("psf_z_step", 0.05)),
        )

    def view_spec(self):
        """Return the declarative AutoForm layout.

        Returns
        -------
        chisurf.core.dataspec.ModelView
            AutoForm section tree.
        """
        return ds.ModelView(
            sections=(
                ds.PanelSection(
                    title="Acquisition",
                    n_col=2,
                    sections=(
                        ds.ChoiceSection(
                            attr="excitation_mode",
                            label="Mode",
                            options=("CW", "Pulsed"),
                            style="combo",
                            description="Excitation mode: CW (continuous) or Pulsed (pulsed laser; enables the micro-time / TCSPC axis).",
                        ),
                        ds.ValueSection(
                            attr="n_ph_max",
                            label="Photons",
                            kind="int",
                            minimum=1,
                            maximum=2_000_000_000,
                            description=_help("N_ph_max"),
                        ),
                        ds.ValueSection(
                            attr="n_ph_per_file",
                            label="Photons/file",
                            kind="int",
                            minimum=1,
                            maximum=2_000_000_000,
                            description="Split the streamed SPC output into files of this many photons each.",
                        ),
                        ds.ValueSection(attr="output_path", label="Output folder", kind="str",
                            description="Folder where SPC/photon-stream files are written."),
                        ds.ValueSection(
                            attr="seed_diffusion",
                            label="Diffusion seed",
                            kind="int",
                            minimum=0,
                            description=_help("rmt1seed"),
                        ),
                        ds.ValueSection(
                            attr="seed_emission",
                            label="Emission seed",
                            description=_help("rmt2seed"),
                            kind="int",
                            minimum=0,
                        ),
                    ),
                ),
                ds.PanelSection(
                    title="Sample",
                    n_col=2,
                    sections=(
                        ds.ValueSection(
                            attr="n_species",
                            label="Species",
                            kind="int",
                            minimum=1,
                            maximum=999,
                            description=_help("N_species"),
                        ),
                        ds.ValueSection(
                            attr="molecules",
                            label="Molecules",
                            kind="float",
                            minimum=0.0,
                            maximum=1_000_000.0,
                            decimals=3,
                            description=_help("M"),
                        ),
                        ds.ValueSection(
                            attr="diffusion",
                            label="Diffusion",
                            kind="float",
                            minimum=0.0,
                            maximum=1_000_000.0,
                            decimals=4,
                            description=_help("D"),
                        ),
                    ),
                ),
                ds.PanelSection(
                    title="Channels",
                    n_col=2,
                    description="Per-colour detection channels and their P/S (parallel/perpendicular) brightness q and background (photons per macro-time unit). Enable the colours present in the setup.",
                    sections=(
                        ds.ToggleSection(attr="green_enabled", label="Green",
                            description="Enable the green (donor) detection channels."),
                        ds.ToggleSection(attr="red_enabled", label="Red",
                            description="Enable the red (acceptor) detection channels."),
                        ds.ToggleSection(attr="yellow_enabled", label="Yellow",
                            description="Enable the yellow detection channels (3-colour)."),
                        self._float_field("q_green_p", "Green P", decimals=4),
                        self._float_field("q_green_s", "Green S", decimals=4),
                        self._float_field("q_red_p", "Red P", decimals=4),
                        self._float_field("q_red_s", "Red S", decimals=4),
                        self._float_field("q_yellow_p", "Yellow P", decimals=4),
                        self._float_field("q_yellow_s", "Yellow S", decimals=4),
                        self._float_field("bg_green_p", "BG Green P", decimals=6),
                        self._float_field("bg_green_s", "BG Green S", decimals=6),
                        self._float_field("bg_red_p", "BG Red P", decimals=6),
                        self._float_field("bg_red_s", "BG Red S", decimals=6),
                        self._float_field("bg_yellow_p", "BG Yellow P", decimals=6),
                        self._float_field("bg_yellow_s", "BG Yellow S", decimals=6),
                    ),
                ),
                ds.PanelSection(
                    title="Geometry",
                    n_col=2,
                    sections=(
                        self._float_field("box_xy", "Box XY", minimum=0.001, decimals=4,
                            description=_help("box_xy")),
                        self._float_field("box_z", "Box Z", minimum=0.001, decimals=4,
                            description=_help("box_z")),
                        self._float_field("focus_w0", "Focus w0", minimum=0.001, decimals=4,
                            description=_help("focus_param")),
                        self._float_field("focus_z0", "Focus z0", minimum=0.001, decimals=4,
                            description=_help("focus_param")),
                        self._float_field("dt", "Step", minimum=0.000001, decimals=6,
                            description=_help("dt")),
                    ),
                ),
                ds.PanelSection(
                    title="Microtime",
                    n_col=2,
                    sections=(
                        ds.ValueSection(
                            attr="n_tac_channels",
                            label="TAC channels",
                            kind="int",
                            minimum=1,
                            maximum=1_000_000,
                            description=_help("N_tac_channels"),
                        ),
                        self._float_field("tac_dt", "TAC dt", minimum=0.000001, decimals=6,
                            description=_help("tac_dt")),
                        self._float_field(
                            "laser_period",
                            "Laser period",
                            minimum=0.000001,
                            decimals=6,
                            description=_help("laser_period"),
                        ),
                    ),
                ),
                ds.PanelSection(
                    title="Performance",
                    n_col=2,
                    description="Optional throughput knobs (native tttrlib Sim* engine). Speed/accuracy trade-offs — see each tooltip. Defaults reproduce the exact fixed-step engine.",
                    sections=(
                        ds.ToggleSection(
                            attr="per_molecule_skip", label="Coasting",
                            description=_help("per_molecule_skip"),
                        ),
                        ds.ToggleSection(
                            attr="fast_grid_bbox", label="Two-step field lookup",
                            description=_help("fast_grid_bbox"),
                        ),
                        ds.ToggleSection(
                            attr="independent_molecules", label="Independent molecules",
                            description=_help("independent_molecules"),
                        ),
                        ds.ToggleSection(
                            attr="analytic_excitation", label="Analytic Gaussian focus",
                            description=_help("analytic_excitation"),
                        ),
                        ds.ChoiceSection(
                            attr="psf_type", label="PSF / focus model",
                            options=("gaussian3d", "analytic_gaussian3d",
                                     "gaussian_lorentzian", "radial"),
                            labels=("Gaussian 3D", "Analytic Gaussian",
                                    "Gaussian-Lorentzian", "Numeric (radial)"),
                            description=_help("psf_type"),
                        ),
                        self._float_field(
                            "psf_zR", "Rayleigh range zR (µm)", minimum=0.0, decimals=3,
                            description=_help("psf_zR"),
                        ),
                        ds.ValueSection(
                            attr="psf_file", label="PSF file (radial)", kind="string",
                            description=_help("psf_file"),
                        ),
                        self._float_field(
                            "psf_r_step", "PSF r step (µm)", minimum=0.0, decimals=4,
                            description=_help("psf_r_step"),
                        ),
                        self._float_field(
                            "psf_z_step", "PSF z step (µm)", minimum=0.0, decimals=4,
                            description=_help("psf_z_step"),
                        ),
                        self._float_field(
                            "active_margin", "Active margin (µm)", minimum=0.0, decimals=3,
                            description=_help("active_margin"),
                        ),
                        ds.ValueSection(
                            attr="max_windows", label="Max windows", kind="int", minimum=0,
                            maximum=2_000_000_000,
                            description=_help("max_windows"),
                        ),
                        self._float_field(
                            "coast_safety", "Coast safety", minimum=1.0, decimals=2,
                            description=_help("coast_safety"),
                        ),
                        ds.ValueSection(
                            attr="min_coast_windows", label="Min coast windows", kind="int",
                            minimum=1, maximum=1_000_000,
                            description=_help("min_coast_windows"),
                        ),
                        self._float_field(
                            "focus_threshold", "Focus threshold", minimum=0.0, decimals=6,
                            description=_help("focus_threshold"),
                        ),
                    ),
                ),
                ds.ButtonRowSection(
                    buttons=(
                        {"label": "Load JSON", "action": "load_json"},
                        {"label": "Save JSON", "action": "save_json"},
                        {"label": "View JSON", "action": "view_json"},
                    )
                ),
            )
        )

    @staticmethod
    def _float_field(
        attr: str,
        label: str,
        minimum: float = 0.0,
        maximum: float = 1_000_000.0,
        decimals: int = 4,
        description: str = "",
    ) -> ds.ValueSection:
        """Create a bounded float ``ValueSection``.

        Parameters
        ----------
        attr : str
            Bound model attribute.
        label : str
            Form label.
        minimum : float, optional
            Minimum accepted value.
        maximum : float, optional
            Maximum accepted value.
        decimals : int, optional
            Number of decimal places.

        Returns
        -------
        chisurf.core.dataspec.ValueSection
            Configured float field.
        """
        return ds.ValueSection(
            attr=attr,
            label=label,
            kind="float",
            minimum=minimum,
            maximum=maximum,
            decimals=decimals,
            description=description,
        )

    def _enabled_channel_values(self) -> tuple[list[float], list[float], list[int]]:
        """Return brightness, background, and detector mappings.

        Returns
        -------
        tuple
            ``(q, q_bg, ch_conversion)`` for enabled color channels.
        """
        if not (self.green_enabled or self.red_enabled or self.yellow_enabled):
            self.green_enabled = True
        q = []
        q_bg = []
        ch_conversion = []
        next_source = 0
        for enabled, values, bg_values, detectors in (
            (
                self.green_enabled,
                (self.q_green_p, self.q_green_s),
                (self.bg_green_p, self.bg_green_s),
                (8, 9),
            ),
            (
                self.red_enabled,
                (self.q_red_p, self.q_red_s),
                (self.bg_red_p, self.bg_red_s),
                (10, 11),
            ),
            (
                self.yellow_enabled,
                (self.q_yellow_p, self.q_yellow_s),
                (self.bg_yellow_p, self.bg_yellow_s),
                (12, 13),
            ),
        ):
            if not enabled:
                continue
            for value, bg_value, detector in zip(values, bg_values, detectors):
                q.append(float(value))
                q_bg.append(float(bg_value))
                ch_conversion.extend([int(detector), int(next_source)])
                next_source += 1
        return q, q_bg, ch_conversion

    def to_parameters(self) -> dict[str, Any]:
        """Serialize the view model to acquisition ``simulation_params``.

        Returns
        -------
        dict
            Parameters consumed by ``SimulationDevice`` and the tttrlib backend.
        """
        q_one_species, q_bg, ch_conversion = self._enabled_channel_values()
        n_channels = len(q_one_species)
        n_species = max(1, int(self.n_species))
        q = q_one_species * n_species
        return {
            "N_species": n_species,
            "M": [float(self.molecules)] * n_species,
            "D": [float(self.diffusion)] * n_species,
            "N_channels": n_channels,
            "q": q,
            "q_bg": q_bg,
            "k_rad": [0.0] * (n_species * n_species),
            "k_nrad": [0.0] * (n_species * n_species),
            "box_xy": float(self.box_xy),
            "box_z": float(self.box_z),
            "focus_type": 0,
            "focus_param": [float(self.focus_w0), float(self.focus_z0)],
            "dt": float(self.dt),
            "tw": float(self.dt),
            "N_ph_max": max(1, int(self.n_ph_max)),
            "N_ph_per_file": max(1, int(self.n_ph_per_file)),
            "spc_output_path": str(self.output_path or ""),
            "pulsed_exc": 1 if self.excitation_mode == "Pulsed" else 0,
            "excitation_mode": str(self.excitation_mode),
            "ch_conversion": ch_conversion,
            "N_tac_channels": max(1, int(self.n_tac_channels)),
            "tac_dt": float(self.tac_dt),
            "laser_period": float(self.laser_period),
            "rmt1seed": int(self.seed_diffusion),
            "rmt2seed": int(self.seed_emission),
            "green_enabled": bool(self.green_enabled),
            "red_enabled": bool(self.red_enabled),
            "yellow_enabled": bool(self.yellow_enabled),
            "max_windows": int(self.max_windows),
            "analytic_excitation": bool(self.analytic_excitation),
            "per_molecule_skip": bool(self.per_molecule_skip),
            "fast_grid_bbox": bool(self.fast_grid_bbox),
            "independent_molecules": bool(self.independent_molecules),
            "active_margin": float(self.active_margin),
            "coast_safety": float(self.coast_safety),
            "min_coast_windows": int(self.min_coast_windows),
            "focus_threshold": float(self.focus_threshold),
            "psf_type": str(self.psf_type),
            "psf_zR": float(self.psf_zR),
            "psf_file": str(self.psf_file or ""),
            "psf_r_step": float(self.psf_r_step),
            "psf_z_step": float(self.psf_z_step),
        }

    def apply_parameters(self, params: dict[str, Any]) -> None:
        """Apply a parameter dictionary to this model.

        Parameters
        ----------
        params : dict
            Parameter dictionary to load.
        """
        fresh = self.from_parameters(params)
        self.__dict__.update(fresh.__dict__)

    def load_json(self) -> None:
        """Load simulation parameters from a JSON file selected by the user."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Load simulation JSON",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        with open(path, encoding="utf-8") as handle:
            self.apply_parameters(json.load(handle))
        sync = getattr(self, "_sync_fields", None)
        if callable(sync):
            sync()

    def save_json(self) -> None:
        """Save the current simulation parameters to a JSON file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save simulation JSON",
            "simulation_config.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_parameters(), handle, indent=2)

    def view_json(self) -> None:
        """Show the current serialized parameter JSON."""
        QtWidgets.QMessageBox.information(
            None,
            "Simulation Parameters",
            json.dumps(self.to_parameters(), indent=2),
        )


class EnhancedSimulationSetupDialog(QtWidgets.QDialog):
    """AutoForm-backed setup dialog for tttrlib simulation parameters."""

    def __init__(self, device=None, parent=None):
        """Initialize the setup dialog.

        Parameters
        ----------
        device : object, optional
            Simulation device or acquisition wrapper carrying
            ``simulation_params``.
        parent : QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.device = device
        self.setWindowTitle("Simulation Setup")
        self.resize(720, 720)
        self.model = SimulationSettingsModel.from_parameters(self._device_parameters())
        self.form = AutoForm(self.model)
        self.model._sync_fields = self.form.sync_fields
        self._build_ui()

    def _device_parameters(self) -> dict[str, Any]:
        """Read simulation parameters from the device-like object.

        Returns
        -------
        dict
            Existing parameters or an empty dictionary.
        """
        if self.device is None:
            return {}
        if hasattr(self.device, "simulation_params"):
            return dict(getattr(self.device, "simulation_params") or {})
        inner = getattr(self.device, "device", None)
        if inner is not None and hasattr(inner, "simulation_params"):
            return dict(getattr(inner, "simulation_params") or {})
        return {}

    def _build_ui(self) -> None:
        """Build the dialog around the AutoForm widget."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.form, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _flush_editor_focus(self) -> None:
        """Commit any AutoForm editor that is waiting for focus-out."""
        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()

    def get_parameters(self) -> dict[str, Any]:
        """Return current simulation parameters.

        Returns
        -------
        dict
            Parameters consumed by ``SimulationDevice``.
        """
        self._flush_editor_focus()
        return self.model.to_parameters()

    def _apply_parameters(self, params: dict[str, Any]) -> None:
        """Apply saved parameters to the form.

        Parameters
        ----------
        params : dict
            Saved simulation parameter dictionary.
        """
        self.model.apply_parameters(params)
        self.form.sync_fields()

    def accept(self) -> None:
        """Accept the dialog and update the attached device parameters."""
        params = self.get_parameters()
        target = self.device
        if target is not None and hasattr(target, "simulation_params"):
            target.simulation_params.update(params)
        inner = getattr(target, "device", None) if target is not None else None
        if inner is not None and hasattr(inner, "simulation_params"):
            inner.simulation_params.update(params)
        super().accept()

    def load_json(self) -> None:
        """Load JSON parameters into the form."""
        self.model.load_json()
        self.form.sync_fields()

    def save_json(self) -> None:
        """Save the current form parameters as JSON."""
        self.model.save_json()

    def view_json(self) -> None:
        """Show the current form parameters as JSON."""
        self.model.view_json()


SimulationSetupDialog = EnhancedSimulationSetupDialog
