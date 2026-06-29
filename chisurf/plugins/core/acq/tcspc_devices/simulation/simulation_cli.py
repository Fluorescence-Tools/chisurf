"""Unified TCSPC simulation CLI with separate steps.

This script combines two logical steps:

1. ``config`` – generate a JSON configuration for the simulation
   device (compatible with EnhancedSimulationSetupDialog.get_parameters()).
2. ``run`` – read such a JSON file and generate SPC-132 files using the
   Burbulator DLL.

The JSON schema matches the GUI dialog so configurations can be moved
between CLI and GUI without conversion.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import click

try:
    # When executed as a module (e.g. ``python -m chisurf.plugins...``)
    # use the package-relative import.
    from .burbulator_dll_wrapper import BurbulatorDLL, BurbulatorError
except ImportError:  # pragma: no cover - script mode fallback
    # When executed as a standalone script (``python simulation_cli.py``)
    # fall back to importing from the same directory.
    from burbulator_dll_wrapper import BurbulatorDLL, BurbulatorError


def make_json_serializable(data: Any) -> Any:
    """Recursively convert NumPy types/arrays to Python standard types for JSON serialization."""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(x) for x in data]
    elif hasattr(data, 'tolist'):  # Handles NumPy arrays and scalars
        return data.tolist()
    elif hasattr(data, 'item'):  # Handles NumPy scalars
        return data.item()
    else:
        return data


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Any] = {
    # Species parameters
    "N_species": 1,
    "M": [5.0],
    "D": [3.0],

    # Detection parameters - 6 channels for green/red/yellow P/S
    "N_channels": 6,
    "q": [50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
    "q_bg": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],

    # Species transitions (radiative/non-radiative) - 2D arrays for NxN transitions
    "k_rad": [[0.0]],
    "k_nrad": [[0.0]],

    # Geometry
    "box_xy": 2.0,
    "box_z": 4.0,
    "focus_type": 0,
    "focus_param": [0.3, 2.0],

    # Simulation timing
    "dt": 0.01,
    "N_ph_max": 50000,

    # Excitation mode
    "excitation_mode": "CW",

    # TAC/IRF parameters
    "N_tac_channels": 4096,
    "tac_dt": 0.004069,
    "laser_period": 13.596,
    "use_gaussian_irf": True,
    "gaussian_irf_fwhm": 0.11,
    "gaussian_irf_mean": 1.5,
    "gaussian_irf_sigma": 0.0467,
    "irf_file": "",

    # Anisotropy
    "r0": 0.38,
    "g_factor": 1.0,
    "l1": 0.0308,
    "l2": 0.0368,

    # Background/scattering
    "parallel_scatter": 0.0,
    "perp_scatter": 0.0,
    "parallel_dark": 0.0,
    "perp_dark": 0.0,

    # Output
    "spc_output_path": "",
    "N_ph_per_file": 50000,

    # BH_SPC conversion - 6 channel mapping
    "pulsed_exc": 0,
    "ch_conversion": [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5],

    # RNG parameters
    "rng_mode": 0,
    "rmt1seed": 12345,
    "rmt2seed": 67890,

    # Channel enable flags
    "green_enabled": True,
    "red_enabled": False,
    "yellow_enabled": False,

    # Fluorescence decay parameters
    "decay_lifetimes": [[4.0, 4.0, 4.0]],
    "decay_patterns": [""],
    "rotational_correlation_times": [[0.4, 0.4, 0.4]],

    # Dark state parameters
    "k_bd": [0.0],
    "k_bb": [0.0],
    "k_db": [0.0],
    "darkstate_interconvert": False,

    # Focus-specific parameters
    "w0": 0.3,
    "z0": 2.0,
    "Rph": 0.15,
    "z0_CEF": 1.0,

    # Additional simulation parameters
    "tw": 0.01,
}


def _resize_list(values: List[float], n: int, default: float) -> List[float]:
    values = list(values)
    if len(values) >= n:
        return values[:n]
    return values + [default] * (n - len(values))


def _init_kinetics(n_species: int) -> Dict[str, Any]:
    """Create default kinetic scheme arrays for a given number of species."""

    zeros_matrix = [[0.0 for _ in range(n_species)] for _ in range(n_species)]
    k_rad = zeros_matrix
    k_nrad = zeros_matrix

    k_bd = [0.0 for _ in range(n_species)]
    k_bb = [0.0 for _ in range(n_species)]
    k_db = [0.0 for _ in range(n_species)]

    return {
        "k_rad": k_rad,
        "k_nrad": k_nrad,
        "k_bd": k_bd,
        "k_bb": k_bb,
        "k_db": k_db,
    }


def _flatten_rates(matrix: Any, n_species: int) -> List[float]:
    """Flatten a k_rad / k_nrad matrix from JSON into a 1D list."""

    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        flat: List[float] = []
        for i in range(n_species):
            row = matrix[i] if i < len(matrix) else []
            for j in range(n_species):
                val = 0.0
                if j < len(row):
                    try:
                        val = float(row[j])
                    except Exception:
                        val = 0.0
                flat.append(val)
        return flat

    seq = [float(v) for v in (matrix or [])]
    needed = n_species * n_species
    if len(seq) >= needed:
        return seq[:needed]
    return seq + [0.0] * (needed - len(seq))


def _prepare_dll_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration dictionary for BurbulatorDLL.simulate_ov3."""

    n_species = int(config.get("N_species", 1))
    if n_species <= 0:
        raise click.ClickException("N_species must be positive in the configuration")

    N_channels = int(config.get("N_channels", 6))

    M = list(config.get("M", [5.0] * n_species))
    if len(M) < n_species:
        M.extend([M[-1]] * (n_species - len(M)))
    M = M[:n_species]

    D = list(config.get("D", [3.0] * n_species))
    if len(D) < n_species:
        D.extend([D[-1]] * (n_species - len(D)))
    D = D[:n_species]

    q = list(config.get("q", []))
    expected_q_len = n_species * N_channels
    if len(q) < expected_q_len:
        q.extend([0.0] * (expected_q_len - len(q)))
    q = q[:expected_q_len]

    q_bg = list(config.get("q_bg", []))
    if len(q_bg) < expected_q_len:
        q_bg.extend([0.0] * (expected_q_len - len(q_bg)))
    q_bg = q_bg[:expected_q_len]

    k_rad = _flatten_rates(config.get("k_rad", []), n_species)
    k_nrad = _flatten_rates(config.get("k_nrad", []), n_species)

    box_xy = float(config.get("box_xy", 2.0))
    box_z = float(config.get("box_z", 4.0))
    focus_type = int(config.get("focus_type", 0))
    focus_param = list(config.get("focus_param", [0.3, 2.0]))

    dt = float(config.get("dt", 0.01))
    N_ph_max = int(config.get("N_ph_max", 50000))

    return {
        "N_species": n_species,
        "M": M,
        "D": D,
        "N_channels": N_channels,
        "q": q,
        "q_bg": q_bg,
        "k_rad": k_rad,
        "k_nrad": k_nrad,
        "box_xy": box_xy,
        "box_z": box_z,
        "focus_type": focus_type,
        "focus_param": focus_param,
        "dt": dt,
        "N_ph_max": N_ph_max,
    }


def _prepare_conversion_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare parameters for convert_to_spc132 (CW only)."""

    pulsed_exc = int(config.get("pulsed_exc", 0))
    if pulsed_exc:
        raise click.ClickException(
            "Pulsed excitation (pulsed_exc=1) is not yet supported by this CLI.\n"
            "Please use a CW configuration (excitation_mode='CW', pulsed_exc=0) "
            "or extend this tool to provide F/lookup tables."
        )

    N_channels = int(config.get("N_channels", 6))
    ch_conversion = list(
        config.get("ch_conversion", [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5])
    )

    N_tac_channels = int(config.get("N_tac_channels", 4096))
    tac_dt = float(config.get("tac_dt", 0.004069))
    laser_period = float(config.get("laser_period", 13.596))

    dt = float(config.get("dt", config.get("tw", 0.01)))

    return {
        "pulsed_exc": pulsed_exc,
        "N_channels": N_channels,
        "ch_conversion": ch_conversion,
        "N_tac_channels": N_tac_channels,
        "tac_dt": tac_dt,
        "laser_period": laser_period,
        "tw": dt,
    }


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group(help="TCSPC simulation CLI with separate 'config' and 'run' steps.")
def main() -> None:
    """Entry point for the unified CLI."""


# ---------------------------------------------------------------------------
# config subcommand
# ---------------------------------------------------------------------------


@main.command("config", help=(
    "Generate a JSON configuration for the TCSPC simulation device. "
    "Did you know that the EnhancedSimulationSetupDialog in the GUI "
    "uses the same parameter structure? This step lets you build such a "
    "configuration without opening the GUI."
))
@click.option(
    "--output", "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default="simulation_config.json",
    show_default=True,
    help="Path to write the JSON configuration file.",
)
@click.option(
    "--n-species",
    type=int,
    default=1,
    show_default=True,
    help="Number of molecular species in the simulation.",
)
@click.option(
    "--M",
    "M_values",
    type=float,
    multiple=True,
    help=(
        "Initial number of molecules per species. "
        "Provide one value per species; missing values are filled with the default."
    ),
)
@click.option(
    "--D",
    "D_values",
    type=float,
    multiple=True,
    help=(
        "Diffusion coefficient per species (µm^2/s). "
        "Provide one value per species; missing values are filled with the default."
    ),
)
@click.option(
    "--q-first-species",
    "q_first",
    type=float,
    multiple=True,
    help=(
        "Brightness for the first species, six values for "
        "[G_P, G_S, R_P, R_S, Y_P, Y_S]. "
        "If omitted, defaults from the GUI are used."
    ),
)
@click.option(
    "--excitation-mode",
    type=click.Choice(["CW", "Pulsed"], case_sensitive=False),
    default="CW",
    show_default=True,
    help="Excitation mode flag stored in the configuration.",
)
@click.option(
    "--n-ph-max",
    type=int,
    default=None,
    help="Override the maximum number of photons to simulate (N_ph_max).",
)
@click.option(
    "--spc-output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=None,
    help=(
        "Optional default output directory for SPC files. "
        "This can be overridden by the 'run' step."
    ),
)
@click.option(
    "--fret-states",
    type=int,
    default=0,
    show_default=False,
    help=(
        "Number of FRET states (>= 2). When >0, configures an N-state CW "
        "FRET/FCS system with donor (green) and acceptor (red) brightness "
        "gradient and symmetric nearest-neighbor exchange between states. "
        "Assumes N_channels=6 with [G_P, G_S, R_P, R_S, Y_P, Y_S]."
    ),
)
@click.option(
    "--tau0-ns",
    type=float,
    default=4.0,
    show_default=True,
    help=(
        "Donor lifetime in the absence of FRET, in ns. When used together "
        "with --fret-states and --fret-eff, the CLI computes per-state "
        "fluorescence lifetimes tau_i = tau0 * (1 - E_i) and stores them "
        "in decay_lifetimes."
    ),
)
@click.option(
    "--fret-eff",
    "fret_eff",
    type=float,
    multiple=True,
    help=(
        "FRET efficiencies E_i (0-1) for each FRET state when using "
        "--fret-states. If provided, must have exactly --fret-states "
        "values; lifetimes are computed from tau0-ns and E_i."
    ),
)
@click.option(
    "--exchange-rate-ms",
    type=float,
    default=1.0,
    show_default=True,
    help=(
        "Symmetric exchange rate in 1/ms for the FRET preset (--fret-states). "
        "Internally converted to 1/us (k = value / 1000)."
    ),
)
@click.option(
    "--irf-mean-ns",
    type=float,
    default=None,
    show_default=False,
    help=(
        "Mean of the Gaussian IRF in ns. If provided, overrides the default "
        "gaussian_irf_mean stored in the JSON."
    ),
)
@click.option(
    "--irf-sigma-ns",
    type=float,
    default=None,
    show_default=False,
    help=(
        "Sigma of the Gaussian IRF in ns. If provided, overrides the "
        "default gaussian_irf_sigma and updates gaussian_irf_fwhm = 2.3548*sigma."
    ),
)
@click.option(
    "--irf-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    show_default=False,
    help=(
        "Optional IRF file path. If set, the JSON will use this file "
        "(irf_file) and disable the Gaussian IRF (use_gaussian_irf = False)."
    ),
)
@click.option(
    "--fret-low-green",
    type=float,
    default=50.0,
    show_default=True,
    help=(
        "Brightness in green channels (G_P,G_S) for the lowest-FRET state "
        "when using --fret-states."
    ),
)
@click.option(
    "--fret-low-red",
    type=float,
    default=5.0,
    show_default=True,
    help=(
        "Brightness in red channels (R_P,R_S) for the lowest-FRET state "
        "when using --fret-states."
    ),
)
@click.option(
    "--fret-high-green",
    type=float,
    default=5.0,
    show_default=True,
    help=(
        "Brightness in green channels (G_P,G_S) for the highest-FRET state "
        "when using --fret-states."
    ),
)
@click.option(
    "--fret-high-red",
    type=float,
    default=50.0,
    show_default=True,
    help=(
        "Brightness in red channels (R_P,R_S) for the highest-FRET state "
        "when using --fret-states."
    ),
)
def config_cmd(
    output_path: str,
    n_species: int,
    M_values: List[float],
    D_values: List[float],
    q_first: List[float],
    excitation_mode: str,
    n_ph_max: int | None,
    spc_output_dir: str | None,
    fret_states: int,
    tau0_ns: float,
    fret_eff: List[float],
    exchange_rate_ms: float,
    fret_low_green: float,
    fret_low_red: float,
    fret_high_green: float,
    fret_high_red: float,
    irf_mean_ns: float | None,
    irf_sigma_ns: float | None,
    irf_file: str | None,
) -> None:
    """Generate simulation configuration JSON (step 1)."""

    if n_species <= 0:
        raise click.ClickException("n-species must be positive")

    params: Dict[str, Any] = dict(DEFAULT_PARAMS)

    # If an N-state FRET preset is requested, it defines the number of
    # species. Otherwise, use the user-provided n_species.
    if fret_states > 0:
        if fret_states < 2:
            raise click.ClickException("--fret-states must be >= 2 when used")
        n_species = fret_states

    params["N_species"] = n_species

    params["M"] = _resize_list(M_values or params["M"], n_species, params["M"][0])
    params["D"] = _resize_list(D_values or params["D"], n_species, params["D"][0])

    default_lifetimes = params.get("decay_lifetimes", [[4.0, 4.0, 4.0]])[0]
    default_rot = params.get("rotational_correlation_times", [[0.4, 0.4, 0.4]])[0]
    params["decay_lifetimes"] = [list(default_lifetimes) for _ in range(n_species)]
    params["rotational_correlation_times"] = [list(default_rot) for _ in range(n_species)]
    params["decay_patterns"] = ["" for _ in range(n_species)]

    kinetics = _init_kinetics(n_species)
    params.update(kinetics)

    n_channels = params["N_channels"]
    total_q_len = n_species * n_channels
    q = [0.0] * total_q_len

    default_q = params.get("q", [50.0, 50.0, 0.0, 0.0, 0.0, 0.0])
    if len(default_q) < n_channels:
        default_q = default_q + [0.0] * (n_channels - len(default_q))
    for i in range(n_channels):
        q[i] = default_q[i]

    if q_first:
        if len(q_first) != n_channels:
            raise click.ClickException(
                f"--q-first-species requires exactly {n_channels} values, "
                f"got {len(q_first)}."
            )
        for i, val in enumerate(q_first):
            q[i] = float(val)

    params["q"] = q

    # Optional N-state FRET/FCS preset: N species, same diffusion,
    # brightness gradient from low- to high-FRET, and symmetric
    # nearest-neighbor exchange between states.
    if fret_states > 0:
        if n_channels != 6:
            raise click.ClickException(
                "--fret-states currently assumes N_channels=6 with "
                "[G_P, G_S, R_P, R_S, Y_P, Y_S]."
            )

        rate_per_us = float(exchange_rate_ms) / 1000.0
        if rate_per_us < 0.0:
            raise click.ClickException(
                "--exchange-rate-ms must be non-negative"
            )

        # Linearly interpolate brightness between endpoints for each
        # FRET state index i in [0, fret_states-1].
        def _lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        q_fret: List[float] = []
        for i in range(fret_states):
            t = i / (fret_states - 1) if fret_states > 1 else 0.0

            green = _lerp(float(fret_low_green), float(fret_high_green), t)
            red = _lerp(float(fret_low_red), float(fret_high_red), t)

            q_fret.extend([
                green, green,  # G_P, G_S
                red, red,      # R_P, R_S
                0.0, 0.0,      # Y_P, Y_S (unused)
            ])

        params["q"] = q_fret

        # Symmetric nearest-neighbor exchange matrix in k_nrad.
        k_nrad = [[0.0 for _ in range(fret_states)] for _ in range(fret_states)]
        for i in range(fret_states - 1):
            k_nrad[i][i + 1] = rate_per_us
            k_nrad[i + 1][i] = rate_per_us
        params["k_nrad"] = k_nrad

        # Radiative matrix remains zero for this preset.
        params["k_rad"] = [[0.0 for _ in range(fret_states)] for _ in range(fret_states)]

        # Optional: compute per-state fluorescence lifetimes from tau0 and
        # user-specified FRET efficiencies E_i (0-1) via
        #   tau_i = tau0 * (1 - E_i)
        # Only affects the JSON metadata (decay_lifetimes); the CW DLL path
        # uses only the kinetic parameters.
        if fret_eff:
            if len(fret_eff) != fret_states:
                raise click.ClickException(
                    "--fret-eff must have exactly --fret-states values "
                    f"(got {len(fret_eff)} for {fret_states} states)."
                )

            lifetimes = []
            for i, E in enumerate(fret_eff):
                if not (0.0 <= E <= 1.0):
                    raise click.ClickException(
                        f"FRET efficiency E[{i}]={E} is outside [0,1]."
                    )
                tau_i = float(tau0_ns) * (1.0 - float(E))
                lifetimes.append(tau_i)

            # Update decay_lifetimes per species (first component used).
            dl = params.get("decay_lifetimes", [])
            if not dl or len(dl) != fret_states:
                dl = [list(default_lifetimes) for _ in range(fret_states)]

            for i, tau_i in enumerate(lifetimes):
                if not dl[i]:
                    dl[i] = list(default_lifetimes)
                dl[i][0] = tau_i

            params["decay_lifetimes"] = dl

    excitation_mode = excitation_mode.upper()
    params["excitation_mode"] = excitation_mode
    params["pulsed_exc"] = 1 if excitation_mode == "PULSED" else 0

    if n_ph_max is not None:
        if n_ph_max <= 0:
            raise click.ClickException("n-ph-max must be positive if specified")
        params["N_ph_max"] = int(n_ph_max)

    if spc_output_dir:
        params["spc_output_path"] = os.path.abspath(spc_output_dir)

    # Apply IRF / convolution settings.
    # For CW mode the DLL does not currently use the IRF, but these
    # fields keep the JSON consistent with the GUI and future pulsed
    # extensions.
    if irf_file:
        # External IRF from file takes precedence over Gaussian IRF.
        params["use_gaussian_irf"] = False
        params["irf_file"] = os.path.abspath(irf_file)
    else:
        if irf_mean_ns is not None:
            params["gaussian_irf_mean"] = float(irf_mean_ns)
        if irf_sigma_ns is not None:
            sigma = float(irf_sigma_ns)
            params["gaussian_irf_sigma"] = sigma
            params["gaussian_irf_fwhm"] = 2.354820045 * sigma

    out_dir = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    click.echo(f"Simulation configuration written to {os.path.abspath(output_path)}")
    click.echo(
        "Did you know: you can also load this JSON in the GUI setup dialog "
        "or use it directly with the DLL-based 'run' step."
    )


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


@main.command("run", help=(
    "Run a TCSPC simulation from a JSON configuration and write SPC files. "
    "The configuration can be generated by the GUI setup dialog or by the "
    "'config' step of this script."
))
@click.argument(
    "config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=None,
    help=(
        "Optional override for the SPC output directory. "
        "If omitted, the CLI uses 'spc_output_path' from the JSON or "
        "creates a folder next to the config file."
    ),
)
@click.option(
    "--n-ph-max",
    type=int,
    default=None,
    help="Optional override of N_ph_max (maximum photons) for the simulation.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help=(
        "Optional override of photons per SPC file (N_ph_per_file). "
        "If omitted, the JSON value or a reasonable default is used."
    ),
)
def run_cmd(
    config_file: str,
    output_dir: str | None,
    n_ph_max: int | None,
    batch_size: int | None,
) -> None:
    """Run DLL-based simulation and write SPC files (step 2)."""

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    dll_params = _prepare_dll_params(config)
    conv_params = _prepare_conversion_params(config)

    if n_ph_max is not None:
        if n_ph_max <= 0:
            raise click.ClickException("n-ph-max must be positive if specified")
        dll_params["N_ph_max"] = int(n_ph_max)

    if output_dir:
        spc_output_path = os.path.abspath(output_dir)
    else:
        spc_output_path = config.get("spc_output_path") or os.path.join(
            os.path.dirname(os.path.abspath(config_file)),
            "simulation_output",
        )
    os.makedirs(spc_output_path, exist_ok=True)

    # Write the config JSON to the output directory to enable reproduction of the simulation
    config_output_path = os.path.join(spc_output_path, "simulation_config.json")
    try:
        serializable_config = make_json_serializable(config)
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2)
        click.echo(f"Simulation configuration written to {config_output_path}")
    except Exception as e:
        click.echo(f"Warning: Failed to write simulation config file to output directory: {e}")

    N_ph_per_file = batch_size or int(config.get("N_ph_per_file", 100000))
    if N_ph_per_file <= 0:
        raise click.ClickException("batch-size / N_ph_per_file must be positive")

    try:
        dll = BurbulatorDLL()
    except (FileNotFoundError, BurbulatorError) as e:
        raise click.ClickException(str(e))

    click.echo(
        f"Running Burbulator simulation: N_species={dll_params['N_species']}, "
        f"N_channels={dll_params['N_channels']}, N_ph_max={dll_params['N_ph_max']}"
    )

    sim = dll.simulate_ov3(
        Nspecies=dll_params["N_species"],
        M=dll_params["M"],
        D=dll_params["D"],
        Nchannels=dll_params["N_channels"],
        q=dll_params["q"],
        q_bg=dll_params["q_bg"],
        k_rad=dll_params["k_rad"],
        k_nrad=dll_params["k_nrad"],
        box_xy=dll_params["box_xy"],
        box_z=dll_params["box_z"],
        focus_type=dll_params["focus_type"],
        focus_param=dll_params["focus_param"],
        dt=dll_params["dt"],
        N_ph_max=dll_params["N_ph_max"],
        rmt1seed=int(config.get("rmt1seed", 12345)),
        rmt2seed=int(config.get("rmt2seed", 67890)),
    )

    N_ph = int(sim.get("N_ph", 0))
    click.echo(f"DLL returned {N_ph} photons")

    if N_ph <= 0:
        click.echo("No photons generated; nothing to write.")
        return

    data_T = sim["data_T"]
    data_t = sim["data_t"]
    data_N = sim["data_N"]
    data_species = sim["data_species"]
    data_molecule = sim["data_molecule"]

    filenumber = 0
    photon_start = 0

    while photon_start < N_ph:
        photon_end = min(photon_start + N_ph_per_file, N_ph)
        batch_photons = photon_end - photon_start

        batch_data_T = data_T[photon_start:photon_end]
        batch_data_t = data_t[photon_start:photon_end]
        batch_data_N = data_N[photon_start:photon_end]
        batch_data_species = data_species[photon_start:photon_end]
        batch_data_molecule = data_molecule[photon_start:photon_end]

        spc_bytes, MT_ov, spc_i = dll.convert_to_spc132(
            pulsed_exc=conv_params["pulsed_exc"],
            Nchannels=conv_params["N_channels"],
            data_T=batch_data_T,
            data_t=batch_data_t,
            data_N=batch_data_N,
            data_species=batch_data_species,
            data_molecule=batch_data_molecule,
            tw=conv_params["tw"],
            ch_conversion=conv_params["ch_conversion"],
            N_tac_channels=conv_params["N_tac_channels"],
            tac_dt=conv_params["tac_dt"],
            laser_period=conv_params["laser_period"],
            N_photons=batch_photons,
        )

        filename = os.path.join(spc_output_path, f"m{filenumber:03d}.spc")
        dll.write_spc132_file(filename, spc_bytes)

        click.echo(
            f"Wrote {len(spc_bytes)} bytes (MT_ov={MT_ov}, photons={batch_photons}) "
            f"to {filename}"
        )

        filenumber += 1
        photon_start = photon_end

    click.echo(
        f"Simulation completed, wrote {filenumber} SPC file(s) to {spc_output_path}"
    )


# ---------------------------------------------------------------------------
# pipeline subcommand    (import runtime to avoid circular deps at module level)
# ---------------------------------------------------------------------------
try:
    from .pipeline_debug import pipeline_cmd
    main.add_command(pipeline_cmd, "pipeline")
except ImportError:
    pass


if __name__ == "__main__":
    main()
