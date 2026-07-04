"""Qt-free simulation core over tttrlib's photon simulator.

Maps the acquisition plugin's flat ``simulation_params`` dict onto tttrlib
``SimEngine`` objects and produces Becker & Hickl SPC-132 records as a ``uint32``
array — the same word format the streaming ``SimulationDevice`` feeds through
``read_fifo``. Replaces the ctypes/Burbulator-DLL path; no Qt, headless-testable.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict

import numpy as np


def tttrlib_available() -> bool:
    """Return whether the installed ``tttrlib`` exposes photon simulation.

    Returns
    -------
    bool
        ``True`` when the runtime has the simulator classes required by this
        acquisition backend.
    """
    try:
        import tttrlib  # noqa: F401
        required = (
            "SimEngine",
            "SimSystem",
            "SimSpecies",
            "SimGrid",
            "SimIntegrator",
            "SimMicrotimeEncoder",
        )
        return all(hasattr(tttrlib, name) for name in required)
    except Exception:
        return False


def _vd(values):
    """Convert numeric values to a ``tttrlib.VectorDouble``.

    Parameters
    ----------
    values : iterable
        Values that can be converted to floats.

    Returns
    -------
    tttrlib.VectorDouble
        C++ vector proxy consumed by ``tttrlib``.
    """
    import tttrlib

    return tttrlib.VectorDouble([float(v) for v in values])


def _flatten(values: Any) -> list[float]:
    """Flatten legacy nested list settings into one numeric vector.

    Parameters
    ----------
    values : object
        Scalar, list, tuple, or NumPy-like object.

    Returns
    -------
    list of float
        Flat list suitable for tttrlib vector construction.
    """
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        out = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [float(values)]


def _sized(values: Any, size: int, default: float = 0.0) -> list[float]:
    """Return a flat list padded or truncated to ``size``.

    Parameters
    ----------
    values : object
        Scalar or sequence value.
    size : int
        Desired output length.
    default : float, optional
        Padding value.

    Returns
    -------
    list of float
        Vector with exactly ``size`` entries.
    """
    flat = _flatten(values)
    if len(flat) < size:
        flat = flat + [float(default)] * (size - len(flat))
    return flat[:size]


def build_engine(params: Dict[str, Any]):
    """Build a configured ``tttrlib.SimEngine`` from plugin parameters.

    Parameters
    ----------
    params : dict
        Acquisition simulation parameter dictionary.

    Returns
    -------
    tttrlib.SimEngine
        Configured simulator ready for ``run()``.
    """
    import tttrlib

    ns = int(params.get("N_species", 1))
    nc = int(params.get("N_channels", 2))
    q = _sized(params.get("q", [50.0] * (ns * nc)), ns * nc, 0.0)
    D = _sized(params.get("D", [3.0] * ns), ns, 0.0)
    M = _sized(params.get("M", [50.0] * ns), ns, 0.0)
    box_xy = float(params.get("box_xy", 2.0))
    box_z = float(params.get("box_z", 4.0))
    dt = float(params.get("dt", 0.01))
    focus = list(params.get("focus_param", [0.3, 2.0]))
    w0 = float(focus[0]) if len(focus) > 0 else 0.3
    z0 = float(focus[1]) if len(focus) > 1 else 2.0

    sample = tttrlib.SimSystem()
    for i in range(ns):
        sp = tttrlib.SimSpecies()
        sp.D = float(D[i])
        sp.q = _vd(q[i * nc:(i + 1) * nc])
        sp.r0 = float(params.get("r0", 0.0))
        sp.l1 = float(params.get("l1", 0.0))
        sp.l2 = float(params.get("l2", 0.0))
        sp.D_rot = float(params.get("D_rot", 0.0))
        sample.add_species(sp)

    k_rad = _sized(params.get("k_rad", [0.0] * (ns * ns)), ns * ns, 0.0)
    k_nrad = _sized(params.get("k_nrad", [0.0] * (ns * ns)), ns * ns, 0.0)
    # tttrlib 0.27.0's SWIG binding accepts Python sequences for these two
    # setters, while explicit VectorDouble proxies are rejected on some builds.
    sample.set_rate_matrices(k_rad, k_nrad)
    sample.set_background(_sized(params.get("q_bg", [0.0] * nc), nc, 0.0))
    sample.set_box(box_xy, box_z)
    for i in range(ns):
        sample.set_population(i, float(M[i]))

    # Excitation focus / point-spread function. Selected by ``psf_type``:
    #   • "gaussian3d"          separable 3D Gaussian (w0, z0) on a trilinear voxel grid;
    #   • "analytic_gaussian3d" the same Gaussian evaluated on the fly (no voxels/construction);
    #   • "gaussian_lorentzian" confocal MDF with a z-expanding waist (w0, Rayleigh range zR);
    #   • "radial"              numeric/measured radially-symmetric PSF from a file (r,z grid).
    # The grid extent defaults to the box, but may be sized to the focus (``excitation_extent``)
    # since the field is ~0 outside it — far fewer voxels, exact, ~50x cheaper to build.
    spacing = max(min(w0, z0) / 4.0, 1e-3)
    ext = params.get("excitation_extent", None)
    ext_xy = float(ext[0]) if isinstance(ext, (list, tuple)) and len(ext) > 0 else box_xy
    ext_z = float(ext[1]) if isinstance(ext, (list, tuple)) and len(ext) > 1 else box_z

    psf_type = str(params.get("psf_type", "gaussian3d")).strip().lower()
    # Legacy: the ``analytic_excitation`` flag selects the analytic Gaussian for the default PSF.
    if params.get("analytic_excitation", False) and psf_type == "gaussian3d":
        psf_type = "analytic_gaussian3d"

    if psf_type in ("gaussian_lorentzian", "gauss_lorentz") and hasattr(
        tttrlib.SimGrid, "gaussian_lorentzian"
    ):
        zR = float(params.get("psf_zR", z0))
        excitation = tttrlib.SimGrid.gaussian_lorentzian(w0, zR, ext_xy, ext_z, spacing, 1.0)
    elif (
        psf_type in ("radial", "numeric", "measured")
        and params.get("psf_file")
        and hasattr(tttrlib.SimGrid, "numeric_from_file")
    ):
        excitation = tttrlib.SimGrid.numeric_from_file(
            str(params["psf_file"]),
            r_step=float(params.get("psf_r_step", spacing)),
            z_step=float(params.get("psf_z_step", spacing)),
            extent_xy=ext_xy, extent_z=ext_z, spacing=spacing,
        )
    elif psf_type == "analytic_gaussian3d" and hasattr(tttrlib.SimGrid, "analytic_gaussian3d"):
        excitation = tttrlib.SimGrid.analytic_gaussian3d(w0, z0, 1.0)
    else:  # "gaussian3d" (default / fallback)
        excitation = tttrlib.SimGrid.gaussian3d(w0, z0, ext_xy, ext_z, spacing, 1.0)

    settings = tttrlib.SimIntegrator()
    settings.dt = dt
    settings.n_channels = nc
    settings.n_ph_max = int(params.get("N_ph_max", 1_000_000))
    settings.max_windows = int(params.get("max_windows", 0))
    settings.seed_diffusion = int(params.get("rmt1seed", 12345))
    settings.seed_emission = int(params.get("rmt2seed", 54321))
    settings.n_microtime_channels = int(params.get("N_tac_channels", 4096))
    settings.microtime_resolution = float(params.get("tac_dt", 0.004069))
    settings.laser_period = float(params.get("laser_period", 13.596))

    # Opt-in throughput knobs (native tttrlib Sim* engine; PRD-007). Guarded with hasattr so
    # an older tttrlib without them still runs — the flag is simply ignored there.
    _throughput = (
        ("per_molecule_skip", "per_molecule_skip", bool, False),
        ("coast_safety", "coast_safety", float, 3.0),
        ("min_coast_windows", "min_coast_windows", int, 8),
        ("focus_threshold", "focus_threshold", float, 1e-3),
        ("fast_grid_bbox", "fast_grid_bbox", bool, False),
        ("independent_molecules", "independent_molecules", bool, False),
        ("active_margin", "active_margin", float, 0.0),
    )
    for attr, key, cast, default in _throughput:
        if hasattr(settings, attr):
            setattr(settings, attr, cast(params.get(key, default)))

    return tttrlib.SimEngine(sample, excitation, tttrlib.VectorSimGrid([]), settings)


def generate_spc132_uint32(params: Dict[str, Any]) -> np.ndarray:
    """Run tttrlib and return encoded SPC records as ``uint32`` words.

    Parameters
    ----------
    params : dict
        Acquisition simulation parameter dictionary.

    Returns
    -------
    numpy.ndarray
        Encoded SPC records viewed as unsigned 32-bit words.
    """
    engine = build_engine(params)
    engine.run()
    return encode_records(engine, params)


def encode_records(engine, params: Dict[str, Any]) -> np.ndarray:
    """Encode an already-run ``SimEngine`` as SPC-132 ``uint32`` words.

    Split out from :func:`generate_spc132_uint32` so callers that already ran the
    engine (e.g. the RPC handler, which also needs ``engine.n_photons()``) encode
    without simulating twice.
    """
    import tttrlib

    enc = tttrlib.SimMicrotimeEncoder()
    enc.pulsed_exc = int(params.get("pulsed_exc", 0))
    enc.n_channels = int(params.get("N_channels", 2))
    enc.tw = float(params.get("dt", 0.01))
    enc.ch_conversion = tttrlib.VectorUint16([
        int(v) for v in params.get("ch_conversion", [8, 0, 9, 1, 10, 2])
    ])
    enc.n_microtime_channels = int(params.get("N_tac_channels", 4096))
    enc.microtime_resolution = float(params.get("tac_dt", 0.004069))
    enc.laser_period = float(params.get("laser_period", 13.596))

    rng = tttrlib.SimRandom(int(params.get("rmt2seed", 54321)))
    rec = engine.encode(enc, rng)
    b = bytes(bytearray(rec.bytes))
    # SPC-132 records are 4 bytes; view as uint32 words.
    if len(b) % 4:
        b = b[: len(b) - (len(b) % 4)]
    return np.frombuffer(b, dtype=np.uint32).copy()
