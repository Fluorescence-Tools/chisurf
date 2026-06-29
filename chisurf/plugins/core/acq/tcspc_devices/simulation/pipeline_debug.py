"""Pipeline debug CLI.

Exercises the full BH SPC-130 simulation → decode → accumulate → correlation
pipeline outside of the GUI, printing detailed diagnostics at each step.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import click
import numpy as np

try:
    import tttrlib
except ImportError:
    tttrlib = None  # type: ignore[assignment]

try:
    from .burbulator_dll_wrapper import BurbulatorDLL, BurbulatorError
except ImportError:
    from burbulator_dll_wrapper import BurbulatorDLL, BurbulatorError


# ---------------------------------------------------------------------------
# Decode BH SPC-130 – mirrors _process_bh_spc_records_numba in tool.py
# ---------------------------------------------------------------------------

def decode_bh_spc130(data: np.ndarray, initial_overflow: int = 0):
    """Decode BH SPC-130 uint32 records into macrotimes, microtimes, channels.

    Parameters
    ----------
    data : np.ndarray[uint32]
        Raw BH SPC-130 records.
    initial_overflow : int
        Overflow counter from previous chunk.

    Returns
    -------
    (photons, microtimes, channels, total_overflows, final_overflow)
    """
    n = len(data)
    photons = np.zeros(n, dtype=np.uint64)
    microtimes = np.zeros(n, dtype=np.uint16)
    channels = np.zeros(n, dtype=np.uint8)

    overflow_counter = np.uint64(initial_overflow)
    total_overflows = np.uint64(0)
    idx = 0

    for i in range(n):
        rec = data[i]
        mt = rec & 0xFFF
        rout = (rec >> 12) & 0xF
        adc = (rec >> 16) & 0xFFF
        mtov = (rec >> 30) & 0x1
        invalid = (rec >> 31) & 0x1

        if invalid == 0:
            overflow_counter += np.uint64(mtov)
            total_overflows += np.uint64(mtov)
            true_nsync = np.uint64(mt) + overflow_counter * np.uint64(4096)
            photons[idx] = true_nsync
            microtimes[idx] = np.uint16(4095 - adc)
            channels[idx] = np.uint8(rout)
            idx += 1
        elif invalid == 1 and mtov == 1:
            cnt = rec & 0x0FFFFFFF
            overflow_counter += np.uint64(cnt)
            total_overflows += np.uint64(cnt)

    return (
        photons[:idx].copy(),
        microtimes[:idx].copy(),
        channels[:idx].copy(),
        int(total_overflows),
        int(overflow_counter),
    )


# ---------------------------------------------------------------------------
# Correlation via tttrlib – mirrors _correlate_pair in tool.py
# ---------------------------------------------------------------------------

def correlate_all(macrotimes: np.ndarray, macro_clock: float = 50e-9):
    """Compute auto-correlation of all photons (channel -1).

    Parameters
    ----------
    macrotimes : np.ndarray[uint64]
    macro_clock : float
        Clock period in seconds.

    Returns
    -------
    (tau_ms, g) or None
    """
    if tttrlib is None:
        return None
    if macrotimes.size < 10:
        return None
    try:
        mt = np.asarray(macrotimes, dtype=np.uint64)
        corr = tttrlib.Correlator(n_bins=9, n_casc=15, make_fine=False)
        w = np.ones_like(mt, dtype=np.float64)
        corr.set_events(mt, w, mt, w)
        tau = corr.x * macro_clock * 1e3
        g = corr.y
        return (tau.copy(), g.copy())
    except Exception as e:
        return None


def correlate_channel(macrotimes: np.ndarray, channels: np.ndarray,
                      ch: int, macro_clock: float = 50e-9):
    """Compute auto-correlation for a specific routing channel.

    Parameters
    ----------
    macrotimes : np.ndarray[uint64]
    channels : np.ndarray[int]
    ch : int
        Routing channel value.
    macro_clock : float
        Clock period in seconds.

    Returns
    -------
    (tau_ms, g) or None
    """
    if tttrlib is None:
        return None
    mask = channels == ch
    mt = macrotimes[mask]
    if mt.size < 10:
        return None
    try:
        mt = np.asarray(mt, dtype=np.uint64)
        corr = tttrlib.Correlator(n_bins=9, n_casc=15, make_fine=False)
        w = np.ones_like(mt, dtype=np.float64)
        corr.set_events(mt, w, mt, w)
        tau = corr.x * macro_clock * 1e3
        g = corr.y
        return (tau.copy(), g.copy())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Simulation defaults (matches wrapper.py SimulationDevice)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Any] = {
    "N_species": 1,
    "M": [50.0],
    "D": [3.0],
    "N_channels": 2,
    "q": [50.0, 50.0],
    "q_bg": [0.001, 0.001],
    "k_rad": [1.0],
    "k_nrad": [0.0],
    "box_xy": 2.0,
    "box_z": 4.0,
    "focus_type": 0,
    "focus_param": [0.3, 2.0],
    "dt": 0.01,
    "N_ph_max": 50000,
    "pulsed_exc": 0,
    "ch_conversion": [8, 0, 9, 1, 10, 2],
    "N_tac_channels": 4096,
    "tac_dt": 0.004069,
    "laser_period": 13.596,
}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    params: Dict[str, Any],
    dll: BurbulatorDLL,
    macro_clock: float = 50e-9,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full simulation→decode→accumulate→correlate pipeline.

    Parameters
    ----------
    params : dict
        Simulation parameters (see DEFAULT_PARAMS).
    dll : BurbulatorDLL
        Loaded library wrapper.
    macro_clock : float
        Macrotime clock period in seconds.
    verbose : bool
        Print detailed per-step diagnostics.

    Returns
    -------
    dict with keys: simulate, decode, accumulate, correlation, summary
    """
    results: Dict[str, Any] = {}
    N_ph_max = int(params["N_ph_max"])

    # --- Step 1: simulate_ov3 ------------------------------------------------
    t0 = time.monotonic()
    sim = dll.simulate_ov3(
        Nspecies=int(params["N_species"]),
        M=params["M"],
        D=params["D"],
        Nchannels=int(params["N_channels"]),
        q=params["q"],
        q_bg=params["q_bg"],
        k_rad=params["k_rad"],
        k_nrad=params["k_nrad"],
        box_xy=float(params["box_xy"]),
        box_z=float(params["box_z"]),
        focus_type=int(params["focus_type"]),
        focus_param=params["focus_param"],
        dt=float(params["dt"]),
        N_ph_max=N_ph_max,
        rmt1seed=int(params.get("rmt1seed", 12345)),
        rmt2seed=int(params.get("rmt2seed", 54321)),
    )
    t_sim = time.monotonic() - t0
    results["simulate"] = {
        "N_ph": sim["N_ph"],
        "T0": sim["T0"],
        "Nmolecules": sim["Nmolecules"],
        "time_s": t_sim,
    }

    if verbose:
        click.echo(f"\n{'='*60}")
        click.echo(f"STEP 1: simulate_ov3  ({t_sim:.3f} s)")
        click.echo(f"{'='*60}")
        click.echo(f"  N_ph       = {sim['N_ph']}")
        click.echo(f"  T0         = {sim['T0']}")
        click.echo(f"  Nmolecules = {sim['Nmolecules']}")
        click.echo(f"  data_T     = {sim['data_T'][:10].tolist()}{'...' if sim['N_ph'] > 10 else ''}")
        click.echo(f"  data_t     = {sim['data_t'][:5].tolist()}{'...' if sim['N_ph'] > 5 else ''}")
        click.echo(f"  data_N     = {sim['data_N'][:10].tolist()}{'...' if sim['N_ph'] > 10 else ''}")
        click.echo(f"  data_species unique = {np.unique(sim['data_species']).tolist()}")
        ch_conv = params.get("ch_conversion", [8, 0, 9, 1, 10, 2])
        click.echo(f"  ch_conversion        = {ch_conv}")
        click.echo(f"  data_N values used for channel routing: {np.unique(sim['data_N']).tolist()}")

    # --- Step 2: convert_to_spc132 -------------------------------------------
    N_ph = sim["N_ph"]
    if N_ph == 0:
        results["decode"] = {"error": "No photons to convert"}
        results["summary"] = {"status": "NO_PHOTONS"}
        return results

    t0 = time.monotonic()
    spc_bytes, MT_ov, spc_i = dll.convert_to_spc132(
        pulsed_exc=int(params.get("pulsed_exc", 0)),
        Nchannels=int(params["N_channels"]),
        data_T=sim["data_T"],
        data_t=sim["data_t"],
        data_N=sim["data_N"],
        data_species=sim["data_species"],
        data_molecule=sim["data_molecule"],
        tw=float(params["dt"]),
        ch_conversion=list(params.get("ch_conversion", [8, 0, 9, 1, 10, 2])),
        N_tac_channels=int(params.get("N_tac_channels", 4096)),
        tac_dt=float(params.get("tac_dt", 0.004069)),
        laser_period=float(params.get("laser_period", 13.596)),
        N_photons=N_ph,
    )
    t_conv = time.monotonic() - t0
    results["convert"] = {
        "n_bytes": spc_i,
        "MT_ov": MT_ov,
        "time_s": t_conv,
    }

    if verbose:
        click.echo(f"\n{'='*60}")
        click.echo(f"STEP 2: convert_to_spc132  ({t_conv:.3f} s)")
        click.echo(f"{'='*60}")
        click.echo(f"  n_bytes  = {spc_i}")
        click.echo(f"  MT_ov    = {MT_ov}")

    # --- Step 3: Decode BH SPC-130 records -----------------------------------
    spc_data = np.frombuffer(spc_bytes, dtype=np.uint32)
    if verbose:
        click.echo(f"\n{'='*60}")
        click.echo(f"STEP 3: Decode BH SPC-130")
        click.echo(f"{'='*60}")
        click.echo(f"  raw records       = {len(spc_data)}")
        click.echo(f"  first 5 records   = {spc_data[:5].tolist()}")
        click.echo(f"  last 5 records    = {spc_data[-5:].tolist()}")
        # Show bit fields for first record
        r0 = spc_data[0]
        click.echo(f"  record[0] fields:")
        click.echo(f"    mt={r0 & 0xFFF}, rout={(r0 >> 12) & 0xF}, "
                   f"adc={(r0 >> 16) & 0xFFF}, mtov={(r0 >> 30) & 0x1}, "
                   f"invalid={(r0 >> 31) & 0x1}")

    t0 = time.monotonic()
    photons, microtimes, channels, total_ov, final_ov = decode_bh_spc130(spc_data)
    t_dec = time.monotonic() - t0
    results["decode"] = {
        "n_photons": len(photons),
        "total_overflows": total_ov,
        "final_overflow": final_ov,
        "time_s": t_dec,
        "unique_channels": sorted(np.unique(channels).tolist()),
    }

    if verbose:
        click.echo(f"  decoded photons   = {len(photons)}  ({t_dec*1000:.2f} ms)")
        click.echo(f"  unique channels   = {sorted(np.unique(channels).tolist())}")
        click.echo(f"  total overflows   = {total_ov}")
        click.echo(f"  final overflow    = {final_ov}")
        click.echo(f"  macrotime range   = [{photons[0]}, {photons[-1]}]")
        click.echo(f"  macrotime dtype   = {photons.dtype}")
        click.echo(f"  photons[:10]      = {photons[:10].tolist()}")

    # --- Step 4: Accumulate (simulate multi-chunk) ---------------------------
    absolute_macrotimes = photons.astype(np.uint64)
    routing_channels = channels.astype(np.int16)

    results["accumulate"] = {
        "n_total": len(absolute_macrotimes),
        "macrotime_min": int(absolute_macrotimes[0]),
        "macrotime_max": int(absolute_macrotimes[-1]),
        "macrotime_span_ticks": int(absolute_macrotimes[-1] - absolute_macrotimes[0]),
        "macrotime_span_s": float(absolute_macrotimes[-1] - absolute_macrotimes[0]) * macro_clock,
    }

    if verbose:
        macro_span_ticks = absolute_macrotimes[-1] - absolute_macrotimes[0]
        macro_span_s = macro_span_ticks * macro_clock
        click.echo(f"\n{'='*60}")
        click.echo(f"STEP 4: Accumulate")
        click.echo(f"{'='*60}")
        click.echo(f"  total photons     = {len(absolute_macrotimes)}")
        click.echo(f"  macrotime span    = {macro_span_ticks} ticks ({macro_span_s*1000:.2f} ms)")
        click.echo(f"  mean rate         = {len(absolute_macrotimes)/macro_span_s/1000:.1f} kHz")

    # --- Step 5: tttrlib correlation -----------------------------------------
    if tttrlib is None:
        click.echo("\n[SKIP] tttrlib not available — correlation skipped")
        results["correlation"] = {"status": "SKIPPED_NO_TTTRLIB"}
        results["summary"] = {"status": "OK", "tttrlib": False}
        return results

    t0 = time.monotonic()
    # All-channel (auto)
    corr_all = correlate_all(absolute_macrotimes, macro_clock)

    # Per-channel
    corr_per_ch = {}
    for ch_val in sorted(np.unique(channels).tolist()):
        corr_per_ch[str(ch_val)] = correlate_channel(
            absolute_macrotimes, routing_channels, ch_val, macro_clock
        )
    t_corr = time.monotonic() - t0

    results["correlation"] = {
        "n_curves": 1 + len(corr_per_ch),
        "time_s": t_corr,
    }
    if corr_all is not None:
        tau, g = corr_all
        results["correlation"]["all"] = {
            "tau_len": len(tau),
            "g_min": float(g.min()),
            "g_max": float(g.max()),
            "g_mean": float(g.mean()),
            "g_nonzero": bool(np.any(g > 0)),
            "tau_first_5": tau[:5].tolist(),
            "g_first_5": g[:5].tolist(),
        }
    for ch_key, corr in corr_per_ch.items():
        if corr is not None:
            tau, g = corr
            results["correlation"][f"ch{ch_key}"] = {
                "n_photons": int(np.count_nonzero(routing_channels == int(ch_key))),
                "g_min": float(g.min()),
                "g_max": float(g.max()),
                "g_mean": float(g.mean()),
                "g_nonzero": bool(np.any(g > 0)),
            }
        else:
            results["correlation"][f"ch{ch_key}"] = {
                "n_photons": int(np.count_nonzero(routing_channels == int(ch_key))),
                "status": "TOO_FEW_PHOTONS",
            }

    if verbose:
        click.echo(f"\n{'='*60}")
        click.echo(f"STEP 5: tttrlib Correlation  ({t_corr*1000:.2f} ms)")
        click.echo(f"{'='*60}")
        if corr_all is not None:
            tau, g = corr_all
            click.echo(f"  ALL channels ({len(absolute_macrotimes)} photons):")
            click.echo(f"    tau shape = {tau.shape}")
            click.echo(f"    tau[:5]   = {tau[:5]}")
            click.echo(f"    g[:5]     = {g[:5]}")
            click.echo(f"    g range   = [{g.min():.4f}, {g.max():.4f}]")
            click.echo(f"    g mean    = {g.mean():.4f}")
            click.echo(f"    non-zero  = {np.any(g > 0)}")
        else:
            click.echo(f"  ALL channels: too few photons")

        for ch_key, corr in sorted(corr_per_ch.items()):
            n_ph = int(np.count_nonzero(routing_channels == int(ch_key)))
            if corr is not None:
                tau, g = corr
                click.echo(f"  Channel {ch_key} ({n_ph} photons):")
                click.echo(f"    g range   = [{g.min():.4f}, {g.max():.4f}]")
                click.echo(f"    g mean    = {g.mean():.4f}")
                click.echo(f"    non-zero  = {np.any(g > 0)}")
            else:
                click.echo(f"  Channel {ch_key} ({n_ph} photons): too few (< 10)")

    # Summary
    ok = (
        corr_all is not None
        and np.any(corr_all[1] > 0)
    )
    results["summary"] = {
        "status": "OK" if ok else "CORRELATION_ZERO",
        "tttrlib": True,
        "total_photons": len(absolute_macrotimes),
    }

    if verbose:
        click.echo(f"\n{'='*60}")
        click.echo(f"SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"  status         = {results['summary']['status']}")
        click.echo(f"  total_photons  = {len(absolute_macrotimes)}")
        click.echo(f"  unique channels = {sorted(np.unique(channels).tolist())}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("pipeline",
               help="Run the full simulation→decode→correlation pipeline.")
@click.option("--n-ph-max", type=int, default=50000,
              help="Number of photons to simulate (default: 50000).")
@click.option("--n-ph-max-2nd", type=int, default=None,
              help="Number for a second run (default: same as n-ph-max).")
@click.option("--json", "output_json",
              type=click.Path(dir_okay=False, writable=True),
              default=None,
              help="Write results as JSON.")
@click.option("--quiet", is_flag=True, default=False,
              help="Suppress per-step verbose output (JSON only).")
def pipeline_cmd(
    n_ph_max: int,
    n_ph_max_2nd: Optional[int],
    output_json: Optional[str],
    quiet: bool,
) -> None:
    """Run and debug the photon pipeline."""
    verbose = not quiet

    # Load the DLL
    try:
        dll = BurbulatorDLL()
        click.echo(f"DLL loaded: {dll.path}")
    except (FileNotFoundError, BurbulatorError) as e:
        raise click.ClickException(str(e))

    params = dict(DEFAULT_PARAMS)
    params["N_ph_max"] = n_ph_max

    results = run_pipeline(params, dll, verbose=verbose)

    if n_ph_max_2nd is not None and n_ph_max_2nd > 0:
        params2 = dict(DEFAULT_PARAMS)
        params2["N_ph_max"] = n_ph_max_2nd
        results_2nd = run_pipeline(params2, dll, verbose=verbose)
        results["second_run"] = results_2nd

    if output_json:
        # Convert numpy types
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_json, "w") as f:
            json.dump(_convert(results), f, indent=2)
        click.echo(f"\nResults written to {output_json}")

    if results.get("summary", {}).get("status") == "CORRELATION_ZERO":
        sys.exit(1)


if __name__ == "__main__":
    pipeline_cmd()
