#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .export_fitting_parameters import _load_existing_registry, _save_registry


def _describe_tcspc_param(symbol: str) -> Tuple[str, List[str]]:
    """Return (description, keywords) for a given TCSPC parameter symbol.

    Descriptions are intentionally generic but physically meaningful so they
    can be reused across many TCSPC decay models that share the same
    parameter names.
    """

    s = (symbol or "").strip()
    base = s.lower()
    desc = ""
    kw: List[str] = ["TCSPC", "lifetime"]

    if not s:
        return "", []

    # --- Pre-exponential amplitudes / populations ----------------------------
    if base in {"a", "a1", "a2", "a3", "ad1", "ado", "ada", "adaa", "af1a"}:
        desc = (
            "Pre-exponential amplitude or population fraction of this decay "
            "component (dimensionless contribution to the overall "
            "time-resolved signal)."
        )
        kw += ["amplitude", "population fraction"]

    # --- Overall intensity / normalization -----------------------------------
    elif base == "p0":
        desc = (
            "Overall normalization or intensity at time zero of the decay "
            "curve."
        )
        kw += ["normalization", "initial intensity"]

    # --- Empirical transient-quenching coefficients -------------------------
    elif base == "b":
        desc = (
            "Empirical coefficient of the sqrt(time) term in the "
            "transient-quenching decay model; controls deviations from a "
            "pure single-exponential decay."
        )
        kw += ["empirical", "transient quenching"]

    # --- Lifetimes / decay times ---------------------------------------------
    elif base.startswith("tau") or base in {"td1", "td2", "t0"}:
        desc = (
            "Fluorescence lifetime or characteristic decay time of this "
            "component."
        )
        kw += ["lifetime", "decay time"]

    # --- Quenching / FRET rate constants -------------------------------------
    elif base == "kq":
        desc = (
            "Quenching rate constant added to the inverse lifetime "
            "(1/tau + kQ)."
        )
        kw += ["quenching", "rate constant"]

    elif base in {"kf1a", "kf2a", "kf1aa"}:
        desc = "Effective FRET or energy-transfer rate constant for this channel."
        kw += ["FRET", "rate constant", "energy transfer"]

    # --- Transient-quenching geometry / transport ----------------------------
    elif base == "rdye":
        desc = (
            "Characteristic interaction radius or distance parameter for the "
            "dye-quencher system."
        )
        kw += ["distance", "interaction radius", "quenching"]

    elif base == "ddye":
        desc = "Translational diffusion coefficient of the fluorescent dye."
        kw += ["diffusion", "coefficient"]

    elif base == "nq":
        desc = (
            "Effective number or concentration of quenchers in the "
            "interaction volume."
        )
        kw += ["quenchers", "concentration", "quenching"]

    elif base == "vav":
        desc = "Effective averaging volume used in the transient-quenching model."
        kw += ["volume", "quenching model"]

    # --- Fractions related to additional rate channels -----------------------
    elif base in {"xd", "ad", "xdonly"}:
        desc = (
            "Fraction of donors subject to additional quenching or rate "
            "channels (dimensionless)."
        )
        kw += ["donor fraction", "population fraction"]

    # --- Generic TCSPC background and scatter parameters --------------------
    elif base == "sc":
        desc = (
            "Relative amplitude of a prompt scattering contribution that is "
            "added to the model decay."
        )
        kw += ["scatter", "prompt", "amplitude"]

    elif base in {"bg", "bg"}:
        desc = (
            "Constant background level added to the time-resolved decay "
            "curve (counts per time channel)."
        )
        kw += ["background", "offset", "counts"]

    elif base == "tbg":
        desc = "Measurement time of the background acquisition."
        kw += ["background", "measurement time"]

    elif base == "tmeas":
        desc = "Measurement time of the main TCSPC experiment."
        kw += ["measurement time", "experiment"]

    # --- Convolution / IRF related parameters -------------------------------
    elif base == "n0":
        desc = (
            "Initial number of excited donor molecules used to scale the "
            "model decay to the experimental counts."
        )
        kw += ["normalization", "excited molecules"]

    elif base == "dt":
        desc = "Time bin width of the TCSPC histogram (time per channel)."
        kw += ["time bin", "resolution"]

    elif base == "rep":
        desc = "Laser repetition rate of the excitation source."
        kw += ["repetition rate", "laser", "frequency"]

    elif base == "start":
        desc = "Start time (or channel) of the fit/convolution window."
        kw += ["fit window", "start"]

    elif base == "stop":
        desc = "Stop time (or channel) of the fit/convolution window."
        kw += ["fit window", "stop"]

    elif base == "irf_start":
        desc = "Start index (or time) of the IRF region used for convolution."
        kw += ["IRF", "window", "start"]

    elif base == "irf_stop":
        desc = "Stop index (or time) of the IRF region used for convolution."
        kw += ["IRF", "window", "stop"]

    elif base == "lb":
        desc = "Lamp background level subtracted from the instrument response function."
        kw += ["lamp", "background", "IRF"]

    elif base == "ts":
        desc = "Additional temporal shift applied to align IRF and decay."
        kw += ["timeshift", "alignment"]

    elif base == "iw":
        desc = "Width parameter of the synthetic IRF model."
        kw += ["IRF", "width"]

    elif base == "ik":
        desc = "Shape parameter of the synthetic IRF model."
        kw += ["IRF", "shape"]

    # --- FRET / distance-distribution specific parameters -------------------
    elif base == "r0":
        desc = "Frster radius R0 of the donor-acceptor pair."
        kw += ["FRET", "Forster radius", "distance"]

    elif base == "k2":
        desc = "Orientation factor  governing dipole-dipole coupling in FRET."
        kw += ["FRET", "orientation factor", "kappa2"]

    elif base == "e_fret":
        desc = "Apparent FRET efficiency parameter E_FRET (0e00..1)."
        kw += ["FRET", "efficiency"]

    elif base.startswith("r("):
        desc = (
            "Center distance of a component in the FRET distance distribution "
            "(e.g. mean of a Gaussian peak)."
        )
        kw += ["FRET", "distance distribution", "mean distance"]

    elif base.startswith("x("):
        desc = (
            "Relative amplitude (population) of a component in the FRET "
            "distance distribution."
        )
        kw += ["FRET", "distance distribution", "amplitude"]

    elif base.startswith("s("):
        desc = "Width (sigma) of a component in the FRET distance distribution."
        kw += ["FRET", "distance distribution", "width"]

    elif base.startswith("k("):
        desc = "Shape parameter of a component in the FRET distance distribution."
        kw += ["FRET", "distance distribution", "shape"]

    # Fallback: provide a very generic but safe description so that all
    # tcspc.* parameters can show some help text in the GUI.
    if not desc:
        desc = (
            f"Model parameter {s} used in TCSPC decay models. Refer to the "
            "specific model documentation for the detailed physical meaning."
        )

    # Deduplicate keywords while preserving order
    seen = set()
    merged_kw: List[str] = []
    for k in kw:
        if k not in seen:
            seen.add(k)
            merged_kw.append(k)

    return desc, merged_kw


def _update_tcspc_descriptions(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing descriptions for all TCSPC-related registry entries.

    This includes both tcspc.* keys (from parsed models) and plain parameter
    names whose sources live in chisurf.core.models.tcspc.* modules.

    Existing non-empty descriptions are preserved. Keywords are merged with the
    auto-generated ones without removing any user-provided terms.
    """

    if not isinstance(params, dict):
        return params

    updated = dict(params)

    for key, entry in list(updated.items()):
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue

        # Determine whether this entry is TCSPC-related.
        is_tcspc_registry_key = key.startswith("tcspc.")
        is_tcspc_code_param = False

        sources = entry.get("sources") or []
        if isinstance(sources, list):
            for src in sources:
                if not isinstance(src, dict):
                    continue
                mod = str(src.get("module", ""))
                fil = str(src.get("file", ""))
                if "tcspc" in mod or "tcspc" in fil:
                    is_tcspc_code_param = True
                    break

        if not (is_tcspc_registry_key or is_tcspc_code_param):
            continue

        existing_desc = entry.get("description")
        if isinstance(existing_desc, str) and existing_desc.strip():
            # User or earlier tooling already provided a description; keep it.
            continue

        symbol = entry.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            if is_tcspc_registry_key:
                symbol = key.split(".", 1)[-1]
            else:
                symbol = key

        desc, auto_kw = _describe_tcspc_param(symbol)
        if not desc:
            # Leave untouched if we do not have a sensible description.
            continue

        # Merge keywords
        kw_existing = entry.get("keywords")
        if not isinstance(kw_existing, list):
            kw_existing = []
        for kw in auto_kw:
            if kw not in kw_existing:
                kw_existing.append(kw)

        entry["description"] = desc
        entry["keywords"] = kw_existing
        updated[key] = entry

    return updated


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing descriptions for TCSPC parameters in "
            "parameter_registry.json using generic TCSPC-aware heuristics."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the chisurf package root (defaults to this file's parent).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path (defaults to chisurf/settings/constants/"
            "parameter_registry.json under --root)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    here = Path(__file__).resolve()
    default_root = here.parent.parent  # chisurf package directory
    source_root = Path(args.root).resolve() if args.root is not None else default_root

    if args.output is not None:
        out_path = Path(args.output).resolve()
    else:
        out_path = source_root / "settings" / "constants" / "parameter_registry.json"

    existing = _load_existing_registry(out_path)
    if not isinstance(existing, dict):
        existing = {}

    merged = _update_tcspc_descriptions(existing)
    _save_registry(out_path, merged)


if __name__ == "__main__":
    main()
