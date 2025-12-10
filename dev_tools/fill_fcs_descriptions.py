#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .export_fitting_parameters import _load_existing_registry, _save_registry


def _describe_fcs_param(symbol: str) -> Tuple[str, List[str]]:
    """Return (description, keywords) for a given FCS parameter symbol.

    Descriptions are intentionally generic but physically meaningful so they
    can be reused across many FCS models that share the same parameter names.
    """

    s = (symbol or "").strip()
    base = s.lower()
    desc = ""
    kw: List[str] = ["FCS"]

    if not s:
        return "", []

    # --- Global particle-number parameters ---------------------------------
    if s in {"N", "N3"}:
        desc = (
            "Average number of fluorescent particles in the observation volume. "
            "In standard FCS models the correlation amplitude scales roughly as 1/N."
        )
        kw += ["number of molecules", "concentration", "amplitude"]

    # --- Baseline / structure ----------------------------------------------
    elif base == "b":
        desc = "Additive baseline/offset of the correlation function."
        kw += ["baseline", "offset", "background"]

    elif base == "s":
        desc = (
            "Structure parameter s = z0 / w0 describing the axial-to-radial "
            "extent of the detection volume."
        )
        kw += ["structure parameter", "PSF", "geometry"]

    # --- Diffusion times ----------------------------------------------------
    elif base.startswith("td"):
        desc = (
            "Characteristic diffusion time of this component (time scale on "
            "which particles traverse the observation volume)."
        )
        kw += ["diffusion", "correlation time"]

    # --- Rotational / anisotropy terms -------------------------------------
    elif base == "c":
        desc = "Dimensionless factor C used in the rotational correlation / anisotropy term."
        kw += ["anisotropy", "rotation"]

    elif base == "aroc":
        desc = "Amplitude of the rotational correlation (anisotropy) contribution."
        kw += ["anisotropy", "rotation", "amplitude"]

    elif base.startswith("trc"):
        desc = "Rotational correlation time used in the anisotropy / rotational diffusion term."
        kw += ["anisotropy", "rotation", "correlation time"]

    # --- Relaxation / (anti)correlation amplitudes -------------------------
    elif s in {"a", "a1", "a2", "a21", "a22", "a31"}:
        desc = (
            "Fractional amplitude or population of this diffusion/kinetic "
            "component (dimensionless, typically between 0 and 1)."
        )
        kw += ["amplitude", "population fraction"]

    elif base in {"ar", "ar1", "ar2", "ara"}:
        desc = "Amplitude of a relaxation or anticorrelation term (dimensionless)."
        kw += ["relaxation", "anticorrelation", "amplitude"]

    elif base == "af":
        desc = "Overall amplitude factor for a set of anticorrelation components."
        kw += ["anticorrelation", "amplitude"]

    # --- Antibunching / anisotropy amplitudes ------------------------------
    elif base == "aab":
        desc = "Amplitude of the photon antibunching term (very fast correlation component)."
        kw += ["antibunching", "triplet", "amplitude"]

    elif base == "abf":
        desc = "Total amplitude factor for a group of anticorrelation components."
        kw += ["anticorrelation", "amplitude"]

    elif base.startswith("ab") and base not in {"abf", "abt"}:
        desc = (
            "Amplitude of an individual antibunching / anisotropy / "
            "anticorrelation component (dimensionless)."
        )
        kw += ["antibunching", "anisotropy", "amplitude"]

    # --- Dark / bunching amplitudes ----------------------------------------
    elif base.startswith("ba"):
        desc = (
            "Fractional amplitude of a dark or bunching state (e.g. triplet or "
            "blinking); dimensionless and typically between 0 and 1."
        )
        kw += ["bunching", "triplet", "dark state", "amplitude"]

    # --- Time constants: dark/bunching, anti-correlation, antibunching -----
    elif base == "tab":
        desc = "Correlation time of the photon antibunching term (fast time scale)."
        kw += ["antibunching", "correlation time"]

    elif base.startswith("bt"):
        desc = "Correlation/relaxation time constant of a dark or bunching state."
        kw += ["bunching", "triplet", "dark state", "correlation time"]

    elif base.startswith("tr") or base.startswith("tR".lower()):
        desc = "Relaxation or anticorrelation time constant of a kinetic component."
        kw += ["relaxation", "kinetics", "correlation time"]

    elif base.startswith("abt"):
        desc = "Time constant of an antibunching or anticorrelation component."
        kw += ["antibunching", "anticorrelation", "correlation time"]

    # --- Stretched-exponential exponents -----------------------------------
    elif base.startswith("bs"):
        desc = "Stretching exponent for a stretched-exponential relaxation term."
        kw += ["stretched exponential", "relaxation", "exponent"]

    # Fallback: leave description empty so we do not invent semantics
    if not desc:
        return "", []

    # Deduplicate keywords while preserving order
    seen = set()
    merged_kw: List[str] = []
    for k in kw:
        if k not in seen:
            seen.add(k)
            merged_kw.append(k)

    return desc, merged_kw


def _update_fcs_descriptions(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing descriptions for all fcs.* registry entries.

    Existing non-empty descriptions are preserved. Keywords are merged with the
    auto-generated ones without removing any user-provided terms.
    """

    if not isinstance(params, dict):
        return params

    updated = dict(params)

    for key, entry in list(updated.items()):
        if not (isinstance(key, str) and key.startswith("fcs.")):
            continue
        if not isinstance(entry, dict):
            continue

        existing_desc = entry.get("description")
        if isinstance(existing_desc, str) and existing_desc.strip():
            # User or earlier tooling already provided a description; keep it.
            continue

        symbol = entry.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            symbol = key.split(".", 1)[-1]

        desc, auto_kw = _describe_fcs_param(symbol)
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
            "Fill missing descriptions for FCS parameters in "
            "fitting_parameters.json using generic FCS-aware heuristics."
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
            "fitting_parameters.json under --root)."
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
        out_path = source_root / "settings" / "constants" / "fitting_parameters.json"

    existing = _load_existing_registry(out_path)
    if not isinstance(existing, dict):
        existing = {}

    merged = _update_fcs_descriptions(existing)
    _save_registry(out_path, merged)


if __name__ == "__main__":
    main()
