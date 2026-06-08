"""
FRET Docking & Screening Plugin
================================

Consolidates functionality from FPS (FRET Positioning System) and
OLGA (Oligonucleotide Library Generation and Analysis) as a
self-contained ChiSurf plugin using LabelLib (Windows) or IMP.bff
(macOS/Linux) for accessible-volume calculations and IMP for
structure I/O and RMF output.

Reference
---------
- Kalinin, S., Peulen, T., et al. (2012). *Nature Methods* 9(12), 1218-1225.
- Dimura, M., Peulen, T., et al. (2016). *Nature Communications* 7, 10947.
"""

from __future__ import annotations

name = "Structure:FRET:Docking & Screening"
description = (
    "Rigid-body FRET-restrained docking, structure-library screening, "
    "and Metropolis Monte Carlo sampling using accessible-volume (AV) "
    "calculations."
)
cli_entrypoint = "fret"

# Expose core modules and actions.
from .core import av, clash, distance, docking, engine, io, results, sampling, screening, refine, bootstrap, evaluate, pair_selection, olga_greedy, trajectory
from .core.io import load_structure, write_pdb
from .core.av import compute_av, compute_avs_for_structure, load_structure_with_vdw
from .core.distance import (
    average_distance,
    mean_fret_distance,
    distance_between_mean_positions,
    model_distance,
    chi2_score,
)
from .core.docking import run_docking
from .core.screening import score_single_structure, screen_structure_library
from .core.sampling import run_metropolis
from .core.engine import SpringEngine, RigidBody, DistanceRestraint, SpringParameters

# ChiSurf plugin entry point.
if __name__ == "plugin":
    from .gui import FretDockWizard

    window = FretDockWizard()
    window.show()

