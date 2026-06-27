"""Fitting backends for 2D-FLC.

* :mod:`ilt` -- fast Tikhonov / NNLS inverse-Laplace (default).
* :mod:`mem_1d` -- explicit 1D maximum-entropy fit with the MATLAB ``mi`` priors.
* :mod:`mem_2d` -- faithful gradient-based 2D maximum-entropy method.
* :mod:`global_mem` -- joint multi-lag 2D-MEM.
* :mod:`dynamics` -- lifetime-filtered (species) correlation + relaxation fitting.
* :mod:`kinetics` -- rate-matrix recovery from species correlation decays.
* :mod:`gaussian` -- multi-Gaussian peak fit of a lifetime distribution.
* :mod:`helpers` -- 1D-FDC, 1D histogram, IRF-rise scan.
"""

from .dynamics import SpeciesCorrelation, filtered_correlation, fit_relaxation, species_filters
from .gaussian import GaussianComponent, GaussianFitResult, fit_gaussian_multi
from .global_mem import GlobalMEMResult, solve_global_mem_2d
from .helpers import RiseIRFResult, create_1d_fdc, histogram_1d, search_rise_irf
from .ilt import (
    ILTResult1D,
    ILTResult2D,
    LCurveData,
    build_exp_basis,
    ilt_1d,
    ilt_2d,
    lcurve_1d,
    lifetime_grid,
)
from .kinetics import (
    RateMatrixResult,
    equilibrium_populations,
    fit_rate_matrix,
    make_generator_matrix,
)
from .mem_1d import OneDMEMResult, mi_prior, solve_mem_1d
from .mem_2d import TwoDMEMFitter, solve_mem_2d

__all__ = [
    "build_exp_basis",
    "ilt_1d",
    "ilt_2d",
    "lifetime_grid",
    "lcurve_1d",
    "ILTResult1D",
    "ILTResult2D",
    "LCurveData",
    "species_filters",
    "filtered_correlation",
    "fit_relaxation",
    "SpeciesCorrelation",
    "solve_mem_1d",
    "mi_prior",
    "OneDMEMResult",
    "solve_mem_2d",
    "TwoDMEMFitter",
    "fit_rate_matrix",
    "make_generator_matrix",
    "equilibrium_populations",
    "RateMatrixResult",
    "solve_global_mem_2d",
    "GlobalMEMResult",
    "fit_gaussian_multi",
    "GaussianComponent",
    "GaussianFitResult",
    "create_1d_fdc",
    "histogram_1d",
    "search_rise_irf",
    "RiseIRFResult",
]
