"""Qt-free core for the FCS confocal (diffusion/volume) calculator."""

from .algorithms import DYE_DATA, N_PER_nM_fL, compute_confocal

__all__ = ["DYE_DATA", "N_PER_nM_fL", "compute_confocal"]
