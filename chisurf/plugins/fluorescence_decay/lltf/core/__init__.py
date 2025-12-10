"""
Lazy Lifetime Fitter (lltf) package.

This package provides tools for fitting fluorescence lifetime data.
"""

from .fitter import fit_lifetime, Decay

__all__ = ['fit_lifetime', 'Decay']