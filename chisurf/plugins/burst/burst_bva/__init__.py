"""
Single Molecule Burst Variance Analysis (BVA) Plugin

This plugin implements Burst Variance Analysis for single-molecule FRET experiments.
BVA is a technique that analyzes the variance of FRET efficiency within individual
bursts to distinguish between static and dynamic heterogeneity in the sample.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest else "Spectroscopy:Single-Molecule:BVA"

from .gui.tool import BVATool

__all__ = ["BVATool", "name"]
