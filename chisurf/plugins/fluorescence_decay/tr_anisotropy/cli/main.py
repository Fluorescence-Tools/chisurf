"""CLI for the anisotropy tools.

Both commands are backed by the Qt-free :mod:`...core`:

* ``irf-correct`` — background-subtract and intensity-match a VV/VH IRF pair
  (single-column intensities, or two-column ``x y``). Fully headless.
* ``spectrum`` — print the stored default lifetime/rotation spectra.
"""

from __future__ import annotations

import click
import numpy as np

from ..core import irf as core_irf
from ..core import spectra as core_spectra


def _load_intensity(path: str) -> np.ndarray:
    """Load intensities from a 1- or 2-column text file (returns the last column)."""
    arr = np.loadtxt(path)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, -1]
    return arr.ravel()


@click.group()
def cli():
    """Time-resolved anisotropy helpers."""


@cli.command("irf-correct")
@click.option("--vv", required=True, type=click.Path(exists=True), help="VV IRF file.")
@click.option("--vh", required=True, type=click.Path(exists=True), help="VH IRF file.")
@click.option("--lb", type=int, default=None, help="Background region lower channel.")
@click.option("--ub", type=int, default=None, help="Background region upper channel.")
@click.option("--out-vv", default="irf_vv_corrected.txt", help="Corrected VV output.")
@click.option("--out-vh", default="irf_vh_corrected.txt", help="Corrected VH output.")
def irf_correct(vv, vh, lb, ub, out_vv, out_vh):
    """Background-subtract and intensity-match a VV/VH IRF pair."""
    y_vv = _load_intensity(vv)
    y_vh = _load_intensity(vh)
    n = min(len(y_vv), len(y_vh))
    if lb is None or ub is None:
        lb, ub = core_irf.initial_region(n)
    cvv, cvh = core_irf.correct_irfs(y_vv, y_vh, lb, ub)
    np.savetxt(out_vv, cvv)
    np.savetxt(out_vh, cvh)
    click.echo(f"Background region [{lb}, {ub}) over {n} channels.")
    click.echo(f"Wrote {out_vv} and {out_vh}.")


@cli.command("spectrum")
def spectrum():
    """Print the stored default lifetime/rotation spectra."""
    data = core_spectra.load_spectra(core_spectra.spk_json_path())
    click.echo("Lifetime spectrum [amplitude, lifetime(ns)]:")
    for a, v in data["lifetime_spectrum"]:
        click.echo(f"  {a:>8.3f}  {v:>8.3f}")
    click.echo("Rotation spectrum [amplitude, rho(ns)]:")
    for a, v in data["rotation_spectrum"]:
        click.echo(f"  {a:>8.3f}  {v:>8.3f}")
