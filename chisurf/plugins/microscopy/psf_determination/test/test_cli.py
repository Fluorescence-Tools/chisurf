"""Headless CLI + multi-page stack-loader tests for psf_determination."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.microscopy.psf_determination.api import psf


def _write_bead_tiff(path, nz=21, ny=50, nx=50, sxy=2.0, sz=4.0):
    imageio = pytest.importorskip("imageio.v2")
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    bead = (
        5.0
        + 1000.0
        * np.exp(
            -0.5
            * (((x - nx / 2) / sxy) ** 2 + ((y - ny / 2) / sxy) ** 2 + ((z - nz / 2) / sz) ** 2)
        )
    ).astype(np.float32)
    imageio.mimwrite(path, [s for s in bead])


def test_load_stack_reads_all_pages(tmp_path):
    """A multi-page TIFF must load as a 3-D (z, y, x) volume, not a single page."""
    p = tmp_path / "bead.tif"
    _write_bead_tiff(p)
    arr = psf.load_stack(str(p))
    assert arr.ndim == 3
    assert arr.shape == (21, 50, 50)
    assert arr.dtype == np.float32


def test_cli_fit_stack(tmp_path):
    """The CLI detects and fits the bead in a multi-page TIFF."""
    pytest.importorskip("imageio.v2")
    from click.testing import CliRunner

    from chisurf.plugins.microscopy.psf_determination.cli import cli

    p = tmp_path / "bead.tif"
    _write_bead_tiff(p)
    result = CliRunner().invoke(cli, ["fit-stack", str(p)])
    assert result.exit_code == 0, result.output
    assert "Detected 1 bead" in result.output
    assert "ok=True" in result.output
