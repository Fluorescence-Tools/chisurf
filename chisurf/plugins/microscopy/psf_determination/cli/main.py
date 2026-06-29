"""CLI for PSF Determination.

Usage:
    psf-determination fit-stack STACK.tif [options]
    psf-determination contract
"""

from __future__ import annotations

import json

import click


@click.group()
def cli() -> None:
    """3D Gaussian PSF fitting for confocal bead stacks."""


@cli.command("fit-stack")
@click.argument("stack_path", type=click.Path(exists=True))
@click.option("--pixel-size-nm", default=100.0, type=float, show_default=True, help="Lateral pixel size in nm.")
@click.option("--z-step-nm", default=200.0, type=float, show_default=True, help="Axial step size in nm.")
@click.option("--roi-xy", default=15, type=int, show_default=True, help="ROI half-size in x/y (pixels).")
@click.option("--roi-z", default=15, type=int, show_default=True, help="ROI half-size in z (slices).")
@click.option("--pixels-per-frame", default=20, type=int, show_default=True, help="Expected bright pixels per frame (for quantile threshold).")
@click.option("--min-distance", default=5.0, type=float, show_default=True, help="Minimum lateral distance between detected beads (pixels).")
@click.option("--json", "json_output", is_flag=True, help="Print results as JSON.")
@click.option("--csv", "csv_output", type=click.Path(), default=None, help="Save results to CSV file.")
def fit_stack(
    stack_path: str,
    pixel_size_nm: float,
    z_step_nm: float,
    roi_xy: int,
    roi_z: int,
    pixels_per_frame: int,
    min_distance: float,
    json_output: bool,
    csv_output: str | None,
) -> None:
    """Detect beads in STACK_PATH and fit 3-D Gaussian PSF to each."""
    from ..backend.services import _handle_fit

    click.echo(f"Loading stack: {stack_path}")
    result = _handle_fit({
        "stack_path": stack_path,
        "pixel_size_nm": pixel_size_nm,
        "z_step_nm": z_step_nm,
        "roi_xy": roi_xy,
        "roi_z": roi_z,
        "pixels_per_frame": pixels_per_frame,
        "min_distance": min_distance,
    })

    if not result.get("ok"):
        click.echo(f"ERROR: {result.get('error')}", err=True)
        raise SystemExit(1)

    fits = result["result"]["fits"]
    n = result["result"]["n_beads"]
    click.echo(f"Detected {n} bead(s).")

    if json_output:
        click.echo(json.dumps(fits, indent=2))
    else:
        for f in fits:
            if f.get("error"):
                click.echo(f"[{f['index']:03d}] x={f['x_px']}, y={f['y_px']}, z={f['z_slice']}: {f['error']}")
            else:
                click.echo(
                    f"[{f['index']:03d}] x={f['x_px']}, y={f['y_px']}, z={f['z_slice']} | "
                    f"FWHMxy={f['fwhm_xy_nm']:.1f} nm, FWHMz={f['fwhm_z_nm']:.1f} nm | "
                    f"axial={f['axial_ratio']:.2f} | ok={f['success']}"
                )

    if csv_output and fits:
        import csv as _csv
        keys = list(fits[0].keys())
        with open(csv_output, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(fits)
        click.echo(f"Saved CSV: {csv_output}")


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def contract(json_output: bool) -> None:
    """Print the RPC contract descriptor."""
    from ..api.contract import contract_descriptor

    desc = contract_descriptor()
    if json_output:
        click.echo(json.dumps(desc, indent=2))
    else:
        click.echo(f"Plugin: {desc['plugin_id']} v{desc['version']}")
        click.echo(f"Methods: {', '.join(desc['methods'])}")
