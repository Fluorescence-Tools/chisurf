"""CLI for PCH analysis.

Usage:
    chisurf pch analyze <file> [--channels CH] [--bin-time US] [--components N]
    chisurf pch fit <npz-file> [--components N]
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np


@click.group()
def cli():
    """Photon Counting Histogram (PCH) analysis."""


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--channels", default="0,2", help="Routing channels (comma-separated)")
@click.option("--bin-time", default=100.0, type=float, help="Bin time in microseconds")
@click.option("--mt-min", default=0, type=int, help="Micro time minimum")
@click.option("--mt-max", default=65535, type=int, help="Micro time maximum")
@click.option("--components", default=1, type=int, help="Number of species")
@click.option("--output", "-o", default=None, help="Output NPZ path")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def analyze(file, channels, bin_time, mt_min, mt_max, components, output, json_output):
    """Compute PCH histogram and fit a multi-species model."""
    from chisurf.plugins.pch.gui.client import PCHClient

    client = PCHClient()
    ch = [int(c.strip()) for c in channels.split(",")]

    click.echo(f"Loading {file}...")
    result = client.compute(
        filename=str(file),
        channels=ch,
        bin_time_us=bin_time,
        micro_time_min=mt_min,
        micro_time_max=mt_max,
    )

    click.echo(f"Fitting with {components} component(s)...")
    fit_result = client.fit(
        k_vals=result["k_vals"],
        p_exp=result["p_exp"],
        hist_counts=result["hist_counts"],
        total_bins=result["total_bins"],
        n_components=components,
        fit_low=0,
        fit_high=int(float(result["k_vals"][-1])),
    )

    if json_output:
        click.echo(json.dumps(fit_result, indent=2))
    else:
        click.echo("Fit results:")
        for i in range(components):
            click.echo(
                f"  Component {i + 1}: ε={fit_result['epsilons'][i]:.4f}  "
                f"⟨N⟩={fit_result['avg_Ns'][i]:.4f}  "
                f"x={fit_result['fractions'][i]:.1f}%"
            )
        click.echo(f"  χ² = {fit_result['chi2']:.2f}")
        click.echo(f"  red. χ² = {fit_result['reduced_chi2']:.3f}")

    if output:
        np.savez(
            output,
            k_vals=result["k_vals"],
            p_exp=result["p_exp"],
            p_fit=fit_result["p_fit"],
            **{k: v for k, v in fit_result.items() if k != "p_fit"},
        )
        click.echo(f"Saved to {output}")


@cli.command()
@click.argument("npz_file", type=click.Path(exists=True))
@click.option("--components", default=None, type=int, help="Number of species")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def refit(npz_file, components, json_output):
    """Re-fit PCH data from a saved NPZ file."""
    from chisurf.plugins.pch.gui.client import PCHClient

    data = np.load(npz_file)
    k_vals = data["k_vals"].tolist()
    p_exp = data["p_exp"].tolist()

    client = PCHClient()
    fit_result = client.fit(
        k_vals=k_vals,
        p_exp=p_exp,
        n_components=components or 1,
        fit_low=0,
        fit_high=int(float(k_vals[-1])),
    )

    if json_output:
        click.echo(json.dumps(fit_result, indent=2))
    else:
        click.echo("Fit results:")
        n = fit_result["n_components"]
        for i in range(n):
            click.echo(
                f"  Component {i + 1}: ε={fit_result['epsilons'][i]:.4f}  "
                f"⟨N⟩={fit_result['avg_Ns'][i]:.4f}  "
                f"x={fit_result['fractions'][i]:.1f}%"
            )
        click.echo(f"  χ² = {fit_result['chi2']:.2f}")
        click.echo(f"  red. χ² = {fit_result['reduced_chi2']:.3f}")
