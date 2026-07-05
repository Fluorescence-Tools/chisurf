"""Click-based CLI for HYDROPRO / HYDRO++ runs (Qt-free)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..core import HydroProSettings, run_hydro
from ..core.runner import parse_diffusion_coefficient


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """HYDROPRO / HYDRO++ hydrodynamics tools."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("run")
@click.argument("structures", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-e", "--exe", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to the HYDROPRO / HYDRO++ executable.")
@click.option("-w", "--work-dir", type=click.Path(file_okay=False),
              help="Working directory for job folders (default ~/.hydropp_gui).")
@click.option("--indmode", type=int, default=1, show_default=True,
              help="1 atomic/shell, 2 residue/shell, 4 residue/bead.")
@click.option("--aer", type=float, default=2.9, show_default=True, help="AER (Å).")
@click.option("--nsig", type=int, default=6, show_default=True, help="NSIG (-1 = automatic).")
@click.option("--t", "temperature", type=float, default=20.0, show_default=True, help="T (°C).")
@click.option("--eta", type=float, default=0.01, show_default=True, help="Solvent viscosity (poise).")
@click.option("--rm", type=float, default=100000.0, show_default=True, help="Molecular weight (Da).")
@click.option("--vbar", type=float, default=0.74, show_default=True, help="Partial specific volume (cm³/g).")
@click.option("--rho", type=float, default=1.0, show_default=True, help="Solution density (g/cm³).")
@click.option("--json", "as_json", is_flag=True, help="Emit results as JSON.")
def run_cmd(structures, exe, work_dir, indmode, aer, nsig, temperature,
            eta, rm, vbar, rho, as_json) -> None:
    """Run HYDRO over one or more STRUCTURES and report diffusion coefficients."""
    settings = HydroProSettings(
        indmode=indmode, aer=aer, nsig=nsig, t=temperature,
        eta=eta, rm=rm, vbar=vbar, rho=rho,
    )
    settings.validate()
    results = run_hydro(
        [Path(s) for s in structures], settings, Path(exe),
        Path(work_dir) if work_dir else None,
        on_log=(lambda m: None) if as_json else click.echo,
    )
    rows = [{"file": r.struct_file, "diffusion_coefficient": r.diffusion_coefficient}
            for r in results]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        click.echo("")
        for r in rows:
            dc = r["diffusion_coefficient"]
            click.echo(f"{r['file']}\t{dc:.3e} cm^2/s" if dc is not None else f"{r['file']}\tN/A")


@cli.command("parse-res")
@click.argument("res_file", type=click.Path(exists=True, dir_okay=False))
def parse_res_cmd(res_file) -> None:
    """Parse a HYDRO ``*.res`` report and print the diffusion coefficient."""
    value = parse_diffusion_coefficient(Path(res_file))
    click.echo("N/A" if value is None else f"{value:.6e}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
