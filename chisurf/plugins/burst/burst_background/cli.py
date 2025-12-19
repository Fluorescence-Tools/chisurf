"""Command Line Interface for Burst Background Estimation.

This module provides a CLI for estimating background count rates
from TTTR files using detector setups defined via the
DetectorWizardPage JSON format.
"""

import json
import os
from typing import Any, Dict, List

import click
import numpy as np
import tttrlib

import chisurf.fluorescence.burst as cs_burst

# Import from chisurf if available, otherwise handle standalone usage
try:  # pragma: no cover - convenience import
    from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups
except Exception:  # pragma: no cover - standalone / minimal environment

    def load_detector_setups(file_path: str) -> Dict[str, Any]:
        """Load detector setups from a JSON file.

        This is a minimal fallback used when the full ChiSurf GUI
        components are not importable.
        """

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover - IO error path
            click.echo(f"Error loading detector setups: {exc}", err=True)
            return {"setups": {}}


def estimate_background_for_files(
    tttr_files: List[str],
    detectors: Dict[str, Dict[str, Any]],
    *,
    binsize_ms: float = 0.1,
    tail_fraction: float = 0.2,
    min_counts: int = 1,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Estimate background rates for all files and detectors.

    Returns a mapping ``{file_path: {detector_name: background_kHz}}``.
    """

    results: Dict[str, Dict[str, float]] = {}

    if not tttr_files:
        click.echo("No TTTR files provided", err=True)
        return results

    if not detectors:
        click.echo("No detectors defined", err=True)
        return results

    for idx, path in enumerate(tttr_files):
        if verbose:
            click.echo(
                f"Processing file {idx + 1}/{len(tttr_files)}: {os.path.basename(path)}"
            )

        try:
            tttr = tttrlib.TTTR(path)
        except Exception as exc:  # pragma: no cover - TTTR load error path
            click.echo(f"Error loading {path}: {exc}", err=True)
            continue

        bg = cs_burst.estimate_background_from_bursts(
            tttr,
            detectors,
            binsize_ms=binsize_ms,
            tail_fraction=tail_fraction,
            min_counts=min_counts,
        )
        results[path] = bg

    return results


def save_results_as_txt(
    backgrounds: Dict[str, Dict[str, float]],
    output_file: str,
) -> None:
    """Save per-file, per-detector background estimates to a text file."""

    if not backgrounds:
        click.echo("No data to save", err=True)
        return

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            headers = ["File", "Detector", "Background (kHz)"]
            f.write("\t".join(headers) + "\n")

            for path, det_bg in backgrounds.items():
                fname = os.path.basename(path)
                for det_name, rate in det_bg.items():
                    row = [fname, str(det_name), f"{rate:.3f}"]
                    f.write("\t".join(row) + "\n")

        click.echo(f"Results saved to {output_file}")

    except Exception as exc:  # pragma: no cover - IO error path
        click.echo(f"Failed to save file: {exc}", err=True)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True
)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Burst Background Estimation CLI.

    Estimate background count rates (kHz) from TTTR files using
    detector setups defined for the DetectorWizard.
    """

    if version:
        click.echo("Burst Background Estimation CLI v1.0.0")
        return

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--setup-file",
    "-s",
    type=click.Path(exists=True),
    help="Path to the detector setups JSON file.",
)
@click.option(
    "--setup-name",
    "-n",
    help="Name of the setup to use from the setups file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save the results as a text file.",
)
@click.option(
    "--binsize-ms",
    type=float,
    default=0.1,
    show_default=True,
    help="Histogram bin width in milliseconds.",
)
@click.option(
    "--tail-fraction",
    type=float,
    default=0.2,
    show_default=True,
    help=(
        "Fraction of the histogram range used for the tail fit. "
        "0.2 corresponds to dt > max(dt)/5."
    ),
)
@click.option(
    "--min-counts",
    type=int,
    default=1,
    show_default=True,
    help="Minimum counts per histogram bin included in the tail fit.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def analyze(
    files: List[str],
    setup_file: str,
    setup_name: str,
    output: str,
    binsize_ms: float,
    tail_fraction: float,
    min_counts: int,
    verbose: bool,
) -> None:
    """Estimate background rates in TTTR FILES.

    FILES: One or more TTTR files to analyze.
    """

    if not files:
        click.echo(
            "No files specified. Use --help for usage information.", err=True
        )
        return

    if not setup_file:
        click.echo(
            "No setup file specified. Use --help for usage information.", err=True
        )
        return

    setups = load_detector_setups(setup_file)

    # Select setup
    if not setup_name:
        available_setups = list(setups.get("setups", {}).keys())
        if not available_setups:
            click.echo(f"No setups found in {setup_file}", err=True)
            return

        click.echo("Available setups:")
        for i, name in enumerate(available_setups, start=1):
            click.echo(f"  {i}. {name}")

        idx = click.prompt(
            "Select a setup (number)", type=int, default=1  # type: ignore[arg-type]
        )
        if idx < 1 or idx > len(available_setups):
            click.echo(
                f"Invalid selection. Please choose between 1 and {len(available_setups)}",
                err=True,
            )
            return

        setup_name = available_setups[idx - 1]

    setup_data = setups.get("setups", {}).get(setup_name)
    if not setup_data:
        click.echo(f"Setup '{setup_name}' not found in {setup_file}", err=True)
        return

    detectors = setup_data.get("detectors", {})
    if not detectors:
        click.echo(f"Setup '{setup_name}' defines no detectors", err=True)
        return

    if verbose:
        click.echo(f"Using setup: {setup_name}")
        click.echo(f"Detectors: {len(detectors)}")
        click.echo(f"Processing {len(files)} files...")

    backgrounds = estimate_background_for_files(
        list(files),
        detectors,
        binsize_ms=binsize_ms,
        tail_fraction=tail_fraction,
        min_counts=min_counts,
        verbose=verbose,
    )

    if not backgrounds:
        click.echo("No background estimates were computed", err=True)
        return

    # Summary across files per detector
    det_names = set()
    for det_bg in backgrounds.values():
        det_names.update(det_bg.keys())

    click.echo("\nResults Summary (per detector):")
    click.echo(f"Files processed: {len(backgrounds)}")
    click.echo(f"Detectors: {len(det_names)}")

    for det in sorted(det_names):
        vals: List[float] = []
        for det_bg in backgrounds.values():
            if det in det_bg:
                vals.append(det_bg[det])
        if not vals:
            continue
        vals_arr = np.asarray(vals, dtype=float)
        mean = float(vals_arr.mean())
        std = float(vals_arr.std())
        click.echo(f"  {det}: {mean:.3f} +/- {std:.3f} kHz")

    if output:
        save_results_as_txt(backgrounds, output)


@cli.command("list-setups")
@click.argument("setup_file", type=click.Path(exists=True))
def list_setups_cmd(setup_file: str) -> None:
    """List available detector setups in SETUP_FILE."""

    setups = load_detector_setups(setup_file)
    available_setups = list(setups.get("setups", {}).keys())
    if not available_setups:
        click.echo(f"No setups found in {setup_file}", err=True)
        return

    click.echo(f"Available setups in {setup_file}:")
    for i, name in enumerate(available_setups, start=1):
        setup_data = setups["setups"][name]
        windows = len(setup_data.get("windows", {}))
        detectors = len(setup_data.get("detectors", {}))
        click.echo(f"  {i}. {name} ({windows} windows, {detectors} detectors)")


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    from click.core import Context

    # Enable "did you mean" suggestions when run as a script
    Context.fail = lambda self, message: self._fail_with_didyoumean(message)

    cli()
