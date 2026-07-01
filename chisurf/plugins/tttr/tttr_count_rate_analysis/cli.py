"""
Command Line Interface for Count Rate Analysis

This module provides a CLI for the Count Rate Analysis functionality,
allowing users to analyze TTTR files from the command line.
"""

import json
import os
from typing import Any

import click
import numpy as np
import tttrlib

# Import from chisurf if available, otherwise handle standalone usage
try:
    from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups
except ImportError:
    # Simplified version for standalone usage
    def load_detector_setups(file_path):
        """Load detector setups from a JSON file."""
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            click.echo(f"Error loading detector setups: {e}", err=True)
            return {"setups": {}}


def calculate_count_rates(
    tttr_files: list[str], channels: dict[str, list[dict[str, Any]]], verbose: bool = False
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, float]]:
    """Calculate count rates for all files and channels.

    Args:
        tttr_files: List of paths to TTTR files
        channels: Dictionary of channel definitions
        verbose: Whether to print verbose output

    Returns:
        Tuple containing:
        - Dictionary mapping file paths to dictionaries of channel count rates
        - Dictionary mapping file paths to dictionaries of channel photon counts
        - Dictionary mapping file paths to measurement times
    """
    if not tttr_files:
        click.echo("No TTTR files provided", err=True)
        return {}, {}, {}

    if not channels:
        click.echo("No channels defined", err=True)
        return {}, {}, {}

    # Data storage
    count_rates = {}
    n_photons = {}
    measurement_times = {}

    # Process each file
    for file_idx, file_path in enumerate(tttr_files):
        if verbose:
            click.echo(
                f"Processing file {file_idx + 1}/{len(tttr_files)}: {os.path.basename(file_path)}"
            )

        try:
            # Load TTTR file
            tttr = tttrlib.TTTR(file_path)

            # Get macro time resolution (in seconds)
            header = tttr.header
            macro_time_resolution = header.macro_time_resolution

            # Calculate count rates for each channel
            file_count_rates = {}
            file_n_photons = {}
            measurement_time = tttr.macro_times[-1] * macro_time_resolution  # in seconds

            # Store measurement time for this file
            measurement_times[file_path] = measurement_time

            for channel_name, channel_info_list in channels.items():
                channel_count_rates = []
                channel_n_photons = []

                for channel_info in channel_info_list:
                    # Extract channel parameters
                    window_range = channel_info["window_range"]
                    detector_chs = channel_info["detector_chs"]
                    micro_time_range = channel_info["micro_time_range"]

                    # Filter TTTR data by detector channels
                    tttr_filtered = tttr.get_tttr_by_channel(detector_chs)

                    # Filter by micro time range if specified
                    if micro_time_range:
                        micro_times = tttr_filtered.micro_times
                        micro_mask = (micro_times >= micro_time_range[0]) & (
                            micro_times <= micro_time_range[1]
                        )
                        # Convert int64 indices to int32 to avoid TypeError
                        selection_indices = np.where(micro_mask)[0].astype(np.int32)
                        tttr_filtered = tttr_filtered.get_tttr_by_selection(selection_indices)

                    # Filter by macro time window if specified
                    if window_range:
                        micro_times = tttr_filtered.micro_times
                        macro_mask = (micro_times >= window_range[0]) & (
                            micro_times <= window_range[1]
                        )
                        # Convert int64 indices to int32 to avoid TypeError
                        selection_indices = np.where(macro_mask)[0].astype(np.int32)
                        tttr_filtered = tttr_filtered.get_tttr_by_selection(selection_indices)

                    # Calculate count rate (photons per second)
                    n_photons_channel = len(tttr_filtered.macro_times)
                    channel_n_photons.append(n_photons_channel)

                    if measurement_time > 0:
                        count_rate = n_photons_channel / measurement_time  # in Hz
                        channel_count_rates.append(count_rate)

                # Store average count rate and total photons for this channel
                if channel_count_rates:
                    file_count_rates[channel_name] = np.mean(channel_count_rates)
                    file_n_photons[channel_name] = sum(channel_n_photons)
                else:
                    file_count_rates[channel_name] = 0.0
                    file_n_photons[channel_name] = 0

            # Store count rates and photon counts for this file
            count_rates[file_path] = file_count_rates
            n_photons[file_path] = file_n_photons

            if verbose:
                click.echo(
                    f"  Processed {os.path.basename(file_path)}: {measurement_time:.2f}s, {sum(file_n_photons.values())} photons"
                )

        except Exception as e:
            click.echo(f"Error processing {file_path}: {str(e)}", err=True)

    return count_rates, n_photons, measurement_times


def get_channels_from_setup(setup_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract channel definitions from setup data.

    Args:
        setup_data: Dictionary containing windows, detectors, and tttr_reading settings

    Returns:
        Dictionary mapping channel names to lists of channel info dictionaries
    """
    channels = {}
    windows = setup_data.get("windows", {})
    detectors = setup_data.get("detectors", {})

    for wname, wrange in windows.items():
        for dname, dinfo in detectors.items():
            cname = f"{wname}_{dname}"
            channels[cname] = []
            for mtr in dinfo["micro_time_ranges"]:
                channels[cname].append(
                    {"window_range": wrange, "detector_chs": dinfo["chs"], "micro_time_range": mtr}
                )

    return channels


def save_results_as_txt(
    count_rates: dict[str, dict[str, float]],
    n_photons: dict[str, dict[str, int]],
    measurement_times: dict[str, float],
    output_file: str,
) -> None:
    """Save the results to a text file.

    Args:
        count_rates: Dictionary mapping file paths to dictionaries of channel count rates
        n_photons: Dictionary mapping file paths to dictionaries of channel photon counts
        measurement_times: Dictionary mapping file paths to measurement times
        output_file: Path to the output file
    """
    if not count_rates:
        click.echo("No data to save", err=True)
        return

    try:
        with open(output_file, "w") as f:
            # Get all channel names
            all_channels = set()
            for file_rates in count_rates.values():
                all_channels.update(file_rates.keys())

            # Calculate statistics for each channel
            channel_stats = {}
            for channel in all_channels:
                # Get count rates for this channel across all files
                rates = [file_rates.get(channel, 0.0) for file_rates in count_rates.values()]
                mean_rate = np.mean(rates) / 1000.0  # Convert to kHz
                std_rate = np.std(rates) / 1000.0  # Convert to kHz

                # Get photon counts for this channel across all files
                photons = [file_photons.get(channel, 0) for file_photons in n_photons.values()]
                total_photons = np.sum(photons)

                # Get measurement times for all files
                times = list(measurement_times.values())
                total_time = np.sum(times)

                # Store statistics
                channel_stats[channel] = (mean_rate, std_rate, total_photons, total_time)

            # Write header
            headers = [
                "Channel",
                "Mean Count Rate (kHz)",
                "Std Count Rate (kHz)",
                "#Photons",
                "Measurement Time (s)",
            ]
            f.write("\t".join(headers) + "\n")

            # Write data rows
            for channel, stats in channel_stats.items():
                mean_rate, std_rate, total_photons, total_time = stats
                row_data = [
                    channel,
                    f"{mean_rate:.2f}",
                    f"{std_rate:.2f}",
                    f"{total_photons:.0f}",
                    f"{total_time:.3f}",
                ]
                f.write("\t".join(row_data) + "\n")

        click.echo(f"Results saved to {output_file}")

    except Exception as e:
        click.echo(f"Failed to save file: {str(e)}", err=True)


# Set up the Click command group with "did you mean" suggestions
@click.group(context_settings=dict(help_option_names=["-h", "--help"]), invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx, version):
    """Count Rate Analysis CLI - Analyze count rates in TTTR files.

    This tool allows you to calculate count rates from TTTR files using
    predefined detector and window settings.
    """
    if version:
        click.echo("Count Rate Analysis CLI v1.0.0")
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
@click.option("--setup-name", "-n", help="Name of the setup to use from the setups file.")
@click.option("--output", "-o", type=click.Path(), help="Path to save the results.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def analyze(files, setup_file, setup_name, output, verbose):
    """Analyze count rates in TTTR files.

    FILES: One or more TTTR files to analyze.
    """
    if not files:
        click.echo("No files specified. Use --help for usage information.", err=True)
        return

    if not setup_file:
        click.echo("No setup file specified. Use --help for usage information.", err=True)
        return

    # Load detector setups
    setups = load_detector_setups(setup_file)

    if not setup_name:
        # List available setups if no setup name is provided
        available_setups = list(setups.get("setups", {}).keys())
        if not available_setups:
            click.echo(f"No setups found in {setup_file}", err=True)
            return

        click.echo("Available setups:")
        for i, name in enumerate(available_setups):
            click.echo(f"  {i + 1}. {name}")

        # Prompt user to select a setup
        setup_idx = click.prompt("Select a setup (number)", type=int, default=1)
        if setup_idx < 1 or setup_idx > len(available_setups):
            click.echo(
                f"Invalid selection. Please choose a number between 1 and {len(available_setups)}",
                err=True,
            )
            return

        setup_name = available_setups[setup_idx - 1]

    # Get the selected setup
    setup_data = setups.get("setups", {}).get(setup_name)
    if not setup_data:
        click.echo(f"Setup '{setup_name}' not found in {setup_file}", err=True)
        return

    # Get channels from setup
    channels = get_channels_from_setup(setup_data)

    if verbose:
        click.echo(f"Using setup: {setup_name}")
        click.echo(f"Found {len(channels)} channels")
        click.echo(f"Processing {len(files)} files...")

    # Calculate count rates
    count_rates, n_photons, measurement_times = calculate_count_rates(files, channels, verbose)

    # Print summary
    all_channels = set()
    for file_rates in count_rates.values():
        all_channels.update(file_rates.keys())

    click.echo("\nResults Summary:")
    click.echo(f"Files processed: {len(count_rates)}")
    click.echo(f"Channels: {len(all_channels)}")

    # Calculate and print statistics for each channel
    for channel in sorted(all_channels):
        rates = [file_rates.get(channel, 0.0) for file_rates in count_rates.values()]
        mean_rate = np.mean(rates) / 1000.0  # Convert to kHz
        std_rate = np.std(rates) / 1000.0  # Convert to kHz
        click.echo(f"  {channel}: {mean_rate:.2f} ± {std_rate:.2f} kHz")

    # Save results if output file is specified
    if output:
        save_results_as_txt(count_rates, n_photons, measurement_times, output)


@cli.command()
@click.argument("setup_file", type=click.Path(exists=True))
def list_setups(setup_file):
    """List available detector setups in a setup file.

    SETUP_FILE: Path to the detector setups JSON file.
    """
    # Load detector setups
    setups = load_detector_setups(setup_file)

    # List available setups
    available_setups = list(setups.get("setups", {}).keys())
    if not available_setups:
        click.echo(f"No setups found in {setup_file}", err=True)
        return

    click.echo(f"Available setups in {setup_file}:")
    for i, name in enumerate(available_setups):
        setup_data = setups["setups"][name]
        windows = len(setup_data.get("windows", {}))
        detectors = len(setup_data.get("detectors", {}))
        click.echo(f"  {i + 1}. {name} ({windows} windows, {detectors} detectors)")


if __name__ == "__main__":
    # Enable "did you mean" suggestions
    from click.core import Context

    Context.fail = lambda self, message: self._fail_with_didyoumean(message)

    # Run the CLI
    cli()
