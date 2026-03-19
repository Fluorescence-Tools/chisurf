"""
Command-line interface for lltf.

This module provides a command-line interface for fitting lifetime spectra to
fluorescence decay data.
"""

import os
import sys

import click
import yaml
from click_didyoumean import DYMGroup

from .fitter import fit_lifetime


@click.group(cls=DYMGroup)
def cli():
    """Lifetime fitter command-line interface."""
    pass


@cli.command()
@click.argument('decay_file', type=click.Path(exists=True))
@click.argument('irf_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON file')
@click.option('--plot', '-p', type=click.Path(), help='Output plot file')
@click.option('--save-path', '-sp', type=click.Path(), help='Path to save all output files (creates a subfolder named "lltf_output" in data location if not specified)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration YAML file')
@click.option('--n-lifetimes', '-n', type=int, default=1, help='Number of lifetimes to fit (ignored if --find-optimal is used)')
@click.option('--find-optimal', '-f', is_flag=True, help='Find optimal number of lifetimes automatically')
@click.option('--max-lifetimes', '-m', type=int, default=6, help='Maximum number of lifetimes to try when finding optimal')
@click.option('--prob-threshold', '-pt', type=float, default=0.68, help='Probability threshold for selecting the best number of lifetimes')
@click.option('--selection-mode', '-sm', type=click.Choice(['lower', 'upper']), default='lower', help='Mode for selecting the best number of lifetimes (lower: fewer components, upper: more components)')
@click.option('--skiprows', '-s', type=int, default=0, help='Number of rows to skip in data files')
@click.option('--delimiter', '-d', type=str, default=None, help='Delimiter used in data files')
@click.option('--time-column', '-t', type=int, default=0, help='Column index for time data')
@click.option('--counts-column', '-y', type=int, default=1, help='Column index for counts data')
@click.option('--verbose', '-v', is_flag=True, help='Print verbose output')
@click.option('--save-intermediate', '-si', is_flag=True, default=False, help='Save intermediate results when finding optimal number of lifetimes (default: disabled)')
@click.option('--intermediate-base', '-ib', type=click.Path(), help='Base filename for intermediate results (defaults to output filename without extension)')
def fit(
        decay_file,
        irf_file,
        output,
        plot,
        save_path,
        config,
        n_lifetimes,
        find_optimal,
        max_lifetimes,
        prob_threshold,
        selection_mode,
        skiprows,
        delimiter,
        time_column,
        counts_column,
        verbose,
        save_intermediate,
        intermediate_base
):
    """Fit a lifetime to decay data.

    DECAY_FILE is the path to the decay data file.
    IRF_FILE is the path to the IRF data file.
    """
    # Load configuration
    config_dict = {}
    if config:
        try:
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            click.echo(f"Error loading configuration file: {e}", err=True)
            sys.exit(1)

    # Handle save path and create output directory if needed
    output_dir = None

    if save_path is not None:
        # Use user-provided save path
        output_dir = save_path
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        click.echo(f"Using save path: {output_dir}")
    else:
        # Create a subfolder in the data file location
        data_dir = os.path.dirname(os.path.abspath(decay_file))
        output_dir = os.path.join(data_dir, "lltf_output")
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        click.echo(f"Created output directory: {output_dir}")

    # Get the base filename without path and extension
    base_filename = os.path.splitext(os.path.basename(decay_file))[0]

    # Set default output file if not specified
    if output is None:
        output = os.path.join(output_dir, f"{base_filename}_fit.json")
    elif not os.path.isabs(output):
        # If output is a relative path, make it relative to the output directory
        output = os.path.join(output_dir, output)

    # Set default plot file if not specified
    if plot is None:
        plot = os.path.join(output_dir, f"{base_filename}_fit.png")
    elif not os.path.isabs(plot):
        # If plot is a relative path, make it relative to the output directory
        plot = os.path.join(output_dir, plot)

    # Fit lifetime
    # If find_optimal is specified, add it to the config
    if find_optimal:
        if 'lifetime_fit_parameter' not in config_dict:
            config_dict['lifetime_fit_parameter'] = {}
        config_dict['lifetime_fit_parameter']['find_optimal'] = find_optimal
        config_dict['lifetime_fit_parameter']['maximum_number_of_lifetimes'] = max_lifetimes
        config_dict['lifetime_fit_parameter']['prob_threshold'] = prob_threshold
        config_dict['lifetime_fit_parameter']['selection_mode'] = selection_mode

    # Set default intermediate_base if not specified but save_intermediate is enabled
    if save_intermediate and not intermediate_base:
        # Use the same directory as the output file, but with a base name derived from the decay file
        intermediate_base = os.path.join(output_dir, f"{base_filename}_intermediate")
        click.echo(f"No intermediate base filename specified, using: {intermediate_base}")
    elif intermediate_base and not os.path.isabs(intermediate_base):
        # If intermediate_base is a relative path, make it relative to the output directory
        intermediate_base = os.path.join(output_dir, intermediate_base)

    # Determine whether to save intermediate results
    # Only save intermediates if find_optimal is True AND save_intermediate is True
    should_save_intermediates = find_optimal and save_intermediate

    # Handle save_intermediate based on find_optimal flag
    if save_intermediate and not find_optimal:
        # User explicitly enabled save_intermediate but didn't use find_optimal
        click.echo("Note: --save-intermediate is only effective with --find-optimal")
        # We don't automatically enable find_optimal anymore
        # Instead, we'll pass save_intermediate=True to fit_lifetime, but it will only
        # have an effect if find_optimal is enabled in the config

    # Inform user about saving intermediate results
    if should_save_intermediates:
        click.echo(f"Intermediate results will be saved with base filename: {intermediate_base}")
    elif save_intermediate and not find_optimal:
        click.echo("Note: Intermediate results will not be saved because --find-optimal is not enabled")

    # Always disable plotting to screen
    if 'plot_resulting_fit' in config_dict:
        config_dict['plot_resulting_fit'] = False

    result = fit_lifetime(
        decay_file=decay_file,
        irf_file=irf_file,
        n_lifetimes=n_lifetimes,
        skiprows=skiprows,
        delimiter=delimiter,
        time_column=time_column,
        counts_column=counts_column,
        output_file=output,
        plot_file=plot,
        verbose=verbose,
        config=config_dict,
        save_intermediate_results=should_save_intermediates,
        intermediate_results_base_filename=intermediate_base
    )

    click.echo(f"Fit completed successfully.")
    click.echo(f"Results saved to {output}")
    click.echo(f"Plot saved to {plot}")

    # Print summary of results
    click.echo("\nFit results summary:")
    for i in range(result['n_lifetimes']):
        lifetime = result['lifetime_spectrum'][2*i+1]
        amplitude = result['lifetime_spectrum'][2*i]
        click.echo(f"Lifetime {i+1}: {lifetime:.3f} ns, Amplitude: {amplitude:.3f}")

    click.echo(f"IRF shift: {result['irf_shift']:.3f} ns")
    click.echo(f"Decay background: {result['decay_background']:.2f}")
    click.echo(f"Reduced chi-square: {result['reduced_chi_square']:.2f}")


def main():
    """Entry point for the command-line interface."""
    cli()


if __name__ == '__main__':
    main()
