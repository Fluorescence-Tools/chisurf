"""Command-line interface for filtered FCS filter computation."""

import sys
from pathlib import Path  # noqa: F401  (kept for click.Path users / scripts)

import click

from ..api import FilterResult, compute_filters_from_files


@click.group()
def cli():
    """Filtered FCS Lifetime Filter Calculator - CLI"""
    pass


@cli.command()
@click.option('--total', '-t', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to total decay histogram file')
@click.option('--species', '-s', multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to species decay histogram file (repeatable)')
@click.option('--output', '-o', default='fcs_filter.json', type=click.Path(),
              help='Output JSON file path (default: fcs_filter.json)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def compute(total, species, output, verbose):
    """Compute fFCS filters from decay histogram files."""
    try:
        if verbose:
            click.echo(f"Loading total decay: {total}")
            click.echo(f"Loading {len(species)} species decay(s):")
            for i, sp in enumerate(species, 1):
                click.echo(f"  {i}. {sp}")
        result = compute_filters_from_files(total, species)
        if verbose:
            click.echo("\nComputed filters:")
            click.echo(f"  Species: {result.n_species}")
            click.echo(f"  TAC bins: {result.n_bins}")
            click.echo(f"  Filter shape: {result.filters.shape}")
        result.to_json(output, indent=2)
        click.echo(f"\n✓ Filters exported to: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
def info(input_file):
    """Show information about a saved filter JSON file."""
    try:
        result = FilterResult.from_json(input_file)
        click.echo(f"Filter file: {input_file}")
        click.echo("=" * 60)
        click.echo(f"Species: {result.n_species}")
        click.echo(f"TAC bins: {result.n_bins}")
        click.echo("\nMetadata:")
        for key, value in result.metadata.items():
            click.echo(f"  {key}: {value}")
        click.echo(f"\nFilter matrix shape: {result.filters.shape}")
        click.echo(f"Reconstruction error (max): {abs(result.weighted_residuals).max():.4f}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def gui():
    """Launch the GUI version of the filter calculator."""
    try:
        from qtpy import QtWidgets

        from ..gui_parts.main_window import FcsFilterCalculatorWidget

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        window = FcsFilterCalculatorWidget()
        window.show()
        sys.exit(app.exec_())
    except ImportError as e:
        click.echo(f"✗ GUI not available: {e}", err=True)
        sys.exit(1)


def main(argv=None):
    """Console-script entry point."""
    return cli.main(args=argv, standalone_mode=False)


if __name__ == '__main__':
    cli()
