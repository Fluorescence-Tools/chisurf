"""
Command-line interface for filtered FCS filter computation.
"""
import sys
import click
from pathlib import Path

from .api import compute_filters_from_files, FilterResult


@click.group()
def cli():
    """Filtered FCS Lifetime Filter Calculator - CLI"""
    pass


@cli.command()
@click.option(
    '--total', '-t',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to total decay histogram file'
)
@click.option(
    '--species', '-s',
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to species decay histogram file (can be specified multiple times)'
)
@click.option(
    '--output', '-o',
    default='fcs_filter.json',
    type=click.Path(),
    help='Output JSON file path (default: fcs_filter.json)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def compute(total, species, output, verbose):
    """Compute fFCS filters from decay histogram files.
    
    Examples:
    
        # Two species
        fcs-filter-calc compute -t total.txt -s species1.txt -s species2.txt
        
        # Three species with custom output
        fcs-filter-calc compute -t total.txt -s sp1.txt -s sp2.txt -s sp3.txt -o my_filters.json
    """
    try:
        if verbose:
            click.echo(f"Loading total decay: {total}")
            click.echo(f"Loading {len(species)} species decay(s):")
            for i, sp in enumerate(species, 1):
                click.echo(f"  {i}. {sp}")
        
        # Compute filters
        result = compute_filters_from_files(total, species)
        
        if verbose:
            click.echo(f"\nComputed filters:")
            click.echo(f"  Species: {result.n_species}")
            click.echo(f"  TAC bins: {result.n_bins}")
            click.echo(f"  Filter shape: {result.filters.shape}")
        
        # Export
        result.to_json(output, indent=2)
        
        click.echo(f"\n✓ Filters exported to: {output}")
        
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
def info(input_file):
    """Show information about a saved filter JSON file.
    
    Example:
    
        fcs-filter-calc info fcs_filter.json
    """
    try:
        result = FilterResult.from_json(input_file)
        
        click.echo(f"Filter file: {input_file}")
        click.echo(f"=" * 60)
        click.echo(f"Species: {result.n_species}")
        click.echo(f"TAC bins: {result.n_bins}")
        click.echo(f"\nMetadata:")
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
        from .widget import FcsFilterCalculatorWidget
        
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        
        window = FcsFilterCalculatorWidget()
        window.show()
        
        sys.exit(app.exec_())
        
    except ImportError as e:
        click.echo(f"✗ GUI not available: {e}", err=True)
        click.echo("Install GUI dependencies: pip install qtpy pyqtgraph", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
