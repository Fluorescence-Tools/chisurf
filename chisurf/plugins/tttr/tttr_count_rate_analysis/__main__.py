"""
Entry point for running the Count Rate Analysis CLI directly.

This allows the CLI to be run with:
python -m chisurf.plugins.tttr.tttr_count_rate_analysis
"""

from chisurf.plugins.tttr.tttr_count_rate_analysis.cli import cli

if __name__ == "__main__":
    # Enable "did you mean" suggestions
    from click.core import Context

    Context.fail = lambda self, message: self._fail_with_didyoumean(message)

    # Run the CLI
    cli()
