"""Count Rate Analysis plugin.

Drop TTTR files and compute the count rate per detector channel in each file,
then plot the count rate vs file and report the mean and standard deviation over
all files. Ported to the new-style AutoForm + JSON view (``gui/``): a Qt-free
:class:`~.gui.view_model.CountRateViewModel` drives the detector channel-
definition page, the file list, the count-rate plot and the results table laid
out from ``count_rate.view.json``.

Aggregated into the TTTR Tools toolbox (``tttr_toolbox``); ``menu_hidden`` keeps
it out of the ribbon but it stays importable and standalone-launchable, and the
``count-rate`` CLI is unchanged.

CLI Usage::

    csc_count_rate --help                    # Show help
    csc_count_rate list-setups SETUP_FILE    # List available detector setups
    csc_count_rate analyze FILE1 FILE2...    # Analyze TTTR files
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "TTTR:Analysis:Count Rate Analysis"

# Hidden from the ribbon (aggregated into the TTTR Tools toolbox); still
# importable, standalone-launchable and CLI-exposed.
menu_hidden = True

# Expose the plugin CLI through chisurf.core.cli.
cli_entrypoint = "count-rate=chisurf.plugins.tttr.tttr_count_rate_analysis.cli:cli"

__all__ = ["CountRateAnalyzer"]


def __getattr__(attr_name: str):
    """Lazy Qt import gate (no Qt import as a package side effect)."""
    if attr_name == "CountRateAnalyzer":
        from .gui.tool import CountRateAnalyzer as _cls

        globals()["CountRateAnalyzer"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    from .gui.tool import CountRateAnalyzer

    window = CountRateAnalyzer()
    window.show()
