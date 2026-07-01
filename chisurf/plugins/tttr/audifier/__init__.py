"""TTTR Audifier plugin.

Convert TTTR photon streams to audio, with a live micro-time / lifetime waterfall
preview. Ported to the new-style AutoForm + JSON view (``gui/``): a Qt-free
:class:`~.gui.view_model.AudifierViewModel` drives the detector setup, the
per-detector/channel mixer, declarative audio + waterfall parameter panels, and
the reusable general ``waterfall`` AutoForm section, laid out from
``audifier.view.json``.

Aggregated into the TTTR Tools toolbox (``tttr_toolbox``); ``menu_hidden`` keeps
it out of the ribbon but it stays importable and standalone-launchable.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "TTTR:Analysis:Audifier"
menu_hidden = True

# Dynamic icon that changes based on plugin state (idle/playing/paused/...).
icon = "🎵"
try:
    from chisurf.plugins.tttr.audifier.dynamic_icons import get_icon_manager

    _icon_manager = get_icon_manager()

    def get_current_icon():
        """Return the current dynamic icon."""
        return _icon_manager.get_current_icon()

    def set_icon_state(state: str):
        """Set the icon state (idle, playing, paused, processing, error, loaded)."""
        _icon_manager.set_state(state)

except ImportError:  # pragma: no cover

    def get_current_icon():
        """Return the default icon (dynamic icons unavailable)."""
        return icon

    def set_icon_state(state: str):
        """No-op fallback when dynamic icons are unavailable."""


# Qt-free lifetime-analysis helpers exposed for direct access.
try:
    from chisurf.plugins.tttr.audifier.lifetime_analysis import (
        compute_lifetime_waterfall,
        lifetime_spectrum_ilt,
        plot_lifetime_waterfall,
        plot_lifetime_waterfall_multichannel,
    )
except ImportError:  # pragma: no cover
    lifetime_spectrum_ilt = None
    compute_lifetime_waterfall = None
    plot_lifetime_waterfall = None
    plot_lifetime_waterfall_multichannel = None


__all__ = [
    "TTTRAudifierWidget",
    "name",
    "menu_hidden",
    "icon",
    "get_current_icon",
    "set_icon_state",
    "load",
    "lifetime_spectrum_ilt",
    "compute_lifetime_waterfall",
    "plot_lifetime_waterfall",
    "plot_lifetime_waterfall_multichannel",
]


def load():
    """Return the plugin's main widget instance."""
    from chisurf.plugins.tttr.audifier.gui.tool import TTTRAudifierWidget

    return TTTRAudifierWidget()


def __getattr__(attr_name: str):
    """Lazy Qt import gate (no Qt import as a package side effect)."""
    if attr_name == "TTTRAudifierWidget":
        from .gui.tool import TTTRAudifierWidget as _cls

        globals()["TTTRAudifierWidget"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":  # pragma: no cover - GUI bootstrap
    widget = load()
    widget.show()
