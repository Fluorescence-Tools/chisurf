"""Boarding wizard plugin for first-run ChiSurf setup."""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Help:Boarding Wizard"


if __name__ == "plugin":
    from .wizard import show_onboarding

    show_onboarding()
