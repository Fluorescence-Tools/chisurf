"""Per-detector IRF & background calibration step for the imaging pipeline."""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "Imaging:IRF & BG"
