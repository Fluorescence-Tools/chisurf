"""ImagingToolsTool — unified imaging panel using NavigationPanelTool.

A single **"Setup"** panel (the detector/PIE-window wizard) is the one place a
user defines channels for the whole imaging workflow.  Its definition is
published to the central ``detector_setups.*`` RPC store and pulled by every
sub-tool through a uniform ``apply_setup_settings(payload)`` adapter — so the
detector wizard is no longer duplicated across the embedded tools.

Each sub-tool still works **standalone** (with its own embedded wizard) when
launched on its own; the embedded copy is only suppressed here, where the shared
Setup panel feeds it via RPC.

Panels (lazy-loaded via factory functions):

  1. Setup            — SetupChannelDefinitionWidget (shared, publishes via RPC)
  2. Browser          — TTTRImageBrowserTool
  3. Pixel-wise MLE   — ImgPixelMleTool (embedded)
  4. Molecule-wise MLE — SmImageMleTool (embedded)
  5. CLSM Draw        — CLSMPixelSelect
  ─────────────────── (separator)
  6. PSF Determination — PsfDeterminationTool
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.widgets.navigation import NavigationPanelTool

from .client import DetectorSetupClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Panel factory functions — each imported lazily to keep startup fast.
# Every factory receives the ImagingToolsTool coordinator as its argument and
# registers the created widget so the coordinator can feed it the shared setup.
# ---------------------------------------------------------------------------

def _setup(parent: ImagingToolsTool) -> QtWidgets.QWidget:
    """Channel / detector setup — the single shared definition (RPC publisher)."""
    from chisurf.plugins.core.setup_channel_definition.gui.tool import (
        SetupChannelDefinitionWidget,
    )
    widget = SetupChannelDefinitionWidget(parent=parent)
    parent._register_setup_panel(widget)
    return widget


def _browser(parent: ImagingToolsTool) -> QtWidgets.QWidget:
    from chisurf.plugins.tttr.tttr_image_browser.gui.tool import TTTRImageBrowserTool
    widget = TTTRImageBrowserTool(parent=parent)
    parent._register_panel("browser", widget)
    return widget


def _pixel_mle(parent: ImagingToolsTool) -> QtWidgets.QWidget:
    from chisurf.plugins.microscopy.img_pixel_mle.gui.tool import ImgPixelMleTool
    widget = ImgPixelMleTool(parent=parent, embedded=True)
    parent._register_panel("pixel_mle", widget)
    return widget


def _molecule_mle(parent: ImagingToolsTool) -> QtWidgets.QWidget:
    from chisurf.plugins.microscopy.sm_image_mle.gui.tool import SmImageMleTool
    widget = SmImageMleTool(parent=parent, embedded=True)
    parent._register_panel("molecule_mle", widget)
    return widget


def _clsm_draw(parent: ImagingToolsTool) -> QtWidgets.QWidget:
    from clsmview.gui import CLSMPixelSelect
    widget = CLSMPixelSelect(parent=parent)
    parent._register_panel("clsm_draw", widget)
    return widget


def _psf(parent: ImagingToolsTool) -> QtWidgets.QWidget:
    from chisurf.plugins.microscopy.psf_determination.gui.tool import PsfDeterminationTool
    widget = PsfDeterminationTool(parent=parent)
    parent._register_panel("psf", widget)
    return widget


# ---------------------------------------------------------------------------
# Panel list
# ---------------------------------------------------------------------------

IMAGING_PANELS: list[dict] = [
    {
        "name": "1. Setup",
        "icon": "🧭",
        "description": "Define detector channels and PIE time windows once for all imaging tools.",
        "factory": _setup,
        "role": "setup",
    },
    {
        "name": "2. Browser",
        "icon": "📂",
        "description": "Browse TTTR image files and explore intensity maps.",
        "factory": _browser,
        "role": "browser",
    },
    {
        "name": "3. Pixel-wise MLE",
        "icon": "🗺️",
        "description": "Pixel-wise MLE lifetime analysis for TTTR imaging data.",
        "factory": _pixel_mle,
        "role": "pixel_mle",
    },
    {
        "name": "4. Molecule-wise MLE",
        "icon": "💠",
        "description": "Molecule-wise MLE lifetime analysis from TTTR imaging data.",
        "factory": _molecule_mle,
        "role": "molecule_mle",
    },
    {
        "name": "5. CLSM Draw",
        "icon": "✏️",
        "description": "Interactive CLSM pixel selection, ROI drawing and decay histogram extraction.",
        "factory": _clsm_draw,
        "role": "clsm_draw",
    },
    {
        "name": "────────",
        "icon": "",
        "separator": True,
        "role": "separator",
    },
    {
        "name": "PSF Determination",
        "icon": "🔭",
        "description": "3D Gaussian PSF fitting and bead detection.",
        "factory": _psf,
        "role": "psf",
    },
]


class ImagingToolsTool(NavigationPanelTool):
    """Unified imaging toolbox with a left-navigation panel.

    Holds one :class:`DetectorSetupClient`; the Setup panel publishes the active
    detector definition to the central ``detector_setups.*`` RPC store, and each
    analysis sub-tool pulls it through ``apply_setup_settings``.
    """

    def __init__(self, parent=None):
        # Initialise coordination state *before* super().__init__, because the
        # base class loads the first panel (Setup) during construction.
        self._setup_client = DetectorSetupClient()
        self._setup_page = None
        self._panels_by_role: dict[str, QtWidgets.QWidget] = {}
        super().__init__(
            title="🔬 Image Tools",
            panels=IMAGING_PANELS,
            parent=parent,
            minimum_size=(900, 600),
            initial_size=(1200, 750),
            navigation_width=210,
        )

    # ── panel registration / setup propagation ─────────────────────────
    def _register_setup_panel(self, widget: QtWidgets.QWidget) -> None:
        """Wire the shared Setup panel to publish definitions over RPC."""
        page = getattr(widget, "page", None)
        if page is None:
            return
        self._setup_page = page
        signal = getattr(page, "detectorsChanged", None)
        if signal is not None:
            try:
                signal.connect(self._on_setup_changed)
            except Exception:  # pragma: no cover - signal wiring is best-effort
                logger.debug("Could not connect detectorsChanged", exc_info=True)
        # Publish whatever the wizard currently holds so panels loaded later
        # (or already loaded) immediately see a definition.
        self._on_setup_changed()

    def _register_panel(self, role: str, widget: QtWidgets.QWidget) -> None:
        """Track a sub-tool panel and push the current setup to it."""
        self._panels_by_role[role] = widget
        self._apply_setup_to_panel(widget)

    def _on_setup_changed(self, *args) -> None:
        """Publish the current Setup definition and refresh all sub-tools."""
        if self._setup_page is None:
            return
        try:
            settings = self._setup_page.get_settings()
        except Exception:  # pragma: no cover - depends on wizard state
            logger.debug("Could not read setup settings", exc_info=True)
            return
        if settings:
            self._setup_client.set_current(settings)
        for widget in self._panels_by_role.values():
            self._apply_setup_to_panel(widget)

    def _apply_setup_to_panel(self, widget: QtWidgets.QWidget) -> None:
        """Apply the shared detector definition to one sub-tool, if it accepts it."""
        apply = getattr(widget, "apply_setup_settings", None)
        if not callable(apply):
            return
        payload = self._setup_client.get_current()
        if not payload:
            return
        try:
            apply(payload)
        except Exception:  # pragma: no cover - sub-tool apply is best-effort
            logger.debug("apply_setup_settings failed for %r", widget, exc_info=True)


__all__ = ["ImagingToolsTool"]
