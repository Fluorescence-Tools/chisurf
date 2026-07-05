"""Qt-free-ish view-model backing the onboarding wizard.

Holds the small amount of state the onboarding wizard needs and exposes the
``source`` methods (HTML for the info panels), the zero-arg ``action`` methods
(the button-row callbacks) and the ``complete_when`` booleans that drive the step
✓ marks. All layout lives in ``boarding.view.json``; this class carries no Qt
layout code (it only opens auxiliary windows on demand, importing Qt lazily).

Mirrors :class:`chisurf.plugins.tttr.ptu_alex_creator.gui.view_model.AlexViewModel`.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

from . import utils

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "boarding.view.json"

_WELCOME_HTML = """
<h3>Welcome to ChiSurf</h3>
<p>This wizard helps you get ChiSurf ready for your data and workflows. It focuses
on the user settings folder (<code>~/.chisurf</code>) and a few key configuration
files.</p>
<p>What you can do here:</p>
<ol>
<li>Check whether your settings files exist and look sane.</li>
<li>Create or restore defaults if something is missing or broken.</li>
<li>Configure TTTR detector setups and (optionally) FCS channel presets.</li>
</ol>
<p>You can run this wizard any time from the Plugin Manager.</p>
"""

_REPAIR_INTRO_HTML = """
<p>Use this page if ChiSurf reports missing settings, or to reset to a known-good
baseline. <b>Create missing files</b> only writes files that do not exist yet;
<b>Restore defaults</b> replaces your current user settings files with packaged
defaults.</p>
"""

_DETECTOR_INTRO_HTML = """
<p>TTTR experiments (TCSPC/PIE) need a detector/channel definition. Define a setup
once for your hardware below and save it — it is stored in your user settings.</p>
"""

_FCS_INTRO_HTML = """
<p>For burst-wise FCS and correlator tools, ChiSurf needs to know which detector
channels (or channel pairs) belong to a correlation setup. Create at least one
setup below and save it.</p>
"""

_FINISH_HTML = """
<h3>You are ready to start working</h3>
<p>Recommended next steps:</p>
<ol>
<li>Open Settings and verify paths/options (working directory, plugins, GUI preferences).</li>
<li>For TTTR data: define detector setups (PIE/windows) once per hardware configuration.</li>
<li>For burst-wise FCS: define at least one FCS channel preset.</li>
<li>Use Help to browse experiment-specific guides and built-in documentation.</li>
</ol>
<p>You can close this wizard now; it does not need to stay open.</p>
"""


class BoardingViewModel:
    """State + view wiring for the onboarding wizard (helpers live in ``utils``)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``boarding.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        self._observers: list[Callable[[str], None]] = []
        self._repair_status_html = ""

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers (the host refreshes info panels / ✓ marks)."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("boarding observer failed", exc_info=True)

    def refresh(self) -> None:
        """Button hook: re-read the status/deps panels (via observer refresh)."""
        self.notify("refresh")

    # ── info sources (HTML) ─────────────────────────────────────────────
    def welcome_html(self) -> str:
        """Return the welcome-step introduction (HTML)."""
        return _WELCOME_HTML

    def repair_intro_html(self) -> str:
        """Return the fix/initialize-step explanation (HTML)."""
        return _REPAIR_INTRO_HTML

    def detector_intro_html(self) -> str:
        """Return the detector-step introduction (HTML)."""
        return _DETECTOR_INTRO_HTML

    def fcs_intro_html(self) -> str:
        """Return the FCS-step introduction (HTML)."""
        return _FCS_INTRO_HTML

    def finish_html(self) -> str:
        """Return the finish-step next-steps text (HTML)."""
        return _FINISH_HTML

    def status_html(self) -> str:
        """Return the live settings-files status table (HTML)."""
        return utils.build_status_html()

    def deps_html(self) -> str:
        """Return the live optional-dependency status table (HTML)."""
        return utils.build_deps_html()

    def repair_status_html(self) -> str:
        """Return the outcome of the last create/restore action (HTML)."""
        return self._repair_status_html

    def experiments_intro_html(self) -> str:
        """Return the experiment-update-step explanation (HTML)."""
        return (
            "<p>Fitting models (e.g. the AutoForm PDA models) are registered from "
            "<code>experiment_configs.yaml</code> in your settings folder. If ChiSurf "
            "ships new or updated models, re-sync your copy so they appear in the "
            "add-fit list.</p>"
            "<p>This updates only <code>experiment_configs.yaml</code>; all other "
            "settings are left untouched. A restart is required afterwards.</p>"
        )

    # ── completion booleans (complete_when) ────────────────────────────
    @property
    def settings_ok(self) -> bool:
        """Whether the user settings folder and main settings file exist."""
        p = utils.settings_paths()
        return p["user_settings_dir"].exists() and p["settings_chisurf_yaml"].is_file()

    @property
    def has_detector_setups(self) -> bool:
        """Whether any detector setup exists (MFDB-aware, not just the JSON file)."""
        return bool(utils.detector_setups_summary().get("count"))

    @property
    def has_fcs_setups(self) -> bool:
        """Whether any FCS channel setup exists (MFDB-aware, not just the JSON file)."""
        return bool(utils.fcs_setups_summary().get("count"))

    # ── actions (button-row callbacks) ─────────────────────────────────
    def open_settings_dir(self) -> None:
        """Open the user settings folder in the OS file manager."""
        utils.open_in_file_manager(utils.settings_paths()["user_settings_dir"])

    def create_missing(self) -> None:
        """Create any missing settings files (never overwrites existing ones)."""
        ok, msg = utils.copy_defaults(overwrite=False)
        self._set_repair_status(ok, msg)

    def overwrite_defaults(self) -> None:
        """Overwrite the user settings files with the packaged defaults."""
        ok, msg = utils.copy_defaults(overwrite=True)
        self._set_repair_status(ok, msg)

    def update_experiments(self) -> None:
        """Re-sync the user experiment configuration so new models appear."""
        ok, msg = utils.update_experiment_config()
        self._set_repair_status(ok, msg)

    def _set_repair_status(self, ok: bool, msg: str) -> None:
        color = "#2e7d32" if ok else "#c62828"
        self._repair_status_html = f"<span style='color:{color}; font-weight:600'>{msg}</span>"
        self.notify("refresh")

    def open_settings_editor(self) -> None:
        """Open the ChiSurf settings editor in a modeless window."""
        try:
            from qtpy import QtCore, QtWidgets

            import chisurf as cs
            from chisurf.gui.widgets.settings_editor import SettingsEditor

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("ChiSurf Settings")
            dialog.setModal(False)
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            layout = QtWidgets.QVBoxLayout(dialog)
            editor = SettingsEditor(
                filename=str(utils.settings_paths()["settings_chisurf_yaml"]),
                window_title="ChiSurf Settings",
            )
            layout.addWidget(editor)
            dialog.resize(1000, 700)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            cs.__init_chisurf_settings_editor__ = dialog
        except Exception:
            logger.warning("boarding: could not open settings editor", exc_info=True)

    def open_help(self) -> None:
        """Open the ChiSurf help browser, if the main GUI is available."""
        try:
            import chisurf as cs

            gui = getattr(cs, "cs", None)
            if gui is not None and hasattr(gui, "onOpenHelp"):
                gui.onOpenHelp()
        except Exception:
            logger.debug("boarding: open help failed", exc_info=True)

    def open_plugin_manager(self) -> None:
        """Open the Plugin Manager window."""
        try:
            import importlib

            import chisurf as cs

            pm = importlib.import_module("chisurf.plugins.core.plugin_manager")
            w = pm.PluginManagerWidget()
            w.show()
            cs.__init_chisurf_plugin_manager__ = w
        except Exception:
            logger.debug("boarding: open plugin manager failed", exc_info=True)


__all__ = ["BoardingViewModel"]
