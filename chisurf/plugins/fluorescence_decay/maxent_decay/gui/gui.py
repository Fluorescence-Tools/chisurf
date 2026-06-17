"""Qt/pyqtgraph GUI front-end for the MaxEnt TCSPC lifetime plugin."""

from __future__ import annotations

from typing import Optional

from .gui_actions import _MaxentActionsMixin
from .gui_data import _MaxentDataMixin
from .gui_helpers import _MaxentHelpersMixin
from .gui_mode import _MaxentModeMixin
from .gui_priors import _MaxentPriorsMixin
from .gui_plotting import _MaxentPlottingMixin
from .gui_run import _MaxentRunMixin
from .gui_ui import _MaxentUIMixin
from chisurf.plugins.fluorescence_decay.maxent_decay.core.settings import load_maxent_settings as maxent_load_settings

try:
    from qtpy import QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtWidgets = None

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


@persist_plugin_state("maxent_decay")
class MaxentDecayWidget(
    _MaxentActionsMixin,
    _MaxentDataMixin,
    _MaxentHelpersMixin,
    _MaxentModeMixin,
    _MaxentPriorsMixin,
    _MaxentRunMixin,
    _MaxentPlottingMixin,
    _MaxentUIMixin,
    QtWidgets.QMainWindow,
):  # type: ignore[misc]
    """GUI front-end for the MaxEnt lifetime MEM analysis."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MaxEnt TCSPC lifetime (dev)")
        self.resize(900, 700)

        self._irf_dataset = None
        self._prior_vec: Optional[np.ndarray] = None
        self._donly_vec: Optional[np.ndarray] = None
        self._dist_prior_vec: Optional[np.ndarray] = None
        self._last_result = None
        self._sample_stats = None

        self._t_axis = None
        self._fit_range = None

        self._mode_fret = False

        self._donor_style_missing = "QToolButton { background-color: #9b2c2c; color: white; font-weight: bold; }"
        self._donor_label_style_missing = "QLabel { color: #ff6b6b; font-weight: bold; }"
        self._donor_btn_style_normal = None
        self._donor_btn_fit_style_normal = None
        self._donor_label_style_normal = None

        # Sampling worker/thread handles background MCMC so the UI remains
        # responsive while sampling.
        self._sampling_thread = None
        self._sampling_worker = None

        # Cached settings dict loaded from the user JSON file. This is used
        # to initialize defaults and can later be edited via the JSON
        # settings editor.
        try:
            self._settings = maxent_load_settings()
        except Exception:
            self._settings = {}

        self._init_ui()


__all__ = ["MaxentDecayWidget"]
