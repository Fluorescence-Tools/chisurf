"""AutoForm-based editor for the slow-storage staging settings.

The fields are declared in ``staged_loading_view.json`` and bound (via
``attr``) to :class:`DataLoadingSettingsModel`, whose property setters persist
each change to ``settings_chisurf.yaml`` and the live ``cs_settings`` dict (see
:func:`chisurf.core.settings.settings_utils.set_data_loading_settings`). The
byte-valued core settings are exposed here in friendlier units (MB).
"""

from __future__ import annotations

import pathlib

from chisurf.core.fio import staging
from chisurf.core.settings.settings_utils import set_data_loading_settings
from chisurf.gui import QtWidgets
from chisurf.gui.autoform import AutoForm

_VIEW_JSON = pathlib.Path(__file__).with_name("staged_loading_view.json")
_MIB = 1024 * 1024


class DataLoadingSettingsModel:
    """Backing model for the data-loading settings AutoForm.

    Properties mirror the ``data_loading`` settings section; setters persist
    immediately. Values that the core stores in bytes are presented in MB.
    """

    def view_spec(self):
        """Return the AutoForm view-spec loaded from ``staged_loading_view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _cfg() -> dict:
        return staging._settings()

    @staticmethod
    def _save(key: str, value) -> None:
        set_data_loading_settings({key: value})

    # -- bound fields ----------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether slow-storage staging is enabled."""
        return bool(self._cfg()["enabled"])

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._save("enabled", bool(value))

    @property
    def threshold_mbps(self) -> float:
        """Throughput (MB/s) below which a source is staged locally."""
        return float(self._cfg()["threshold_mbps"])

    @threshold_mbps.setter
    def threshold_mbps(self, value: float) -> None:
        self._save("threshold_mbps", float(value))

    @property
    def min_size_mb(self) -> float:
        """Minimum file size (MB) for staging to apply."""
        return float(self._cfg()["min_size"]) / _MIB

    @min_size_mb.setter
    def min_size_mb(self, value: float) -> None:
        self._save("min_size", int(round(float(value) * _MIB)))

    @property
    def chunk_mb(self) -> float:
        """Read/copy chunk size in MB."""
        return float(self._cfg()["chunk_bytes"]) / _MIB

    @chunk_mb.setter
    def chunk_mb(self, value: float) -> None:
        self._save("chunk_bytes", int(round(float(value) * _MIB)))

    @property
    def probe_mb(self) -> float:
        """Throughput-probe head size in MB."""
        return float(self._cfg()["probe_bytes"]) / _MIB

    @probe_mb.setter
    def probe_mb(self, value: float) -> None:
        self._save("probe_bytes", int(round(float(value) * _MIB)))


class DataLoadingSettingsWidget(QtWidgets.QWidget):
    """Renders :class:`DataLoadingSettingsModel` via :class:`AutoForm`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = DataLoadingSettingsModel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        self.form = AutoForm(self.model, parent=self)
        layout.addWidget(self.form)
