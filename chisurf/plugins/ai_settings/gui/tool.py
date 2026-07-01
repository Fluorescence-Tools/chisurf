from __future__ import annotations

import logging
import threading

from qtpy import QtCore, QtWidgets

from chisurf.gui.autoform import AutoForm

from .model import AISettingsModel

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:

    def persist_plugin_state(name):
        """Fallback no-op decorator."""

        def decorator(cls):
            return cls

        return decorator


_LOG = logging.getLogger(__name__)


class _AsyncBridge(QtCore.QObject):
    """Run a callable on a background thread, deliver its result on the GUI thread.

    Network calls (test connection / fetch models) go through here so a slow or
    unreachable endpoint never blocks the UI. The result is marshalled back via a
    queued signal, so the model's status/state update runs on the GUI thread.
    """

    _done = QtCore.Signal(object, object)  # (done_callback, result)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._done.connect(self._deliver)

    def run(self, work, done) -> None:
        """Execute ``work()`` off-thread; call ``done(result)`` on the GUI thread."""

        def task():
            try:
                result = work()
            except Exception:  # pragma: no cover - work() handles its own errors
                _LOG.warning("AI settings background task failed", exc_info=True)
                return
            self._done.emit(done, result)

        threading.Thread(target=task, name="ai-settings-net", daemon=True).start()

    def _deliver(self, done, result) -> None:
        done(result)


@persist_plugin_state("ai_settings")
class AISettingsWidget(QtWidgets.QWidget):
    """Configure AI/LLM API settings via the AutoForm JSON view."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QtWidgets.QLabel("<h2>AI Settings</h2>")
        layout.addWidget(title)

        info = QtWidgets.QLabel(
            "Configure one endpoint per provider, with separate models for "
            "chat/code tasks and image generation."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._bridge = _AsyncBridge(self)
        self.model = AISettingsModel()
        self.model.on_change = self._refresh
        self.model.async_runner = self._bridge.run  # network off the UI thread
        self.form = AutoForm(self.model, parent=self)
        layout.addWidget(self.form)
        layout.addStretch()

    def _refresh(self) -> None:
        """Re-read model values and status into the form without a teardown.

        Called from model methods (provider switch, fetch, test, save, reset)
        that run inside Qt signal handlers, so it must not rebuild the layout.
        The deferral keeps the update out of the emitting widget's own callback.
        """
        QtCore.QTimer.singleShot(0, self._do_refresh)

    def _do_refresh(self) -> None:
        self.form.sync_fields()
        self.form.refresh_plots()
