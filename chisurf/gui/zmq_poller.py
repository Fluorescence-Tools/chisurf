from __future__ import annotations

from qtpy import QtCore


class ZmqSubscriberPoller(QtCore.QObject):
    """Main-thread poller for ZMQ subscriber events.

    Drives ``ZmqClient.drain()`` from the Qt event loop so that
    callbacks are **always** invoked in the main thread, eliminating
    thread-safety violations that cause memory corruption / SIGBUS.

    Parameters
    ----------
    client : ZmqClient
        The ZMQ client whose subscriber queue should be drained.
    poll_interval_ms : int
        How often (in ms) the queue is checked.  The default of 50 ms
        keeps latency low without busy-spinning.
    """

    def __init__(
        self,
        client: "ZmqClient",
        poll_interval_ms: int = 50,
        parent: QtCore.QObject | None = None,
    ):
        """Initialize the ZmqSubscriberPoller instance."""
        super().__init__(parent)
        self._client = client
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(poll_interval_ms)

    def _on_timer(self) -> None:
# TODO: needs docstring
        try:
            self._client.drain()
        except Exception:
            pass

    def stop(self) -> None:
        """Stop the operation."""
        self._timer.stop()
