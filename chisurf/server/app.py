from __future__ import annotations

import logging

from chisurf.core.plugin.registry import PluginRegistry
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.eventbus import EventBus, InProcessEventBus
from chisurf.server.jobs import JobManager
from chisurf.server.service_startup import ServiceStartupManager
from chisurf.server.session import SessionState
from chisurf.server.transport.zmq import ZmqServer

_log = logging.getLogger(__name__)


class ChiSurfServer:
    """Wires together all server components with an authoritative ``SessionState``.

    Usage::

        server = ChiSurfServer(cmd_port=8765, pub_port=8766)
        # Run serve_forever in a separate thread if non-blocking operation is needed.
        ...
        server.stop()
    """

    def __init__(
        self,
        cmd_port: int = 8765,
        pub_port: int = 8766,
        host: str = "127.0.0.1",
        state: SessionState | None = None,
    ):
        """Initialise the server, wiring together all components.

        Parameters
        ----------
        cmd_port : int
            TCP port for the REQ/REP command socket.
        pub_port : int
            TCP port for the PUB event broadcast socket.
        host : str
            Bind address.
        state : SessionState, optional
            Pre-existing session state, or ``None`` to create a fresh one.

        """
        self.state = state or SessionState()
        self.state.flr_database = self._init_flr_database()
        self.job_manager = JobManager()
        self.event_bus: EventBus = InProcessEventBus()
        self._cmd_port = cmd_port
        self._pub_port = pub_port
        self._host = host
        self._zmq_server = ZmqServer(
            handler=self._zmq_dispatch,
            cmd_port=cmd_port,
            pub_port=pub_port,
            host=host,
        )
        self.dispatcher = ServiceDispatcher(self.state, event_bus=self.event_bus)
        self.dispatcher._build_default_registry()
        self.service_startup_manager = ServiceStartupManager(
            dispatcher=self.dispatcher,
            state=self.state,
            event_bus=self.event_bus,
            job_manager=self.job_manager,
        )
        self.service_startup_manager.start()
        # Auto-discover and register all plugin services from manifests.
        # Services declared in startup config files are owned by that flow.
        self.plugin_registry = PluginRegistry()
        self.plugin_registry.discover()
        self.plugin_registry.register_services(
            self.dispatcher,
            exclude_entrypoints=self.service_startup_manager.entrypoints,
        )
        # Bridge in-process event bus to ZMQ broadcast
        self.event_bus.subscribe("*", self._broadcast_bridge)

    @property
    def cmd_port(self) -> int:
        """Return the command-socket TCP port."""
        return self._cmd_port

    @property
    def pub_port(self) -> int:
        """Return the event-broadcast TCP port."""
        return self._pub_port

    @property
    def host(self) -> str:
        """Return the bind address."""
        return self._host

    def serve_forever(self) -> None:
        """Run the ZMQ event loop (blocking)."""
        _log.info(
            "ChiSurfServer starting on %s:%s (cmd) and %s:%s (pub)",
            self._host, self._cmd_port, self._host, self._pub_port,
        )
        self._zmq_server.serve_forever()

    def stop(self) -> None:
        """Gracefully stop the server."""
        _log.info("ChiSurfServer stopping")
        if hasattr(self, "service_startup_manager"):
            self.service_startup_manager.stop()
        self._zmq_server.stop()

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _init_flr_database():
        """Create or attach the FLR database."""
        from chisurf.core.fio.mmcif.db import FluorescenceDatabase
        return FluorescenceDatabase()

    def _zmq_dispatch(self, method: str, params: dict | None = None) -> dict:
        """Bridge ZMQ REQ → dispatcher."""
        return self.dispatcher.dispatch(method, params)

    def _broadcast_bridge(self, event: dict) -> None:
        """Bridge in-process event → ZMQ PUB broadcast."""
        topic = event.get("topic", "")
        self._zmq_server.broadcast_event(topic, event)
