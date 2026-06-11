from __future__ import annotations

import logging
from typing import Any


class RpcLogWriter:
    """Write logs through ZMQ JSON-RPC when available, with local fallback."""

    def __init__(
        self,
        logger_name: str,
        *,
        host: str = "127.0.0.1",
        cmd_port: int = 8765,
        timeout_ms: int = 50,
    ) -> None:
        """Initialize the RPC log writer.

        Parameters
        ----------
        logger_name : str
            Name of the logger used remotely and locally.
        host : str
            JSON-RPC server host.
        cmd_port : int
            JSON-RPC command port.
        timeout_ms : int
            Short timeout used for best-effort GUI logging.

        """
        self.logger_name = logger_name
        self.host = host
        self.cmd_port = cmd_port
        self.timeout_ms = timeout_ms
        self._client: Any | None = None
        self._rpc_available: bool | None = None
        self._local = logging.getLogger(logger_name)

    def debug(self, message: str, **extra: Any) -> None:
        """Log a debug message.

        Parameters
        ----------
        message : str
            Message text.
        **extra : Any
            Optional structured fields.

        """
        self.write("debug", message, extra or None)

    def info(self, message: str, **extra: Any) -> None:
        """Log an info message.

        Parameters
        ----------
        message : str
            Message text.
        **extra : Any
            Optional structured fields.

        """
        self.write("info", message, extra or None)

    def warning(self, message: str, **extra: Any) -> None:
        """Log a warning message.

        Parameters
        ----------
        message : str
            Message text.
        **extra : Any
            Optional structured fields.

        """
        self.write("warning", message, extra or None)

    def error(self, message: str, **extra: Any) -> None:
        """Log an error message.

        Parameters
        ----------
        message : str
            Message text.
        **extra : Any
            Optional structured fields.

        """
        self.write("error", message, extra or None)

    def exception(self, message: str, **extra: Any) -> None:
        """Log an exception message.

        Parameters
        ----------
        message : str
            Message text.
        **extra : Any
            Optional structured fields.

        """
        self.write("error", message, extra or None)

    def write(self, level: str, message: str, extra: dict[str, Any] | None = None) -> None:
        """Write a log record through JSON-RPC if possible.

        Parameters
        ----------
        level : str
            Logging level name.
        message : str
            Message text.
        extra : dict, optional
            Optional structured fields appended by the remote service.

        """
        if self._rpc_available is not False:
            try:
                client = self._get_client()
                response = client.call(
                    "log.write",
                    {
                        "level": level,
                        "message": message,
                        "logger_name": self.logger_name,
                        "extra": extra,
                    },
                )
                result = response.get("result", response)
                if response.get("error") or result.get("ok") is False:
                    raise RuntimeError(str(response.get("error") or result.get("error")))
                self._rpc_available = True
                return
            except Exception:
                self._rpc_available = False
                self._close_client()

        self._local.log(getattr(logging, level.upper(), logging.INFO), "%s%s", message, f" | {extra}" if extra else "")

    def _get_client(self) -> Any:
        """Return the cached ZMQ JSON-RPC client.

        Returns
        -------
        Any
            Connected client instance.

        """
        if self._client is None:
            from chisurf.server.transport.zmq import ZmqClient

            self._client = ZmqClient(
                host=self.host,
                cmd_port=self.cmd_port,
                timeout_ms=self.timeout_ms,
            )
        return self._client

    def _close_client(self) -> None:
        """Close and clear the cached ZMQ JSON-RPC client."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
