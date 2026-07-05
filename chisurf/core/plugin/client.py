from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol


class PluginClient(Protocol):
    """Protocol that every plugin client must implement.

    This is the **only** way GUI code should communicate with the backend.
    """

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute an RPC method and return the parsed response.

        Parameters
        ----------
        method : str
            Dotted method name (e.g. ``"burst_selection.jobs.analyze_files"``).
        params : dict, optional
            Method parameters.
        timeout : float, optional
            Request timeout in seconds.

        Returns
        -------
        dict
            The response dict (``{"ok": True, ..., ...}``).

        Raises
        ------
        RemoteError
            On transport/protocol failures or application-level errors.

        """
        ...

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """Subscribe to event topic glob.

        Parameters
        ----------
        topic : str
            Topic string. May contain ``*`` and ``?`` wildcards.
        callback : callable
            Callback receiving the event dict.

        Returns
        -------
        str
            Subscription token for :meth:`unsubscribe`.

        """
        ...

    def unsubscribe(self, token: str) -> None:
        """Remove a subscription by token.

        Parameters
        ----------
        token : str
            Token returned by :meth:`subscribe`.

        """
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected to a server."""
        ...


class InProcessClient:
    """In-process client wrapping a ServiceDispatcher directly.

    For development and contract tests only. **Never used in production.**
    Same API as ``ZmqClient``, but no network involved.
    """

    def __init__(self, dispatcher: Any):
        """Wrap a ``ServiceDispatcher``.

        Parameters
        ----------
        dispatcher : ServiceDispatcher
            The server's service dispatcher instance.

        """
        self._dispatcher = dispatcher
        self._subscriptions: Dict[str, List[tuple[str, Callable]]] = {}
        self._pattern_subs: List[tuple[str, str, Callable]] = []

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Dispatch *method* directly to the ServiceDispatcher.

        Parameters
        ----------
        method : str
            Dotted method name.
        params : dict, optional
            Method parameters.
        timeout : float, optional
            Ignored (no network).

        Returns
        -------
        dict
            Service result dict.

        """
        return self._dispatcher.dispatch(method, params)

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """Register a subscription.

        Uses fnmatch-style pattern matching.
        """
        import fnmatch
        import uuid

        token = str(uuid.uuid4())
        if "*" in topic or "?" in topic:
            self._pattern_subs.append((topic, token, callback))
        else:
            self._subscriptions.setdefault(topic, []).append((token, callback))
        return token

    def unsubscribe(self, token: str) -> None:
        """Remove a subscription by token."""
        for topic in list(self._subscriptions):
            self._subscriptions[topic] = [
                (t, h) for t, h in self._subscriptions[topic] if t != token
            ]
            if not self._subscriptions[topic]:
                del self._subscriptions[topic]
        self._pattern_subs = [
            (p, t, h) for p, t, h in self._pattern_subs if t != token
        ]

    @property
    def is_connected(self) -> bool:
        return True
